"""
Real Streets Pipeline for EV Routing
=====================================

This module implements the complete workflow:
1. Load customer locations from CSV
2. Group customers into exactly 10 per group
3. Assign 4 nearest charging stations per group
4. Build complete graph using Dijkstra shortest paths (real road network)
5. Run DRL model inference on each group
6. Project solution onto real street map

The key insight: instead of Euclidean distances, we use real road network
distances computed via Dijkstra's algorithm between all POI pairs.
"""

import math
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch
import networkx as nx
import requests
import polyline as pl

# DRL Model imports
from problems.EVRP import VehicleRoutingDataset
from nets.DRLModel import AttentionModel, set_decode_type
from utils.functions import torch_load_cpu


@dataclass
class GroupResult:
    """Results for a single group of 10 customers"""
    group_id: int
    depot_coords: Tuple[float, float]
    customer_coords: np.ndarray  # (10, 2) lat/lon
    station_coords: np.ndarray   # (4, 2) lat/lon
    tour_indices: List[int]       # DRL model output
    route_cost: float
    inference_time: float
    real_paths: List[List[Tuple[float, float]]]  # Real street geometries
    total_distance_m: float
    poi_latlons: List[Tuple[float, float]]  # All POIs in order: depot, depot_chg, stations, customers


class OSRMRouter:
    """Router using OSRM API for real street routing"""
    
    def __init__(self, base_url: str = "http://router.project-osrm.org"):
        self.base_url = base_url
        self.cache: Dict[Tuple[float, float, float, float], List[Tuple[float, float]]] = {}
    
    def get_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> Tuple[List[Tuple[float, float]], float]:
        """
        Get route geometry and distance between two points using OSRM
        
        Args:
            start: (lat, lon) tuple
            end: (lat, lon) tuple
            
        Returns:
            (path_coords, distance_meters)
        """
        cache_key = (start[0], start[1], end[0], end[1])
        if cache_key in self.cache:
            return self.cache[cache_key], self._geodesic_dist(start, end)
        
        try:
            # OSRM expects lon,lat format
            url = f"{self.base_url}/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}"
            params = {'overview': 'full', 'geometries': 'polyline'}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and len(data['routes']) > 0:
                    route = data['routes'][0]
                    encoded = route['geometry']
                    distance = route.get('distance', 0)  # meters
                    decoded = pl.decode(encoded)  # Returns list of (lat, lon)
                    self.cache[cache_key] = decoded
                    return decoded, distance
        except Exception:
            pass
        
        # Fallback to straight line
        fallback = [start, end]
        return fallback, self._geodesic_dist(start, end)
    
    def _geodesic_dist(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Haversine distance in meters"""
        R = 6371000.0
        lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
        lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1-a)))
        return R * c
    
    def _build_geodesic_matrix(self, coords: List[Tuple[float, float]]) -> np.ndarray:
        """Fallback: build distance matrix using geodesic (Haversine) distances"""
        N = len(coords)
        D = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = self._geodesic_dist(coords[i], coords[j])
        return D

    def build_distance_matrix(self, coords: List[Tuple[float, float]]) -> np.ndarray:
        """
        Build complete distance matrix between all points using OSRM Table API.
        
        Uses a single API call instead of N×N calls for massive speedup.
        
        Args:
            coords: List of (lat, lon) coordinates
            
        Returns:
            D: (N, N) distance matrix in meters
        """
        N = len(coords)
        
        # Build coordinates string for OSRM Table API (lon,lat format)
        coords_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
        
        try:
            # OSRM Table API - returns full N×N matrix in ONE request
            url = f"{self.base_url}/table/v1/driving/{coords_str}"
            params = {
                'annotations': 'distance'  # Request distances (not durations)
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == 'Ok' and 'distances' in data:
                    # OSRM returns distances in meters
                    D = np.array(data['distances'], dtype=np.float64)
                    # Replace None values (unreachable) with geodesic fallback
                    for i in range(N):
                        for j in range(N):
                            if D[i, j] is None or D[i, j] < 0:
                                D[i, j] = self._geodesic_dist(coords[i], coords[j])
                    return D
        except Exception as e:
            # Log but don't crash - fall back to geodesic
            print(f"OSRM Table API failed: {e}, using geodesic fallback")
        
        # Fallback to geodesic distances if API fails
        return self._build_geodesic_matrix(coords)


class CustomerGrouper:
    """Groups customers into exactly 10 per group using K-means clustering"""
    
    def __init__(self, group_size: int = 10):
        self.group_size = group_size
    
    def group_customers(self, customer_coords: np.ndarray) -> List[np.ndarray]:
        """
        Group customers into groups of exactly 10
        
        Args:
            customer_coords: (N, 2) array of (lat, lon)
            
        Returns:
            List of customer coordinate arrays, each (10, 2)
        """
        from sklearn.cluster import KMeans
        
        n_customers = len(customer_coords)
        n_groups = n_customers // self.group_size
        
        if n_groups == 0:
            raise ValueError(f"Need at least {self.group_size} customers, got {n_customers}")
        
        # Use exactly n_groups * group_size customers
        n_used = n_groups * self.group_size
        coords_used = customer_coords[:n_used]
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords_used)
        
        # Post-process to ensure exactly group_size per cluster
        groups = self._balance_clusters(coords_used, labels, n_groups)
        
        return groups
    
    def _balance_clusters(self, coords: np.ndarray, labels: np.ndarray, n_groups: int) -> List[np.ndarray]:
        """Ensure each cluster has exactly group_size members"""
        from collections import defaultdict
        
        # Group indices by label
        cluster_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_indices[label].append(idx)
        
        # Calculate centroids
        centroids = {}
        for label, indices in cluster_indices.items():
            centroids[label] = coords[indices].mean(axis=0)
        
        # Balance: move excess to nearest under-filled cluster
        balanced = {i: [] for i in range(n_groups)}
        assigned = set()
        
        # First pass: assign up to group_size to each cluster
        for label, indices in cluster_indices.items():
            for idx in indices[:self.group_size]:
                balanced[label].append(idx)
                assigned.add(idx)
        
        # Second pass: distribute remaining to nearest under-filled clusters
        remaining = [i for i in range(len(coords)) if i not in assigned]
        
        for idx in remaining:
            coord = coords[idx]
            # Find under-filled cluster nearest to this point
            best_cluster = None
            best_dist = float('inf')
            for label in range(n_groups):
                if len(balanced[label]) < self.group_size:
                    dist = np.linalg.norm(coord - centroids[label])
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = label
            if best_cluster is not None:
                balanced[best_cluster].append(idx)
                assigned.add(idx)
        
        # Build output arrays
        groups = []
        for label in range(n_groups):
            indices = balanced[label][:self.group_size]
            if len(indices) == self.group_size:
                groups.append(coords[indices])
        
        return groups


class ChargingStationManager:
    """Manages fixed charging stations and assigns nearest ones to groups"""
    
    def __init__(self, station_coords: np.ndarray):
        """
        Args:
            station_coords: (M, 2) array of station (lat, lon) coordinates
        """
        self.station_coords = station_coords
    
    def get_nearest_stations(self, group_centroid: Tuple[float, float], n_stations: int = 4) -> np.ndarray:
        """
        Get the n nearest stations to the group centroid
        
        Args:
            group_centroid: (lat, lon) of group center
            n_stations: Number of stations to return
            
        Returns:
            (n_stations, 2) array of station coordinates
        """
        centroid = np.array(group_centroid)
        distances = np.linalg.norm(self.station_coords - centroid, axis=1)
        nearest_indices = np.argsort(distances)[:n_stations]
        return self.station_coords[nearest_indices]


class RealStreetsPipeline:
    """
    Complete pipeline for EV routing with real street network
    
    Flow:
    1. Load customers from CSV
    2. Group into sets of 10
    3. For each group:
       a. Assign 4 nearest charging stations
       b. Build complete distance matrix using real roads (OSRM/Dijkstra)
       c. Create NPZ-format input for DRL model
       d. Run inference
       e. Project solution onto real streets
    """
    
    def __init__(self, 
                 station_coords: np.ndarray,
                 depot_coords: Tuple[float, float],
                 checkpoint_path: str = None,
                 device: str = None):
        """
        Args:
            station_coords: (M, 2) array of all charging station coordinates
            depot_coords: (lat, lon) of the depot
            checkpoint_path: Path to DRL model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.depot_coords = depot_coords
        self.station_manager = ChargingStationManager(station_coords)
        self.grouper = CustomerGrouper(group_size=10)
        self.router = OSRMRouter()
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = str(Path(__file__).parent / 'ExperimentalLog' / 'train' / '10' / 'rollout' / 
                                  'C10_02_42_53.467564' / 'best.pt')
        self.checkpoint_path = checkpoint_path
        
        # Model parameters
        self.model_params = {
            'embedding_dim': 128,
            'hidden_dim': 128,
            'n_encode_layers': 3,
            'normalization': 'batch',
            'tanh_clipping': 10.0,
            'start_soc': 120.0,
            't_limit': 10.0,
            'max_load': 4.0,
            'max_demand': 4,
            'velocity': 50.0,
            'num_nodes': 10,
            'charging_num': 4
        }
        
        self.model = None
        self.dataset = None
    
    def _load_model(self):
        """Load the DRL model if not already loaded"""
        if self.model is not None:
            return
        
        params = self.model_params
        
        # Create dataset for update functions
        self.dataset = VehicleRoutingDataset(
            num_samples=1,
            input_size=params['num_nodes'],
            t_limit=params['t_limit'],
            Start_SOC=params['start_soc'],
            velocity=params['velocity'],
            max_load=params['max_load'],
            max_demand=int(params['max_demand']),
            charging_num=params['charging_num'],
            seed=1234,
            args=type('obj', (), {
                'CVRP_lib_test': False,
                'num_nodes': params['num_nodes'],
                'charging_num': params['charging_num'],
                'Start_SOC': params['start_soc'],
                't_limit': params['t_limit']
            })()
        )
        
        # Create model
        self.model = AttentionModel(
            embedding_dim=params['embedding_dim'],
            hidden_dim=params['hidden_dim'],
            args=type('obj', (), {
                'Start_SOC': params['start_soc'],
                't_limit': params['t_limit'],
                'num_nodes': params['num_nodes'],
                'charging_num': params['charging_num']
            })(),
            n_encode_layers=params['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=params['normalization'],
            tanh_clipping=params['tanh_clipping'],
            update_dynamic=self.dataset.update_dynamic,
            update_mask=self.dataset.update_mask,
        ).to(self.device)
        
        # Load checkpoint
        load_path = Path(self.checkpoint_path)
        if load_path.exists():
            ckpt = torch_load_cpu(str(load_path))
            self.model.load_state_dict({**self.model.state_dict(), **ckpt.get('model', {})})
            print(f"✅ Loaded model checkpoint: {load_path}")
        else:
            print(f"⚠️ Checkpoint not found: {load_path}. Using randomly initialized model.")
        
        set_decode_type(self.model, 'greedy')
        self.model.eval()
    
    def _normalize_coords(self, coords: List[Tuple[float, float]]) -> np.ndarray:
        """
        Normalize lat/lon coordinates to [0, 100] range for DRL model
        
        Args:
            coords: List of (lat, lon) tuples
            
        Returns:
            (2, N) array with x (lon) and y (lat) normalized
        """
        lats = np.array([c[0] for c in coords], dtype=np.float64)
        lons = np.array([c[1] for c in coords], dtype=np.float64)
        
        min_lat, max_lat = lats.min(), lats.max()
        min_lon, max_lon = lons.min(), lons.max()
        
        lat_span = max(1e-9, max_lat - min_lat)
        lon_span = max(1e-9, max_lon - min_lon)
        
        y = (lats - min_lat) / lat_span * 100.0
        x = (lons - min_lon) / lon_span * 100.0
        
        return np.stack([x, y], axis=0)
    
    def _build_group_tensors(self, 
                             depot: Tuple[float, float],
                             station_coords: np.ndarray,
                             customer_coords: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build input tensors for DRL model using real road distances
        
        POI order: depot, depot_charging, stations (4), customers (10)
        Total: 16 nodes
        """
        params = self.model_params
        
        # Depot charging is near depot (slight offset)
        depot_chg = (depot[0] + 0.0005, depot[1] + 0.0005)
        
        # Build POI list in correct order
        poi_latlons = [depot, depot_chg]
        for i in range(len(station_coords)):
            poi_latlons.append((station_coords[i, 0], station_coords[i, 1]))
        for i in range(len(customer_coords)):
            poi_latlons.append((customer_coords[i, 0], customer_coords[i, 1]))
        
        N = len(poi_latlons)  # Should be 16 (1 + 1 + 4 + 10)
        
        # Build distance matrix using OSRM
        print(f"    📍 Building {N}x{N} distance matrix via real roads...")
        D = self.router.build_distance_matrix(poi_latlons)
        
        # Normalize coordinates
        xy = self._normalize_coords(poi_latlons)  # (2, N)
        
        # Normalize distances to training scale
        # Training uses Euclidean on [0,100], max ~141
        D_max = float(np.max(D[np.isfinite(D)]))
        if D_max > 200:  # Real-world meters
            D_normalized = D / D_max * 141.0
        else:
            D_normalized = D
        
        # Build slope matrix (assume flat for OSRM, or estimate from elevation if available)
        S = np.zeros((N, N), dtype=np.float32)
        
        # Build tensors
        static = torch.from_numpy(xy.astype(np.float32)).unsqueeze(0)  # (1, 2, N)
        
        # Dynamic: loads, demands, SOC, time
        loads = torch.full((1, 1, N), 1.0, dtype=torch.float32)
        
        # Generate random demands for customers only
        demands = torch.zeros((1, 1, N), dtype=torch.float32)
        customer_start = 2 + params['charging_num']  # After depot, depot_chg, and stations
        rng = np.random.RandomState(42)
        dem_values = rng.uniform(0.05, 0.15, size=params['num_nodes'])
        scale = min(1.0, 0.9 / max(1e-6, float(dem_values.sum())))
        dem_values *= scale
        demands[0, 0, customer_start:customer_start + params['num_nodes']] = torch.from_numpy(dem_values.astype(np.float32))
        
        soc = torch.full((1, 1, N), float(params['start_soc']), dtype=torch.float32)
        time_left = torch.full((1, 1, N), float(params['t_limit']), dtype=torch.float32)
        
        dynamic = torch.cat([loads, demands, soc, time_left], dim=1)  # (1, 4, N)
        
        distances = torch.from_numpy(D_normalized.astype(np.float32)).unsqueeze(0)  # (1, N, N)
        slopes = torch.from_numpy(S).unsqueeze(0)  # (1, N, N)
        
        return static, dynamic, distances, slopes, poi_latlons
    
    def process_group(self, 
                      group_id: int,
                      customer_coords: np.ndarray,
                      progress_callback=None) -> GroupResult:
        """
        Process a single group of 10 customers
        
        Args:
            group_id: Group identifier
            customer_coords: (10, 2) array of customer (lat, lon)
            progress_callback: Optional callback for progress updates
        """
        import time
        
        self._load_model()
        
        if progress_callback:
            progress_callback(f"Group {group_id + 1}: Finding nearest charging stations...")
        
        # Get group centroid
        centroid = (customer_coords[:, 0].mean(), customer_coords[:, 1].mean())
        
        # Get 4 nearest stations
        station_coords = self.station_manager.get_nearest_stations(centroid, n_stations=4)
        
        if progress_callback:
            progress_callback(f"Group {group_id + 1}: Building road network distance matrix...")
        
        # Build tensors with real road distances
        static, dynamic, distances, slopes, poi_latlons = self._build_group_tensors(
            self.depot_coords, station_coords, customer_coords
        )
        
        batch = (
            static.to(self.device),
            dynamic.to(self.device),
            distances.to(self.device),
            slopes.to(self.device),
        )
        
        if progress_callback:
            progress_callback(f"Group {group_id + 1}: Running DRL inference...")
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            pi, ll, cost = self.model(batch)
        inference_time = time.time() - start_time
        
        route_cost = cost.sum(1).cpu().numpy()[0]
        tour_indices = pi[0].detach().cpu().numpy().tolist()
        
        if progress_callback:
            progress_callback(f"Group {group_id + 1}: Fetching real street paths...")
        
        # Get real street paths for visualization
        real_paths = []
        total_distance = 0.0
        
        for i in range(len(tour_indices) - 1):
            idx_from = tour_indices[i]
            idx_to = tour_indices[i + 1]
            
            if idx_from < len(poi_latlons) and idx_to < len(poi_latlons):
                start = poi_latlons[idx_from]
                end = poi_latlons[idx_to]
                path_coords, dist = self.router.get_route(start, end)
                real_paths.append(path_coords)
                total_distance += dist
        
        return GroupResult(
            group_id=group_id,
            depot_coords=self.depot_coords,
            customer_coords=customer_coords,
            station_coords=station_coords,
            tour_indices=tour_indices,
            route_cost=float(route_cost),
            inference_time=inference_time,
            real_paths=real_paths,
            total_distance_m=total_distance,
            poi_latlons=poi_latlons
        )
    
    def run_pipeline(self, 
                     customer_coords: np.ndarray,
                     progress_callback=None) -> Dict[str, Any]:
        """
        Run complete pipeline on customer coordinates
        
        Args:
            customer_coords: (N, 2) array of all customer (lat, lon) coordinates
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with results for all groups
        """
        import time
        
        print(f"\n{'='*60}")
        print(f"🚗 Real Streets EV Routing Pipeline")
        print(f"{'='*60}")
        print(f"📍 Depot: {self.depot_coords}")
        print(f"👥 Total customers: {len(customer_coords)}")
        print(f"⚡ Available stations: {len(self.station_manager.station_coords)}")
        print(f"📦 Group size: 10 customers")
        
        # Group customers
        if progress_callback:
            progress_callback("Grouping customers into sets of 10...")
        
        groups = self.grouper.group_customers(customer_coords)
        print(f"📊 Created {len(groups)} groups of 10 customers each")
        
        # Process each group
        all_results = []
        total_start = time.time()
        
        for i, group_coords in enumerate(groups):
            print(f"\n--- Processing Group {i + 1}/{len(groups)} ---")
            result = self.process_group(i, group_coords, progress_callback)
            all_results.append(result)
            print(f"    ✅ Route cost: {result.route_cost:.4f}")
            print(f"    📏 Total distance: {result.total_distance_m/1000:.2f} km")
            print(f"    ⏱️ Inference time: {result.inference_time*1000:.1f} ms")
        
        total_time = time.time() - total_start
        
        # Aggregate results
        results = {
            'depot_coords': self.depot_coords,
            'n_groups': len(groups),
            'n_customers_total': len(groups) * 10,
            'total_time': total_time,
            'total_cost': sum(r.route_cost for r in all_results),
            'total_distance_km': sum(r.total_distance_m for r in all_results) / 1000,
            'group_results': [
                {
                    'group_id': r.group_id,
                    'depot_coords': r.depot_coords,
                    'customer_coords': r.customer_coords,
                    'station_coords': r.station_coords,
                    'tour_indices': r.tour_indices,
                    'route_cost': r.route_cost,
                    'inference_time': r.inference_time,
                    'real_paths': r.real_paths,
                    'total_distance_m': r.total_distance_m,
                    'poi_latlons': r.poi_latlons,
                    'n_customers': len(r.customer_coords),
                    'n_stations': len(r.station_coords),
                    'group_info': {
                        'customer_coords': r.customer_coords,
                        'station_coords': r.station_coords
                    }
                }
                for r in all_results
            ]
        }
        
        print(f"\n{'='*60}")
        print(f"✅ Pipeline Complete!")
        print(f"📊 Total groups: {results['n_groups']}")
        print(f"💰 Total cost: {results['total_cost']:.4f}")
        print(f"📏 Total distance: {results['total_distance_km']:.2f} km")
        print(f"⏱️ Total time: {results['total_time']:.2f}s")
        print(f"{'='*60}\n")
        
        return results


def load_customers_from_csv(csv_path: str) -> np.ndarray:
    """
    Load customer coordinates from CSV file
    
    Expected format: latitude,longitude (with or without header)
    
    Returns:
        (N, 2) array of (lat, lon) coordinates
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Try to find latitude/longitude columns
    lat_col = None
    lon_col = None
    
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['latitude', 'lat']:
            lat_col = col
        elif col_lower in ['longitude', 'lon', 'lng']:
            lon_col = col
    
    if lat_col is None or lon_col is None:
        # Assume first two columns are lat, lon
        lat_col = df.columns[0]
        lon_col = df.columns[1]
    
    coords = df[[lat_col, lon_col]].values.astype(np.float64)
    return coords


# Default NYC charging stations (50 stations across NYC)
NYC_CHARGING_STATIONS = np.array([
    [40.7580, -73.9855], [40.7484, -73.9857], [40.7527, -73.9772],
    [40.7614, -73.9776], [40.7549, -73.9840], [40.7425, -73.9883],
    [40.7282, -73.9942], [40.7359, -73.9911], [40.7193, -73.9978],
    [40.7081, -73.9571], [40.6892, -73.9442], [40.6782, -73.9442],
    [40.6501, -73.9496], [40.6732, -73.9772], [40.6455, -73.9631],
    [40.6892, -73.9819], [40.7061, -73.9969], [40.7134, -73.9893],
    [40.7580, -73.9515], [40.7489, -73.9680], [40.7648, -73.9731],
    [40.7738, -73.9654], [40.7829, -73.9654], [40.7919, -73.9545],
    [40.8006, -73.9653], [40.8075, -73.9458], [40.8176, -73.9395],
    [40.8282, -73.9257], [40.8389, -73.9429], [40.8452, -73.9328],
    [40.7282, -74.0007], [40.7359, -74.0073], [40.7435, -74.0046],
    [40.7512, -74.0044], [40.7282, -74.0073], [40.7359, -74.0139],
    [40.7127, -74.0134], [40.7058, -74.0090], [40.6988, -74.0421],
    [40.7550, -73.9230], [40.7648, -73.9176], [40.7738, -73.9065],
    [40.7829, -73.9176], [40.7919, -73.9065], [40.7648, -73.9508],
    [40.7738, -73.9564], [40.7829, -73.9508], [40.7373, -73.8795],
    [40.7282, -73.8685], [40.7580, -73.8650]
], dtype=np.float64)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EV routing with real street network')
    parser.add_argument('--csv', type=str, required=True, help='Path to customer CSV file')
    parser.add_argument('--depot-lat', type=float, default=40.7128, help='Depot latitude')
    parser.add_argument('--depot-lon', type=float, default=-74.0060, help='Depot longitude')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--output', type=str, default='results.json', help='Output JSON path')
    args = parser.parse_args()
    
    # Load customers
    customer_coords = load_customers_from_csv(args.csv)
    print(f"Loaded {len(customer_coords)} customers from {args.csv}")
    
    # Create pipeline
    pipeline = RealStreetsPipeline(
        station_coords=NYC_CHARGING_STATIONS,
        depot_coords=(args.depot_lat, args.depot_lon),
        checkpoint_path=args.checkpoint
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(customer_coords)
    
    # Save results
    import json
    
    # Convert numpy arrays for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj
    
    with open(args.output, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"Results saved to {args.output}")
