import pickle
import gzip
from pathlib import Path
from Graph_Generation.config.ev_config import *
import osmnx as ox
import networkx as nx
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
from geopy.distance import geodesic
from Graph_Generation.config.logging_config import *
from Graph_Generation.utils.logger import info, warning, error, debug
from scipy.spatial import KDTree

# Import our configurations
import sys
sys.path.append(os.path.join('..', '..'))
from Graph_Generation.config.ev_config import *

# 🚀 PHASE 1: KDTree for fast spatial queries
NYC_LOCATIONS = {
    'manhattan': (40.7831, -73.9712),
    'brooklyn': (40.6782, -73.9442),
    'queens': (40.7282, -73.7949),
    'bronx': (40.8448, -73.8648),
    'staten_island': (40.5795, -74.1502),
    'albany': (42.6526, -73.7562),
    'buffalo': (42.8864, -78.8784),
    'rochester': (43.1566, -77.6088),
    'syracuse': (43.0481, -76.1474),
    'yonkers': (40.9312, -73.8988),
    'new_rochelle': (40.9115, -73.7824),
    'long_island_nassau': (40.7407, -73.5895),
    'long_island_suffolk': (40.8834, -72.9804)
}

class NetworkDatabase:
    """Manages persistent storage and loading of road networks"""
    
    def __init__(self, db_path: str = "New_York_network_ny.pkl.gz"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.network = None
        self.metadata = {}
        # 🚀 PHASE 1: KDTree for fast spatial queries
        self._kdtree = None
        self._node_coords = None
        self._node_ids = None
    
    def network_exists(self) -> bool:
        """Check if network file exists"""
        return self.db_path.exists()
    
    def save_network(self, network: nx.MultiDiGraph, metadata: Dict = None):
        """Save network to compressed pickle file"""
        info(f"Saving network to {self.db_path}", 'road_network_db')
        
        data = {
            'network': network,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'nodes_count': len(network.nodes),
            'edges_count': len(network.edges)
        }
        
        with gzip.open(self.db_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        info(f"Network saved: {len(network.nodes)} nodes, {len(network.edges)} edges", 'road_network_db')
    
    def load_network(self) -> nx.MultiDiGraph:
        """Load network from pickle file"""
        if not self.network_exists():
            raise FileNotFoundError(f"Network file not found: {self.db_path}")
        
        info(f"Loading network from {self.db_path}", 'road_network_db')
        
        with gzip.open(self.db_path, 'rb') as f:
            data = pickle.load(f)
        
        self.network = data['network']
        self.metadata = data.get('metadata', {})
        
        info(f"Network loaded: {len(self.network.nodes)} nodes, {len(self.network.edges)} edges", 'road_network_db')
        info(f"Created: {data.get('created_at', 'Unknown')}", 'road_network_db')
        self._build_kdtree()  # Build KDTree after loading
        return self.network
    
    def validate_network_connectivity(self, test_locations: List[Tuple[float, float]]) -> float:
        """Test network connectivity between locations"""
        if not self.network:
            return 0.0
        
        successful_routes = 0
        total_tests = 0
        
        for i, origin in enumerate(test_locations):
            for j, dest in enumerate(test_locations):
                if i != j:
                    total_tests += 1
                    try:
                        origin_node = self._find_nearest_node(origin[0], origin[1])
                        dest_node = self._find_nearest_node(dest[0], dest[1])
                        
                        if origin_node and dest_node:
                            nx.shortest_path(self.network, origin_node, dest_node)
                            successful_routes += 1
                    except (nx.NetworkXNoPath, Exception):
                        pass
        
        return successful_routes / total_tests if total_tests > 0 else 0
    
    def _find_nearest_node(self, lat: float, lon: float) -> int:
        """Find nearest node in the network using KDTree if available"""
        if self._kdtree is not None and self._node_coords is not None and self._node_ids is not None:
            dist, idx = self._kdtree.query([lat, lon])
            return self._node_ids[idx]
        try:
            # Try OSMnx method first
            return ox.nearest_nodes(self.network, lon, lat)
        except:
            # Fallback: find closest node manually
            min_distance = float('inf')
            nearest_node = None
            for node_id, node_data in self.network.nodes(data=True):
                node_lat = node_data['y']
                node_lon = node_data['x']
                distance = geodesic((lat, lon), (node_lat, node_lon)).meters
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node_id
            return nearest_node
    
    def load_or_create_network(self) -> nx.MultiDiGraph:
        """Load road network from database or create new one for New York State"""
        
        # Try to load existing network from database
        if self.network_exists():
            try:
                info("📁 Loading existing New York network from database...", 'road_network_db')
                road_network = self.load_network()
                
                # Validate connectivity with New York locations
                test_locations = [
                    (40.7831, -73.9712),  # Manhattan
                    (42.6526, -73.7562),  # Albany
                    (42.8864, -78.8784),  # Buffalo
                    (40.6782, -73.9442),  # Brooklyn
                    (43.1566, -77.6088),  # Rochester
                ]
                
                connectivity = self.validate_network_connectivity(test_locations)
                info(f"📊 Loaded network connectivity: {connectivity:.2f}", 'road_network_db')
                              
                if connectivity > 0.5:  # Accept 50%+ connectivity
                    info("✅ Using existing New York network from database", 'road_network_db')
                    return road_network
                else:
                    warning("⚠️ Loaded New York network has poor connectivity, rebuilding...", 'road_network_db')
                    
            except Exception as e:
                warning(f"❌ Failed to load existing network: {e}", 'road_network_db')
        
        # Create new network if none exists or existing one is poor
        info("🔨 Creating new New York road network...", 'road_network_db')
        road_network = self._create_and_save_network()
        return road_network

    def _create_and_save_network(self) -> nx.MultiDiGraph:
        """Create new network for New York State and save to database"""
        
        network = None
        
        # Strategy 1: Chunked bbox approach for New York State
        network = self._create_chunked_bbox_network()
        if network and len(network.nodes) > 10000:
            info(f"✅ Chunked bbox successful for New York: {len(network.nodes)} nodes", 'road_network_db')
            return network
        
        # Strategy 2: Multiple overlapping city networks (merge approach)
        if network is None:
            network = self._create_merged_city_networks()
        
        # Strategy 3: State-level network with filtering
        if network is None:
            network = self._create_filtered_state_network()
        
        # Strategy 4: Enhanced mock network as fallback
        if network is None:
            info("🎭 Creating enhanced mock network for New York...", 'road_network_db')
            network = self._create_comprehensive_mock_network()
        
        # Always enhance connectivity
        if network:
            network = self._enhance_network_connectivity(network)
            
            # Save to database
            metadata = {
                'source': 'osm' if hasattr(network, 'graph') and 'crs' in network.graph else 'mock',
                'new_york_state_bounds': True,
                'connectivity_enhanced': True,
                'creation_method': 'comprehensive_new_york',
                'nodes_count': len(network.nodes),
                'edges_count': len(network.edges)
            }
            
            info("💾 Saving New York network to database...", 'road_network_db')
            self.save_network(network, metadata)
            self._build_kdtree()  # Build KDTree after creating
            return network
        else:
            raise Exception("Failed to create any network - all strategies failed")
    
    def _create_chunked_bbox_network(self) -> nx.MultiDiGraph:
        """Create network by splitting New York State into manageable chunks and merging"""
        info("🧩 Creating chunked bbox network for New York State...", 'road_network_db')
        
        # Define smaller overlapping bbox chunks across New York State
        chunks = [
            # NYC and Long Island
            (41.3, 40.5, -71.8, -74.3, "NYC Metro Area"),
            # Hudson Valley and Capital Region
            (43.0, 41.0, -73.0, -75.0, "Hudson Valley & Capital Region"),
            # Central New York
            (44.0, 42.0, -75.0, -77.0, "Central New York"),
            # Western New York
            (43.5, 42.0, -77.0, -79.8, "Western New York"),
            # North Country
            (45.1, 43.5, -73.2, -76.0, "North Country"),
        ]
        
        merged_network = None
        total_nodes = 0
        successful_chunks = 0
        
        for i, (north, south, west, east, name) in enumerate(chunks):
            try:
                info(f"📍 Loading chunk {i+1}/{len(chunks)}: {name}", 'road_network_db')
                
                chunk_network = ox.graph_from_bbox(
                    bbox=(north, south, east, west),
                    network_type='drive',
                    simplify=True,
                    retain_all=True
                )
                
                if chunk_network and len(chunk_network.nodes) > 100:
                    info(f"  ✅ {name}: {len(chunk_network.nodes)} nodes", 'road_network_db')
                    
                    if merged_network is None:
                        merged_network = chunk_network.copy()
                    else:
                        # Merge networks
                        merged_network = nx.union(merged_network, chunk_network)
                    
                    total_nodes = len(merged_network.nodes)
                    successful_chunks += 1
                    
                    info(f"  🔗 Total merged: {total_nodes} nodes ({successful_chunks} chunks)", 'road_network_db')
                else:
                    warning(f"  ⚠️ {name}: chunk too small or empty", 'road_network_db')
                    
            except Exception as e:
                warning(f"  ❌ {name}: failed - {e}", 'road_network_db')
                continue
        
        if merged_network and successful_chunks >= 3:
            info(f"🎯 Chunked network complete: {total_nodes} nodes from {successful_chunks} chunks", 'road_network_db')
            
            # Keep largest connected component
            largest_cc = max(nx.strongly_connected_components(merged_network), key=len)
            merged_network = merged_network.subgraph(largest_cc).copy()
            info(f"🔗 Largest connected component: {len(merged_network.nodes)} nodes", 'road_network_db')
            
            # Add attributes and enhance
            merged_network = self._add_network_attributes(merged_network)
            return self._enhance_network_connectivity(merged_network)
        
        return None

    def _create_merged_city_networks(self) -> Optional[nx.MultiDiGraph]:
        """Create network by merging multiple city networks in New York"""
        try:
            info("🏙️ Trying merged city networks approach for New York...", 'road_network_db')
            
            # Comprehensive New York cities for maximum coverage
            cities = [
                # Major cities - core coverage
                ("New York, New York, USA", (40.7128, -74.0060), 25000),
                ("Albany, New York, USA", (42.6526, -73.7562), 15000),
                ("Buffalo, New York, USA", (42.8864, -78.8784), 18000),
                ("Rochester, New York, USA", (43.1566, -77.6088), 15000),
                ("Syracuse, New York, USA", (43.0481, -76.1474), 12000),
                ("Yonkers, New York, USA", (40.9312, -73.8988), 10000),
            ]
            
            merged_network = None
            
            for city_name, center_point, radius in cities:
                try:
                    info(f"📍 Loading network for {city_name} (radius: {radius}m)", 'road_network_db')
                    
                    # Use point-based approach for reliability
                    city_network = ox.graph_from_point(
                        center_point,
                        dist=radius,
                        network_type='drive',
                        simplify=True
                    )
                    
                    if merged_network is None:
                        merged_network = city_network
                        info(f"  🏗️ Base network: {len(city_network.nodes)} nodes", 'road_network_db')
                    else:
                        # Merge networks
                        merged_network = nx.compose_all([merged_network, city_network])
                        info(f"  ➕ Merged network: {len(merged_network.nodes)} nodes", 'road_network_db')
                    
                except Exception as e:
                    warning(f"  ❌ Failed to load {city_name}: {e}", 'road_network_db')
                    continue
            
            if merged_network and len(merged_network.nodes) > 5000:
                info(f"✅ Merged network successful: {len(merged_network.nodes)} nodes", 'road_network_db')
                merged_network = self._add_network_attributes(merged_network)
                return merged_network
            else:
                warning("⚠️ Merged network insufficient", 'road_network_db')
                return None
                
        except Exception as e:
            warning(f"❌ Merged city networks failed: {e}", 'road_network_db')
            return None

    def _create_filtered_state_network(self) -> Optional[nx.MultiDiGraph]:
        """Create network from New York state data"""
        try:
            info("🏛️ Trying filtered state network approach for New York...", 'road_network_db')
            
            # Get New York network
            bbox_bounds = (45.1, 40.4, -71.8, -79.8)  # New York State BBox
            
            network = ox.graph_from_bbox(
                bbox=bbox_bounds,
                network_type='drive',
                simplify=True,
                retain_all=True
            )
            
            if network and len(network.nodes) > 3000:
                info(f"✅ State network successful: {len(network.nodes)} nodes", 'road_network_db')
                network = self._add_network_attributes(network)
                return network
            else:
                warning("⚠️ State network insufficient", 'road_network_db')
                return None
                
        except Exception as e:
            warning(f"❌ State network failed: {e}", 'road_network_db')
            return None

    def _create_comprehensive_mock_network(self) -> nx.MultiDiGraph:
        """Create comprehensive mock network covering full New York State"""
        info("Creating comprehensive mock network with full New York State coverage...", 'road_network_db')
        
        G = nx.MultiDiGraph()
        
        # Expanded major locations covering entire New York State
        major_hubs = NYC_LOCATIONS
        
        # Add hub nodes
        node_id = 0
        hub_nodes = {}
        
        for hub_name, (lat, lon) in major_hubs.items():
            G.add_node(node_id, y=lat, x=lon, hub=hub_name)
            hub_nodes[hub_name] = node_id
            node_id += 1
        
        # Define major highway connections (realistic NYS highways)
        major_highways = [
            ('manhattan', 'bronx', 55, 10),
            ('manhattan', 'queens', 55, 12),
            ('manhattan', 'brooklyn', 55, 8),
            ('brooklyn', 'queens', 50, 6),
            ('brooklyn', 'staten_island', 65, 10),
            ('manhattan', 'yonkers', 65, 15),
            ('yonkers', 'albany', 70, 140),
            ('albany', 'syracuse', 70, 140),
            ('syracuse', 'rochester', 70, 85),
            ('rochester', 'buffalo', 70, 75),
            ('queens', 'long_island_nassau', 60, 20),
            ('long_island_nassau', 'long_island_suffolk', 60, 30),
        ]
        
        # Add highway connections
        for hub1, hub2, speed_kmh, distance_km in major_highways:
            if hub1 in hub_nodes and hub2 in hub_nodes:
                node1 = hub_nodes[hub1]
                node2 = hub_nodes[hub2]
                
                distance_m = distance_km * 1000
                travel_time = distance_m / (speed_kmh * 1000 / 3600)
                
                # Add bidirectional edges
                G.add_edge(node1, node2, 0, 
                        length=distance_m, 
                        speed_kph=speed_kmh, 
                        travel_time=travel_time,
                        highway='primary')
                G.add_edge(node2, node1, 0, 
                        length=distance_m, 
                        speed_kph=speed_kmh, 
                        travel_time=travel_time,
                        highway='primary')
        
        # Add dense local grids around each hub
        for hub_name, (hub_lat, hub_lon) in major_hubs.items():
            hub_node = hub_nodes[hub_name]
            
            grid_size = 5
            grid_spacing = 0.005
            
            local_nodes = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    if i == 2 and j == 2:
                        local_nodes.append(hub_node)
                        continue
                    
                    lat = hub_lat + (i - 2) * grid_spacing
                    lon = hub_lon + (j - 2) * grid_spacing
                    
                    G.add_node(node_id, y=lat, x=lon, hub_area=hub_name)
                    local_nodes.append(node_id)
                    node_id += 1
            
            for i in range(grid_size):
                for j in range(grid_size):
                    current_idx = i * grid_size + j
                    current_node = local_nodes[current_idx]
                    
                    if j < grid_size - 1:
                        right_node = local_nodes[current_idx + 1]
                        self._add_local_edge(G, current_node, right_node)
                    
                    if i < grid_size - 1:
                        bottom_node = local_nodes[current_idx + grid_size]
                        self._add_local_edge(G, current_node, bottom_node)

        info(f"Created comprehensive mock network with {len(G.nodes)} nodes and {len(G.edges)} edges", 'road_network_db')
        return G

    def _add_local_edge(self, G, node1, node2, speed_kmh=40):
        """Add local street edge between two nodes with configurable speed"""
        node1_data = G.nodes[node1]
        node2_data = G.nodes[node2]
        
        coord1 = (node1_data['y'], node1_data['x'])
        coord2 = (node2_data['y'], node2_data['x'])
        
        distance = geodesic(coord1, coord2).meters
        travel_time = distance / (speed_kmh * 1000 / 3600)
        
        # Add bidirectional edges
        G.add_edge(node1, node2, 0, 
                length=distance, 
                speed_kph=speed_kmh, 
                travel_time=travel_time,
                highway='residential')
        G.add_edge(node2, node1, 0, 
                length=distance, 
                speed_kph=speed_kmh, 
                travel_time=travel_time,
                highway='residential')

    def _enhance_network_connectivity(self, network: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Enhanced connectivity improvement with New York specific bridges"""
        
        # Find disconnected components
        undirected = network.to_undirected()
        components = list(nx.connected_components(undirected))
        
        if len(components) > 1:
            info(f"🔗 Found {len(components)} disconnected components, adding bridges...", 'road_network_db')
            
            # Connect largest component to others
            largest_component = max(components, key=len)
            bridges_added = 0
            
            for component in components:
                if component != largest_component and len(component) > 5:  # Only connect significant components
                    # Find closest nodes between components
                    min_distance = float('inf')
                    best_connection = None
                    
                    # Sample nodes for performance
                    sample_size = min(20, len(component), len(largest_component))
                    component_sample = list(component)[:sample_size]
                    largest_sample = list(largest_component)[:sample_size]
                    
                    for node1 in largest_sample:
                        for node2 in component_sample:
                            try:
                                coord1 = (network.nodes[node1]['y'], network.nodes[node1]['x'])
                                coord2 = (network.nodes[node2]['y'], network.nodes[node2]['x'])
                                distance = geodesic(coord1, coord2).meters
                                
                                if distance < min_distance:
                                    min_distance = distance
                                    best_connection = (node1, node2)
                            except:
                                continue
                    
                    # Add synthetic bridge if reasonable distance
                    if best_connection and min_distance < 50000:  # Max 50km bridge
                        node1, node2 = best_connection
                        
                        # Determine bridge type and speed based on distance
                        if min_distance < 5000:  # Local connection
                            speed_kmh = 50
                            highway_type = 'synthetic_local'
                        elif min_distance < 15000:  # Regional connection
                            speed_kmh = 65
                            highway_type = 'synthetic_regional'
                        else:  # Long-distance bridge
                            speed_kmh = 80
                            highway_type = 'synthetic_bridge'
                        
                        travel_time = min_distance / (speed_kmh * 1000 / 3600)
                        
                        # Add bidirectional edges
                        network.add_edge(node1, node2, 0,
                                    length=min_distance,
                                    speed_kph=speed_kmh,
                                    travel_time=travel_time,
                                    highway=highway_type)
                        network.add_edge(node2, node1, 0,
                                    length=min_distance,
                                    speed_kph=speed_kmh,
                                    travel_time=travel_time,
                                    highway=highway_type)
                        
                        bridges_added += 1
                        info(f"🌉 Added {highway_type}: {min_distance/1000:.1f}km", 'road_network_db')
            
            info(f"✅ Added {bridges_added} synthetic connections", 'road_network_db')
        else:
            info("✅ Network is already fully connected", 'road_network_db')
        
        # Add strategic New York connections if this is a mock network
        if not hasattr(network, 'graph') or 'crs' not in network.graph:
            network = self._add_strategic_nyc_connections(network)
        
        return network

    def _add_strategic_nyc_connections(self, network: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """Add strategic connections for key New York routes (manual bridges/tunnels)"""
        info("🌉 Adding strategic New York connections...", 'road_network_db')
        # Key NYC connection points that should always be connected
        strategic_connections = [
            # Major bridge/tunnel equivalents
            (NYC_LOCATIONS['manhattan'], NYC_LOCATIONS['brooklyn'], 60, "Brooklyn Bridge equivalent"),
            (NYC_LOCATIONS['manhattan'], NYC_LOCATIONS['queens'], 65, "Queensboro Bridge equivalent"),
            (NYC_LOCATIONS['manhattan'], NYC_LOCATIONS['bronx'], 60, "Third Avenue Bridge equivalent"),
            (NYC_LOCATIONS['brooklyn'], NYC_LOCATIONS['staten_island'], 70, "Verrazzano-Narrows Bridge equivalent"),
            (NYC_LOCATIONS['manhattan'], (40.7600, -73.9910), 65, "Lincoln Tunnel equivalent"), # Approx NJ side
        ]
        connections_added = 0
        for (lat1, lon1), (lat2, lon2), speed_kmh, description in strategic_connections:
            try:
                node1 = self._find_nearest_node(lat1, lon1)
                node2 = self._find_nearest_node(lat2, lon2)
                if node1 and node2 and node1 != node2:
                    try:
                        nx.shortest_path(network, node1, node2)
                        debug(f"✅ {description} already connected", 'road_network_db')
                        continue
                    except nx.NetworkXNoPath:
                        coord1 = (network.nodes[node1]['y'], network.nodes[node1]['x'])
                        coord2 = (network.nodes[node2]['y'], network.nodes[node2]['x'])
                        distance = geodesic(coord1, coord2).meters
                        travel_time = distance / (speed_kmh * 1000 / 3600)
                        network.add_edge(node1, node2, 0,
                                    length=distance,
                                    speed_kph=speed_kmh,
                                    travel_time=travel_time,
                                    highway='strategic_connection',
                                    description=description)
                        network.add_edge(node2, node1, 0,
                                    length=distance,
                                    speed_kph=speed_kmh,
                                    travel_time=travel_time,
                                    highway='strategic_connection',
                                    description=description)
                        connections_added += 1
                        info(f"🔗 Added {description}: {distance/1000:.1f}km", 'road_network_db')
            except Exception as e:
                debug(f"Could not add {description}: {e}", 'road_network_db')
                continue
        info(f"✅ Added {connections_added} strategic New York connections", 'road_network_db')
        return network

    def get_network_info(self) -> Dict:
        """Get information about the current network"""
        if not self.network:
            if self.network_exists():
                self.load_network()
            else:
                return {"status": "No network available"}
        
        # Test connectivity
        test_locations = [
            (40.7831, -73.9712),  # Manhattan
            (42.6526, -73.7562),  # Albany
            (42.8864, -78.8784),  # Buffalo
            (40.6782, -73.9442),  # Brooklyn
            (43.1566, -77.6088),  # Rochester
        ]
        
        connectivity = self.validate_network_connectivity(test_locations)
        
        return {
            "status": "Available",
            "nodes": len(self.network.nodes),
            "edges": len(self.network.edges),
            "connectivity_score": connectivity,
            "metadata": self.metadata,
            "file_path": str(self.db_path),
            "file_exists": self.network_exists()
        }

    def _add_network_attributes(self, network):  # Add 'network' parameter
        """Add speed and travel time attributes to network edges"""
        try:
            info("Adding edge speeds and travel times...", 'road_network_db')
            network = ox.add_edge_speeds(network)  # Use the parameter
            network = ox.add_edge_travel_times(network)  # Use the parameter
            info("Successfully added network attributes", 'road_network_db')
            return network  # Return the modified network
        except Exception as e:
            warning(f"Failed to add network attributes: {e}", 'road_network_db')
            # Add basic attributes manually
            return self._add_basic_network_attributes(network)  # Pass network parameter


    def _add_basic_network_attributes(self, network):  # Add network parameter
        """Add improved speed and travel time attributes manually"""
        info("Adding improved network attributes manually...", 'road_network_db')
        for u, v, key, data in network.edges(keys=True, data=True):
            length = data.get('length', 100)  # Default 100m if missing
            highway_type = data.get('highway', 'residential')
            if isinstance(highway_type, list):
                highway_type = highway_type[0]
            # Improved speed mapping (km/h) with more types and realism
            speed_map = {
                'motorway': 105,
                'motorway_link': 90,
                'trunk': 90,
                'trunk_link': 80,
                'primary': 70,
                'primary_link': 60,
                'secondary': 55,
                'secondary_link': 50,
                'tertiary': 45,
                'tertiary_link': 40,
                'residential': 32,
                'service': 20,
                'unclassified': 40,
                'synthetic_bridge': 80,
                'synthetic_regional': 65,
                'synthetic_local': 50,
                'strategic_connection': 70,
                'secondary': 55,
                'secondary_link': 50,
                'primary': 70,
                'primary_link': 60,
                'highway': 40
            }
            speed_kmh = speed_map.get(highway_type, 40)  # Default 40 km/h
            network.edges[u, v, key]['speed_kph'] = speed_kmh
            network.edges[u, v, key]['travel_time'] = length / (speed_kmh * 1000 / 3600)
        return network  # Return the modified network
    
    def _build_kdtree(self):
        """Build KDTree for fast node search"""
        if self.network is None:
            return
        coords = []
        node_ids = []
        for node_id, node_data in self.network.nodes(data=True):
            coords.append([node_data['y'], node_data['x']])
            node_ids.append(node_id)
        if coords:
            self._kdtree = KDTree(coords)
            self._node_coords = np.array(coords)
            self._node_ids = np.array(node_ids)
        else:
            self._kdtree = None
            self._node_coords = None
            self._node_ids = None