import argparse
import gzip
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import torch

from problems.EVRP import VehicleRoutingDataset
from nets.DRLModel import AttentionModel, set_decode_type
from utils.functions import torch_load_cpu


def build_tensors_from_npz(npz_path: Path, start_soc: float, t_limit: float, max_load: float,
                           velocity_kmh: float, charging_num: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(npz_path)
    xy = data['xy']  # (2,N)
    D = data['D']  # (N,N) meters
    S = data['S']  # (N,N) slope (rise/run)
    demands_np = data['demands'] if 'demands' in data else None

    N = xy.shape[1]
    # static (1,2,N) - ALREADY in [0,100] range from build script
    static = torch.from_numpy(xy.astype(np.float32)).unsqueeze(0)

    # dynamic (1,4,N): loads, demands, SOC, time
    loads = torch.full((1, 1, N), 1.0, dtype=torch.float32)
    if demands_np is None:
        demands = torch.zeros((1, 1, N), dtype=torch.float32)
    else:
        dem = torch.from_numpy(demands_np.astype(np.float32)).view(1, 1, N)
        # Ensure depot/depot_chg/stations have zero demand
        dem[:, :, : (2 + charging_num)] = 0.0
        demands = dem
    soc = torch.full((1, 1, N), float(start_soc), dtype=torch.float32)
    time_left = torch.full((1, 1, N), float(t_limit), dtype=torch.float32)
    dynamic = torch.cat([loads, demands, soc, time_left], dim=1)

    # CRITICAL: D is in meters, but model expects it scaled similarly to training
    # The training data uses euclidean distance on [0,100] coords
    # We need to normalize D to be comparable
    # Since xy is [0,100], max euclidean distance ≈ sqrt(100^2 + 100^2) ≈ 141
    # Real distances in D are in meters (could be 1000-10000m)
    # We need to scale D to match the [0,141] range approximately
    
    # Get the max distance in D to normalize
    D_max = float(np.max(D[np.isfinite(D)]))
    if D_max > 200:  # Real-world meters
        # Scale to match training distribution: map to ~[0, 141] range
        D_normalized = D / D_max * 141.0
    else:
        D_normalized = D  # Already in training scale
    
    distances = torch.from_numpy(D_normalized.astype(np.float32)).unsqueeze(0)
    
    # S is rise/run (slope), should be fine as-is (typically -0.1 to 0.1)
    slopes = torch.from_numpy(S.astype(np.float32)).unsqueeze(0)

    return static, dynamic, distances, slopes


def load_graph(db_path: Path) -> nx.MultiDiGraph:
    with gzip.open(db_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'network' in data:
        return data['network']
    if isinstance(data, nx.MultiDiGraph):
        return data
    raise RuntimeError("Unsupported graph pickle structure")


def get_node_coords_dict(G: nx.MultiDiGraph) -> Dict[int, Tuple[float, float]]:
    coords = {}
    for n, d in G.nodes(data=True):
        if 'y' in d and 'x' in d:
            coords[n] = (float(d['y']), float(d['x']))
    return coords


def denormalize_xy(xy_normalized: np.ndarray, poi_latlons: List[Tuple[float, float]]) -> np.ndarray:
    """Reverse the [0,100] normalization using the original POI lat/lon bounds."""
    lats = np.array([lat for lat, _ in poi_latlons])
    lons = np.array([lon for _, lon in poi_latlons])
    min_lat, max_lat = float(lats.min()), float(lats.max())
    min_lon, max_lon = float(lons.min()), float(lons.max())
    lat_span = max(1e-9, max_lat - min_lat)
    lon_span = max(1e-9, max_lon - min_lon)
    # xy_normalized shape (2, N)
    x_norm = xy_normalized[0]
    y_norm = xy_normalized[1]
    lon_denorm = x_norm / 100.0 * lon_span + min_lon
    lat_denorm = y_norm / 100.0 * lat_span + min_lat
    return np.stack([lat_denorm, lon_denorm], axis=0)


def plot_solution(G: nx.MultiDiGraph, coords_dict: Dict[int, Tuple[float, float]],
                  poi_nodes: List[int], poi_latlons: List[Tuple[float, float]],
                  tour_indices: List[int], out_png: Path):
    """Plot the inferred tour over the original graph background with real street paths."""
    # Determine bbox around POIs
    lats = [lat for lat, _ in poi_latlons]
    lons = [lon for _, lon in poi_latlons]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    lat_pad = (max_lat - min_lat) * 0.25 + 1e-3
    lon_pad = (max_lon - min_lon) * 0.25 + 1e-3
    box = (min_lat - lat_pad, max_lat + lat_pad, min_lon - lon_pad, max_lon + lon_pad)

    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Draw background street network (denser for better context)
    print("📍 Drawing background street network from graph...")
    drawn = 0
    for u, v in G.edges():
        y1, x1 = coords_dict.get(u, (None, None))
        y2, x2 = coords_dict.get(v, (None, None))
        if y1 is None or y2 is None:
            continue
        if (box[0] <= y1 <= box[1] and box[2] <= x1 <= box[3]) or (box[0] <= y2 <= box[1] and box[2] <= x2 <= box[3]):
            # Draw every 3rd edge instead of every 5th for better context
            if drawn % 3 == 0:
                ax.plot([x1, x2], [y1, y2], color='#d0d0d0', linewidth=0.5, alpha=0.6, zorder=1)
            drawn += 1
    print(f"   Drew {drawn // 3} background edges")

    # Draw actual route paths on real streets with step numbers
    print("🚗 Projecting solution route onto real graph...")
    total_distance = 0.0
    for step in range(len(tour_indices) - 1):
        idx_from = tour_indices[step]
        idx_to = tour_indices[step + 1]
        if idx_from >= len(poi_nodes) or idx_to >= len(poi_nodes):
            continue
        
        node_from = poi_nodes[idx_from]
        node_to = poi_nodes[idx_to]
        
        # Compute shortest path on real graph
        try:
            path = nx.shortest_path(G, node_from, node_to, weight='length')
            # Calculate real distance
            segment_dist = 0.0
            for i in range(len(path) - 1):
                if G.has_edge(path[i], path[i+1]):
                    edge_data = G[path[i]][path[i+1]]
                    if isinstance(edge_data, dict):
                        segment_dist += edge_data.get(0, {}).get('length', 0)
                    else:
                        for k, v in edge_data.items():
                            segment_dist += v.get('length', 0)
                            break
            total_distance += segment_dist
            
            # Draw the path segments with gradient color
            path_lons = [coords_dict[n][1] for n in path]
            path_lats = [coords_dict[n][0] for n in path]
            
            # Use color gradient to show progression through tour
            color_val = step / max(1, len(tour_indices) - 2)
            route_color = plt.cm.coolwarm(color_val)
            
            ax.plot(path_lons, path_lats, color=route_color, linewidth=3.5, alpha=0.9, zorder=3, solid_capstyle='round')
            
            # Add arrow at the end of this segment
            if len(path) >= 2:
                ax.annotate('', xy=(path_lons[-1], path_lats[-1]), 
                           xytext=(path_lons[-2], path_lats[-2]),
                           arrowprops=dict(arrowstyle='->', lw=3, color=route_color, alpha=0.95), zorder=4)
            
            # Add step number at midpoint
            mid_idx = len(path_lons) // 2
            ax.text(path_lons[mid_idx], path_lats[mid_idx], str(step + 1),
                   fontsize=9, ha='center', va='center',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor=route_color, linewidth=2),
                   zorder=6, weight='bold')
            
            print(f"   Step {step + 1}: {idx_from} → {idx_to} ({segment_dist:.1f}m via {len(path)} nodes)")
            
        except nx.NetworkXNoPath:
            # Fallback: direct line if no path found
            lat1, lon1 = poi_latlons[idx_from]
            lat2, lon2 = poi_latlons[idx_to]
            ax.plot([lon1, lon2], [lat1, lat2], color='red', linewidth=3, linestyle='--', alpha=0.7, zorder=3)
            ax.annotate('', xy=(lon2, lat2), xytext=(lon1, lat1),
                       arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.9), zorder=4)
            print(f"   Step {step + 1}: {idx_from} → {idx_to} (NO PATH - direct line)")

    print(f"✓ Total real-world distance: {total_distance:.1f} meters ({total_distance/1000:.2f} km)")

    # Draw POI nodes with enhanced styling
    legend_elements = []
    for idx, (lat, lon) in enumerate(poi_latlons):
        if idx == 0:
            ax.scatter([lon], [lat], c='red', s=200, marker='s', edgecolors='darkred', linewidths=2.5, zorder=7)
            ax.text(lon, lat, 'D', fontsize=11, ha='center', va='center', color='white', weight='bold')
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                                             markersize=10, label='Depot', markeredgecolor='darkred', markeredgewidth=2))
        elif idx == 1:
            ax.scatter([lon], [lat], c='orange', s=180, marker='D', edgecolors='darkorange', linewidths=2.5, zorder=7)
            ax.text(lon, lat, 'DC', fontsize=10, ha='center', va='center', color='white', weight='bold')
            legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='orange', 
                                             markersize=10, label='Depot Charging', markeredgecolor='darkorange', markeredgewidth=2))
        elif 1 < idx <= (1 + len([x for x in poi_latlons[2:] if poi_latlons.index(x) < len(poi_latlons) and poi_latlons.index(x) <= 5])):
            ax.scatter([lon], [lat], c='green', s=150, marker='^', edgecolors='darkgreen', linewidths=2.5, zorder=7)
            ax.text(lon, lat, f'S{idx-1}', fontsize=9, ha='center', va='bottom', color='darkgreen', weight='bold')
            if idx == 2:
                legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
                                                 markersize=10, label='Charging Station', markeredgecolor='darkgreen', markeredgewidth=2))
        else:
            ax.scatter([lon], [lat], c='#9C27B0', s=120, marker='o', edgecolors='#6A1B9A', linewidths=2, zorder=7)
            ax.text(lon, lat, f'{idx-5}', fontsize=8, ha='center', va='center', color='white', weight='bold')
            if idx == 6:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#9C27B0', 
                                                 markersize=9, label='Customer', markeredgecolor='#6A1B9A', markeredgewidth=2))

    ax.set_xlabel('Longitude', fontsize=13, weight='bold')
    ax.set_ylabel('Latitude', fontsize=13, weight='bold')
    ax.set_title(f'EM-EVRP Solution Projected on Real NYC Street Network\n{len(tour_indices)} stops | {total_distance/1000:.2f} km total distance', 
                fontsize=15, weight='bold', pad=20)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim(box[2] - lon_pad*0.1, box[3] + lon_pad*0.1)
    ax.set_ylim(box[0] - lat_pad*0.1, box[1] + lat_pad*0.1)
    
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Enhanced solution plot saved: {out_png}")


def plot_solution_folium(G: nx.MultiDiGraph, coords_dict: Dict[int, Tuple[float, float]],
                        poi_nodes: List[int], poi_latlons: List[Tuple[float, float]],
                        tour_indices: List[int], out_html: Path):
    """Create an interactive folium map with the solution projected on real-world OpenStreetMap."""
    
    # Calculate center of map
    lats = [lat for lat, _ in poi_latlons]
    lons = [lon for _, lon in poi_latlons]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    
    # Create folium map with OpenStreetMap tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Add additional tile layers
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    
    print("🗺️  Creating interactive Folium map...")
    
    # Draw route paths on real streets with step numbers
    total_distance = 0.0
    route_group = folium.FeatureGroup(name='Route', show=True)
    
    for step in range(len(tour_indices) - 1):
        idx_from = tour_indices[step]
        idx_to = tour_indices[step + 1]
        if idx_from >= len(poi_nodes) or idx_to >= len(poi_nodes):
            continue
        
        node_from = poi_nodes[idx_from]
        node_to = poi_nodes[idx_to]
        
        # Compute shortest path on real graph
        try:
            path = nx.shortest_path(G, node_from, node_to, weight='length')
            
            # Calculate real distance
            segment_dist = 0.0
            for i in range(len(path) - 1):
                if G.has_edge(path[i], path[i+1]):
                    edge_data = G[path[i]][path[i+1]]
                    if isinstance(edge_data, dict):
                        segment_dist += edge_data.get(0, {}).get('length', 0)
                    else:
                        for k, v in edge_data.items():
                            segment_dist += v.get('length', 0)
                            break
            total_distance += segment_dist
            
            # Get path coordinates
            path_coords = [(coords_dict[n][0], coords_dict[n][1]) for n in path]
            
            # Color gradient for route progression
            color_val = step / max(1, len(tour_indices) - 2)
            cmap = plt.cm.coolwarm
            rgba = cmap(color_val)
            hex_color = mcolors.rgb2hex(rgba[:3])
            
            # Draw the route segment
            folium.PolyLine(
                path_coords,
                color=hex_color,
                weight=5,
                opacity=0.8,
                popup=f"<b>Step {step + 1}</b><br>From: {idx_from} → To: {idx_to}<br>Distance: {segment_dist:.1f}m<br>Nodes: {len(path)}",
                tooltip=f"Step {step + 1}: {segment_dist:.0f}m"
            ).add_to(route_group)
            
            # Add arrow marker at end of segment
            if len(path_coords) >= 2:
                mid_lat = (path_coords[-1][0] + path_coords[-2][0]) / 2
                mid_lon = (path_coords[-1][1] + path_coords[-2][1]) / 2
                
                # Calculate angle for arrow
                import math
                lat1, lon1 = path_coords[-2]
                lat2, lon2 = path_coords[-1]
                angle = math.degrees(math.atan2(lon2 - lon1, lat2 - lat1))
                
                folium.RegularPolygonMarker(
                    location=[mid_lat, mid_lon],
                    fill_color=hex_color,
                    color=hex_color,
                    number_of_sides=3,
                    radius=8,
                    rotation=angle,
                    fill_opacity=0.9,
                    opacity=0.9
                ).add_to(route_group)
            
            print(f"   Step {step + 1}: {idx_from} → {idx_to} ({segment_dist:.1f}m via {len(path)} nodes)")
            
        except nx.NetworkXNoPath:
            # Fallback: direct line if no path found
            lat1, lon1 = poi_latlons[idx_from]
            lat2, lon2 = poi_latlons[idx_to]
            folium.PolyLine(
                [(lat1, lon1), (lat2, lon2)],
                color='red',
                weight=4,
                opacity=0.7,
                dash_array='10',
                popup=f"<b>Step {step + 1}</b><br>NO PATH FOUND<br>Direct line",
                tooltip=f"Step {step + 1}: NO PATH"
            ).add_to(route_group)
            print(f"   Step {step + 1}: {idx_from} → {idx_to} (NO PATH - direct line)")
    
    route_group.add_to(m)
    
    print(f"✓ Total real-world distance: {total_distance:.1f} meters ({total_distance/1000:.2f} km)")
    
    # Add POI markers
    poi_group = folium.FeatureGroup(name='Points of Interest', show=True)
    
    for idx, (lat, lon) in enumerate(poi_latlons):
        if idx == 0:
            # Depot
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>Depot</b><br>Index: {idx}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
                tooltip="Depot (Start/End)",
                icon=folium.Icon(color='red', icon='home', prefix='fa')
            ).add_to(poi_group)
        elif idx == 1:
            # Depot Charging
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>Depot Charging</b><br>Index: {idx}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
                tooltip="Depot Charging Station",
                icon=folium.Icon(color='orange', icon='bolt', prefix='fa')
            ).add_to(poi_group)
        elif 1 < idx <= 5:  # Charging stations (assuming 4 stations)
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>Charging Station {idx-1}</b><br>Index: {idx}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
                tooltip=f"Station S{idx-1}",
                icon=folium.Icon(color='green', icon='charging-station', prefix='fa')
            ).add_to(poi_group)
        else:
            # Customer
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>Customer {idx-5}</b><br>Index: {idx}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
                tooltip=f"Customer {idx-5}",
                icon=folium.Icon(color='purple', icon='user', prefix='fa')
            ).add_to(poi_group)
    
    poi_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topleft',
        title='Fullscreen',
        title_cancel='Exit fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Add measure control
    plugins.MeasureControl(position='bottomleft', primary_length_unit='meters').add_to(m)
    
    # Add title/legend as HTML overlay
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: auto; height: auto; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <h4 style="margin:0; padding-bottom:5px; border-bottom: 1px solid #ddd;">🗺️ EM-EVRP Solution on Real NYC Map</h4>
        <p style="margin:5px 0;"><b>Total Stops:</b> {len(tour_indices)}</p>
        <p style="margin:5px 0;"><b>Total Distance:</b> {total_distance/1000:.2f} km ({total_distance:.0f} m)</p>
        <p style="margin:5px 0;"><b>Legend:</b></p>
        <ul style="margin:5px 0; padding-left:20px; font-size:12px;">
            <li>🏠 <span style="color:red;">Red</span> - Depot</li>
            <li>⚡ <span style="color:orange;">Orange</span> - Depot Charging</li>
            <li>🔌 <span style="color:green;">Green</span> - Charging Stations</li>
            <li>👤 <span style="color:purple;">Purple</span> - Customers</li>
        </ul>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save the map
    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    print(f"✅ Interactive Folium map saved: {out_html}")
    print(f"   Open in browser to explore the solution on real-world map!")


def main():
    parser = argparse.ArgumentParser(description='Run EM-EVRP inference on NPZ graph input')
    parser.add_argument('--npz', required=True, type=str, help='Path to NPZ built from real graph')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('ExperimentalLog', 'train', '10', 'rollout','C10_02_42_53.467564', 'best.pt'))
    parser.add_argument('--nodes', type=int, default=10, help='Number of customers (C)')
    parser.add_argument('--charging_num', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n_encode_layers', type=int, default=3)
    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--tanh_clipping', type=float, default=10.0)
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'sample'])
    parser.add_argument('--velocity', type=float, default=50.0, help='km/h for time conversions if needed')
    parser.add_argument('--start_soc', type=float, default=150.0)
    parser.add_argument('--t_limit', type=float, default=10.0)
    parser.add_argument('--max_load', type=float, default=4.0)
    parser.add_argument('--max_demand', type=int, default=4, help='Integer upper bound for torch.randint; must be int')
    parser.add_argument('--graph-db', type=str, default=None, help='Path to NYC graph pickle for plotting (optional)')
    parser.add_argument('--out-plot', type=str, default='inference_solution.png', help='Output PNG for matplotlib plot')
    parser.add_argument('--out-html', type=str, default='inference_solution.html', help='Output HTML for interactive Folium map')
    parser.add_argument('--use-folium', action='store_true', help='Generate interactive Folium map instead of/in addition to matplotlib')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a tiny dataset instance to supply update functions and args
    # Create a tiny dataset only to reuse update_dynamic/update_mask; ensure max_demand is int
    ds = VehicleRoutingDataset(num_samples=1,
                               input_size=args.nodes,
                               t_limit=args.t_limit,
                               Start_SOC=args.start_soc,
                               velocity=args.velocity,
                               max_load=args.max_load,
                               max_demand=int(args.max_demand),
                               charging_num=args.charging_num,
                               seed=1234,
                               args=type('obj', (), {
                                   'CVRP_lib_test': False,
                                   'num_nodes': args.nodes,
                                   'charging_num': args.charging_num,
                                   'Start_SOC': args.start_soc,
                                   't_limit': args.t_limit
                               })())

    model = AttentionModel(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden,
        args=type('obj', (), {
            'Start_SOC': args.start_soc,
            't_limit': args.t_limit,
            'num_nodes': args.nodes,
            'charging_num': args.charging_num
        })(),
        n_encode_layers=args.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=args.normalization,
        tanh_clipping=args.tanh_clipping,
        update_dynamic=ds.update_dynamic,
        update_mask=ds.update_mask,
    ).to(device)

    load_path = Path(args.checkpoint)
    if load_path.exists():
        ckpt = torch_load_cpu(str(load_path))
        model.load_state_dict({**model.state_dict(), **ckpt.get('model', {})})
    else:
        print(f"[!] Checkpoint not found: {load_path}. Using randomly initialized model.")

    set_decode_type(model, args.decode)

    static, dynamic, distances, slopes = build_tensors_from_npz(Path(args.npz),
                                                                 start_soc=args.start_soc,
                                                                 t_limit=args.t_limit,
                                                                 max_load=args.max_load,
                                                                 velocity_kmh=args.velocity,
                                                                 charging_num=args.charging_num)

    batch = (
        static.to(device),
        dynamic.to(device),
        distances.to(device),
        slopes.to(device),
    )

    with torch.no_grad():
        pi, ll, cost = model(batch)

    # cost is per-step energy, sum along steps
    route_cost = cost.sum(1).cpu().numpy()[0]
    tour = pi[0].detach().cpu().numpy().tolist()
    # Trim trailing zeros beyond mask end
    print('Greedy route indices:', tour)
    print('Total route energy cost:', float(route_cost))

    # Plot solution if graph DB provided
    if args.graph_db:
        npz_data = np.load(Path(args.npz))
        xy_norm = npz_data['xy']  # (2, N)
       
        try:
            G = load_graph(Path(args.graph_db))
            coords_dict = get_node_coords_dict(G)
            # Rebuild POI selection with same params (user must ensure consistency)
            # For safety, we need poi_node_ids from NPZ. Let's check if NPZ has it.
            if 'poi_node_ids' in npz_data:
                poi_nodes = npz_data['poi_node_ids'].tolist()
                poi_latlons = [coords_dict[n] for n in poi_nodes]
            else:
                # Fallback: denormalize xy using its own bounds (approx lat/lon)
                # This won't align with real graph, so skip background
                print("⚠️  NPZ missing poi_node_ids; plotting without graph background overlay.")
                # Use xy as synthetic coords
                x = xy_norm[0]
                y = xy_norm[1]
                poi_latlons = [(y[i], x[i]) for i in range(xy_norm.shape[1])]
                coords_dict = {}  # empty so no background drawn
                poi_nodes = []
            print('✅ Loaded graph for plotting.')
            
            # Generate Folium interactive map if requested
            if args.use_folium and poi_nodes:
                print("\n🌍 Generating interactive Folium map...")
                plot_solution_folium(G, coords_dict, poi_nodes, poi_latlons, tour, Path(args.out_html))
            
            # Generate matplotlib static plot (default or if folium not requested)
            if not args.use_folium or poi_nodes:
                print("\n📊 Generating matplotlib static plot...")
                plot_solution(G, coords_dict, poi_nodes, poi_latlons, tour, Path(args.out_plot))
                
        except Exception as e:
            print(f"⚠️  Could not plot solution: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("ℹ️  Skipping plot (use --graph-db to enable solution visualization)")


if __name__ == '__main__':
    main()
