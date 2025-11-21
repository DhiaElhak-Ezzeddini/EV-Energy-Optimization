import argparse
import gzip
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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

    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Draw background edges (subsampled)
    drawn = 0
    for u, v in G.edges():
        y1, x1 = coords_dict.get(u, (None, None))
        y2, x2 = coords_dict.get(v, (None, None))
        if y1 is None or y2 is None:
            continue
        if (box[0] <= y1 <= box[1] and box[2] <= x1 <= box[3]) or (box[0] <= y2 <= box[1] and box[2] <= x2 <= box[3]):
            if drawn % 5 == 0:
                ax.plot([x1, x2], [y1, y2], color='#e0e0e0', linewidth=0.3, alpha=0.5, zorder=1)
            drawn += 1

    # Draw actual route paths on real streets
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
            # Draw the path segments
            path_lons = [coords_dict[n][1] for n in path]
            path_lats = [coords_dict[n][0] for n in path]
            ax.plot(path_lons, path_lats, color='blue', linewidth=2.5, alpha=0.85, zorder=3, solid_capstyle='round')
            
            # Add arrow at the end of this segment
            if len(path) >= 2:
                # Arrow from second-to-last to last point
                ax.annotate('', xy=(path_lons[-1], path_lats[-1]), 
                           xytext=(path_lons[-2], path_lats[-2]),
                           arrowprops=dict(arrowstyle='->', lw=2.5, color='blue', alpha=0.9), zorder=4)
        except nx.NetworkXNoPath:
            # Fallback: direct line if no path found
            lat1, lon1 = poi_latlons[idx_from]
            lat2, lon2 = poi_latlons[idx_to]
            ax.plot([lon1, lon2], [lat1, lat2], color='red', linewidth=2, linestyle='--', alpha=0.7, zorder=3)
            ax.annotate('', xy=(lon2, lat2), xytext=(lon1, lat1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.8), zorder=4)

    # Draw POI nodes
    for idx, (lat, lon) in enumerate(poi_latlons):
        if idx == 0:
            ax.scatter([lon], [lat], c='red', s=120, marker='s', edgecolors='black', linewidths=1.5, label='Depot', zorder=5)
            ax.text(lon, lat, 'D', fontsize=9, ha='center', va='center', color='white', weight='bold')
        elif idx == 1:
            ax.scatter([lon], [lat], c='orange', s=100, marker='D', edgecolors='black', linewidths=1.5, label='Depot Chg', zorder=5)
            ax.text(lon, lat, 'DC', fontsize=8, ha='center', va='center', color='white', weight='bold')
        elif 1 < idx <= 5:  # stations (assuming 4 stations)
            ax.scatter([lon], [lat], c='green', s=90, marker='^', edgecolors='black', linewidths=1.5, zorder=5)
            ax.text(lon, lat, f'S{idx-1}', fontsize=8, ha='center', va='bottom', color='black', weight='bold')
        else:
            ax.scatter([lon], [lat], c='purple', s=70, marker='o', edgecolors='black', linewidths=1, zorder=5)
            ax.text(lon, lat, f'{idx-5}', fontsize=7, ha='center', va='center', color='white')

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Inferred EM-EVRP Tour on NYC Graph', fontsize=14, weight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Solution plot saved: {out_png}")


def main():
    parser = argparse.ArgumentParser(description='Run EM-EVRP inference on NPZ graph input')
    parser.add_argument('--npz', required=True, type=str, help='Path to NPZ built from real graph')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('ExperimentalLog', 'train', '10', 'rollout', 'best.pt'))
    parser.add_argument('--nodes', type=int, default=10, help='Number of customers (C)')
    parser.add_argument('--charging_num', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n_encode_layers', type=int, default=3)
    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--tanh_clipping', type=float, default=10.0)
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'sample'])
    parser.add_argument('--velocity', type=float, default=50.0, help='km/h for time conversions if needed')
    parser.add_argument('--start_soc', type=float, default=80.0)
    parser.add_argument('--t_limit', type=float, default=10.0)
    parser.add_argument('--max_load', type=float, default=4.0)
    parser.add_argument('--max_demand', type=int, default=4, help='Integer upper bound for torch.randint; must be int')
    parser.add_argument('--graph-db', type=str, default=None, help='Path to NYC graph pickle for plotting (optional)')
    parser.add_argument('--out-plot', type=str, default='inference_solution.png', help='Output PNG for solution plot')
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
        # We need original POI lat/lons to denormalize; load from NPZ metadata if stored, else recompute from graph
        # For now, load graph and reconstruct POI list from NPZ types and node selection
        # Simpler: just load the graph, pick same POIs, and use their coords
        # Alternate: store POI node IDs in NPZ at build time. For demo, we'll assume graph-db + npz align.
        
        # Workaround: we saved xy normalized; to get original coords we need the graph.
        # Load graph and map NPZ back. Ideally build_poi_graph should save poi_node_ids in NPZ.
        # For a quick solution: denormalize using xy bounds from npz itself, or load graph and extract coords.
        # Let's load graph and get coords for the POI nodes used.
        # Since we don't have poi_node_ids saved, we'll infer lat/lon by denormalizing with the span used.
        # Actually, the build scripts know the POIs; let's just denormalize using the xy bounds themselves.
        
        # Simplest: assume xy was normalized from some latlons; reverse it approximately.
        # But we need the original min/max. Let's load the graph and get a representative set.
        # Actually, let's just convert normalized xy back using a heuristic or load graph fully.
        
        # To avoid complexity: load graph, reconstruct POI selection (not ideal but workable).
        # Better: have build_poi_graph save 'poi_node_ids' in NPZ so we can look them up.
        # For now, let's assume user provides --graph-db and we load and map POIs by regenerating.
        
        # Quick fix: denormalize using the NPZ xy itself (treat it as approximate lat/lon scaled).
        # Properly: we should store poi_node_ids or original latlons in NPZ. Let's add that.
        
        # For demo: use xy as-is (scaled 0-100) and overlay on a small grid, or skip graph BG.
        # Let's do a simple version: load graph, assume POI order matches, get coords.
        
        # Attempt: load graph, get coords for all nodes, then use NPZ to map.
        # Since we don't have poi_node_ids, let's just plot xy normalized as a simple diagram.
        # Or: user must run build_poi_graph with same seed/params so we can regenerate POI list.
        
        # Practical: add a fallback to plot without background if graph unavailable.
        # Or: store poi_latlons in NPZ at build time. Let's do that in a follow-up or assume it's there.
        
        # For now: implement a minimal plot using xy and tour, and overlay on graph if available.
        # We'll load the graph and denormalize using a best-guess or skip background.
        
        # Simplest working solution: save poi_node_ids in NPZ at build time, then load here.
        # Let's update build_poi_graph to save poi_node_ids and poi_latlons.
        
        # For immediate demo: just plot the tour on normalized coordinates without background.
        # Or load graph and attempt to match POIs by re-running selection (fragile).
        
        # Compromise: if graph-db provided, load it, run POI selection with same params (must match!),
        # get coords, and plot. Otherwise, plot on normalized grid only.
        
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
            plot_solution(G, coords_dict, poi_nodes, poi_latlons, tour, Path(args.out_plot))
        except Exception as e:
            print(f"⚠️  Could not plot solution: {e}")
    else:
        print("ℹ️  Skipping plot (use --graph-db to enable solution visualization)")


if __name__ == '__main__':
    main()
