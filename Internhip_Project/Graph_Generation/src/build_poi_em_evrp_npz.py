import argparse
import csv
import gzip
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

# Local import: use the NetworkDatabase helper if available
try:
    from Graph_Generation.src.network_gen import NetworkDatabase
except Exception:
    NetworkDatabase = None  # Fallback to direct pickle loading


def load_network(db_path: Path) -> nx.MultiDiGraph:
    if NetworkDatabase is not None:
        db = NetworkDatabase(str(db_path))
        return db.load_network()
    # Fallback: load pickled structure with dict({'network': nxgraph, ...})
    with gzip.open(db_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'network' in data:
        return data['network']
    if isinstance(data, nx.MultiDiGraph):
        return data
    raise RuntimeError("Unsupported network pickle structure")


def graph_node_coords(G: nx.MultiDiGraph) -> Dict[int, Tuple[float, float]]:
    coords = {}
    for n, d in G.nodes(data=True):
        if 'y' in d and 'x' in d:
            coords[n] = (float(d['y']), float(d['x']))
    if not coords:
        raise RuntimeError("Graph nodes missing 'y'/'x' coordinates")
    return coords


def geodesic_approx_m(lat1, lon1, lat2, lon2) -> float:
    # Haversine simplified for small distances (meters)
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
    return R * c


def farthest_point_sampling(nodes: List[int], coords: Dict[int, Tuple[float, float]], k: int, seed: int,
                            start_node: Optional[int] = None) -> List[int]:
    rng = np.random.RandomState(seed)
    chosen: List[int] = []
    if start_node is None:
        start_node = nodes[rng.randint(len(nodes))]
    chosen.append(start_node)
    remaining = set(nodes)
    remaining.discard(start_node)

    def mindist_to_chosen(cand: int) -> float:
        latc, lonc = coords[cand]
        best = float('inf')
        for s in chosen:
            lats, lons = coords[s]
            d = geodesic_approx_m(latc, lonc, lats, lons)
            if d < best:
                best = d
        return best

    while len(chosen) < k and remaining:
        best_node = None
        best_dist = -1.0
        for cand in list(remaining):
            dmin = mindist_to_chosen(cand)
            if dmin > best_dist:
                best_dist = dmin
                best_node = cand
        if best_node is None:
            break
        chosen.append(best_node)
        remaining.discard(best_node)
    return chosen


def pick_pois(G: nx.MultiDiGraph, coords: Dict[int, Tuple[float, float]], customers: int, stations: int,
              seed: int,
              station_min_km: float,
              station_max_km: float,
              customer_min_km: float,
              customer_max_km: float,
              customer_min_pair_km: float,
              station_min_pair_km: float,
              station_strategy: str = 'farthest') -> Tuple[int, int, List[int], List[int]]:
    # Depot: nearest to centroid of all nodes
    lats = np.array([lat for lat, _ in coords.values()])
    lons = np.array([lon for _, lon in coords.values()])
    lat0 = float(np.median(lats))
    lon0 = float(np.median(lons))

    def nearest_node(lat, lon):
        return min(coords.keys(), key=lambda n: geodesic_approx_m(lat, lon, coords[n][0], coords[n][1]))

    depot = nearest_node(lat0, lon0)
    # Depot charging: nearest neighbor of depot's adjacency or nearest by distance not equal to depot
    dep_neighbors = list(G.neighbors(depot))
    if dep_neighbors:
        depot_chg = dep_neighbors[0]
    else:
        depot_chg = min((n for n in coords.keys() if n != depot),
                        key=lambda n: geodesic_approx_m(coords[depot][0], coords[depot][1], coords[n][0], coords[n][1]))

    candidates = [n for n in coords.keys() if n not in (depot, depot_chg)]

    def band(nodes: List[int], center: int, min_km: float, max_km: float) -> List[int]:
        out = []
        for n in nodes:
            d = geodesic_approx_m(coords[center][0], coords[center][1], coords[n][0], coords[n][1])
            if (min_km * 1000.0) <= d <= (max_km * 1000.0):
                out.append(n)
        return out

    station_pool = band(candidates, depot, station_min_km, station_max_km)
    relax = 0
    while len(station_pool) < stations and relax < 3:
        relax += 1
        station_pool = band(candidates, depot, max(0.0, station_min_km - 0.5 * relax), station_max_km + 1.0 * relax)
    if station_strategy == 'degree':
        degs = {n: G.degree(n) for n in station_pool}
        station_nodes: List[int] = []
        for n, _ in sorted(degs.items(), key=lambda kv: kv[1], reverse=True):
            if len(station_nodes) >= stations:
                break
            ok = True
            for s in station_nodes:
                if geodesic_approx_m(coords[s][0], coords[s][1], coords[n][0], coords[n][1]) < station_min_pair_km * 1000.0:
                    ok = False
                    break
            if ok:
                station_nodes.append(n)
    else:
        station_nodes = farthest_point_sampling(station_pool, coords, k=stations, seed=seed,
                                                start_node=depot, min_pair_m=station_min_pair_km * 1000.0)[:stations]

    rng = np.random.RandomState(seed + 1)
    cust_pool = [n for n in candidates if n not in station_nodes]
    cust_pool = band(cust_pool, depot, customer_min_km, customer_max_km)
    relax = 0
    while len(cust_pool) < customers and relax < 3:
        relax += 1
        cust_pool = band([n for n in candidates if n not in station_nodes], depot,
                         max(0.0, customer_min_km - 0.5 * relax), customer_max_km + 1.0 * relax)
    customers_nodes = farthest_point_sampling(cust_pool, coords, k=customers, seed=seed + 2,
                                              start_node=None, min_pair_m=customer_min_pair_km * 1000.0)
    if len(customers_nodes) < customers:
        rng.shuffle(cust_pool)
        chosen: List[int] = []
        for n in cust_pool:
            ok = True
            for s in chosen:
                if geodesic_approx_m(coords[s][0], coords[s][1], coords[n][0], coords[n][1]) < customer_min_pair_km * 1000.0:
                    ok = False
                    break
            if ok:
                chosen.append(n)
            if len(chosen) >= customers:
                break
        customers_nodes = chosen

    return depot, depot_chg, station_nodes, customers_nodes


def dijkstra_metrics(G: nx.MultiDiGraph, path: List[int]) -> Tuple[float, float, float]:
    total_len = 0.0
    total_time = 0.0
    weighted_slope = 0.0
    for u, v in zip(path[:-1], path[1:]):
        # pick the shortest among parallel edges by length
        best_key = None
        best_len = float('inf')
        best_edge = None
        for key, data in G[u][v].items():
            L = float(data.get('length', float('inf')))
            if L < best_len:
                best_len = L
                best_edge = data
                best_key = key
        if best_edge is None:
            continue
        L = float(best_edge.get('length', 0.0))
        T = float(best_edge.get('travel_time', L / max(1e-6, float(best_edge.get('speed_kph', 40))) * 3.6))
        slope_deg = float(best_edge.get('slope_deg', 0.0))
        slope_val = math.tan(math.radians(slope_deg))  # convert degrees to rise/run
        total_len += L
        total_time += T
        weighted_slope += slope_val * L
    avg_slope = (weighted_slope / total_len) if total_len > 0 else 0.0
    return total_len, total_time, avg_slope


def build_complete_poi_graph(G: nx.MultiDiGraph, pois: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    N = len(pois)
    D = np.zeros((N, N), dtype=np.float64)
    T = np.zeros((N, N), dtype=np.float64)
    S = np.zeros((N, N), dtype=np.float64)
    paths: List[List[Optional[List[int]]]] = [[None for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            try:
                path = nx.shortest_path(G, pois[i], pois[j], weight='length')
                L, tt, slope = dijkstra_metrics(G, path)
                D[i, j] = L
                T[i, j] = tt
                S[i, j] = slope
                paths[i][j] = path
            except nx.NetworkXNoPath:
                D[i, j] = np.inf
                T[i, j] = np.inf
                S[i, j] = 0.0
                paths[i][j] = []
    # Make symmetric by taking min over both directions for D/T and mean for S
    for i in range(N):
        for j in range(i + 1, N):
            if not np.isfinite(D[i, j]) and np.isfinite(D[j, i]):
                D[i, j] = D[j, i]
                T[i, j] = T[j, i]
                S[i, j] = S[j, i]
                paths[i][j] = list(reversed(paths[j][i])) if paths[j][i] else []
            elif not np.isfinite(D[j, i]) and np.isfinite(D[i, j]):
                D[j, i] = D[i, j]
                T[j, i] = T[i, j]
                S[j, i] = S[i, j]
                paths[j][i] = list(reversed(paths[i][j])) if paths[i][j] else []
    return D, T, S, paths  # noqa


def normalize_xy(latlons: List[Tuple[float, float]]) -> np.ndarray:
    lats = np.array([lat for lat, _ in latlons], dtype=np.float64)
    lons = np.array([lon for _, lon in latlons], dtype=np.float64)
    min_lat, max_lat = float(lats.min()), float(lats.max())
    min_lon, max_lon = float(lons.min()), float(lons.max())
    # avoid divide-by-zero if degenerate
    lat_span = max(1e-9, max_lat - min_lat)
    lon_span = max(1e-9, max_lon - min_lon)
    y = (lats - min_lat) / lat_span * 100.0
    x = (lons - min_lon) / lon_span * 100.0
    return np.stack([x, y], axis=0)


def read_poi_csv(csv_path: Path, G: nx.MultiDiGraph, coords: Dict[int, Tuple[float, float]]) -> Tuple[int, int, List[int], List[int]]:
    # CSV schema: role,node_id OR role,lat,lon. roles: depot,depot_charging,station,customer
    depot = None
    depot_chg = None
    stations: List[int] = []
    customers: List[int] = []
    def nearest(lat, lon):
        return min(coords.keys(), key=lambda n: geodesic_approx_m(lat, lon, coords[n][0], coords[n][1]))
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row.get('role', '').strip().lower()
            node_id = row.get('node_id')
            lat = row.get('lat')
            lon = row.get('lon')
            nid: Optional[int] = None
            if node_id:
                nid = int(node_id)
            elif lat and lon:
                nid = nearest(float(lat), float(lon))
            if nid is None or nid not in G:
                continue
            if role == 'depot':
                depot = nid
            elif role in ('depot_charging', 'depotcharging', 'depot_charge'):
                depot_chg = nid
            elif role in ('station', 'charging_station'):
                stations.append(nid)
            elif role in ('customer', 'cus'):
                customers.append(nid)
    if depot is None or depot_chg is None or not stations or not customers:
        raise RuntimeError("Incomplete POIs in CSV. Need depot, depot_charging, >=1 station, >=1 customer")
    return depot, depot_chg, stations, customers


def save_npz(out_path: Path, xy: np.ndarray, D: np.ndarray, T: np.ndarray, S: np.ndarray,
             types: np.ndarray, charging_num: int, demands: np.ndarray, poi_node_ids: np.ndarray = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        'xy': xy.astype(np.float32),
        'D': D.astype(np.float32),
        'T': T.astype(np.float32),
        'S': S.astype(np.float32),
        'types': types.astype(np.int32),
        'charging_num': int(charging_num),
        'demands': demands.astype(np.float32)
    }
    if poi_node_ids is not None:
        save_dict['poi_node_ids'] = poi_node_ids.astype(np.int64)
    np.savez_compressed(out_path, **save_dict)


def plot_overlay(G: nx.MultiDiGraph, coords: Dict[int, Tuple[float, float]], poi_nodes: List[int], paths: List[List[List[int]]],
                 out_png: Path):
    import matplotlib.pyplot as plt
    # bbox around POIs
    latv = [coords[n][0] for n in poi_nodes]
    lonv = [coords[n][1] for n in poi_nodes]
    min_lat, max_lat = min(latv), max(latv)
    min_lon, max_lon = min(lonv), max(lonv)
    lat_pad = (max_lat - min_lat) * 0.2 + 1e-3
    lon_pad = (max_lon - min_lon) * 0.2 + 1e-3
    box = (min_lat - lat_pad, max_lat + lat_pad, min_lon - lon_pad, max_lon + lon_pad)

    fig, ax = plt.subplots(figsize=(9, 9))
    # background edges within bbox (sampled for speed)
    drawn = 0
    for u, v, data in G.edges(data=True):
        y1, x1 = coords.get(u, (None, None))
        y2, x2 = coords.get(v, (None, None))
        if y1 is None or y2 is None:
            continue
        if (box[0] <= y1 <= box[1] and box[2] <= x1 <= box[3]) or (box[0] <= y2 <= box[1] and box[2] <= x2 <= box[3]):
            if drawn % 4 == 0:  # thin subsample for performance
                ax.plot([x1, x2], [y1, y2], color='#cccccc', linewidth=0.4, alpha=0.6)
            drawn += 1

    # overlay paths
    N = len(poi_nodes)
    for i in range(N):
        for j in range(i + 1, N):
            p = paths[i][j]
            if not p:
                continue
            xs = [coords[n][1] for n in p]
            ys = [coords[n][0] for n in p]
            ax.plot(xs, ys, color='#1f78b4', linewidth=1.2, alpha=0.8)

    # plot POIs
    for idx, n in enumerate(poi_nodes):
        x = coords[n][1]
        y = coords[n][0]
        if idx == 0:
            ax.scatter([x], [y], c='red', s=60, label='Depot')
            ax.text(x, y, 'Depot', fontsize=8)
        elif idx == 1:
            ax.scatter([x], [y], c='orange', s=50, label='DepotChg')
            ax.text(x, y, 'DepotChg', fontsize=8)
        elif 1 < idx <= 1 + 4:
            ax.scatter([x], [y], c='green', s=40)
            ax.text(x, y, f'S{idx-1}', fontsize=8)
        else:
            ax.scatter([x], [y], c='purple', s=30)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('POI complete graph paths over original network')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Build EM-EVRP NPZ from NYC network graph')
    parser.add_argument('--db-path', type=str, default=str(Path(__file__).parent / 'New_York_network_enhanced_attributes.pkl.gz'))
    parser.add_argument('--customers', type=int, default=10)
    parser.add_argument('--stations', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--poi-csv', type=str, default=None, help='Optional CSV with role,node_id or role,lat,lon')
    parser.add_argument('--out-npz', type=str, default=str(Path(__file__).parent / 'NYC_C10_S4.npz'))
    parser.add_argument('--out-plot', type=str, default=str(Path(__file__).parent / 'NYC_C10_S4.png'))
    parser.add_argument('--station-min-km', type=float, default=1.0)
    parser.add_argument('--station-max-km', type=float, default=8.0)
    parser.add_argument('--customer-min-km', type=float, default=1.0)
    parser.add_argument('--customer-max-km', type=float, default=6.0)
    parser.add_argument('--customer-min-pair-km', type=float, default=0.7)
    parser.add_argument('--station-min-pair-km', type=float, default=1.5)
    parser.add_argument('--station-strategy', type=str, default='farthest', choices=['farthest', 'degree'])
    parser.add_argument('--demands-min', type=float, default=0.05)
    parser.add_argument('--demands-max', type=float, default=0.15)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Network pickle not found: {db_path}")
    G = load_network(db_path)
    coords = graph_node_coords(G)

    if args.poi_csv:
        depot, depot_chg, stations, customers = read_poi_csv(Path(args.poi_csv), G, coords)
        # trim to requested counts if longer
        stations = stations[: args.stations]
        customers = customers[: args.customers]
    else:
        depot, depot_chg, stations, customers = pick_pois(
            G, coords, args.customers, args.stations, args.seed,
            station_min_km=args.station_min_km,
            station_max_km=args.station_max_km,
            customer_min_km=args.customer_min_km,
            customer_max_km=args.customer_max_km,
            customer_min_pair_km=args.customer_min_pair_km,
            station_min_pair_km=args.station_min_pair_km,
            station_strategy=args.station_strategy)

    poi_nodes = [depot, depot_chg] + stations + customers
    D, T, S, paths = build_complete_poi_graph(G, poi_nodes)

    # Normalize XY for model static input
    latlons = [coords[n] for n in poi_nodes]
    xy = normalize_xy(latlons)

    # Types: 0 depot, 1 depot_charging, 2 station, 3 customer
    types = np.zeros((len(poi_nodes),), dtype=np.int32)
    types[1] = 1
    types[2:2 + len(stations)] = 2
    types[2 + len(stations):] = 3

    # Demands for customers only, scaled 0..1
    rng = np.random.RandomState(args.seed + 7)
    dem = np.zeros((len(poi_nodes),), dtype=np.float64)
    if customers:
        vals = rng.uniform(args.demands_min, args.demands_max, size=len(customers))
        # ensure total <= 1.0 nominal capacity
        scale = min(1.0, 0.9 / max(1e-6, float(vals.sum())))
        vals *= scale
        dem[2 + len(stations):] = vals

    poi_ids_array = np.array(poi_nodes, dtype=np.int64)
    save_npz(Path(args.out_npz), xy, D, T, S, types, charging_num=len(stations), demands=dem, poi_node_ids=poi_ids_array)
    plot_overlay(G, coords, poi_nodes, paths, Path(args.out_plot))


if __name__ == '__main__':
    main()
