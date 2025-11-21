import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

# Allow running as a standalone script
from pathlib import Path as _P
sys.path.append(str(_P(__file__).parent))

from loaders import load_pickled_graph, get_node_coords, normalize_xy, geodesic_approx_m
from dijkstra import build_complete_matrices
from export_em_evrp_npz import save_em_evrp_npz


def farthest_point_sampling(nodes: List[int], coords: Dict[int, Tuple[float, float]], k: int, seed: int,
                            start_node: Optional[int] = None,
                            min_pair_m: float = 0.0) -> List[int]:
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
        # enforce min separation
        if min_pair_m > 0 and chosen:
            ok = True
            for s in chosen:
                if geodesic_approx_m(coords[s][0], coords[s][1], coords[best_node][0], coords[best_node][1]) < min_pair_m:
                    ok = False
                    break
            if not ok:
                remaining.discard(best_node)
                continue
        chosen.append(best_node)
        remaining.discard(best_node)
    return [n for n in chosen if n in coords]


def pick_pois(G: nx.MultiDiGraph,
              coords: Dict[int, Tuple[float, float]],
              customers: int,
              stations: int,
              seed: int,
              station_min_km: float,
              station_max_km: float,
              customer_min_km: float,
              customer_max_km: float,
              customer_min_pair_km: float,
              station_min_pair_km: float,
              station_strategy: str = 'farthest') -> Tuple[int, int, List[int], List[int]]:
    lats = np.array([lat for lat, _ in coords.values()])
    lons = np.array([lon for _, lon in coords.values()])
    lat0 = float(np.median(lats))
    lon0 = float(np.median(lons))

    def nearest_node(lat, lon):
        return min(coords.keys(), key=lambda n: geodesic_approx_m(lat, lon, coords[n][0], coords[n][1]))

    depot = nearest_node(lat0, lon0)
    dep_neighbors = list(G.neighbors(depot))
    depot_chg = dep_neighbors[0] if dep_neighbors else nearest_node(lat0 + 1e-3, lon0 + 1e-3)

    cand = [n for n in coords.keys() if n not in (depot, depot_chg)]

    def filter_band(center: int, nodes: List[int], min_km: float, max_km: float) -> List[int]:
        res = []
        for n in nodes:
            d = geodesic_approx_m(coords[center][0], coords[center][1], coords[n][0], coords[n][1])
            if (min_km * 1000.0) <= d <= (max_km * 1000.0):
                res.append(n)
        return res

    # Stations: within ring [station_min_km, station_max_km], spread apart
    station_pool = filter_band(depot, cand, station_min_km, station_max_km)
    # Relax if insufficient
    relax = 0
    while len(station_pool) < stations and relax < 3:
        relax += 1
        station_pool = filter_band(depot, cand, max(0.0, station_min_km - relax * 0.5), station_max_km + relax * 1.0)
    if station_strategy == 'degree':
        degs = {n: G.degree(n) for n in station_pool}
        # greedy with min separation
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
        station_nodes = farthest_point_sampling(
            station_pool, coords, k=stations, seed=seed, start_node=depot, min_pair_m=station_min_pair_km * 1000.0)[:stations]

    # Customers: within ring [customer_min_km, customer_max_km], spread apart
    rng = np.random.RandomState(seed + 1)
    cust_pool = [n for n in cand if n not in station_nodes]
    cust_pool = filter_band(depot, cust_pool, customer_min_km, customer_max_km)
    relax = 0
    while len(cust_pool) < customers and relax < 3:
        relax += 1
        cust_pool = filter_band(depot, [n for n in cand if n not in station_nodes],
                                max(0.0, customer_min_km - relax * 0.5), customer_max_km + relax * 1.0)
    customers_nodes = farthest_point_sampling(
        cust_pool, coords, k=customers, seed=seed + 2, start_node=None, min_pair_m=customer_min_pair_km * 1000.0)
    if len(customers_nodes) < customers:
        # fallback: random with min separation best-effort
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


def read_poi_csv(csv_path: Path, coords: Dict[int, Tuple[float, float]]) -> Tuple[int, int, List[int], List[int]]:
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
                try:
                    nid = int(node_id)
                except ValueError:
                    nid = None
            elif lat and lon:
                try:
                    nid = nearest(float(lat), float(lon))
                except Exception:
                    nid = None
            if nid is None:
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
        raise RuntimeError("CSV must contain depot, depot_charging, >=1 station, >=1 customer")
    return depot, depot_chg, stations, customers


def plot_overlay(G: nx.MultiDiGraph, coords: Dict[int, Tuple[float, float]], poi_nodes: List[int], paths: List[List[List[int]]],
                 out_png: Path):
    latv = [coords[n][0] for n in poi_nodes]
    lonv = [coords[n][1] for n in poi_nodes]
    min_lat, max_lat = min(latv), max(latv)
    min_lon, max_lon = min(lonv), max(lonv)
    lat_pad = (max_lat - min_lat) * 0.2 + 1e-3
    lon_pad = (max_lon - min_lon) * 0.2 + 1e-3
    box = (min_lat - lat_pad, max_lat + lat_pad, min_lon - lon_pad, max_lon + lon_pad)

    fig, ax = plt.subplots(figsize=(9, 9))
    drawn = 0
    for u, v in G.edges():
        y1, x1 = coords.get(u, (None, None))
        y2, x2 = coords.get(v, (None, None))
        if y1 is None or y2 is None:
            continue
        if (box[0] <= y1 <= box[1] and box[2] <= x1 <= box[3]) or (box[0] <= y2 <= box[1] and box[2] <= x2 <= box[3]):
            if drawn % 4 == 0:
                ax.plot([x1, x2], [y1, y2], color='#cccccc', linewidth=0.4, alpha=0.6)
            drawn += 1

    N = len(poi_nodes)
    for i in range(N):
        for j in range(i + 1, N):
            p = paths[i][j]
            if not p:
                continue
            xs = [coords[n][1] for n in p]
            ys = [coords[n][0] for n in p]
            ax.plot(xs, ys, color='#1f78b4', linewidth=1.2, alpha=0.8)

    for idx, n in enumerate(poi_nodes):
        x = coords[n][1]
        y = coords[n][0]
        if idx == 0:
            ax.scatter([x], [y], c='red', s=60)
            ax.text(x, y, 'Depot', fontsize=8)
        elif idx == 1:
            ax.scatter([x], [y], c='orange', s=50)
            ax.text(x, y, 'DepotChg', fontsize=8)
        elif 1 < idx <= 5:
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
    parser = argparse.ArgumentParser(description='Build EM-EVRP NPZ from large NYC graph')
    parser.add_argument('--db-path', type=str, required=True)
    parser.add_argument('--customers', type=int, default=10)
    parser.add_argument('--stations', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--poi-csv', type=str, default=None)
    parser.add_argument('--station-min-km', type=float, default=1.0)
    parser.add_argument('--station-max-km', type=float, default=8.0)
    parser.add_argument('--customer-min-km', type=float, default=1.0)
    parser.add_argument('--customer-max-km', type=float, default=6.0)
    parser.add_argument('--customer-min-pair-km', type=float, default=0.7)
    parser.add_argument('--station-min-pair-km', type=float, default=1.5)
    parser.add_argument('--station-strategy', type=str, default='farthest', choices=['farthest', 'degree'])
    parser.add_argument('--out-npz', type=str, required=True)
    parser.add_argument('--out-plot', type=str, required=True)
    parser.add_argument('--demands-min', type=float, default=0.05)
    parser.add_argument('--demands-max', type=float, default=0.15)
    args = parser.parse_args()

    G = load_pickled_graph(Path(args.db_path))
    coords = get_node_coords(G)

    if args.poi_csv:
        depot, depot_chg, stations, customers = read_poi_csv(Path(args.poi_csv), coords)
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
    D, T, S, paths = build_complete_matrices(G, poi_nodes)

    xy = normalize_xy([coords[n] for n in poi_nodes])
    types = np.zeros((len(poi_nodes),), dtype=np.int32)
    types[1] = 1
    types[2:2 + len(stations)] = 2
    types[2 + len(stations):] = 3

    rng = np.random.RandomState(args.seed + 7)
    dem = np.zeros((len(poi_nodes),), dtype=np.float64)
    if len(customers) > 0:
        vals = rng.uniform(args.demands_min, args.demands_max, size=len(customers))
        scale = min(1.0, 0.9 / max(1e-6, float(vals.sum())))
        vals *= scale
        dem[2 + len(stations):] = vals

    poi_ids_array = np.array(poi_nodes, dtype=np.int64)
    save_em_evrp_npz(Path(args.out_npz), xy, D, T, S, types, charging_num=len(stations), demands=dem, poi_node_ids=poi_ids_array)
    plot_overlay(G, coords, poi_nodes, paths, Path(args.out_plot))


if __name__ == '__main__':
    main()
