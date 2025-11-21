from typing import Dict, List, Optional, Tuple

import math
import networkx as nx
import numpy as np


def edge_best_data(G: nx.MultiDiGraph, u: int, v: int) -> Tuple[float, float, float]:
    """Return (length_m, travel_time_s, slope_rise_run) for the best parallel edge.
    Picks the parallel edge with minimum length.
    """
    best = None
    best_len = float('inf')
    for _, data in G[u][v].items():
        L = float(data.get('length', float('inf')))
        if L < best_len:
            best_len = L
            best = data
    if best is None:
        return 0.0, 0.0, 0.0
    L = float(best.get('length', 0.0))
    speed_kph = float(best.get('speed_kph', 40.0))
    T = float(best.get('travel_time', L / max(1e-6, speed_kph * 1000 / 3600)))
    slope_deg = float(best.get('slope_deg', 0.0))
    slope = math.tan(math.radians(slope_deg))
    return L, T, slope


def path_metrics(G: nx.MultiDiGraph, path: List[int]) -> Tuple[float, float, float]:
    total_len = 0.0
    total_time = 0.0
    weighted_slope = 0.0
    for u, v in zip(path[:-1], path[1:]):
        L, T, s = edge_best_data(G, u, v)
        total_len += L
        total_time += T
        weighted_slope += s * L
    avg_slope = (weighted_slope / total_len) if total_len > 0 else 0.0
    return total_len, total_time, avg_slope


def build_complete_matrices(G: nx.MultiDiGraph, pois: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[List[int]]]]:
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
                L, tt, slope = path_metrics(G, path)
                D[i, j] = L
                T[i, j] = tt
                S[i, j] = slope
                paths[i][j] = path
            except nx.NetworkXNoPath:
                D[i, j] = np.inf
                T[i, j] = np.inf
                S[i, j] = 0.0
                paths[i][j] = []
    # Symmetrize missing entries
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
