import gzip
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


def load_pickled_graph(db_path: Path) -> nx.MultiDiGraph:
    """Load graph from compressed pickle.
    Supports dict({'network': graph, ...}) or a direct graph pickled payload.
    """
    with gzip.open(db_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'network' in data:
        return data['network']
    if isinstance(data, nx.MultiDiGraph):
        return data
    raise RuntimeError(f"Unsupported pickled structure at {db_path}")


def get_node_coords(G: nx.MultiDiGraph) -> Dict[int, Tuple[float, float]]:
    coords = {}
    for n, d in G.nodes(data=True):
        if 'y' in d and 'x' in d:
            coords[n] = (float(d['y']), float(d['x']))
    if not coords:
        raise RuntimeError("Graph nodes are missing 'y'/'x' attributes")
    return coords


def normalize_xy(latlons: List[Tuple[float, float]]) -> np.ndarray:
    lats = np.array([lat for lat, _ in latlons], dtype=np.float64)
    lons = np.array([lon for _, lon in latlons], dtype=np.float64)
    min_lat, max_lat = float(lats.min()), float(lats.max())
    min_lon, max_lon = float(lons.min()), float(lons.max())
    lat_span = max(1e-9, max_lat - min_lat)
    lon_span = max(1e-9, max_lon - min_lon)
    y = (lats - min_lat) / lat_span * 100.0
    x = (lons - min_lon) / lon_span * 100.0
    return np.stack([x, y], axis=0)


def geodesic_approx_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))
    return R * c
