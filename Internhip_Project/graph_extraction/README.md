Graph Extraction Toolkit (NYC → EM-EVRP)

Overview
- Build a POI-complete graph from a large road network (NYC).
- Compute pairwise shortest paths between POIs (distance, time, slope).
- Export an NPZ compatible with EM-EVRP: `xy, D, T, S, types, charging_num, demands`.
- Plot POI links over the original graph for sanity checks.

File roles
- `loaders.py`: Load the pickled MultiDiGraph (`New_York_network_enhanced_attributes.pkl.gz`). Normalize XY.
- `dijkstra.py`: Shortest path with metrics and matrix builder across POIs.
- `export_em_evrp_npz.py`: Save EM-EVRP input arrays to NPZ.
- `build_poi_graph.py`: CLI to select POIs (auto or CSV), run Dijkstra, export NPZ and PNG overlay.

POI ordering and types
- Order: `[depot, depot_charging, stations..., customers...]`.
- `types`: 0 depot, 1 depot_charging, 2 station, 3 customer.

Quick start (PowerShell)
```powershell
# Paths relative to repo root
python .\Internhip_Project\graph_extraction\build_poi_graph.py `
  --db-path .\Internhip_Project\Graph_Generation\src\New_York_network_enhanced_attributes.pkl.gz `
  --customers 10 `
  --stations 4 `
  --seed 42 `
  --out-npz .\Internhip_Project\graph_extraction\NYC_C10_S4.npz `
  --out-plot .\Internhip_Project\graph_extraction\NYC_C10_S4.png
```

Optional POI CSV (headers: `role,node_id` or `role,lat,lon`)
```csv
role,node_id,lat,lon
depot,12345,,
depot_charging,67890,,
station,,40.75,-73.98
customer,,40.76,-73.97
```

Notes
- Slope aggregation uses length-weighted average of rise/run (tan(theta)) along the path.
- Demands are generated only for customers, small fractions summing <= 0.9 by default.
- XY normalization maps POI lat/lon to [0,100] as expected by EM-EVRP static input.
