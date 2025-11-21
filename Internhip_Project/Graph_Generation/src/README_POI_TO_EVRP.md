POI graph builder for EM-EVRP

What this does
- Loads the saved NYC MultiDiGraph (`New_York_network_enhanced_attributes.pkl.gz`).
- Picks POIs: depot, depot_charging, 4 charging stations, 10 customers (deterministic, seedable) or from a CSV.
- Runs Dijkstra between all POI pairs to compute distance (m), travel time (s), and average slope (rise/run).
- Normalizes POI coordinates to [0,100] for the EM-EVRP static input and exports an NPZ.
- Generates a PNG overlay showing POI-to-POI shortest paths on top of the original graph.

Files
- `build_poi_em_evrp_npz.py`: main script.
- `New_York_network_enhanced_attributes.pkl.gz`: graph DB with node lat/lon and edge attributes.

Optional CSV schema
Use a CSV to override POIs instead of auto selection. Headers must include either `node_id` or `lat,lon`.

role,node_id,lat,lon
depot,12345,,
depot_charging,67890,,
station,13579,,
station,,40.75,-73.98
customer,24680,,
customer,,40.76,-73.97

Outputs
- NPZ keys: `xy(2,N)`, `D(N,N)`, `T(N,N)`, `S(N,N)`, `types(N)`, `charging_num`, `demands(N)`.
  - Node ordering: [depot, depot_charging, stations..., customers...]
  - `types`: 0 depot, 1 depot_charging, 2 station, 3 customer
- PNG overlay visualizing the POI complete graph atop the base network.

Quick usage (PowerShell)
Assumes Python can import `networkx` and `numpy`. No environment setup changes are made.

```powershell
# From repo root
python .\Internhip_Project\Graph_Generation\src\build_poi_em_evrp_npz.py `
  --db-path .\Internhip_Project\Graph_Generation\src\New_York_network_enhanced_attributes.pkl.gz `
  --customers 10 `
  --stations 4 `
  --seed 42 `
  --out-npz .\Internhip_Project\Graph_Generation\src\NYC_C10_S4.npz `
  --out-plot .\Internhip_Project\Graph_Generation\src\NYC_C10_S4.png

# Optional: with POI CSV override
python .\Internhip_Project\Graph_Generation\src\build_poi_em_evrp_npz.py `
  --poi-csv .\Internhip_Project\Graph_Generation\src\pois.csv
```

Run inference with the trained model
```powershell
python .\EM-EVRP\infer_from_npz.py `
  --npz .\Internhip_Project\Graph_Generation\src\NYC_C10_S4.npz `
  --checkpoint .\EM-EVRP\ExperimentalLog\train\10\rollout\best.pt `
  --nodes 10 `
  --charging_num 4 `
  --decode greedy
```

Notes
- Slope is averaged in rise/run (tan(theta)) to align with EM-EVRP energy equations.
- Demands are generated small (<=0.9 total) and only for customers; stations/depot have zero.
- The overlay plot restricts the background to a padded bbox around the POIs to remain readable.
