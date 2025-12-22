# 🗺️ Interactive Folium Map Visualization

## Overview
The inference script now supports generating **interactive real-world maps** using Folium, projecting your EM-EVRP solution onto OpenStreetMap tiles.

## Features

### 🌍 Interactive Map Features
- **Real OpenStreetMap tiles** - See actual NYC streets
- **Multiple tile layers** - Switch between OpenStreetMap, CartoDB Positron, and Dark mode
- **Interactive route** - Click on route segments to see details (distance, nodes, step number)
- **POI markers** - Clickable markers showing depot, charging stations, and customers
- **Layer control** - Toggle route and POIs on/off
- **Fullscreen mode** - Expand map to fullscreen
- **Measure tool** - Measure distances directly on the map
- **Route progression colors** - Color gradient shows tour sequence (blue→red)
- **Distance info** - Total distance displayed in km and meters

## Usage

### Generate Interactive Map Only
```bash
python infer_from_npz.py \
    --npz "../Internhip_Project/graph_extraction/NYC_C10_S4.npz" \
    --graph-db "New_York_network_enhanced_attributes.pkl.gz" \
    --use-folium \
    --out-html "solution_interactive.html"
```

### Generate Both Static and Interactive Maps
```bash
python infer_from_npz.py \
    --npz "../Internhip_Project/graph_extraction/NYC_C10_S4.npz" \
    --graph-db "New_York_network_enhanced_attributes.pkl.gz" \
    --use-folium \
    --out-plot "solution_static.png" \
    --out-html "solution_interactive.html"
```

### Full Example with All Parameters
```bash
python infer_from_npz.py \
    --npz "../Internhip_Project/graph_extraction/NYC_C10_S4.npz" \
    --checkpoint "ExperimentalLog/train/10/rollout/C10_02_42_53.467564/best.pt" \
    --nodes 10 \
    --charging_num 4 \
    --graph-db "New_York_network_enhanced_attributes.pkl.gz" \
    --use-folium \
    --out-html "NYC_solution_interactive.html"
```

## Output Files

### HTML File (Folium Map)
- **Format**: Interactive HTML file
- **Default name**: `inference_solution.html`
- **How to view**: Open in any web browser (Chrome, Firefox, Edge, etc.)
- **File size**: Typically 200-500 KB
- **Features**: Fully interactive, zoomable, clickable

### PNG File (Matplotlib - Optional)
- **Format**: Static PNG image
- **Default name**: `inference_solution.png`
- **Resolution**: 300 DPI (high quality)
- **Features**: Static snapshot, good for reports/papers

## Map Legend

### POI Markers
- 🏠 **Red Home Icon** - Depot (Start/End point)
- ⚡ **Orange Bolt Icon** - Depot Charging Station
- 🔌 **Green Charging Icon** - Public Charging Stations (S1-S4)
- 👤 **Purple User Icon** - Customer locations (numbered)

### Route Display
- **Colored lines** - Route segments following actual streets
- **Color gradient** - Shows progression (blue = early, red = late in tour)
- **Arrows** - Direction indicators on each segment
- **Numbers in circles** - Step numbers along the route

## Interactive Features

### Click Actions
- **POI markers**: Shows popup with details (index, lat/lon, type)
- **Route segments**: Shows step number, distance, number of nodes
- **Map layers button**: Switch between different map styles

### Controls
- **Zoom**: Mouse wheel or +/- buttons
- **Pan**: Click and drag
- **Fullscreen**: Button in top-left corner
- **Measure**: Button in bottom-left for distance measurement
- **Layers**: Control in top-right to toggle route/POIs

## Requirements

### Python Packages
```bash
pip install folium matplotlib networkx numpy torch
```

All packages are already installed in your environment! ✅

## Advantages of Folium Map

1. **Real-world context**: See actual NYC streets and geography
2. **Interactive exploration**: Zoom, pan, click for details
3. **Better presentation**: Impressive for demos and reports
4. **Shareable**: Easy to share HTML file with anyone
5. **No dependencies**: Recipient only needs a web browser
6. **Multiple views**: Switch between different map styles
7. **Distance validation**: Measure tool to verify routes
8. **Accurate projection**: Uses actual lat/lon coordinates

## Tips

- For **presentations**: Use fullscreen mode and dark map theme
- For **analysis**: Use measure tool to verify distances
- For **reports**: Take screenshots at different zoom levels
- For **sharing**: Compress HTML with zip if file is large
- For **printing**: Use the matplotlib PNG output instead

## Troubleshooting

### Map doesn't load
- Check internet connection (needs to download map tiles)
- Try different browser
- Check if HTML file is not corrupted

### No route displayed
- Ensure `--graph-db` parameter points to valid .pkl.gz file
- Verify NPZ file contains `poi_node_ids` field
- Check console output for errors

### Markers in wrong location
- Verify NPZ was built from same graph as --graph-db
- Check that coordinates are in lat/lon format (not normalized)

## Example Output

When you run with `--use-folium`, you'll see:
```
🗺️  Creating interactive Folium map...
   Step 1: 0 → 5 (1234.5m via 45 nodes)
   Step 2: 5 → 7 (987.3m via 32 nodes)
   ...
✓ Total real-world distance: 15234.5 meters (15.23 km)
✅ Interactive Folium map saved: inference_solution.html
   Open in browser to explore the solution on real-world map!
```

Then open `inference_solution.html` in your browser to explore! 🌍
