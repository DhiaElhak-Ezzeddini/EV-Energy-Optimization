# Data Exploration & Visualization Tools

## Quick Reference

### 1. Run the Pipeline
```bash
cd DRL-Energy-optimal-Routing-for-Electric-Vehicles/DM-EVRP
python run.py --test --nodes 20 --test_size 10
```

### 2. Quick Analysis (Terminal Output)
```bash
cd ..
python quick_analysis.py
```

Output example:
```
Metric                    Mean         Std          Min          Max
-------------------------------------------------------------------------
Distance (km)              1160.13      39.95      1116.52      1213.03
Energy (kWh)                512.89      17.75       496.45       537.54
Time (seconds)               28.55       0.81        27.63        29.61

Energy Efficiency: 0.442 kWh/km
```

### 3. Full Visualization (Generates PNG Files)
```bash
python explore_visualize_data.py
```

Creates 3 visualization files:
- `evrp_instance_detailed.png` - Problem instance structure (159 KB)
- `evrp_results_analysis.png` - Performance distributions (110 KB)
- `evrp_metrics_boxplot.png` - Statistical summaries (81 KB)

## File Organization

```
DRL-Energy-optimal-Routing-for-Electric-Vehicles/
│
├── explore_visualize_data.py      # Main visualization script
├── quick_analysis.py              # Quick terminal summary
├── DATA_EXPLORATION_GUIDE.md      # Detailed documentation
├── EXPLORATION_README.md          # This file
│
├── DM-EVRP/                       # Distance Minimization
│   ├── run.py                     # Main execution script
│   ├── ExperimentalData/
│   │   └── test_data/
│   │       └── 20/
│   │           └── *.pkl          # Test instances (Pickle)
│   └── ExperimentalLog/
│       └── test/
│           └── 20/
│               ├── data_record/
│               │   └── *.csv      # Results (CSV)
│               └── graph/
│                   └── *.png      # Route visualizations
│
└── EM-EVRP/                       # Energy Minimization
    └── (same structure as DM-EVRP)
```

## Data Files Explained

### Pickle Files (.pkl)
**What**: Test problem instances
**Where**: `ExperimentalData/test_data/{nodes}/{size}_seed{seed}.pkl`
**Contains**:
- Node locations (depot, charging stations, customers)
- Demands, distances, slopes
- Initial state (load, SOC, time)

### CSV Files (.csv)
**What**: Solution results
**Where**: `ExperimentalLog/test/{nodes}/data_record/{timestamp}.csv`
**Format**:
```csv
cost,duration,energy
1160.13,28.55,512.89
```

### PNG Files (.png)
**What**: Route visualizations (if --plot_num > 0)
**Where**: `ExperimentalLog/test/{nodes}/graph/batch{idx}_{cost}.png`
**Shows**: Optimized vehicle routes with nodes and paths

## Key Metrics

### Performance Indicators
- **Cost (Distance)**: Total route length in km - LOWER is better
- **Energy**: Battery consumption in kWh - LOWER is better
- **Duration**: Computation time in seconds - LOWER is better
- **Energy Efficiency**: kWh per km - LOWER is better

### Problem Parameters
- **Nodes**: 27 total (2 depots + 5 stations + 20 customers)
- **Vehicle**: BYD van, 4000 KG capacity, 80 kWh battery
- **Constraints**: Capacity, battery, time (10 hours), service time

## Common Tasks

### Compare Different Test Sizes
```bash
cd DM-EVRP

# Small test (fast)
python run.py --test --nodes 20 --test_size 5

# Medium test
python run.py --test --nodes 20 --test_size 50

# Large test (slow but statistical)
python run.py --test --nodes 20 --test_size 200
```

### Compare Node Counts
```bash
# 20 customers
python run.py --test --nodes 20 --test_size 10

# 50 customers (harder problem)
python run.py --test --nodes 50 --test_size 10

# 100 customers (very hard)
python run.py --test --nodes 100 --test_size 5
```

### Generate Route Plots
```bash
# Generate visualizations for first 5 solutions
python run.py --test --nodes 20 --test_size 10 --plot_num 5
```

### Compare DM vs EM
```bash
# Run both
cd DM-EVRP
python run.py --test --nodes 20 --test_size 20

cd ../EM-EVRP
python run.py --test --nodes 20 --test_size 20

# Analyze both
cd ..
python quick_analysis.py

# Uncomment compare_dm_vs_em() in quick_analysis.py to see comparison
```

## Visualization Gallery

### Instance Structure Visualization
Shows 4 panels:
1. **Node Map**: Spatial layout with color-coded node types
2. **Demand Bar Chart**: Customer demand distribution
3. **Distance Heatmap**: Inter-node distances
4. **Slope Heatmap**: Elevation changes (affects energy)

### Results Analysis
Shows 4 panels:
1. **Cost Histogram**: Distribution of route distances
2. **Energy Histogram**: Distribution of energy consumption
3. **Cost vs Energy Scatter**: Correlation analysis
4. **Time Histogram**: Computation time distribution

### Metrics Box Plots
Shows 3 panels:
1. **Distance Box Plot**: Statistical distribution
2. **Energy Box Plot**: Statistical distribution
3. **Time Box Plot**: Statistical distribution

## Interpreting Results

### Untrained Model (Random Init)
- High variance in solutions
- Suboptimal routes
- Good for baseline comparison
- Typical: ~1160 km, ~513 kWh

### Trained Model (After 100 epochs)
- Lower variance
- Near-optimal routes
- 20-30% improvement expected
- Typical: ~800-900 km, ~350-400 kWh

### Good Solution Indicators
✓ Low distance and energy
✓ All customers served
✓ No constraint violations
✓ Fast computation (<30 seconds)

### Red Flags
✗ Extremely high energy (>600 kWh for 20 customers)
✗ Very long routes (>1500 km for 20 customers)
✗ Slow computation (>60 seconds for 20 customers)
✗ Failed instances (check for errors)

## Advanced Usage

### Batch Processing
```python
# Run multiple experiments
import subprocess

node_counts = [20, 30, 40, 50]
test_sizes = [10, 20, 50]

for nodes in node_counts:
    for size in test_sizes:
        cmd = f"python run.py --test --nodes {nodes} --test_size {size}"
        subprocess.run(cmd, shell=True, cwd="DM-EVRP")
```

### Custom Visualizations
```python
from explore_visualize_data import EVRPDataExplorer
import matplotlib.pyplot as plt

explorer = EVRPDataExplorer()
pkl_files, csv_files, _ = explorer.list_available_data()

# Load data
data = explorer.load_test_instance(pkl_files[0])
df = explorer.load_results_csv(csv_files[0])

# Custom plot: Energy per distance
plt.figure(figsize=(10, 6))
plt.scatter(df['cost'], df['energy'] / df['cost'])
plt.xlabel('Route Distance (km)')
plt.ylabel('Energy Efficiency (kWh/km)')
plt.title('Energy Efficiency vs Route Length')
plt.grid(True)
plt.savefig('energy_efficiency.png')
```

### Statistical Analysis
```python
import pandas as pd
import scipy.stats as stats

# Load results
df = pd.read_csv("DM-EVRP/ExperimentalLog/test/20/data_record/*.csv")

# Correlation analysis
corr = df.corr()
print("Correlation Matrix:")
print(corr)

# Hypothesis testing: Is energy correlated with distance?
statistic, pvalue = stats.pearsonr(df['cost'], df['energy'])
print(f"Correlation: {statistic:.3f}, p-value: {pvalue:.6f}")

# Confidence intervals
ci_cost = stats.t.interval(0.95, len(df)-1,
                           loc=df['cost'].mean(),
                           scale=stats.sem(df['cost']))
print(f"95% CI for cost: [{ci_cost[0]:.2f}, {ci_cost[1]:.2f}]")
```

## Troubleshooting

### No data files found
**Problem**: Scripts can't find data files
**Solution**: Make sure you ran the pipeline first:
```bash
cd DM-EVRP && python run.py --test --nodes 20 --test_size 10
```

### Visualization errors
**Problem**: Can't generate plots
**Solution**: Install dependencies:
```bash
pip install matplotlib seaborn pandas
```

### Memory errors
**Problem**: Out of memory with large datasets
**Solution**: Reduce sizes:
```bash
python run.py --test --nodes 20 --test_size 5 --train-size 100 --valid-size 100
```

### Encoding errors (Windows)
**Problem**: Unicode characters not displaying
**Solution**: Already fixed in scripts (uses ASCII-safe characters)

## Next Steps

1. **Explore baseline**: Run tests with untrained model
2. **Train model**: Remove `--test` flag to train
3. **Compare results**: Before/after training
4. **Tune parameters**: Adjust hyperparameters
5. **Scale up**: Test larger problems
6. **Benchmark**: Compare with classical algorithms

## Support Files

- **DATA_EXPLORATION_GUIDE.md**: Comprehensive documentation (180+ lines)
- **explore_visualize_data.py**: Full visualization suite (350+ lines)
- **quick_analysis.py**: Quick terminal summary (150+ lines)

## Questions?

- Check the main README.md for project overview
- See DATA_EXPLORATION_GUIDE.md for detailed explanations
- Examine the code in explore_visualize_data.py for customization
- Refer to the paper for algorithm details

Happy exploring! 🚗🔋⚡
