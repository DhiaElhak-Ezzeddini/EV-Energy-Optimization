"""
Data Exploration and Visualization for EVRP Results
This script loads, explores, and visualizes results from the DRL-EVRP pipeline.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class EVRPDataExplorer:
    def __init__(self, base_dir="DM-EVRP"):
        self.base_dir = Path(base_dir)
        self.test_data_dir = self.base_dir / "ExperimentalData" / "test_data"
        self.test_log_dir = self.base_dir / "ExperimentalLog" / "test"

    def list_available_data(self):
        """List all available data files"""
        print("=" * 80)
        print("AVAILABLE DATA FILES")
        print("=" * 80)

        print("\n[Pickle Files - Test Instances]:")
        pkl_files = list(self.test_data_dir.rglob("*.pkl"))
        for i, f in enumerate(pkl_files, 1):
            size_kb = f.stat().st_size / 1024
            print(f"  {i}. {f.relative_to(self.base_dir)} ({size_kb:.1f} KB)")

        print("\n[CSV Files - Test Results]:")
        csv_files = list(self.test_log_dir.rglob("*.csv"))
        for i, f in enumerate(csv_files, 1):
            size_kb = f.stat().st_size / 1024
            print(f"  {i}. {f.relative_to(self.base_dir)} ({size_kb:.1f} KB)")

        print("\n[PNG Files - Route Visualizations]:")
        png_files = list(self.test_log_dir.rglob("*.png"))
        if png_files:
            for i, f in enumerate(png_files, 1):
                print(f"  {i}. {f.relative_to(self.base_dir)}")
        else:
            print("  (None generated - use --plot_num > 0 to generate route plots)")

        return pkl_files, csv_files, png_files

    def load_test_instance(self, pkl_file):
        """Load a test instance from pickle file"""
        print(f"\n[Loading test instance: {pkl_file.name}]")

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        # Data is a list of tuples: (static, dynamic, distances, slopes)
        print(f"  [OK] Loaded {len(data)} test instances")

        # Examine first instance
        if len(data) > 0:
            first_item = data[0]
            # Handle both tuple and list formats
            if isinstance(first_item, (tuple, list)):
                static, dynamic, distances, slopes = first_item
                # Convert to tensor if needed
                if not hasattr(static, 'shape'):
                    static = torch.tensor(static)
                if not hasattr(dynamic, 'shape'):
                    dynamic = torch.tensor(dynamic)
                if not hasattr(distances, 'shape'):
                    distances = torch.tensor(distances)
                if not hasattr(slopes, 'shape'):
                    slopes = torch.tensor(slopes)

                print(f"\n  Instance Structure:")
                print(f"    - Static (locations): {static.shape}")
                print(f"    - Dynamic (load, demand, SOC, time): {dynamic.shape}")
                print(f"    - Distances matrix: {distances.shape}")
                print(f"    - Slopes matrix: {slopes.shape}")
            else:
                print(f"\n  Unknown data format: {type(first_item)}")

            # Decode the structure
            num_nodes = static.shape[1]
            print(f"\n  Total nodes: {num_nodes}")
            print(f"    - 1 depot")
            print(f"    - 1 depot_charging")
            print(f"    - 5 charging stations (default)")
            print(f"    - {num_nodes - 7} customer nodes")

        return data

    def load_results_csv(self, csv_file):
        """Load and parse results CSV"""
        print(f"\n[Loading results: {csv_file.name}]")

        # Read CSV - format: cost, duration, energy
        with open(csv_file, 'r') as f:
            lines = f.readlines()

        results = []
        for line in lines:
            if '#' not in line and line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        cost = float(parts[0])
                        duration = float(parts[1])
                        energy = float(parts[2])
                        results.append({'cost': cost, 'duration': duration, 'energy': energy})
                    except ValueError:
                        continue

        df = pd.DataFrame(results)

        if len(df) > 0:
            print(f"\n  [OK] Loaded {len(df)} test results")
            print(f"\n  Summary Statistics:")
            print(df.describe())

        return df

    def visualize_test_instance(self, data, instance_idx=0, save_path=None):
        """Visualize a single test instance"""
        if instance_idx >= len(data):
            print(f"Instance {instance_idx} not found. Only {len(data)} instances available.")
            return

        static, dynamic, distances, slopes = data[instance_idx]

        # Convert to tensors if needed
        if not isinstance(static, torch.Tensor):
            static = torch.tensor(static)
        if not isinstance(dynamic, torch.Tensor):
            dynamic = torch.tensor(dynamic)
        if not isinstance(distances, torch.Tensor):
            distances = torch.tensor(distances)
        if not isinstance(slopes, torch.Tensor):
            slopes = torch.tensor(slopes)

        # Extract coordinates (x, y) - static is [2, num_nodes]
        coords = static.numpy().T  # [num_nodes, 2]
        demands = dynamic[1, :].numpy()  # demand values

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Node locations with types
        ax = axes[0, 0]
        depot = coords[0]
        depot_charging = coords[1]
        stations = coords[2:7]  # 5 charging stations
        customers = coords[7:]

        ax.scatter(depot[0], depot[1], c='red', s=300, marker='s', label='Depot', zorder=5, edgecolors='black', linewidth=2)
        ax.scatter(depot_charging[0], depot_charging[1], c='orange', s=250, marker='D', label='Depot Charging', zorder=4, edgecolors='black', linewidth=2)
        ax.scatter(stations[:, 0], stations[:, 1], c='green', s=200, marker='^', label='Charging Stations', zorder=3, edgecolors='black', linewidth=1.5)
        ax.scatter(customers[:, 0], customers[:, 1], c='blue', s=100, marker='o', label='Customers', zorder=2, alpha=0.7)

        # Add node labels
        for i, (x, y) in enumerate(coords):
            ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title(f'EVRP Instance #{instance_idx}: Node Locations')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Demand heatmap
        ax = axes[0, 1]
        customer_demands = demands[7:]  # Only customer demands
        bars = ax.bar(range(len(customer_demands)), customer_demands, color='steelblue', edgecolor='black')
        ax.set_xlabel('Customer Node Index')
        ax.set_ylabel('Normalized Demand')
        ax.set_title('Customer Demands')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Distance matrix heatmap
        ax = axes[1, 0]
        dist_matrix = distances.numpy()
        im = ax.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xlabel('To Node')
        ax.set_ylabel('From Node')
        ax.set_title('Distance Matrix (Euclidean)')
        plt.colorbar(im, ax=ax, label='Distance')

        # Plot 4: Slope matrix heatmap
        ax = axes[1, 1]
        slope_matrix = slopes.numpy()
        im = ax.imshow(slope_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
        ax.set_xlabel('To Node')
        ax.set_ylabel('From Node')
        ax.set_title('Slope Matrix (Elevation Change)')
        plt.colorbar(im, ax=ax, label='Slope')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  [OK] Saved visualization to {save_path}")
        else:
            plt.savefig(f'evrp_instance_{instance_idx}.png', dpi=150, bbox_inches='tight')
            print(f"\n  [OK] Saved visualization to evrp_instance_{instance_idx}.png")

        plt.close()  # Close instead of show for non-interactive

    def visualize_results_distribution(self, df, save_path=None):
        """Visualize the distribution of results"""
        if df is None or len(df) == 0:
            print("No results data to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Cost distribution
        ax = axes[0, 0]
        ax.hist(df['cost'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(df['cost'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["cost"].mean():.2f}')
        ax.set_xlabel('Total Distance (Cost)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Route Distances')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Energy distribution
        ax = axes[0, 1]
        ax.hist(df['energy'], bins=20, color='green', edgecolor='black', alpha=0.7)
        ax.axvline(df['energy'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["energy"].mean():.2f}')
        ax.set_xlabel('Energy Consumption (kWh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Energy Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Cost vs Energy scatter
        ax = axes[1, 0]
        ax.scatter(df['cost'], df['energy'], alpha=0.6, s=100, c='purple', edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Total Distance (Cost)')
        ax.set_ylabel('Energy Consumption (kWh)')
        ax.set_title('Cost vs Energy Trade-off')
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = df['cost'].corr(df['energy'])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), verticalalignment='top')

        # Plot 4: Computation time
        ax = axes[1, 1]
        ax.hist(df['duration'], bins=20, color='orange', edgecolor='black', alpha=0.7)
        ax.axvline(df['duration'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {df["duration"].mean():.2f}s')
        ax.set_xlabel('Computation Time (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Solution Times')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  [OK] Saved results visualization to {save_path}")
        else:
            plt.savefig('evrp_results_distribution.png', dpi=150, bbox_inches='tight')
            print(f"\n  [OK] Saved results visualization to evrp_results_distribution.png")

        plt.close()  # Close instead of show for non-interactive

    def compare_metrics(self, df, save_path=None):
        """Create box plots comparing metrics"""
        if df is None or len(df) == 0:
            print("No results data to visualize")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Normalize data for comparison
        metrics = ['cost', 'energy', 'duration']
        titles = ['Route Distance', 'Energy Consumption (kWh)', 'Computation Time (s)']
        colors = ['steelblue', 'green', 'orange']

        for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[idx]

            # Box plot
            bp = ax.boxplot([df[metric]], widths=0.6, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           whiskerprops=dict(color='black', linewidth=1.5),
                           capprops=dict(color='black', linewidth=1.5))

            # Add scatter of actual points
            y = df[metric]
            x = np.random.normal(1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=50, color='black')

            # Statistics text
            stats_text = f'Mean: {df[metric].mean():.2f}\nStd: {df[metric].std():.2f}\nMin: {df[metric].min():.2f}\nMax: {df[metric].max():.2f}'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_ylabel(title)
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Performance Metrics Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n  [OK] Saved metrics comparison to {save_path}")
        else:
            plt.savefig('evrp_metrics_comparison.png', dpi=150, bbox_inches='tight')
            print(f"\n  [OK] Saved metrics comparison to evrp_metrics_comparison.png")

        plt.close()  # Close instead of show for non-interactive


def main():
    """Main exploration workflow"""
    print("\n*** EVRP Data Explorer and Visualizer ***\n")

    # Initialize explorer
    explorer = EVRPDataExplorer()

    # List available data
    pkl_files, csv_files, png_files = explorer.list_available_data()

    if not pkl_files and not csv_files:
        print("\n[!] No data found! Run the pipeline first:")
        print("   cd DM-EVRP && python run.py --test --nodes 20 --test_size 10")
        return

    # Load and explore test instances
    if pkl_files:
        print("\n" + "=" * 80)
        print("EXPLORING TEST INSTANCES")
        print("=" * 80)

        # Load first pickle file
        data = explorer.load_test_instance(pkl_files[0])

        # Visualize first instance
        print("\n[Creating visualizations...]")
        explorer.visualize_test_instance(data, instance_idx=0,
                                        save_path='evrp_instance_detailed.png')

    # Load and explore results
    if csv_files:
        print("\n" + "=" * 80)
        print("EXPLORING TEST RESULTS")
        print("=" * 80)

        # Load first CSV file
        df = explorer.load_results_csv(csv_files[0])

        if len(df) > 0:
            # Visualize results
            print("\n[Creating result visualizations...]")
            explorer.visualize_results_distribution(df, save_path='evrp_results_analysis.png')
            explorer.compare_metrics(df, save_path='evrp_metrics_boxplot.png')

    print("\n" + "=" * 80)
    print("[*] EXPLORATION COMPLETE!")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("  [*] evrp_instance_detailed.png - Test instance structure")
    print("  [*] evrp_results_analysis.png - Results distribution")
    print("  [*] evrp_metrics_boxplot.png - Metrics comparison")


if __name__ == "__main__":
    main()
