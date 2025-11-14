"""
Quick Analysis Script for EVRP Data
Run this for a fast overview of your results.
"""

import pandas as pd
import pickle
from pathlib import Path
import torch

def quick_summary():
    """Print a quick summary of all test results"""

    print("\n" + "="*80)
    print("EVRP QUICK ANALYSIS SUMMARY")
    print("="*80)

    # Find all CSV result files
    base_dir = Path("DM-EVRP")
    csv_files = list(base_dir.glob("ExperimentalLog/test/*/data_record/*.csv"))

    if not csv_files:
        print("\n[!] No results found. Run the pipeline first:")
        print("    cd DM-EVRP && python run.py --test --nodes 20 --test_size 10")
        return

    print(f"\nFound {len(csv_files)} result file(s)")

    for csv_file in csv_files:
        print("\n" + "-"*80)
        print(f"File: {csv_file.name}")
        print("-"*80)

        # Read CSV
        try:
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
                            results.append({
                                'cost': cost,
                                'duration': duration,
                                'energy': energy
                            })
                        except ValueError:
                            continue

            if results:
                df = pd.DataFrame(results)

                print(f"\nNumber of test instances: {len(df)}")
                print(f"\n{'Metric':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
                print("-" * 73)

                for metric in ['cost', 'energy', 'duration']:
                    mean = df[metric].mean()
                    std = df[metric].std()
                    min_val = df[metric].min()
                    max_val = df[metric].max()

                    if metric == 'cost':
                        name = "Distance (km)"
                    elif metric == 'energy':
                        name = "Energy (kWh)"
                    else:
                        name = "Time (seconds)"

                    print(f"{name:<25} {mean:>11.2f}  {std:>11.2f}  {min_val:>11.2f}  {max_val:>11.2f}")

                # Calculate efficiency metrics
                energy_per_km = (df['energy'] / df['cost']).mean()
                print(f"\nEnergy Efficiency: {energy_per_km:.3f} kWh/km")

                # Best and worst solutions
                best_idx = df['cost'].idxmin()
                worst_idx = df['cost'].idxmax()

                print(f"\nBest solution (shortest route):")
                print(f"  Distance: {df.loc[best_idx, 'cost']:.2f} km")
                print(f"  Energy: {df.loc[best_idx, 'energy']:.2f} kWh")
                print(f"  Time: {df.loc[best_idx, 'duration']:.2f} seconds")

                print(f"\nWorst solution (longest route):")
                print(f"  Distance: {df.loc[worst_idx, 'cost']:.2f} km")
                print(f"  Energy: {df.loc[worst_idx, 'energy']:.2f} kWh")
                print(f"  Time: {df.loc[worst_idx, 'duration']:.2f} seconds")

        except Exception as e:
            print(f"[!] Error reading file: {e}")

    # Check pickle files
    print("\n" + "="*80)
    print("TEST INSTANCES")
    print("="*80)

    pkl_files = list(base_dir.glob("ExperimentalData/test_data/*/*.pkl"))

    if pkl_files:
        print(f"\nFound {len(pkl_files)} test instance file(s):")
        for pkl_file in pkl_files:
            size_kb = pkl_file.stat().st_size / 1024
            print(f"  - {pkl_file.relative_to(base_dir)} ({size_kb:.1f} KB)")

            # Load first one to show structure
            if pkl_file == pkl_files[0]:
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)

                    if data:
                        first = data[0]
                        if isinstance(first, (tuple, list)):
                            static, dynamic, distances, slopes = first
                            if isinstance(static, torch.Tensor):
                                num_nodes = static.shape[1]
                            else:
                                num_nodes = len(static[0]) if hasattr(static[0], '__len__') else 0

                            print(f"\n  Sample structure (first file):")
                            print(f"    Instances: {len(data)}")
                            print(f"    Nodes per instance: {num_nodes}")
                            print(f"      └─ 2 depots + 5 stations + {num_nodes-7} customers")
                except Exception as e:
                    print(f"    [!] Could not analyze: {e}")
    else:
        print("\n[!] No test instance files found")

    print("\n" + "="*80)
    print("\nTo generate visualizations, run:")
    print("  python explore_visualize_data.py")
    print("\nFor detailed guide, see:")
    print("  DATA_EXPLORATION_GUIDE.md")
    print("="*80 + "\n")


def compare_dm_vs_em():
    """Compare DM-EVRP vs EM-EVRP if both have results"""

    dm_csv = list(Path("DM-EVRP").glob("ExperimentalLog/test/*/data_record/*.csv"))
    em_csv = list(Path("EM-EVRP").glob("ExperimentalLog/test/*/data_record/*.csv"))

    if not dm_csv or not em_csv:
        print("\n[!] Need results from both DM-EVRP and EM-EVRP to compare")
        return

    print("\n" + "="*80)
    print("DM-EVRP vs EM-EVRP COMPARISON")
    print("="*80)

    def load_csv(csv_file):
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        results = []
        for line in lines:
            if '#' not in line and line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        results.append({
                            'cost': float(parts[0]),
                            'energy': float(parts[2])
                        })
                    except ValueError:
                        continue
        return pd.DataFrame(results)

    df_dm = load_csv(dm_csv[0])
    df_em = load_csv(em_csv[0])

    if len(df_dm) > 0 and len(df_em) > 0:
        print(f"\n{'Objective':<30} {'Avg Distance (km)':<20} {'Avg Energy (kWh)':<20}")
        print("-" * 70)
        print(f"{'DM-EVRP (Distance Min)':<30} {df_dm['cost'].mean():>18.2f}  {df_dm['energy'].mean():>18.2f}")
        print(f"{'EM-EVRP (Energy Min)':<30} {df_em['cost'].mean():>18.2f}  {df_em['energy'].mean():>18.2f}")

        dist_diff = ((df_em['cost'].mean() - df_dm['cost'].mean()) / df_dm['cost'].mean()) * 100
        energy_diff = ((df_dm['energy'].mean() - df_em['energy'].mean()) / df_em['energy'].mean()) * 100

        print(f"\nTrade-offs:")
        print(f"  EM-EVRP routes are {abs(dist_diff):.1f}% {'longer' if dist_diff > 0 else 'shorter'} but save {abs(energy_diff):.1f}% energy")

    print("="*80 + "\n")


if __name__ == "__main__":
    quick_summary()

    # Uncomment to compare DM vs EM if you have both:
    # compare_dm_vs_em()
