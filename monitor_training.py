"""
Monitor Training Progress
This script monitors the training progress by checking log files and checkpoints.
"""

import os
import time
from pathlib import Path
import pandas as pd

def monitor_training(base_dir="EM-EVRP", nodes=10, baseline="rollout"):
    """Monitor training progress in real-time"""

    print("\n" + "="*80)
    print("TRAINING MONITOR")
    print("="*80)

    # Paths
    log_dir = Path(base_dir) / "ExperimentalLog" / "train" / str(nodes) / baseline
    data_dir = Path(base_dir) / "ExperimentalData" / "train_data" / str(nodes) / baseline

    print(f"\nMonitoring directory: {log_dir}")
    print(f"Data directory: {data_dir}")

    # Check if training has started
    if not log_dir.exists():
        print("\n[!] Training not started yet. Waiting for log directory...")
        print(f"    Expected path: {log_dir}")
        return

    # Check for checkpoint files
    print("\n" + "-"*80)
    print("CHECKPOINTS")
    print("-"*80)

    best_checkpoint = log_dir / "best.pt"
    checkpoints_dir = log_dir / "checkpoints"

    if best_checkpoint.exists():
        size_mb = best_checkpoint.stat().st_size / (1024 * 1024)
        mtime = time.ctime(best_checkpoint.stat().st_mtime)
        print(f"[OK] Best checkpoint found: {size_mb:.2f} MB (modified: {mtime})")
    else:
        print("[*] Best checkpoint not yet created")

    if checkpoints_dir.exists():
        epoch_checkpoints = list(checkpoints_dir.glob("*/epoch-*.pt"))
        if epoch_checkpoints:
            print(f"[OK] Found {len(epoch_checkpoints)} epoch checkpoints")
            latest = max(epoch_checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"    Latest: {latest.name}")
        else:
            print("[*] No epoch checkpoints yet")
    else:
        print("[*] Checkpoints directory not created yet")

    # Check for training data logs
    print("\n" + "-"*80)
    print("TRAINING LOGS")
    print("-"*80)

    if data_dir.exists():
        csv_files = list(data_dir.glob("Epoch_*.csv"))
        xls_files = list(data_dir.glob("train*.xls"))

        if csv_files:
            print(f"[OK] Found {len(csv_files)} epoch log files")

            # Read latest epoch log
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            print(f"    Latest: {latest_csv.name}")

            try:
                # Try to read the CSV
                df = pd.read_csv(latest_csv)
                if len(df) > 0:
                    print(f"\n    Epochs completed: {len(df)}")
                    print(f"    Columns: {', '.join(df.columns.tolist())}")

                    if 'epoch' in df.columns and 'avg_cost' in df.columns:
                        print("\n    Recent Progress:")
                        print(f"    {'Epoch':<10} {'Avg Cost':<15} {'Best Cost':<15}")
                        print("    " + "-"*40)
                        for _, row in df.tail(5).iterrows():
                            epoch = int(row.get('epoch', 0))
                            avg_cost = float(row.get('avg_cost', 0))
                            best_cost = float(row.get('best_cost', avg_cost))
                            print(f"    {epoch:<10} {avg_cost:<15.2f} {best_cost:<15.2f}")
            except Exception as e:
                print(f"    [!] Could not read CSV: {e}")

        else:
            print("[*] No epoch log files yet")

        if xls_files:
            print(f"[OK] Found {len(xls_files)} Excel log files")
        else:
            print("[*] No Excel log files yet")

    else:
        print("[*] Training data directory not created yet")

    # Check for validation plots
    print("\n" + "-"*80)
    print("VALIDATION PLOTS")
    print("-"*80)

    plot_files = list(log_dir.glob("*.png"))
    if plot_files:
        print(f"[OK] Found {len(plot_files)} validation plots")
        for plot in plot_files:
            print(f"    - {plot.name}")
    else:
        print("[*] No validation plots yet")

    print("\n" + "="*80)


def check_training_status():
    """Quick check if training is running"""

    import subprocess

    print("\nChecking for running Python processes...")

    try:
        # Check for Python processes
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True,
            text=True
        )

        if "python.exe" in result.stdout.lower():
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                print(f"[OK] Found {len(lines)-1} Python process(es) running")
                return True
        else:
            print("[!] No Python processes found")
            return False

    except Exception as e:
        print(f"[!] Could not check processes: {e}")
        return False


if __name__ == "__main__":
    print("\n*** EVRP Training Monitor ***")

    # Check if training is running
    is_running = check_training_status()

    if is_running:
        print("\n[*] Training appears to be running...")
    else:
        print("\n[!] Training may not be running")
        print("    Start training with: python run.py --nodes 20")

    # Monitor training progress
    monitor_training()

    print("\n[*] To continuously monitor, run this script repeatedly or use:")
    print("    watch -n 10 python monitor_training.py")
    print("\n" + "="*80)
