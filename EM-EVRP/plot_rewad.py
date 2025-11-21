import argparse, os, glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="", help="Path to Epoch_*.csv (e.g., ExperimentalData/train_data/10/rollout/Epoch_...csv)")
    ap.add_argument("--out", type=str, default="", help="Output PNG path (defaults next to CSV)")
    ap.add_argument("--show", action="store_true", help="Show the plot window")
    args = ap.parse_args()

    if not args.csv:
        candidates = glob.glob(os.path.join("ExperimentalData", "train_data", "**", "Epoch_*.csv"), recursive=True)
        if not candidates:
            raise SystemExit("No Epoch_*.csv found. Pass --csv explicitly.")
        args.csv = max(candidates, key=os.path.getmtime)

    data = np.loadtxt(args.csv, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    epochs = np.arange(1, data.shape[0] + 1)
    train_reward = data[:, 0]
    valid_reward = data[:, 3] if data.shape[1] >= 4 else None

    # Derive descriptive output name if not provided: include nodes and baseline from path
    if not args.out:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_dir = os.path.dirname(args.csv)
        norm_dir = os.path.normpath(out_dir)
        parts = norm_dir.split(os.sep)
        # Expect .../ExperimentalData/train_data/<nodes>/<baseline>
        nodes = parts[-2] if len(parts) >= 2 else "unknown"
        baseline = parts[-1] if len(parts) >= 1 else "unknown"
        args.out = os.path.join(out_dir, f"reward_nodes-{nodes}_baseline-{baseline}_{base}.png")

    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(epochs, train_reward, "-o", label="Train reward (energy cost)")
    if valid_reward is not None:
        plt.plot(epochs, valid_reward, "-s", label="Validation reward")
    plt.xlabel("Epoch")
    plt.ylabel("Mean energy consumption (lower is better)")
    # Improve title with metadata if available
    title_suffix = ""
    try:
        title_suffix = f" (nodes={nodes}, baseline={baseline})"
    except Exception:
        title_suffix = ""
    plt.title(f"Reward evolution over {len(epochs)} epochs{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved: {args.out}")
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()