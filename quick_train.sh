#!/bin/bash
# Quick Training Demo - Very small scale for fast results (~5-10 minutes)

cd DM-EVRP

echo "Starting QUICK training demo..."
echo "This uses minimal parameters for fast demonstration"
echo ""

python run.py \
  --nodes 20 \
  --train-size 1280 \
  --valid-size 256 \
  --iterations 5 \
  --batch_size 64 \
  --baselines rollout \
  --actor_lr 1e-4

echo ""
echo "Training complete! Check results at:"
echo "  - Logs: ExperimentalLog/train/20/rollout/"
echo "  - Data: ExperimentalData/train_data/20/rollout/"
