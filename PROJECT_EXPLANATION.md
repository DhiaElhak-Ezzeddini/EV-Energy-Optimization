# Project Explanation — EV-Energy-Optimization

This document explains the repository structure, data flow, and how to run
and extend the EV-Energy-Optimization project step-by-step.

> Location: `PROJECT_EXPLANATION.md` (repo root)

---

## Overview

This project implements Deep Reinforcement Learning (DRL) approaches for
Electric Vehicle Routing Problems (EVRP). Two parallel experiment folders
exist: `DM-EVRP` and `EM-EVRP` (distance-minimizing and energy-minimizing
variants). Each folder contains training and evaluation pipelines, neural
network models, problem/environment definitions, utilities, and baseline
algorithms (Gurobi, ACO, ALNS).

High-level components:
- Top-level helper scripts: `quick_analysis.py`, `explore_visualize_data.py`,
  `monitor_training.py`, `quick_train.sh`
- Experiment folders: `DM-EVRP/` and `EM-EVRP/` with: `run.py`, `trainer.py`,
  `nets/`, `problems/`, `utils/`, and `BaselineAlgorithms/`
- `ExperimentalData/` and `ExperimentalLog/` store datasets and results

---

## Top-level scripts

- `quick_analysis.py`
  - Quick console summary of CSV test outputs found under
    `*/ExperimentalLog/test/*/data_record/`.
  - Parses lines in CSVs (expected format `cost,duration,energy`) and prints
    mean/std/min/max and best/worst instances.

- `explore_visualize_data.py`
  - `EVRPDataExplorer` class to list files, load pickled test instances,
    create route/demand/distance/slope visualizations and metrics plots.
  - Saves PNG visualizations used for reports.

- `monitor_training.py`
  - Monitors training artifacts: checkpoint files under
    `ExperimentalLog/train/<nodes>/<baseline>/`, epoch CSV logs under
    `ExperimentalData/train_data/...`, and validation plots.
  - Includes a Windows `tasklist` check for running python processes.

- `quick_train.sh` (unix shell)
  - Convenience wrapper for launching training (on Windows use PowerShell
    commands shown later).

---

## Per-experiment layout (`DM-EVRP/` and `EM-EVRP/`)

Each folder follows the same structure. Key files and responsibilities:

- `run.py`
  - CLI argument parsing and orchestration that calls `train_EVRP(args)` in
    `trainer.py`.

- `trainer.py`
  - Core training & evaluation:
    1. Constructs `VehicleRoutingDataset` (from `problems/EVRP.py`) for
       train/validation/test.
    2. Builds the actor network (options: `DRL`, `AM`, `pointer`).
    3. Loads checkpoint (if provided) with `utils.functions.torch_load_cpu`.
    4. Training flow: choose baseline (`rollout`, `critic`, `exponential`),
       optimizer + LR scheduler, then run `train()`.
    5. Test flow: generate or load test dataset and call `eval_dataset`.
  - Checkpoints and logs saved under `ExperimentalLog` and `ExperimentalData`.

- `nets/`
  - `DRLModel.py`: Attention-based `AttentionModel` used as actor. Encodes
    nodes with a `GraphAttentionEncoder` and decodes sequences via
    attention, returning log-probabilities and rewards (cost, energy).
  - `AM.py`: a variant of the attention model with a slightly different
    step/context projection.
  - `GraphEncoder.py`: `GraphAttentionEncoder` and supporting multi-head
    attention layers used to produce node embeddings.
  - `PointNetwork.py`: an alternative pointer-network based policy.

- `problems/EVRP.py`
  - `VehicleRoutingDataset` (subclass of `torch.utils.data.Dataset`).
  - Responsibilities:
    - Generate random EVRP instances (or load CVRPLIB instances if
      `--CVRP_lib_test` used).
    - Provide `static` (coordinates), `dynamic` (load, demand, soc, time),
      `distances`, and `slope` matrices.
    - Implement `update_mask()` and `update_dynamic()` used during decoding
      to enforce feasibility (time, SOC, demand) and to update state.
  - Data conventions used across the codebase:
    - `static`: shape `[2, num_nodes]` coordinates
    - `dynamic`: shape `[4, num_nodes]` fields (load, demand, soc, time)
    - `distances`: `[num_nodes, num_nodes]` Euclidean distances
    - `slopes`: `[num_nodes, num_nodes]` elevation gradients

- `utils/`
  - `data_utils.py`: `save_dataset`, `load_dataset` (pickle helpers).
  - `functions.py`: misc helpers such as `torch_load_cpu`, `move_to`, CVRPLIB
    parser (`read_file`), `parse_softmax_temperature`, and batch helpers.
  - `tensor_functions.py`: `compute_in_batches` to process memory-heavy
    operations in sub-batches.
  - `beam_search.py`: batched beam search tools and `CachedLookup`.
  - `reinforce_baselines.py`: baseline classes used for REINFORCE
    (RolloutBaseline, CriticBaseline, ExponentialBaseline, WarmupBaseline).
  - `plot_delivery_graph.py`: route plotting helper used during evaluation.

- `BaselineAlgorithms/`
  - Implementations for classical baselines like `Gurobi.py` (MILP model
    for EVRP), `ACO.py`, `ALNS.py` for comparison.

---

## Training: step-by-step

1. Launch training from a folder (example uses `DM-EVRP`):

```powershell
# Example PowerShell command (Windows)
cd DM-EVRP
python .\run.py --test False --nodes 20 --iterations 50 --batch_size 1024
```

2. `run.py` builds an `args` namespace and calls `train_EVRP(args)`.

3. `trainer.train_EVRP(args)`:
   - Creates `VehicleRoutingDataset` instances for train & valid sets.
   - Instantiates the actor network (default `DRL` AttentionModel).
   - Loads checkpoint if provided.
   - Configures baseline and optimizer.
   - Runs `train()` which:
     - Wraps training data with the baseline and creates DataLoader.
     - For each batch: actor samples trajectories, compute reward R,
       baseline value `bl_val`, advantage = reward - baseline.
     - Loss = (advantage * logp).mean() + baseline loss; backprop + clip
       gradients + optimizer step.
     - Save epoch checkpoint and update best model if validation improves.

4. Checkpoints and logs appear under:
   - `ExperimentalLog/train/<nodes>/<baseline>/...` for model artifacts
   - `ExperimentalData/train_data/<nodes>/<baseline>/` for epoch/batch CSVs

---

## Evaluation / Testing: step-by-step

1. Create or provide test dataset. If none provided, `trainer.py` generates
   and saves a pickle under `ExperimentalData/test_data/<nodes>/`.

2. Run `run.py` in test mode (example):

```powershell
cd DM-EVRP
python .\run.py --test --nodes 20 --test_size 256 --checkpoint ExperimentalLog\train\20\rollout\best.pt
```

3. Evaluation (`trainer.eval_dataset` / `_eval_dataset`):
   - Model chooses decode strategy: `sample`, `greedy`, or `bs` (beam search).
   - For `sample` or `greedy` it may use `sample_many` to draw multiple
     rollouts (controlled by `--width`). For `bs`, it uses batched beam
     search utilities.
   - Results `(cost, duration, energy)` are saved to `ExperimentalLog/test/<nodes>/data_record/<timestamp>.csv`.
   - Optional route PNGs saved to `ExperimentalLog/test/<nodes>/graph/` when plotting is enabled.

4. Use `explore_visualize_data.py` or `quick_analysis.py` to inspect results
   and create plots.

---

## Algorithmic notes

- The policy is an attention-based model which encodes nodes with a
  `GraphAttentionEncoder` and decodes next-node choices via attention
  between a context and node embeddings.
- Feasibility (masking) and environment updates (SOC, remaining demand,
  time) are implemented in `problems/EVRP.py` (`update_mask`, `update_dynamic`)
  and are used inside the actor's decoding loop to ensure valid tours.
- Energy consumption is modeled with simple physics-based formulas (mass,
  drag, slope, motor/battery efficiency) inside `EVRP.py`.
- Training uses REINFORCE-style policy gradient with a baseline to reduce
  variance (rollout baseline is the default).

---

## Debugging & Extension pointers

- If tours are infeasible: inspect `problems/EVRP.py` (`update_mask` and
  `update_dynamic`). This logic enforces time/SOC/demand constraints.

- If gradients explode: check `trainer.clip_grad_norms` and learning rates
  in `run.py`.

- For device/checkpoint loading issues: use `utils.functions.torch_load_cpu`
  to load weights on CPU and then move to GPU.

- To add a new model: implement a new module in `nets/` and provide method
  signatures used by `trainer.py`: `forward`, `sample_many`, `beam_search`,
  `set_decode_type`.

---

## Quick commands (PowerShell)

Train (example):

```powershell
cd DM-EVRP
python .\run.py --test False --nodes 20 --iterations 50 --batch_size 1024
```

Test (example):

```powershell
cd DM-EVRP
python .\run.py --test --nodes 20 --test_size 256 --checkpoint ExperimentalLog\train\20\rollout\best.pt
```

Visualize outputs:

```powershell
python .\explore_visualize_data.py
```

Quick summary of results:

```powershell
python .\quick_analysis.py
```

---

## Next steps I can take for you

- Create a `README.md` summarizing these usage notes (I can add badges,
  quick-run commands, and a minimal `requirements.txt`).
- Add a `run_example.ps1` PowerShell script to launch a minimal train/test
  example on Windows and demonstrate the expected output files.
- Produce a per-file, line-by-line expanded explanation or a call graph.

Tell me which of the above you'd like next and I will implement it.
