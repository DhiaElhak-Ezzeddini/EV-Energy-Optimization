# EM-EVRP: Energy-Aware Multi-Charging Electric Vehicle Routing (Deep RL Framework)

## 1. Overview
EM-EVRP is a PyTorch-based Deep Reinforcement Learning (DRL) framework for solving an Electric Vehicle Routing Problem variant with multiple charging stations, battery State of Charge (SOC) dynamics, load/demand constraints, elevation-based energy consumption, and time windows (implicit via remaining route time budget). It learns routing policies that minimize cumulative energy consumption (cost) while fulfilling customer demands under physical and operational constraints.

Target scenario (example used in this README):
- Customers (num_nodes): 10
- Charging stations (charging_num): 4
- Total sequence length: 2 (depot + depot_charging) + 4 (stations) + 10 (customers) = 16 nodes
- Training set size: 1,280,000 samples
- Batch size: 1024
- Epochs (iterations): 100

Two attention-based actor network variants are available:
- `nets/DRLModel.AttentionModel`: Augments node attention with route (tour) context embedding + vehicle state embedding.
- `nets/AM.AttentionModel`: Classic graph attention with step context projection, without explicit tour aggregation.

## 2. Problem Definition
Given a depot, a nearby depot charging location, multiple charging stations, and a set of customers with demands, find routes (variable length sequences) that:
1. Satisfy all customer demands (each demand can be split across visits if necessary due to load constraints).
2. Maintain SOC above feasibility thresholds (cannot traverse if insufficient energy).
3. Respect remaining global time budget (`t_limit`), per visit service time (`serve_time`) and charging duration (`charging_time`).
4. Minimize cumulative energy consumption across traversed edges.

The problem instance is stochastic (random coordinates, elevation, demands) unless using CVRPLIB-based deterministic files.

## 3. Data Generation (problems/EVRP.py)
For non-CVRPLIB training/testing:
- Depot: Random integer coordinates in [25,75]×[25,75].
- Depot charging: A jittered coordinate near depot (±3 offset, guaranteed non-identical). Serves to separate the conceptual “start/end” node from a charging variant (design choice for learning distinct semantics).
- Charging stations: Placed across quadrants using four bucketed coordinate ranges (0–50 vs 51–101) to encourage spatial diversity. Stations are ensured not to coincide with depot or depot_charging (resampling loop).
- Customers: Uniform integer coordinates in [0,101].
- Elevations: Random integer in [0,101] per node; normalized by /1000 to approximate kilometers → used to derive slopes.
- Demands: Integer in [1,max_demand] scaled by 0.25 then normalized by `max_load`. Depot, depot_charging and stations have zero demand.

Precomputation: For small instances (input_size ≤ 20 or num_samples ≤ 12800) full pairwise Euclidean distances and slopes are precomputed and stored; large-scale dataset generation defers these computations to model forward pass for memory efficiency.

## 4. Node Indexing Convention
Given `charging_num = 4` and `num_nodes = 10` (customers):
- Index 0: Depot
- Index 1: Depot charging (near depot, same semantics except SOC reset after charging-time if visited?)
- Indices 2–5: Charging stations (4 stations)
- Indices 6–15: Customers (10 customers)

Sequence length: `seq_len = 2 + charging_num + num_nodes`.

Dynamic shape allocation: `(num_samples, 1, seq_len)` per feature channel.

## 5. Dynamic State Tensor Layout
`dynamic` has 4 stacked channels (after concatenation):
1. Load: Remaining normalized vehicle load capacity (scalar replicated across nodes; node 0 holds canonical value).
2. Demand: Per-node remaining demand (customers > 0; depot node encodes residual load as -1 + load after partial service events).
3. SOC: Per-node replicated current battery State of Charge (SOC) value.
4. Time: Remaining route time budget.

Initializations:
- Load: 1.0 (normalized, meaning full capacity = `max_load`).
- SOC: `Start_SOC` (e.g., 100 units).
- Time: `t_limit` (global allowed driving time horizon).
- Demand: Customers randomly initialized; non-customer nodes set to zero.

## 6. Energy Consumption Model
Energy (Wh) consumed per traversed edge is derived from physical approximations:

Let:
- $m_c$: Curb mass (mc)
- $w$: Load scaling factor (w)
- $g$: Gravity (9.81)
- $C_d$: Drag coefficient (Cd)
- $A$: Frontal area (A)
- $\rho$: Air density (Ad)
- $C_r$: Rolling resistance coefficient (Cr)
- $v$: Vehicle speed (velocity, converted from km/h to m/s via $v/3.6$ in power term)
- $\text{slope}$: Elevation slope between nodes
- $m = m_c + (\text{load} \times \text{max\_load}) \times w$

Instantaneous mechanical power approximation:
$$P_m = \Big( 0.5 \cdot C_d \cdot A \cdot \rho \cdot (v/3.6)^2 + m g \cdot \text{slope} + m g C_r \Big) v$$

Traversal time for segment:
$$t_{seg} = \frac{\text{distance}}{v}$$

SOC consumption (charging and regenerative behavior):
$$\Delta SOC = \begin{cases} \text{motor\_d} \cdot \text{battery\_d} \cdot P_m \cdot t_{seg} / 3600 & P_m > 0 \\ \text{motor\_r} \cdot \text{battery\_r} \cdot P_m \cdot t_{seg} / 3600 & P_m < 0 \end{cases}$$

The reinforcement signal collects per-step `soc_consume` (positive depletion dominates; negative values model limited recuperation/regeneration).

## 7. Masking Logic (update_mask)
Mask determines feasible next nodes (1 = selectable, 0 = blocked). Conditions applied sequentially:
1. Base feasibility: Demand > 0 AND demand < current load.
2. Previously selected node is masked next round.
3. Charging access restrictions:
   - If currently at depot or station: block other stations except the selected rule-set.
   - If at customer: re-enable depot/stations for potential return/charge.
4. Force return to depot if (no load) OR (no remaining customer demand).
5. Block nodes that cannot be directly reached due to SOC or time (compute SOC/time cost for `chosen -> candidate`).
6. Block nodes if a two-leg path (`chosen -> candidate -> depot`) infeasible (SOC/time) AND also detour via nearest station path infeasible (`chosen -> candidate -> station -> depot`).
7. If all customers masked, ensure depot is selectable.

This layered masking enforces energy, time, load, and demand constraints while permitting charging detours. It drives termination when no further customer demand is serviceable.

## 8. Dynamic Update (update_dynamic)
Triggered after selecting node `selected` from current `now_idx`:
- Compute distance, slope → energy consumption → SOC decrement.
- Update remaining time budget (`t_limit` reset upon returning depot; subtract service or charging durations for corresponding node types).
- For customer visits:
  - Partial service: Load reduced by served amount; residual customer demand adjusted.
  - `dynamic.demands[0]` (depot slot) set to `-1 + load` encoding remaining capacity.
- For depot or station: Reset SOC to `Start_SOC`; at depot restore load to full.
- The per-step reward is `soc_consume` (energy effect), aggregated along route.

## 9. Decoder Lifecycle (DRLModel vs AM)
Shared high-level loop (max 1000 steps or until mask empties):
1. Build context (route-dependent for DRLModel; step context for AM).
2. Compute attention logits over feasible nodes (masked).
3. Sample or greedily pick next node depending on `decode_type`.
4. Update dynamic state and mask.
5. Accumulate reward (energy consumption of edge).
6. Iterate until termination condition (no feasible customer demands).

Differences:
- DRLModel: Explicit route feature aggregation via `route_feature()` combining mean tour embeddings + normalized vehicle context (load, SOC, time) then fused into query.
- AM: Uses step context projection (`project_step_context`) with current node embedding + (optionally) load scalar.

## 10. Training Pipeline (trainer.py)
Main function: `train_EVRP(args)`
1. Construct `VehicleRoutingDataset` train & validation sets.
2. Instantiate actor model (`DRL` or `AM`).
3. Instantiate baseline:
   - `exponential`, `critic`, or `rollout` (with optional `WarmupBaseline`).
4. For each epoch (iteration):
   - Wrap training dataset via baseline (rollout baseline precomputes greedy costs for variance reduction).
   - Sample batches of size `batch_size` with decode type = `sample`.
   - Forward pass returns (tour_indices, log_probs, per-step energy R).
   - Compute reward = sum(R) per batch element.
   - Advantage: `reward - baseline_value`.
   - REINFORCE loss: mean over batch of `(advantage.detach() * log_prob_sum)` + baseline loss (critic MSE or zero otherwise).
   - Gradient clip to `max_grad_norm`, optimizer step, optional LR decay.
   - Every 100 batches: log mean reward/loss and timing.
   - Validation using greedy decoding; save best checkpoint.

Loss sign convention: Framework minimizes energy directly; REINFORCE uses positive advantage. (If interpreting energy as cost, a negative sign could encourage minimizing by ascending; here, advantage formulation yields correct direction without explicit negation.)

## 11. Baselines (utils/reinforce_baselines.py)
- ExponentialBaseline: Moving average of recent batch reward.
- CriticBaseline: Learned state-value network predicting expected reward (energy) from encoded static/dynamic.
- RolloutBaseline: Maintains a greedy-policy snapshot; statistically tests candidate improvements with paired t-test (one-sided). If p < `bl_alpha`, baseline updated.
- WarmupBaseline: Interpolates between exponential and target baseline during initial epochs to stabilize learning.

## 12. Reward & Optimization Objective
Per-step reward = energy consumption (SOC delta) for traversed edge (positive depletion, negative regeneration). Episode (route) reward is sum of per-step energies. Training minimizes expected cumulative energy.

## 13. Decoding Strategies (Evaluation)
During validation/testing:
- Greedy: `decode_type=greedy`; selects argmax probability node.
- Sampling: Monte Carlo exploration; can be repeated width times.
- Beam Search (`decode_strategy='bs'`): Systematically expands top-K partial solutions using `beam_search.py` utilities.

`sample_many` supports repeated forward passes for multi-sample best cost selection.

## 14. Large-Scale Training Considerations
For `train_size = 1,280,000`:
- Memory: Avoid precomputing full distance/slope matrices if exceeding thresholds (falls back to on-the-fly computation inside model forward for large `num_samples`).
- I/O: Consider sharding dataset creation or generating on-demand if disk footprint is high.
- Throughput: Use GPU with mixed precision (future enhancement) and ensure batch size (1024) fits memory.
- Checkpointing: Frequent epoch checkpoints in `ExperimentalLog/train/<nodes>/<baseline>/...`; best model saved as `best.pt`.

## 15. Example Instance Journey (10 Customers, 4 Stations)
Initial shapes:
- `static`: (batch, 2, 16)
- `dynamic`: (batch, 4, 16)
- `distances`: (batch, 16, 16) (precomputed for small scale)
- `slope`: (batch, 16, 16)

Step 0:
1. `now_idx = 0` (depot). Mask enables first set of customers whose demand < load plus charging stations (according to rules).
2. Attention computes logits from embeddings and context; softmax yields probabilities over feasible nodes.
3. Sample selects a customer (e.g., index 9). Energy consumption computed; SOC decreases accordingly; time budget reduced by travel + service time.
4. Load decreases by served demand portion.

Intermediate Step (after several visits):
- If SOC drops near threshold or remaining time insufficient to serve additional customers, mask promotes returning to depot or detouring via nearest feasible charging station.
- Visiting charging station resets SOC to `Start_SOC` after charging time.

Termination:
- When all customer demands satisfied (`demands[:, charging_num+1:] == 0`) OR load == 0 with no remaining demand, mask collapses to depot, ending sequence.
- Reward trace R aggregated; final cost = sum(R).

Validation pass (greedy):
- Replaces sampling with deterministic selection for stable evaluation metric.

## 16. Logging & Outputs
Training:
- Batch CSV: `ExperimentalData/train_data/<nodes>/<baseline>/Batch_C<nodes>_<timestamp>.csv` (mean reward, mean reinforce loss every 100 batches).
- Epoch CSV: `Epoch_C<nodes>_<timestamp>.csv` (reward, loss, mean batch time, validation cost).
- XLS summary: epoch-level average reward/loss.
- Checkpoints: `ExperimentalLog/train/<nodes>/<baseline>/<run_id>/checkpoints/epoch-X/epoch-X.pt` & `best.pt`.

Testing:
- CSV: `ExperimentalLog/test/<nodes>/data_record/<timestamp>.csv` with per-instance cost and duration + mean summary.
- Rendered routes: `ExperimentalLog/test/<nodes>/graph/*.png` (first few batches if plotting enabled).

## 17. Running the Example Configuration (PowerShell on Windows)
```powershell
cd "c:\Users\YODA\Documents\Dhia_Salem_Tutoré_AIM_25-26\DRL_for_EV\EV-Energy-Optimization\EM-EVRP"

python run.py `
  --model DRL `
  --num_nodes 10 `
  --charging_num 4 `
  --train_size 1280000 `
  --valid_size 2048 `
  --iterations 100 `
  --batch_size 1024 `
  --embedding_dim 128 `
  --hidden_size 128 `
  --n_encode_layers 2 `
  --actor_lr 1e-4 `
  --critic_lr 1e-4 `
  --baselines rollout `
  --bl_alpha 0.05 `
  --bl_warmup_epochs 5 `
  --Start_SOC 100 `
  --t_limit 100 `
  --velocity 60 `
  --seed 1234
```
Adjust `--model AM` to switch architecture. Reduce `--train_size` for quicker prototyping.

Resume from checkpoint (example):
```powershell
python run.py --model DRL --num_nodes 10 --charging_num 4 --train_size 1280000 --iterations 50 --batch_size 1024 --checkpoint ExperimentalLog/train/10/rollout/<run_id>/best.pt
```

Testing (greedy):
```powershell
python run.py --model DRL --num_nodes 10 --charging_num 4 --test --test_size 128 --decode_strategy greedy --eval_batch_size 128 --batch_size 128
```

Beam search (width=5):
```powershell
python run.py --model DRL --num_nodes 10 --charging_num 4 --test --decode_strategy bs --width 5 --eval_batch_size 1
```

## 18. Troubleshooting
- Station overlaps with depot: Ensure jitter/resampling section in `EVRP.py` retained; if reverted, reapply patch to distinguish depot and depot_charging.
- Vehicles appear underutilized (few station visits): Increase demands (`max_demand`), reduce initial load, or penalize early depot returns by adding custom cost terms.
- NaN mean time in logs: Occurs if fewer than 100 batch logging intervals executed; ensure `times` list populated (add guard when computing `mean_time`).
- Negative or unstable rewards: Confirm energy consumption sign convention; if preferring minimization via negative reward, multiply reward by -1 before REINFORCE loss.
- Slow training with large train_size: Consider streaming dataset or using fewer epochs with curriculum (start small `num_nodes`, ramp up).

## 19. Extensibility
- Add time windows: Introduce per-customer earliest/latest service times and incorporate into mask logic.
- Multi-vehicle: Extend dynamic state to track multiple vehicle loads/SOC; adapt decoder to multi-route generation.
- Alternative objective: Combine energy + distance + lateness penalties.
- Mixed precision: Integrate `torch.cuda.amp` for throughput gains.

## 20. Glossary
- SOC: State of Charge (battery energy proxy).
- Mask: Binary feasibility vector over nodes at each decoding step.
- Advantage: Reward minus baseline estimate used in REINFORCE gradient.
- Rollout baseline: Greedy policy snapshot providing variance reduction.
- Beam width: Number of parallel expansions kept during beam search.
- Elevation / slope: Height-based gradient affecting energy consumption.

## 21. File Map (Key EM-EVRP Modules)
- `run.py`: CLI argument parsing, training/testing entrypoint.
- `trainer.py`: Epoch loop, batching, logging, checkpointing, validation.
- `problems/EVRP.py`: Instance & dynamic generation, masking & state transitions, energy model.
- `nets/DRLModel.py`: Route-context attention actor variant.
- `nets/AM.py`: Simpler attention-based actor with step context.
- `nets/GraphEncoder.py`: Multi-head attention encoder layers.
- `utils/reinforce_baselines.py`: Baseline strategies (exponential, critic, rollout, warmup).
- `utils/beam_search.py`: Generic beam search components.
- `utils/plot_delivery_graph.py`: Rendering of a batch’s route (PNG outputs).

## 22. Recommended Next Steps
1. Run a smaller pilot (`train_size=50000`, `iterations=10`) to validate environment stability.
2. Inspect route PNGs to ensure mask feasibility matches intuition.
3. Gradually scale train_size and iterations while monitoring validation cost trajectory.
4. Evaluate beam search performance vs greedy on held-out test set.

---
For further clarifications or extension suggestions, see troubleshooting and extensibility sections.
