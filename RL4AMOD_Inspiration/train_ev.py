import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import networkx as nx # For path visualization
import time # To track step time
import gc # For garbage collection

# Assuming other files are importable
from network_gen_wrapper import load_or_generate_graph
from ev_routing_env import EVRoutingEnv
from graph_parser import GraphParser
from agent import DQNAgent

# --- Matplotlib Setup ---
plt.ion() # Interactive mode on

# === Debugging Function ===
def plot_episode_path(graph, path_node_ids, start_node_id, target_node_id, episode_num, status, reward, energy, steps, save_dir="episode_plots"):
    """ Plots the path taken by the agent during an episode. """
    try:
        if not path_node_ids:
            print(f"Episode {episode_num}: Cannot plot path - Path is empty.")
            return

        save_path = f"{save_dir}/episode_{episode_num}_{status}.png"
        import os
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(12, 9)) # Slightly larger figure
        # Try getting positions safely
        pos = {}
        missing_pos_nodes = []
        for node_id in graph.nodes():
             data = graph.nodes[node_id]
             if 'x' in data and 'y' in data:
                  pos[node_id] = (data['x'], data['y'])
             else:
                  missing_pos_nodes.append(node_id)
                  pos[node_id] = (np.random.rand(), np.random.rand()) # Assign random if missing

        if missing_pos_nodes:
             print(f"Warning: Nodes missing 'x'/'y' data: {len(missing_pos_nodes)}. Using random positions for them.")


        # Draw the full graph lightly - handle potential errors drawing large graphs
        try:
             nx.draw_networkx_nodes(graph, pos, node_size=5, node_color='lightgray', alpha=0.5)
             nx.draw_networkx_edges(graph, pos, width=0.5, edge_color='lightgray', alpha=0.3, arrows=False)
        except Exception as e:
             print(f"Warning: Error drawing base graph: {e}. Skipping base graph drawing.")


        # Highlight the path taken
        path_edges = list(zip(path_node_ids[:-1], path_node_ids[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=path_node_ids, node_size=15, node_color='orange')
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='orange', width=1.5, arrows=True, arrowsize=10)

        # Highlight start and target nodes if they exist in the graph
        start_node_display = start_node_id if start_node_id in graph else None
        target_node_display = target_node_id if target_node_id in graph else None

        if start_node_display:
            nx.draw_networkx_nodes(graph, pos, nodelist=[start_node_display], node_size=50, node_color='green', label=f'Start ({start_node_id})')
        if target_node_display:
            nx.draw_networkx_nodes(graph, pos, nodelist=[target_node_display], node_size=50, node_color='red', label=f'Target ({target_node_id})')


        plt.title(f"Episode {episode_num} ({status}) | Steps: {steps} | Reward: {reward:.2f} | Energy: {energy:.2f} kWh")
        plt.legend(markerscale=2)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved episode path plot to {save_path}")
        plt.close() # Close the figure to free memory
    except Exception as e:
        print(f"Error generating plot for episode {episode_num}: {e}")
        plt.close() # Ensure figure is closed even on error


# === Main Training Function ===
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # --- Setup ---
    device = torch.device(cfg.agent.device if torch.cuda.is_available() and cfg.agent.device.lower()=='cuda' else "cpu")
    print(f"Using device: {device}")

    # Load or generate graph
    print("Loading graph...")
    graph = load_or_generate_graph(cfg.environment.db_path)
    print("Graph loaded.")

    # Initialize Environment, Parser, Agent
    print("Initializing environment...")
    env = EVRoutingEnv(graph, vehicle_params=OmegaConf.to_container(cfg.vehicle, resolve=True))
    print("Initializing parser...")
    parser = GraphParser(graph, device)

    # # --- DEBUG: Check if graph nodes match parser nodes ---
    # if len(graph.nodes()) != parser.num_nodes:
    #     print(f"!!! MISMATCH: Graph nodes ({len(graph.nodes())}) != Parser nodes ({parser.num_nodes})")
    # print(f"Parser node feature dim: {parser.node_feature_dim}")
    # print(f"Parser edge feature dim: {parser.edge_feature_dim}")
    # # --- END DEBUG ---


    # Update config structure if keys are missing (alternative to modifying YAML)
    # OmegaConf.set_struct(cfg.agent, False) # Allow adding keys temporarily
    # cfg.agent.node_feature_dim = parser.node_feature_dim
    # cfg.agent.edge_feature_dim = parser.edge_feature_dim
    # OmegaConf.set_struct(cfg.agent, True) # Re-enable structure checking

    # Ensure config has the keys (best practice is to have them in YAML)
    if 'node_feature_dim' not in cfg.agent or cfg.agent.node_feature_dim is None:
         cfg.agent.node_feature_dim = parser.node_feature_dim
    if 'edge_feature_dim' not in cfg.agent or cfg.agent.edge_feature_dim is None:
         cfg.agent.edge_feature_dim = parser.edge_feature_dim


    print("Initializing agent...")
    agent = DQNAgent(env, parser, OmegaConf.to_container(cfg.agent, resolve=True))
    print("Agent initialized.")

    # --- Training Loop ---
    episode_rewards = []
    episode_energies = []
    episode_steps = []
    losses = []
    all_episode_paths = {} # Store paths for potential later analysis

    print(f"\nStarting Training for {cfg.training.num_episodes} episodes...")

    # --- Debugging Flags ---
    PLOT_EPISODE_INTERVAL = cfg.training.get("plot_interval", 50) # Plot every N episodes (failed or success)
    PRINT_STEP_DETAILS = cfg.training.get("print_step_details", False) # Verbose step output
    # Checkpoint saving config
    SAVE_CHECKPOINTS = cfg.training.get("save_checkpoints", False)
    CHECKPOINT_INTERVAL = cfg.training.get("save_interval", 500)
    CHECKPOINT_DIR = cfg.training.get("checkpoint_dir", "checkpoints")
    if SAVE_CHECKPOINTS:
        import os
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    for i_episode in range(cfg.training.num_episodes):
        episode_start_time = time.time()
        print(f"\n--- Episode {i_episode+1}/{cfg.training.num_episodes} ---")
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
            gc.collect() # Force garbage collection

        state_tuple = env.reset() # Initial state tuple
        start_node_id = env.start_node_id
        target_node_id = env.target_node_id
        print(f"Start Node: {start_node_id} (Idx: {state_tuple[0]}), Target Node: {target_node_id} (Idx: {state_tuple[1]}), Initial SoC: {state_tuple[2]:.3f}")

        total_reward = 0
        total_energy = 0
        done = False
        step = 0
        path_taken_ids = [start_node_id] # Store node IDs for plotting
        step_cumulative_time = 0.0

        last_info = {} # Store last info dict for end-of-episode reporting

        while not done:
            step_start_time = time.time()
            current_node_id = env.current_node_id
            current_node_idx = state_tuple[0]
            current_soc = state_tuple[2]

            # --- Action Selection ---
            action_select_start = time.time()
            action_node_idx = agent.select_action(state_tuple)
            action_select_time = time.time() - action_select_start

            # --- Debug Print Action Selection ---
            if PRINT_STEP_DETAILS or action_node_idx == -1: # Always print if stuck
                 reachable_indices = env.get_reachable_actions_indices(current_node_id, current_soc) # Recalculate for print
                 print(f"  Step {step+1}: Node {current_node_id}({current_node_idx}), SoC {current_soc:.3f}")
                 print(f"    Reachable Neighbors (Indices): {reachable_indices}")
                 print(f"    Chosen Action (Index): {action_node_idx} (Node ID: {env.idx_to_node.get(action_node_idx, 'Stuck')})")
                 # Optional: Print Q-values if exploiting (can be slow)
                 # if random.random() > agent.current_epsilon and action_node_idx != -1:
                 #    with torch.no_grad():
                 #        data = parser.parse_obs(state_tuple).to(device)
                 #        q_vals, q_indices = agent.policy_net(Batch.from_data_list([data]), [current_node_idx], [reachable_indices])
                 #        if q_vals[0].numel() > 0:
                 #             q_dict = {idx: qv.item() for idx, qv in zip(q_indices[0], q_vals[0])}
                 #             print(f"    Q-Values (Reachable): {q_dict}")

                 print(f"    Epsilon: {agent.current_epsilon:.4f}, Action Select Time: {action_select_time:.3f}s")


            if action_node_idx == -1: # Agent indicates no valid action (stuck)
                 print(f"  Agent STUCK at Node {current_node_id} ({current_node_idx}) with SoC {current_soc:.3f}. No reachable actions. Terminating episode.")
                 total_reward += cfg.vehicle.get('fail_penalty_stuck', -500) # Add a specific penalty
                 last_info['fail_reason'] = 'Stuck'
                 done = True # Force done flag if stuck
                 # break # Exit the inner while loop immediately

            # --- Environment Step (only if not stuck) ---
            if not done: # Check if done wasn't set by the stuck condition
                 env_step_start = time.time()
                 next_state_tuple, reward, step_done, info = env.step(action_node_idx)
                 env_step_time = time.time() - env_step_start
                 done = step_done # Update done flag based on environment step
                 last_info = info # Store info from the step

                 total_reward += reward
                 consumed_kwh = info['energy_consumed_kwh']
                 total_energy += consumed_kwh
                 path_taken_ids.append(env.current_node_id) # Log the node *after* the step

                 if PRINT_STEP_DETAILS:
                      print(f"    Env Step Result: Next Node {env.current_node_id}({next_state_tuple[0]}), Next SoC {next_state_tuple[2]:.3f}")
                      print(f"    Energy Cost: {consumed_kwh:.4f} kWh, Reward: {reward:.2f}")
                      print(f"    Done: {done}, Info: {info}")
                      print(f"    Env Step Time: {env_step_time:.3f}s")

                 # --- Store Transition ---
                 agent.store_transition(state_tuple, action_node_idx, reward, next_state_tuple, done)

                 state_tuple = next_state_tuple
                 step += 1

                 # --- Learning Step ---
                 learn_start = time.time()
                 loss = agent.learn() # Perform optimization
                 learn_time = time.time() - learn_start
                 if loss is not None:
                     losses.append(loss)
                     if PRINT_STEP_DETAILS: print(f"    Learn Step Loss: {loss:.4f}, Learn Time: {learn_time:.3f}s")


                 # --- Loop Timing ---
                 step_end_time = time.time()
                 step_duration = step_end_time - step_start_time
                 step_cumulative_time += step_duration
                 if PRINT_STEP_DETAILS: print(f"    Total Step Time: {step_duration:.3f}s\n")
                 # step_start_time = step_end_time # Reset timer for next step # Redundant

                 # --- Check if episode is stuck (optional extra check) ---
                 # Increased threshold slightly
                 if step > env.max_steps * 2.0: # If somehow max_steps check fails or is too low
                     print(f"  Force Terminating Episode {i_episode+1} due to excessive steps ({step})")
                     done = True # Force done
                     # Apply penalty if not already done by env
                     if not info.get('max_steps_reached'):
                         total_reward += cfg.vehicle.fail_penalty_steps
                         last_info['fail_reason'] = 'Excessive Steps'


        # --- End of Episode ---
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        avg_step_time = step_cumulative_time / step if step > 0 else 0

        episode_rewards.append(total_reward)
        episode_energies.append(total_energy)
        episode_steps.append(step)
        all_episode_paths[i_episode+1] = path_taken_ids # Store path

        avg_loss = np.mean(losses[-step:]) if step > 0 and losses else 0 # Avg loss for this episode
        print(f"Episode {i_episode+1} Finished | Steps: {step} | Duration: {episode_duration:.2f}s (Avg Step: {avg_step_time:.3f}s)")
        print(f"  Total Reward: {total_reward:.2f} | Total Energy: {total_energy:.2f} kWh | Avg Loss: {avg_loss:.4f}")

        # Determine Status
        episode_status = "Unknown"
        episode_failed = False
        fail_reason = last_info.get('fail_reason', 'None') # Get reason if set above

        if last_info.get('soc_violation'): fail_reason = 'SoC Violation'; episode_failed = True
        elif last_info.get('max_steps_reached'): fail_reason = 'Max Steps'; episode_failed = True
        elif last_info.get('invalid_action'): fail_reason = 'Invalid Action'; episode_failed = True
        elif fail_reason == 'Stuck': episode_failed = True # Already set reason
        elif fail_reason == 'Excessive Steps': episode_failed = True # Already set reason
        elif done and env.current_node_id == target_node_id: episode_status = 'Success'; episode_failed = False
        elif done: fail_reason = 'Ended without reaching target'; episode_failed = True # Catch unexpected ends

        if episode_failed:
             episode_status = f"Fail ({fail_reason})"
             print(f"   Status: {episode_status}")
        else:
             print(f"   Status: {episode_status}")


        # Plot path every N episodes OR if failed
        plot_now = (i_episode + 1) % PLOT_EPISODE_INTERVAL == 0
        if (episode_failed or plot_now):
            plot_episode_path(
                graph,
                path_taken_ids,
                start_node_id,
                target_node_id,
                i_episode+1,
                status=episode_status.replace(" ", "_").replace("(", "").replace(")", ""), # Sanitize status for filename
                reward=total_reward,
                energy=total_energy,
                steps=step,
                save_dir=cfg.training.get("plot_dir", "episode_plots")
            )


        # Save checkpoint periodically
        if SAVE_CHECKPOINTS and (i_episode + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_ep{i_episode+1}.pth')
            torch.save({
                'episode': i_episode + 1,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.current_epsilon,
                # Add replay buffer saving if desired (can be large)
            }, save_checkpoint_path)
            print(f"Checkpoint saved to {save_checkpoint_path}")

    # --- Final Save & Plot ---
    if cfg.training.save_path:
        final_save_path = cfg.training.save_path
        # Ensure directory exists
        save_dir = os.path.dirname(final_save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)

        torch.save(agent.policy_net.state_dict(), final_save_path)
        print(f"Training complete. Final model saved to {final_save_path}")
    else:
        print("Training complete. No save path specified.")


    # Plot overall results
    if episode_rewards: # Check if list is not empty
         fig, axs = plt.subplots(1, 3, figsize=(18, 5))
         axs[0].plot(episode_rewards)
         axs[0].set_title('Episode Rewards')
         axs[0].set_xlabel('Episode')
         axs[0].set_ylabel('Total Reward')
         axs[0].grid(True)

         if episode_energies:
              axs[1].plot(episode_energies)


if __name__ == "__main__":
    main()