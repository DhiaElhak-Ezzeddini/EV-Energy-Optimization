import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import networkx as nx
import time
import os
import sys
from typing import List

# Import core components
from network_gen_wrapper import load_or_generate_graph
from ev_routing_env import EVRoutingEnv
from graph_parser import GraphParser # Now accepts static embeddings
from agent import DQNAgent

# --- Matplotlib Setup ---
plt.ion() # Interactive mode on
CHECKPOINT_DIR = "checkpoints"

# === Debugging Function ===
def plot_episode_path(graph, path_node_ids, start_node_id, target_node_id, episode_num, status, reward, energy, steps, save_dir="episode_plots"):
    """ Plots the path taken by the agent during an episode. """
    try:
        if not path_node_ids:
            print(f"Episode {episode_num}: Cannot plot path - Path is empty.")
            return

        os.makedirs(save_dir, exist_ok=True)
        save_path = f"{save_dir}/episode_{episode_num}_{status}.png"

        plt.figure(figsize=(12, 9))
        
        # Try getting positions safely
        pos = {}
        for node_id in graph.nodes():
             data = graph.nodes[node_id]
             if 'x' in data and 'y' in data:
                 pos[node_id] = (data['x'], data['y'])
        
        # If positions are available, plot
        if pos:
            # Draw all edges lightly
            nx.draw_networkx_edges(graph, pos, edge_color='lightgray', alpha=0.5)

            # Draw all nodes (size proportional to degree)
            # node_sizes = [graph.degree(n) * 10 for n in graph.nodes()]
            nx.draw_networkx_nodes(graph, pos, 
                                   node_size=5, 
                                   node_color='gray', 
                                   alpha=0.6)

            # Highlight the path edges
            path_edges = list(zip(path_node_ids[:-1], path_node_ids[1:]))
            nx.draw_networkx_edges(graph, pos, 
                                   edgelist=path_edges, 
                                   edge_color='red', 
                                   width=2.0, 
                                   alpha=0.8)

            # Highlight start/target nodes
            special_nodes = {start_node_id: 'green', target_node_id: 'blue'}
            nx.draw_networkx_nodes(graph, pos, 
                                   nodelist=special_nodes.keys(), 
                                   node_color=list(special_nodes.values()), 
                                   node_size=100, 
                                   label={start_node_id: 'Start', target_node_id: 'Target'})

            plt.title(f"Episode {episode_num} Path ({status}) | Reward: {reward:.2f}, Energy: {energy:.2f}kWh, Steps: {steps}")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(save_path)
            plt.close()
            # print(f"Path plot saved to {save_path}") # Suppress for cleaner output
        else:
            print(f"Episode {episode_num}: No position data available for plotting.")

    except Exception as e:
        print(f"Error during plotting episode {episode_num}: {e}")


@hydra.main(version_base=None, config_path="config", config_name="config")
def train_dqn_agent(cfg: DictConfig):
    """Main function to train the DQN agent."""
    
    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Set device and print
    device = torch.device(cfg.agent.device if torch.cuda.is_available() and cfg.agent.device.lower()=='cuda' else "cpu")
    print(f"Using device: {device}")
    
    # --- Load Graph ---
    print("Loading graph...")
    try:
        graph = load_or_generate_graph(cfg.environment.db_path)
    except Exception as e:
        print(f"FATAL: Error loading/generating graph: {e}")
        return

    # --- Load Pre-computed Static Embeddings (Stage 1 Output) ---
    static_embedding_path = cfg.training.get("static_embedding_path", "static_node_embeddings.pt")
    if not os.path.exists(static_embedding_path):
        print(f"FATAL: Static embeddings not found at {static_embedding_path}.")
        print("Please run stage_1_precompute_embeddings.py first.")
        return
    static_embeddings = torch.load(static_embedding_path).float()
    print(f"Loaded static embeddings with shape: {static_embeddings.shape}")

    # --- Initialize Components ---
    # Environment
    env = EVRoutingEnv(graph, cfg.environment.vehicle_params)
    
    # Parser (pass embeddings)
    parser = GraphParser(graph, device)
    
    # Agent
    agent = DQNAgent(env, parser, OmegaConf.to_container(cfg.agent, resolve=True))
    
    # --- Training Loop Setup ---
    num_episodes = cfg.training.num_episodes
    max_steps = cfg.training.max_steps_per_episode
    log_interval = cfg.training.log_interval
    checkpoint_interval = cfg.training.checkpoint_interval
    
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_energy: List[float] = []
    
    start_time = time.time()
    
    print("--- Starting Training ---")
    
    for i_episode in range(num_episodes):
        
        # Reset environment for a new episode
        state = env.reset()
        done = False
        total_reward = 0.0
        
        for t in range(max_steps):
            
            # 1. Select action
            action_id, q_value = agent.select_action(state, explore=True)
            
            # Action is the same as the current node, meaning agent is stuck/cannot move
            if action_id == state[0]:
                next_state, reward, done, info = state, -5.0, True, {'reason': 'Stuck/No Move'}
                # Do not save to memory, as it's a terminal state resulting from no available action
            else:
                 # 2. Take step in environment
                next_state, reward, done, info = env.step(action_id, max_steps)
            
                # 3. Store the transition in memory (only if a valid action was taken)
                agent.store_transition(state, action_id, reward, next_state if not done else None, done)
            
            # 4. Move to next state
            state = next_state
            total_reward += reward

            # 5. Optimize policy network
            if agent.ready_to_learn:
                loss = agent.learn()
            
            if done:
                break

        # --- Episode End Logging ---
        episode_rewards.append(total_reward)
        episode_lengths.append(env.current_step)
        episode_energy.append(env.total_energy_kwh)
        
        # Log episode results periodically
        if (i_episode + 1) % log_interval == 0 or i_episode == num_episodes - 1:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            avg_energy = np.mean(episode_energy[-log_interval:])
            
            print(f"Episode {i_episode+1:5d}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:7.2f} | Avg Len: {avg_length:5.1f} | "
                  f"Avg Energy: {avg_energy:7.2f}kWh | Epsilon: {agent.current_epsilon:.4f} | "
                  f"Total Steps: {agent.train_step_counter}")
            
            # Plot the last completed path
            plot_episode_path(graph, env.path_taken, env.path_taken[0], env.target_node_id, 
                              i_episode+1, info.get('reason', 'Unknown'), total_reward, env.total_energy_kwh, env.current_step)
                              
        # Save checkpoint periodically
        if (i_episode + 1) % checkpoint_interval == 0:
            save_checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_ep{i_episode+1}.pth')
            torch.save({
                'episode': i_episode + 1,
                'policy_net_state_dict': agent.policy_net.state_dict(),
                'target_net_state_dict': agent.target_net.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.current_epsilon,
            }, save_checkpoint_path)
            print(f"Checkpoint saved to {save_checkpoint_path}")

    # --- Final Save & Plot ---
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds.")
    
    if cfg.training.save_path:
        final_save_path = cfg.training.save_path
        save_dir = os.path.dirname(final_save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)

        torch.save(agent.policy_net.state_dict(), final_save_path)
        print(f"Training complete. Final model saved to {final_save_path}")
    
    # Plot overall results
    if episode_rewards: 
         plt.figure(figsize=(18, 5))
         plt.subplot(1, 3, 1)
         plt.plot(episode_rewards)
         plt.title('Episode Rewards')
         plt.xlabel('Episode')
         plt.ylabel('Total Reward')
         
         plt.subplot(1, 3, 2)
         plt.plot(episode_lengths)
         plt.title('Episode Lengths (Steps)')
         plt.xlabel('Episode')
         plt.ylabel('Steps')
         
         plt.subplot(1, 3, 3)
         plt.plot(episode_energy)
         plt.title('Total Energy Consumption (kWh)')
         plt.xlabel('Episode')
         plt.ylabel('Energy (kWh)')
         
         plt.tight_layout()
         plt.savefig("training_progress.png")
         plt.close()

if __name__ == "__main__":
    train_dqn_agent()
