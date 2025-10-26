# train_ev.py
import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
# Assuming other files are importable
from network_gen_wrapper import load_or_generate_graph
from ev_routing_env import EVRoutingEnv
from graph_parser import GraphParser
from agent import DQNAgent

# --- Matplotlib Setup ---
plt.ion() # Interactive mode on

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # --- Setup ---
    device = torch.device(cfg.agent.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load or generate graph
    graph = load_or_generate_graph(cfg.environment.db_path)

    # Initialize Environment, Parser, Agent
    env = EVRoutingEnv(graph, vehicle_params=OmegaConf.to_container(cfg.vehicle, resolve=True))
    parser = GraphParser(graph, device)

    # Update config with dynamic dims if needed (already handled in parser init)
    cfg.agent.node_feature_dim = parser.node_feature_dim
    cfg.agent.edge_feature_dim = parser.edge_feature_dim

    agent = DQNAgent(env, parser, OmegaConf.to_container(cfg.agent, resolve=True))

    # --- Training Loop ---
    episode_rewards = []
    episode_energies = []
    losses = []
    print(f"\nStarting Training for {cfg.training.num_episodes} episodes...")

    for i_episode in range(cfg.training.num_episodes):
        print(f"\n--- Episode {i_episode+1} ---")
        torch.cuda.empty_cache()
        state_tuple = env.reset() # Initial state tuple
        total_reward = 0
        total_energy = 0
        done = False
        step = 0

        while not done:
            action_node_idx = agent.select_action(state_tuple)

            if action_node_idx == -1: # Agent indicates no valid action
                 # print(f"Episode {i_episode+1}: Agent stuck at node {state_tuple[0]}. Terminating.")
                 # Decide if penalty is needed here or if env.step handles it implicitly
                 break # End episode

            next_state_tuple, reward, done, info = env.step(action_node_idx)

            total_reward += reward
            total_energy += info['energy_consumed_kwh']

            # Store transition using tuples
            agent.store_transition(state_tuple, action_node_idx, reward, next_state_tuple, done)

            state_tuple = next_state_tuple
            step += 1

            loss = agent.learn() # Perform optimization
            if loss is not None:
                losses.append(loss)

            if done:
                episode_rewards.append(total_reward)
                episode_energies.append(total_energy)
                if (i_episode + 1) % cfg.training.log_interval == 0:
                    print(f"Ep {i_episode+1}/{cfg.training.num_episodes} | Steps: {step} | R: {total_reward:.2f} | E: {total_energy:.2f} | Eps: {agent.current_epsilon:.3f} | Loss: {np.mean(losses[-cfg.training.log_interval:]) if losses else 0:.4f}")
                    if info.get('soc_violation'): print("   (Fail: SoC)")
                    if info.get('max_steps_reached'): print("   (Fail: Steps)")
                    if info.get('invalid_action'): print("   (Fail: Invalid Action)")

        # Optional: Save checkpoint periodically
        # if (i_episode + 1) % cfg.training.save_interval == 0:
        #     torch.save(agent.policy_net.state_dict(), f'checkpoint_ep{i_episode+1}.pth')

    # --- Final Save & Plot ---
    torch.save(agent.policy_net.state_dict(), cfg.training.save_path)
    print(f"Training complete. Model saved to {cfg.training.save_path}")

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(episode_energies)
    plt.title('Episode Energy Consumed (kWh)')
    plt.xlabel('Episode')
    plt.ylabel('Energy (kWh)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.ioff() # Turn off interactive mode
    plt.show()

if __name__ == "__main__":
    main()