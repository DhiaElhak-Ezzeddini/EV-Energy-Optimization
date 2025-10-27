# agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque, namedtuple
from models import GNN_QNetwork
from graph_parser import GraphParser
from ev_routing_env import EVRoutingEnv # For type hinting and env access
from torch_geometric.data import Batch
from typing import Union,Tuple,Dict, List, Optional

# Experience Replay Buffer
Transition = namedtuple('Transition', ('state_tuple', 'action_node_idx', 'reward', 'next_state_tuple', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Selects a random batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """ DQN Agent using GNN for Q-value estimation. """
    def __init__(self, env: EVRoutingEnv, parser: GraphParser, config: Dict):
        self.env = env
        self.parser = parser
        self.config = config
        self.device = torch.device(config.get('device', 'cpu')) # Safer default
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.current_epsilon = self.epsilon_start

        node_dim = self.parser.node_feature_dim
        edge_dim = self.parser.edge_feature_dim
        hidden_dim = config['hidden_dim']
        print(f"Initializing GNN_QNetwork with node_dim={node_dim}, edge_dim={edge_dim}, hidden_dim={hidden_dim}")

        self.policy_net = GNN_QNetwork(
            node_feature_dim=node_dim,
            edge_feature_dim=edge_dim,
            hidden_dim=hidden_dim
        ).to(self.device)

        self.target_net = GNN_QNetwork(
             node_feature_dim=node_dim,
             edge_feature_dim=edge_dim,
             hidden_dim=hidden_dim
         ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        self.memory = ReplayBuffer(config['buffer_size'])
        self.train_step_counter = 0

    def select_action(self, state_tuple: Tuple[int, int, float]) -> int:
        """ Selects action (neighbor node index) using epsilon-greedy policy. Returns -1 if stuck."""
        self.current_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                               np.exp(-1. * self.train_step_counter / self.epsilon_decay)

        current_node_idx, _, current_soc = state_tuple
        # Use node ID for environment interaction, index for internal logic/model
        current_node_id = self.env.idx_to_node.get(current_node_idx)
        if current_node_id is None:
             print(f"Error: Invalid current_node_idx {current_node_idx} in select_action")
             return -1 # Invalid state index

        reachable_action_indices = self.env.get_reachable_actions_indices(
            current_node_id=current_node_id,
            current_soc=current_soc
        )

        if not reachable_action_indices:
            # Handle case where no actions are possible (e.g., stuck)
            return -1 # Indicate no valid action

        if random.random() > self.current_epsilon:
            # --- Greedy action ---
            with torch.no_grad():
                data = self.parser.parse_obs(state_tuple).to(self.device)
                # Pass the actual graph Data object's edge map
                edge_idx_map = self.parser.edge_idx_map # Get map from parser

                q_values_tensor, actual_indices = self.policy_net(
                    data, # Pass single Data object
                    current_node_idx=current_node_idx, # Pass index
                    reachable_neighbor_indices=reachable_action_indices,
                    edge_idx_map=edge_idx_map # Pass the map
                )

                if q_values_tensor is None or q_values_tensor.numel() == 0: # Check if empty
                    # Fallback if Q-value calculation failed for some reason
                    print(f"Warning: No Q-values computed for reachable actions from {current_node_idx}. Choosing random reachable.")
                    action_node_idx = random.choice(reachable_action_indices)
                else:
                    # Find the index within the q_values tensor that has the maximum value
                    max_q_local_idx = q_values_tensor.argmax().item()
                    # Get the actual node index corresponding to this Q-value
                    action_node_idx = actual_indices[max_q_local_idx]
        else:
            # --- Exploration ---
            action_node_idx = random.choice(reachable_action_indices)

        return action_node_idx # Return the chosen node index

    def learn(self) -> Optional[float]:
        """ Performs a single optimization step on the policy network. """
        if len(self.memory) < self.batch_size:
            return None # Not enough samples

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # --- Prepare batch for GNN ---
        state_pyg_list = [self.parser.parse_obs(st) for st in batch.state_tuple]
        next_state_pyg_list = [self.parser.parse_obs(nst) for nst in batch.next_state_tuple]
        # Batch the Data objects
        state_batch_pyg = Batch.from_data_list(state_pyg_list).to(self.device)
        next_state_batch_pyg = Batch.from_data_list(next_state_pyg_list).to(self.device)

        action_indices_tensor = torch.tensor(batch.action_node_idx, device=self.device, dtype=torch.long) # Shape: [batch_size]
        rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
        dones = torch.tensor(batch.done, device=self.device, dtype=torch.bool)

        # --- Calculate Q(s, a) for the actions taken (More Efficiently) ---
        current_node_indices_batch = [st[0] for st in batch.state_tuple]
        # Get ALL reachable actions for the states in the batch (needed for indexing)
        # Note: This uses the state *at the time of transition*, stored in batch.state_tuple
        reachable_neighbors_batch = []
        for state_tuple in batch.state_tuple:
            c_node_idx, _, c_soc = state_tuple
            c_node_id = self.env.idx_to_node.get(c_node_idx)
            if c_node_id is None: # Should not happen if data is clean
                 reachable_neighbors_batch.append([])
                 continue
            reachable = self.env.get_reachable_actions_indices(c_node_id, c_soc)
            reachable_neighbors_batch.append(reachable)

        # Pass the precomputed edge map from the parser
        edge_idx_map = self.parser.edge_idx_map

        # Get Q-values for all reachable actions for each state in the batch
        q_values_list_all_reachable, actual_indices_list = self.policy_net(
            state_batch_pyg,
            current_node_indices=current_node_indices_batch,
            reachable_neighbor_indices=reachable_neighbors_batch,
            edge_idx_map=edge_idx_map
        )

        # Select the Q-value corresponding to the action actually taken
        state_action_values_list = []
        for i in range(self.batch_size):
            action_taken = action_indices_tensor[i].item()
            q_values_for_state = q_values_list_all_reachable[i]
            indices_for_state = actual_indices_list[i]

            try:
                # Find the position of the action_taken within the indices_for_state list
                action_pos = indices_for_state.index(action_taken)
                state_action_values_list.append(q_values_for_state[action_pos])
            except (ValueError, IndexError):
                # Action taken wasn't found among reachable/calculated Qs.
                # This could happen due to state changes or rare edge cases.
                # Append 0 or handle as an error, depending on strictness.
                state_action_values_list.append(torch.tensor(0.0, device=self.device))
                # print(f"Warning: Action {action_taken} not found in Q-value output indices {indices_for_state} for state {batch.state_tuple[i]}")

        state_action_values = torch.stack(state_action_values_list) # Shape: [batch_size]


        # --- Calculate max Q_target(s', a') for the next states (Efficiently) ---
        next_state_max_q_values = torch.zeros(self.batch_size, device=self.device)
        non_final_mask = ~dones

        # Process only non-final next states
        non_final_next_state_tuples = [nst for i, nst in enumerate(batch.next_state_tuple) if non_final_mask[i]]

        if non_final_next_state_tuples:
            # Prepare batch for target network
            non_final_next_state_pyg_list = [self.parser.parse_obs(nst) for nst in non_final_next_state_tuples]
            non_final_next_state_batch_pyg = Batch.from_data_list(non_final_next_state_pyg_list).to(self.device)

            non_final_next_current_nodes = [nst[0] for nst in non_final_next_state_tuples]
            non_final_next_socs = [nst[2] for nst in non_final_next_state_tuples]

            # Get reachable actions for non-final next states
            next_reachable_neighbors_batch_non_final = []
            for i in range(len(non_final_next_state_tuples)):
                 node_id = self.env.idx_to_node.get(non_final_next_current_nodes[i])
                 if node_id is None:
                      next_reachable_neighbors_batch_non_final.append([])
                      continue
                 reachable = self.env.get_reachable_actions_indices(node_id, non_final_next_socs[i])
                 next_reachable_neighbors_batch_non_final.append(reachable)

            with torch.no_grad():
                # Get Q-values from target net for all reachable actions in next states
                next_q_values_list_target, _ = self.target_net(
                    non_final_next_state_batch_pyg,
                    current_node_indices=non_final_next_current_nodes,
                    reachable_neighbor_indices=next_reachable_neighbors_batch_non_final,
                    edge_idx_map=edge_idx_map
                )

                # Find the max Q value for each non-final next state
                max_q_per_item = []
                for q_vals in next_q_values_list_target:
                    if q_vals is not None and q_vals.numel() > 0:
                        max_q_per_item.append(q_vals.max().item())
                    else:
                        max_q_per_item.append(0.0) # If no reachable actions from next state or error

                # Place these max Q values into the correct positions
                next_state_max_q_values[non_final_mask] = torch.tensor(max_q_per_item, device=self.device)


        # --- Compute TD Target ---
        expected_state_action_values = (next_state_max_q_values * self.gamma) + rewards

        # --- Compute Loss ---
        # Using Huber loss (SmoothL1Loss) is often more robust than MSELoss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # --- Optimize ---
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients if they explode
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_counter += 1

        # --- Update Target Network ---
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print(f"--- Updated Target Network at step {self.train_step_counter} ---") # Debug print

        return loss.item()

    def store_transition(self, state_tuple, action_node_idx, reward, next_state_tuple, done):
         """Stores the transition."""
         # Ensure action index is valid before storing
         if action_node_idx >= 0:
              self.memory.push(state_tuple, action_node_idx, reward, next_state_tuple, done)
         # else: Optionally log or handle the 'stuck' case where action_node_idx is -1
