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
from typing import Union,Tuple,Dict

# Experience Replay Buffer
Transition = namedtuple('Transition', ('state_tuple', 'action_node_idx', 'reward', 'next_state_tuple', 'done'))

class ReplayBuffer:
    # ... (Standard implementation as shown previously) ...
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """ DQN Agent using GNN for Q-value estimation. """
    def __init__(self, env: EVRoutingEnv, parser: GraphParser, config: Dict):
        self.env = env
        self.parser = parser
        self.config = config
        self.device = torch.device(config['device'])
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.current_epsilon = self.epsilon_start

        self.policy_net = GNN_QNetwork(
            node_feature_dim=parser.node_feature_dim,
            edge_feature_dim=parser.edge_feature_dim,
            hidden_dim=config['hidden_dim']
        ).to(self.device)

        self.target_net = GNN_QNetwork(
             node_feature_dim=parser.node_feature_dim,
             edge_feature_dim=parser.edge_feature_dim,
             hidden_dim=config['hidden_dim']
         ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        self.memory = ReplayBuffer(config['buffer_size'])
        self.train_step_counter = 0

    def select_action(self, state_tuple: Tuple[int, int, float]) -> int:
        """ Selects action (neighbor node index) using epsilon-greedy policy. """
        self.current_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                               np.exp(-1. * self.train_step_counter / self.epsilon_decay)

        current_node_idx, _, current_soc = state_tuple
        reachable_action_indices = self.env.get_reachable_actions_indices(
            current_node_id=self.env.idx_to_node[current_node_idx],
            current_soc=current_soc
        )

        if not reachable_action_indices:
            # Handle case where no actions are possible (e.g., stuck)
            # Returning -1 or a specific indicator might be useful.
            return -1 # Indicate no valid action

        if random.random() > self.current_epsilon:
            # --- Greedy action ---
            with torch.no_grad():
                data = self.parser.parse_obs(state_tuple).to(self.device)
                # Wrap in a list for batch-like processing by the model
                q_values_list, actual_indices_list = self.policy_net(
                    Batch.from_data_list([data]), # Create a batch of size 1
                    current_node_indices=[current_node_idx],
                    reachable_neighbor_indices=[reachable_action_indices]
                )
                q_values = q_values_list[0] # Get result for the single item
                actual_indices = actual_indices_list[0]

                if q_values.numel() == 0: # Check if empty (e.g., edge lookup failed)
                    action_node_idx = random.choice(reachable_action_indices) # Fallback to random reachable
                else:
                    # Find the index within the q_values tensor that has the maximum value
                    max_q_local_idx = q_values.argmax().item()
                    # Get the actual node index corresponding to this Q-value
                    action_node_idx = actual_indices[max_q_local_idx]
        else:
            # --- Exploration ---
            action_node_idx = random.choice(reachable_action_indices)

        return action_node_idx # Return the chosen node index

    def learn(self) -> Union[float,None]:
        """ Performs a single optimization step on the policy network. """
        if len(self.memory) < self.batch_size:
            return None # Not enough samples

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # --- Prepare batch for GNN ---
        state_pyg_list = [self.parser.parse_obs(st) for st in batch.state_tuple]
        next_state_pyg_list = [self.parser.parse_obs(nst) for nst in batch.next_state_tuple]
        state_batch_pyg = Batch.from_data_list(state_pyg_list).to(self.device)
        next_state_batch_pyg = Batch.from_data_list(next_state_pyg_list).to(self.device)

        action_indices = torch.tensor(batch.action_node_idx, device=self.device, dtype=torch.long)
        rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float)
        dones = torch.tensor(batch.done, device=self.device, dtype=torch.bool)

        # --- Calculate Q(s, a) for the actions taken ---
        # Get current node indices for each item in the batch
        current_node_indices_batch = [st[0] for st in batch.state_tuple]
        # We need Q values only for the specific action taken in the state
        # Pass ALL valid neighbors for calculation, then select the one corresponding to 'a'
        # For simplicity in gathering Q(s,a): Call forward with only the action taken as the 'reachable' neighbor
        # Note: This is inefficient but avoids complex indexing for now.
        q_values_for_action_list, _ = self.policy_net(
            state_batch_pyg,
            current_node_indices=current_node_indices_batch,
            reachable_neighbor_indices=[[a] for a in batch.action_node_idx] # Only calculate for the action taken
        )
        # Handle cases where the action might not have been reachable/found (empty tensor)
        state_action_values = torch.cat([qv if qv.numel() > 0 else torch.tensor([0.0], device=self.device) for qv in q_values_for_action_list])


        # --- Calculate max Q(s', a') for the next states using Target Network ---
        next_state_max_q_values = torch.zeros(self.batch_size, device=self.device)

        # Get reachable actions for all non-final next states
        non_final_mask = ~dones
        non_final_next_state_tuples = [nst for i, nst in enumerate(batch.next_state_tuple) if non_final_mask[i]]
        non_final_next_state_indices = [idx for idx, mask_val in enumerate(non_final_mask) if mask_val] # Indices in original batch

        if non_final_next_state_tuples:
            non_final_next_current_nodes = [nst[0] for nst in non_final_next_state_tuples]
            non_final_next_socs = [nst[2] for nst in non_final_next_state_tuples]

            next_reachable_neighbors_batch = []
            for i in range(len(non_final_next_state_tuples)):
                 node_id = self.env.idx_to_node[non_final_next_current_nodes[i]]
                 reachable = self.env.get_reachable_actions_indices(node_id, non_final_next_socs[i])
                 next_reachable_neighbors_batch.append(reachable)

            # Create a batch of non-final next states for the target network
            non_final_next_state_pyg_list = [self.parser.parse_obs(nst) for nst in non_final_next_state_tuples]
            non_final_next_state_batch_pyg = Batch.from_data_list(non_final_next_state_pyg_list).to(self.device)


            with torch.no_grad():
                next_q_values_list, next_actual_indices_list = self.target_net(
                    non_final_next_state_batch_pyg,
                    current_node_indices=non_final_next_current_nodes,
                    reachable_neighbor_indices=next_reachable_neighbors_batch
                )

                # Find max Q for each item in the non_final batch
                max_q_per_item = []
                for q_vals in next_q_values_list:
                    if q_vals.numel() > 0:
                        max_q_per_item.append(q_vals.max().item())
                    else:
                        max_q_per_item.append(0.0) # If no reachable actions from next state

                # Place these max Q values into the correct positions in next_state_max_q_values
                next_state_max_q_values[non_final_mask] = torch.tensor(max_q_per_item, device=self.device)


        # --- Compute TD Target ---
        expected_state_action_values = (next_state_max_q_values * self.gamma) + rewards

        # --- Compute Loss ---
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # --- Optimize ---
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_counter += 1

        # --- Update Target Network ---
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def store_transition(self, state_tuple, action_node_idx, reward, next_state_tuple, done):
         self.memory.push(state_tuple, action_node_idx, reward, next_state_tuple, done)