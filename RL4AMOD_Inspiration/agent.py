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
from typing import Union, Tuple, Dict, List, Optional
import math

# Define StateType as a type alias
StateType = Tuple[int, int, float]

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
        self.gamma = config.get('gamma', 0.99)
        self.batch_size = config.get('batch_size', 128)
        self.target_update_freq = config.get('target_update_freq', 500)
        
        # Epsilon management
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay_steps = config.get('epsilon_decay_steps', 50000)
        self.train_step_counter = 0

        # Model initialization
        node_feature_dim = self.parser.node_feature_dim
        edge_feature_dim = self.parser.edge_feature_dim
        hidden_dim = config.get('hidden_dim', 64)
        lr = config.get('lr', 0.0005)

        self.policy_net = GNN_QNetwork(node_feature_dim, edge_feature_dim, hidden_dim).to(self.device)
        self.target_net = GNN_QNetwork(node_feature_dim, edge_feature_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is frozen

        # Optimizer and Buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(config.get('replay_capacity', 100000))
        
        # State: (current_node_id, target_node_id, current_soc_fraction)
        # StateType is defined at module level


    @property
    def current_epsilon(self) -> float:
        """Calculates the current epsilon value based on linear decay."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.train_step_counter / self.epsilon_decay_steps)

    @property
    def ready_to_learn(self) -> bool:
        """Checks if there are enough samples in the replay buffer to start training."""
        return len(self.memory) >= self.batch_size

    @torch.no_grad()
    def select_action(self, state: StateType, explore: bool = True) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Selects an action (next node index) using an epsilon-greedy strategy.
        Returns the action node ID (osmid) and its Q-value (for logging/debugging).
        """
        current_node_id = state[0]
        # Get list of valid action node IDs (osmid)
        action_node_ids: List[int] = self.env.get_reachable_actions(current_node_id, state[2])

        if not action_node_ids:
            return current_node_id, None # Stay put or indicate failure if no actions are possible

        # Convert node IDs to global indices for model processing
        action_node_indices: List[int] = [self.parser.node_to_idx[nid] for nid in action_node_ids if nid in self.parser.node_to_idx]
        
        if not action_node_indices:
             # Fallback: if no indexed neighbors found (shouldn't happen with correct parsing)
             return current_node_id, None 

        if explore and random.random() < self.current_epsilon:
            # Exploration: Choose a random action
            chosen_node_idx = random.choice(action_node_indices)
            chosen_q_value = None # Q-value is not calculated in exploration
        else:
            # Exploitation: Choose the action with the highest Q-value
            
            # Prepare state for model input
            data = self.parser.parse_obs(state).to(self.device)
            if data is None:
                # Fallback on invalid state
                return current_node_id, None 
            
            # Run the policy network on the single state/graph
            # The model is forced to only consider the reachable indices
            # The result is a list of one tensor (q_values_for_neighbors)
            q_values_list, _ = self.policy_net(data, self.parser, [action_node_indices])
            
            if not q_values_list or q_values_list[0].numel() == 0:
                 # Fallback if model returns no Q-values (e.g., error in forward pass)
                 return current_node_id, None 

            q_values = q_values_list[0]
            
            # Select the action with the highest Q-value
            max_q_index_local = q_values.argmax(dim=0).item()
            chosen_node_idx = action_node_indices[max_q_index_local]
            chosen_q_value = q_values[max_q_index_local].item()
            
        # Convert the chosen global index back to the Node ID (osmid) for the environment
        chosen_node_id = self.parser.nodes[chosen_node_idx]

        return chosen_node_id, chosen_q_value

    def learn(self):
        """Performs one step of optimization on the policy network."""
        if len(self.memory) < self.batch_size:
            return None # Not enough samples to learn
        
        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Separate components of the batch
        state_batch: List[StateType] = list(batch.state_tuple)
        action_node_idx_batch: List[int] = list(batch.action_node_idx)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float, device=self.device)
        next_state_batch: List[Optional[StateType]] = list(batch.next_state_tuple)
        done_batch = torch.tensor(batch.done, dtype=torch.float, device=self.device)
        
        # --- Process States into PyG Batch Object ---
        data_list = [self.parser.parse_obs(s) for s in state_batch if self.parser.parse_obs(s) is not None]
        if not data_list: return None # Safety check
        state_batch_pyg = Batch.from_data_list(data_list).to(self.device)

        # --- Policy Net Forward Pass (Q(s, a)) ---
        # 1. Get all *possible* neighbor Q-values for all nodes in the state batch
        # This requires re-running the logic for action selection for *each* state in the batch
        
        # Collect all reachable indices lists for the batch (necessary for the model's forward pass)
        reachable_indices_batch_list: List[List[int]] = []
        for state in state_batch:
            current_node_id = state[0]
            action_node_ids: List[int] = self.env.get_reachable_actions(current_node_id, state[2])
            action_node_indices: List[int] = [self.parser.node_to_idx[nid] for nid in action_node_ids if nid in self.parser.node_to_idx]
            reachable_indices_batch_list.append(action_node_indices)
            
        # Run Policy Net to get Q-values for the sampled actions (state_action_values)
        # q_values_for_all_neighbors is a list of tensors, one for each graph in the batch
        q_values_for_all_neighbors, all_neighbor_indices = self.policy_net(state_batch_pyg, self.parser, reachable_indices_batch_list)
        
        # Identify the Q-value corresponding to the actual taken action (a)
        state_action_values = []
        for i, q_tensor in enumerate(q_values_for_all_neighbors):
            action_idx = action_node_idx_batch[i] # The global index of the action taken
            # Find the position of the taken action (action_idx) within the list of calculated neighbors
            try:
                # The index of the action node within the list of neighbors for this state
                local_idx = all_neighbor_indices[i].index(action_idx)
                state_action_values.append(q_tensor[local_idx])
            except ValueError:
                # This should not happen if transitions are correctly stored
                # If it does, the action was not considered reachable, skip this transition
                print(f"Warning: Taken action {action_idx} not found in calculated neighbors for state {i}. Skipping.")
                return None
                
        # Stack the calculated Q(s,a) values
        state_action_values = torch.stack(state_action_values) # Shape: [batch_size]

        # --- Target Net Forward Pass (max_a' Q(s', a')) ---
        
        # Create mask for non-final (non-done) states
        non_final_mask = (done_batch == 0)
        non_final_next_states_tuple = [s for s, done in zip(next_state_batch, done_batch) if done == 0]
        
        next_state_max_q_values = torch.zeros(self.batch_size, device=self.device)

        if non_final_next_states_tuple:
            # 1. Process Next States into PyG Batch Object
            next_data_list = [self.parser.parse_obs(s) for s in non_final_next_states_tuple if self.parser.parse_obs(s) is not None]
            if next_data_list:
                next_state_batch_pyg = Batch.from_data_list(next_data_list).to(self.device)
                
                # 2. Get the list of reachable indices for each next state
                next_reachable_indices_list: List[List[int]] = []
                for state in non_final_next_states_tuple:
                    current_node_id = state[0]
                    action_node_ids: List[int] = self.env.get_reachable_actions(current_node_id, state[2])
                    action_node_indices: List[int] = [self.parser.node_to_idx[nid] for nid in action_node_ids if nid in self.parser.node_to_idx]
                    next_reachable_indices_list.append(action_node_indices)
                    
                # 3. Target Net Forward Pass: Q'(s', a')
                # q_values_for_next_neighbors is a list of tensors, one for each non-final next state
                q_values_for_next_neighbors, _ = self.target_net(next_state_batch_pyg, self.parser, next_reachable_indices_list)
                
                max_q_per_item = []
                for q_tensor in q_values_for_next_neighbors:
                    if q_tensor.numel() > 0:
                        max_q_per_item.append(q_tensor.max().item())
                    else:
                        # If a non-final state has no reachable actions, max Q is 0
                        max_q_per_item.append(0.0) 
                        
                # Put the max Q values into the correct positions
                next_state_max_q_values[non_final_mask] = torch.tensor(max_q_per_item, device=self.device)


        # --- Compute TD Target ---
        rewards = reward_batch # Shape: [batch_size] - rewards are already tensor
        # Expected Q = R + gamma * max_a'(Q'(s', a'))
        expected_state_action_values = (next_state_max_q_values * self.gamma) + rewards

        # --- Compute Loss ---
        # Using Huber loss (SmoothL1Loss) is often more robust than MSELoss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # --- Optimize ---
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients if they explode (optional but good practice)
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_counter += 1

        # --- Update Target Network ---
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def store_transition(self, state_tuple, action_node_id, reward, next_state_tuple, done):
         """Stores the transition. Converts action node ID to global index."""
         # Ensure action index is available
         action_node_idx = self.parser.node_to_idx.get(action_node_id)
         if action_node_idx is None:
             print(f"Warning: Tried to store transition with invalid action ID {action_node_id}. Skipping.")
             return
             
         self.memory.push(state_tuple, action_node_idx, reward, next_state_tuple, done)
