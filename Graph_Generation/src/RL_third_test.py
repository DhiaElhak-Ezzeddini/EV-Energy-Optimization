from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Any, List, Optional, Tuple
from stable_baselines3 import PPO
from gymnasium import spaces
from scipy import sparse
import gymnasium as gym
import networkx as nx
import torch.nn as nn
import numpy as np
import random
import torch
import math


class ImprovedEnergyModel:
    """Enhanced physical energy consumption model for EVs"""
    
    def __init__(self, vehicle_params: Dict[str, float]):
        self.params = vehicle_params
        self.g = 9.80665  # gravity
        self.rho = 1.225  # air density
        
    def calculate_edge_energy(self, edge_data: Dict[str, Any], speed_m_s: Optional[float] = None) -> float:
        """Calculate energy consumption for traversing an edge"""
        
        # Extract parameters with defaults
        mass = self.params.get('mass_kg', 1800)
        CdA = self.params.get('CdA', 0.62)
        Cr = self.params.get('Cr', 0.015)
        regen_efficiency = self.params.get('regen_efficiency', 0.6)
        
        # Edge properties
        length = float(edge_data.get('length', 100.0))
        slope_deg = float(edge_data.get('slope_deg', 0.0))
        stop_prob = float(edge_data.get('stop_probability', 0.0))
        congestion = float(edge_data.get('congestion_factor', 1.0))
        
        # Convert slope to radians
        theta = np.radians(slope_deg)
        
        # Determine speed
        if speed_m_s is None:
            speed_kph = edge_data.get('speed_kph', 50.0)
            speed_m_s = speed_kph * 1000.0 / 3600.0
        
        # Force calculations
        grade_force = mass * self.g * math.sin(theta)
        rolling_force = mass * self.g * math.cos(theta) * Cr
        air_resistance = 0.5 * self.rho * CdA * (speed_m_s ** 2)
        
        total_resistance = grade_force + rolling_force + air_resistance
        
        # Base energy (Joules)
        base_energy_j = total_resistance * length
        
        # Acceleration energy from stops
        accel_energy_j = stop_prob * 0.5 * mass * (speed_m_s ** 2) * (1.0 - regen_efficiency)
        
        # Congestion losses
        congestion_energy_j = length * max(congestion - 1.0, 0.0) * mass * self.g * 0.01
        
        # Total energy in Joules
        total_energy_j = base_energy_j + accel_energy_j + congestion_energy_j
        
        # Convert to kWh
        total_energy_kwh = total_energy_j / 3.6e6
        
        # Realistic regen clamping (no more than 20% recovery)
        max_regen = 0.2 * (rolling_force + air_resistance) * length / 3.6e6
        total_energy_kwh = max(total_energy_kwh, -max_regen)
        
        return float(total_energy_kwh)


class GraphFeatureExtractor:
    """Efficient graph feature extraction using multiple techniques"""
    
    def __init__(self, G: nx.Graph, embedding_dim: int = 16):
        self.G = G
        self.embedding_dim = embedding_dim
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        
    def compute_features(self) -> np.ndarray:
        """Compute multiple graph-based features"""
        features = []
        
        # 1. Degree features
        degrees = np.array([self.G.degree(node) for node in self.nodes])
        if np.max(degrees) > 0:
            features.append(degrees / np.max(degrees))
        else:
            features.append(np.zeros_like(degrees))
        
        # 2. Betweenness centrality (approximate for large graphs)
        if len(self.G) <= 1000:
            try:
                betweenness = np.array(list(nx.betweenness_centrality(self.G).values()))
                features.append(betweenness)
            except:
                features.append(np.zeros(len(self.G)))
        
        # 3. PageRank
        try:
            pagerank = np.array(list(nx.pagerank(self.G).values()))
            features.append(pagerank)
        except:
            features.append(np.zeros(len(self.G)))
        
        # 4. Spectral embeddings
        spectral_emb = self._compute_robust_spectral_embeddings()
        if spectral_emb.shape[1] > 0:
            features.extend(spectral_emb.T)
        
        # Combine all features
        if features:
            feature_matrix = np.column_stack(features)
        else:
            feature_matrix = np.zeros((len(self.nodes), 1))
        
        # Ensure consistent dimension
        if feature_matrix.shape[1] > self.embedding_dim:
            feature_matrix = feature_matrix[:, :self.embedding_dim]
        elif feature_matrix.shape[1] < self.embedding_dim:
            padding = np.zeros((len(self.nodes), self.embedding_dim - feature_matrix.shape[1]))
            feature_matrix = np.column_stack([feature_matrix, padding])
        
        return feature_matrix.astype(np.float32)
    
    def _compute_robust_spectral_embeddings(self) -> np.ndarray:
        """Compute robust spectral embeddings with fallback"""
        try:
            # Use normalized Laplacian
            L = nx.normalized_laplacian_matrix(self.G).astype(float)
            
            if L.shape[0] <= 10:
                # Direct computation for small graphs
                eigvals, eigvecs = np.linalg.eigh(L.toarray())
            else:
                # Sparse computation for larger graphs
                eigvals, eigvecs = sparse.linalg.eigsh(L, k=min(6, L.shape[0]-2), which='SM')
            
            # Sort by eigenvalue and skip first (zero) eigenvector
            idx = np.argsort(eigvals)
            start_idx = 1 if len(eigvals) > 1 and eigvals[idx[0]] < 1e-8 else 0
            end_idx = start_idx + min(6, len(eigvals) - start_idx)
            
            if end_idx > start_idx:
                embeddings = np.real(eigvecs[:, idx[start_idx:end_idx]])
                # Normalize
                embeddings = (embeddings - np.mean(embeddings, axis=0)) / (np.std(embeddings, axis=0) + 1e-8)
                return embeddings
            else:
                return np.zeros((len(self.G), 0))
            
        except Exception:
            # Fallback to random embeddings with deterministic seed
            np.random.seed(42)
            return np.random.randn(len(self.G), 6).astype(np.float32)


class EVPathFindingEnv(gym.Env):
    """Improved environment for EV pathfinding with energy constraints"""
    
    def __init__(self, 
                 graph: nx.Graph,
                 vehicle_params: Dict[str, Any],
                 max_episode_steps: int = 100,
                 debug: bool = False):
        
        super().__init__()
        self.graph = graph
        self.vehicle_params = vehicle_params
        self.max_episode_steps = max_episode_steps
        self.debug = debug
        
        # Energy model
        self.energy_model = ImprovedEnergyModel(vehicle_params)
        
        # Graph properties
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        
        # Feature extraction
        self.feature_extractor = GraphFeatureExtractor(graph)
        self.node_features = self.feature_extractor.compute_features()
        self.feature_dim = self.node_features.shape[1]
        
        # Battery constraints
        self.battery_capacity = vehicle_params.get('battery_capacity_kwh', 75.0)
        self.min_battery_safety = 0.1  # 10% safety margin
        
        # Action space: choose from neighbors + stay action for robustness
        max_degree = max([degree for _, degree in graph.degree()]) if len(graph) > 0 else 1
        self.action_space = spaces.Discrete(max_degree + 1)  # +1 for stay action
        
        # Observation space
        obs_dim = self.feature_dim * 2 + 4  # current, target, energy, distance, steps, visited
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # State variables (initialized in reset)
        self.current_node = None
        self.target_node = None
        self.energy_used = None
        self.visited_nodes = None
        self.step_count = None
        self.episode_history = None
        
        # Precompute shortest paths for efficiency
        self._precompute_shortest_paths()
        
    def _precompute_shortest_paths(self):
        """Precompute shortest paths for efficient distance calculation"""
        try:
            self.shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(self.graph))
        except:
            # Fallback for disconnected graphs
            self.shortest_path_lengths = {}
            for node in self.graph.nodes():
                try:
                    self.shortest_path_lengths[node] = nx.single_source_shortest_path_length(self.graph, node)
                except:
                    self.shortest_path_lengths[node] = {node: 0}
    
    def _get_node_neighbors(self, node) -> List[int]:
        """Get valid neighbor indices for a node"""
        try:
            neighbors = list(self.graph.neighbors(node))
            neighbor_indices = [self.node_to_idx[n] for n in neighbors]
        except:
            neighbor_indices = []
        
        # Pad with current node to ensure fixed size
        max_neighbors = self.action_space.n - 1
        if len(neighbor_indices) < max_neighbors:
            neighbor_indices += [self.node_to_idx[node]] * (max_neighbors - len(neighbor_indices))
        
        return neighbor_indices[:max_neighbors]
    
    def _get_observation(self) -> np.ndarray:
        """Construct efficient observation vector"""
        current_idx = self.node_to_idx[self.current_node]
        target_idx = self.node_to_idx[self.target_node]
        
        # Node features
        current_features = self.node_features[current_idx]
        target_features = self.node_features[target_idx]
        
        # Energy state
        energy_remaining = max(0.0, self.battery_capacity - self.energy_used)
        energy_ratio = energy_remaining / self.battery_capacity
        
        # Distance to target
        current_distance = self._get_distance(self.current_node, self.target_node)
        max_possible_distance = max(self.n_nodes, 1)  # Avoid division by zero
        distance_ratio = current_distance / max_possible_distance
        
        # Step ratio
        step_ratio = self.step_count / self.max_episode_steps
        
        # Visit ratio (how many unique nodes visited)
        visit_ratio = len(self.visited_nodes) / max(self.n_nodes, 1)
        
        # Combine all features
        observation = np.concatenate([
            current_features,
            target_features,
            np.array([energy_ratio, distance_ratio, step_ratio, visit_ratio], dtype=np.float32)
        ])
        
        return observation
    
    def _get_distance(self, node1, node2) -> float:
        """Get distance between two nodes"""
        try:
            return float(self.shortest_path_lengths[node1].get(node2, self.n_nodes))
        except:
            return float(self.n_nodes)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment with valid start-target pairs"""
        super().reset(seed=seed)
        
        # Reset state
        self.energy_used = 0.0
        self.visited_nodes = set()
        self.step_count = 0
        self.episode_history = []
        
        # Select valid start and target nodes
        if options and 'start_node' in options and 'target_node' in options:
            self.current_node = options['start_node']
            self.target_node = options['target_node']
        else:
            self.current_node, self.target_node = self._select_valid_start_target()
        
        self.visited_nodes.add(self.current_node)
        self.episode_history.append((self.current_node, self.energy_used))
        
        # Validate connectivity
        if not self._is_reachable(self.current_node, self.target_node):
            if self.debug:
                print(f"Warning: Start {self.current_node} and target {self.target_node} not connected")
        
        return self._get_observation(), {}
    
    def _select_valid_start_target(self) -> Tuple[Any, Any]:
        """Select valid start and target nodes with minimum distance"""
        valid_pairs = []
        min_distance = max(2, self.n_nodes // 10)  # Minimum reasonable distance
        
        for start_node in self.graph.nodes():
            if self.graph.degree(start_node) == 0:
                continue
                
            try:
                reachable = nx.node_connected_component(self.graph, start_node)
                possible_targets = [
                    node for node in reachable 
                    if node != start_node and self._get_distance(start_node, node) >= min_distance
                ]
                
                for target in possible_targets:
                    valid_pairs.append((start_node, target))
            except:
                continue
        
        if not valid_pairs:
            # Fallback: any connected pair
            for start_node in self.graph.nodes():
                if self.graph.degree(start_node) > 0:
                    try:
                        reachable = list(nx.node_connected_component(self.graph, start_node))
                        reachable.remove(start_node)
                        if reachable:
                            return start_node, random.choice(reachable)
                    except:
                        continue
            
            # Last resort: any two distinct nodes
            nodes = list(self.graph.nodes())
            if len(nodes) >= 2:
                return random.sample(nodes, 2)
            else:
                raise RuntimeError("Graph has insufficient nodes for start-target pair")
        
        return random.choice(valid_pairs)
    
    def _is_reachable(self, start: Any, target: Any) -> bool:
        """Check if target is reachable from start"""
        try:
            return target in nx.node_connected_component(self.graph, start)
        except:
            return False
    
    def step(self, action: int):
        """Execute one environment step"""
        self.step_count += 1
        
        # Initialize edge_energy to 0 (FIX for UnboundLocalError)
        edge_energy = 0.0
        
        # Get valid neighbors
        neighbors = self._get_node_neighbors(self.current_node)
        
        # Handle stay action (last action)
        if action == len(neighbors):
            next_node = self.current_node
            # edge_energy remains 0.0
        else:
            # Map action to actual neighbor
            if action < len(neighbors):
                next_node_idx = neighbors[action]
                next_node = self.idx_to_node[next_node_idx]
                
                # Calculate energy for movement if we're moving to a different node
                if next_node != self.current_node:
                    try:
                        edge_data = self.graph[self.current_node][next_node]
                        edge_energy = self.energy_model.calculate_edge_energy(edge_data)
                    except:
                        # If edge doesn't exist or other error, stay in place
                        next_node = self.current_node
                        edge_energy = 0.0
            else:
                # Invalid action - stay in place with penalty
                next_node = self.current_node
                # edge_energy remains 0.0
        
        # Update energy usage
        self.energy_used += edge_energy
        
        # Update state
        self.current_node = next_node
        self.visited_nodes.add(self.current_node)
        self.episode_history.append((self.current_node, self.energy_used))
        
        # Calculate rewards
        reward = self._calculate_reward(edge_energy)
        
        # Check termination conditions
        terminated = self.current_node == self.target_node
        truncated = (
            self.step_count >= self.max_episode_steps or
            self.energy_used >= self.battery_capacity * (1 - self.min_battery_safety) or
            not self._has_valid_moves()
        )
        
        info = {
            'energy_used': self.energy_used,
            'distance_to_target': self._get_distance(self.current_node, self.target_node),
            'steps': self.step_count,
            'success': terminated
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _has_valid_moves(self) -> bool:
        """Check if there are valid moves from current position"""
        try:
            neighbors = list(self.graph.neighbors(self.current_node))
            return len(neighbors) > 0
        except:
            return False
    
    def _calculate_reward(self, edge_energy: float) -> float:
        """Calculate reward with balanced weighting"""
        
        # Base rewards/penalties
        success_reward = 10.0
        step_penalty = -0.1
        energy_penalty = -0.5
        
        # Distance improvement
        if len(self.episode_history) > 1:
            old_node = self.episode_history[-2][0]
            old_distance = self._get_distance(old_node, self.target_node)
        else:
            old_distance = self._get_distance(self.current_node, self.target_node)
            
        new_distance = self._get_distance(self.current_node, self.target_node)
        distance_improvement = old_distance - new_distance
        
        # Check if reached target
        if self.current_node == self.target_node:
            return success_reward
        
        # Check energy constraints
        if self.energy_used >= self.battery_capacity * (1 - self.min_battery_safety):
            return -5.0
        
        # Check if stuck (revisiting same node frequently)
        revisit_penalty = 0.0
        if len(self.episode_history) > 3:
            recent_nodes = [hist[0] for hist in self.episode_history[-4:-1]]  # Last 3 nodes before current
            if self.current_node in recent_nodes:
                revisit_penalty = -0.2
        
        # Combined reward
        reward = (
            distance_improvement * 2.0 +  # Emphasize progress
            energy_penalty * abs(edge_energy) * 10.0 +  # Scale energy appropriately
            step_penalty +
            revisit_penalty
        )
        
        return float(reward)
    
    def render(self, mode='human'):
        """Render current environment state"""
        energy_remaining = max(0.0, self.battery_capacity - self.energy_used)
        battery_percent = (energy_remaining / self.battery_capacity) * 100
        
        print(f"Step: {self.step_count}")
        print(f"Current: {self.current_node}, Target: {self.target_node}")
        print(f"Energy: {self.energy_used:.2f}kWh used, {battery_percent:.1f}% remaining")
        print(f"Visited: {len(self.visited_nodes)} nodes")
        print(f"Distance to target: {self._get_distance(self.current_node, self.target_node)}")
        print("-" * 50)


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for EV pathfinding"""
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimension from observation space
        n_input_features = observation_space.shape[0]
        
        self.network = nn.Sequential(
            nn.Linear(n_input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class PathEvaluationCallback(BaseCallback):
    """Enhanced callback for training monitoring and evaluation"""
    
    def __init__(self, eval_env: gym.Env, eval_freq: int = 10000, n_eval_episodes: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            try:
                mean_reward, std_reward = evaluate_policy(
                    self.model, 
                    self.eval_env, 
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=False
                )
                
                print(f"Evaluation at step {self.num_timesteps}:")
                print(f"  Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")
                
                # Save best model
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save("best_ev_path_model")
                    print(f"  New best model saved with reward {mean_reward:.3f}")
                
                print("-" * 50)
            except Exception as e:
                print(f"Evaluation failed: {e}")
            
        return True


def create_ev_environment(graph: nx.Graph, vehicle_params: Dict[str, Any]) -> gym.Env:
    """Factory function to create EV environment"""
    return EVPathFindingEnv(
        graph=graph,
        vehicle_params=vehicle_params,
        max_episode_steps=min(200, max(len(graph.nodes()) * 2, 50)),  # Ensure at least 50 steps
        debug=False
    )


def train_ev_pathfinder(graph: nx.Graph, 
                       vehicle_params: Dict[str, Any],
                       total_timesteps: int = 50000) -> PPO:
    """Train an EV pathfinding agent"""
    
    # Create environments
    train_env = DummyVecEnv([lambda: Monitor(create_ev_environment(graph, vehicle_params))])
    eval_env = create_ev_environment(graph, vehicle_params)
    
    # Training callback
    callback = PathEvaluationCallback(eval_env, eval_freq=5000, n_eval_episodes=5)
    
    # Correct policy configuration
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[128, 128]  # Additional layers after feature extractor
    )
    
    # Initialize agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ev_path_tensorboard/",
        policy_kwargs=policy_kwargs
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name="ev_path_training"
    )
    
    train_env.close()
    return model


def evaluate_ev_pathfinder(model: PPO, 
                          graph: nx.Graph, 
                          vehicle_params: Dict[str, Any],
                          test_episodes: int = 20) -> Dict[str, Any]:
    """Comprehensive evaluation of trained model"""
    
    env = create_ev_environment(graph, vehicle_params)
    results = {
        'success_rate': 0.0,
        'avg_energy_used': 0.0,
        'avg_steps': 0.0,
        'avg_reward': 0.0,
        'paths': []
    }
    
    successful_episodes = 0
    
    for episode in range(test_episodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        total_reward = 0.0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        # Record results
        if info.get('success', False):
            successful_episodes += 1
            results['avg_energy_used'] += info['energy_used']
            results['avg_steps'] += info['steps']
        
        results['avg_reward'] += total_reward
        results['paths'].append({
            'history': env.episode_history,
            'success': info.get('success', False),
            'energy_used': info['energy_used'],
            'steps': info['steps']
        })
    
    # Calculate averages
    results['success_rate'] = successful_episodes / test_episodes
    if successful_episodes > 0:
        results['avg_energy_used'] /= successful_episodes
        results['avg_steps'] /= successful_episodes
    results['avg_reward'] /= test_episodes
    
    env.close()
    return results


# Example usage with error handling
if __name__ == "__main__":
    try:
        # Create sample graph
        print("Creating sample graph...")
        G = nx.erdos_renyi_graph(30, 0.3, seed=42)  # Smaller graph for faster testing
        
        # Add edge attributes
        for u, v in G.edges():
            G[u][v]['length'] = random.uniform(50, 1000)
            G[u][v]['slope_deg'] = random.uniform(-10, 10)
            G[u][v]['speed_kph'] = random.choice([30, 50, 70, 90])
            G[u][v]['stop_probability'] = random.uniform(0.0, 0.3)
            G[u][v]['congestion_factor'] = random.uniform(1.0, 2.0)
        
        # Vehicle parameters
        vehicle_params = {
            'mass_kg': 1800,
            'CdA': 0.62,
            'Cr': 0.015,
            'regen_efficiency': 0.7,
            'battery_capacity_kwh': 75.0
        }
        
        # Test environment creation
        print("Testing environment creation...")
        test_env = create_ev_environment(G, vehicle_params)
        obs, _ = test_env.reset()
        print(f"Observation space: {test_env.observation_space.shape}")
        print(f"Action space: {test_env.action_space.n}")
        
        # Train model
        print("Training EV Pathfinder...")
        model = train_ev_pathfinder(G, vehicle_params, total_timesteps=10000)  # Fewer timesteps for testing
        
        # Evaluate
        print("\nEvaluating trained model...")
        results = evaluate_ev_pathfinder(model, G, vehicle_params)
        
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Average Energy Used: {results['avg_energy_used']:.2f} kWh")
        print(f"Average Steps: {results['avg_steps']:.1f}")
        print(f"Average Reward: {results['avg_reward']:.2f}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()