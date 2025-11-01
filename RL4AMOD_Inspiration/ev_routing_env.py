# ev_routing_env.py
import networkx as nx
import numpy as np
import random
from typing import Tuple, List, Dict, Any, Optional
import math
from geopy.distance import geodesic

class EVRoutingEnv:
    """
    Reinforcement Learning Environment for Energy-Optimized EV Routing.
    The state is a tuple: (current_node_id, target_node_id, current_soc_fraction).
    """
    def __init__(self, graph: nx.MultiDiGraph, vehicle_params: Dict[str, Any], start_node_id: Optional[int] = None, target_node_id: Optional[int] = None):
        self.graph = graph
        
        # --- Precompute node positions ---
        self.node_positions = {nid: (data.get('x'), data.get('y'))
                               for nid, data in graph.nodes(data=True)
                               if 'x' in data and 'y' in data}

        self.vehicle_params = vehicle_params 
        self.nodes = list(graph.nodes) 
        try:
            self.nodes.sort() 
        except TypeError:
            print("Warning: Node IDs are not sortable, order might vary.")

        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.nodes)}
        
        # Store initial parameters
        self.initial_start_node_id = start_node_id
        self.initial_target_node_id = target_node_id

        # Internal state variables
        self.current_node_id: Optional[int] = None
        self.target_node_id: Optional[int] = None
        self.current_soc_fraction: float = 1.0 
        self.current_step: int = 0
        self.path_taken: List[int] = []
        self.total_energy_kwh: float = 0.0

    def reset(self, start_node_id: Optional[int] = None, target_node_id: Optional[int] = None) -> Tuple[int, int, float]:
        """
        Resets the environment for a new episode. Randomly selects start/target if none provided.
        Returns the initial state tuple (current_node_id, target_node_id, current_soc_fraction).
        """
        valid_nodes = [nid for nid in self.nodes if nid in self.node_positions]
        
        # 1. Select Start Node
        if start_node_id is None:
            self.current_node_id = random.choice(valid_nodes)
        else:
            self.current_node_id = start_node_id
            
        # 2. Select Target Node (must be different from start)
        if target_node_id is None:
            temp_nodes = [n for n in valid_nodes if n != self.current_node_id]
            if not temp_nodes:
                raise ValueError("Graph must have at least two nodes for routing.")
            self.target_node_id = random.choice(temp_nodes)
        else:
            self.target_node_id = target_node_id
            
        # 3. Initialize State
        self.current_soc_fraction = 1.0 # Start with a full battery
        self.current_step = 0
        self.path_taken = [self.current_node_id]
        self.total_energy_kwh = 0.0

        if self.current_node_id is None or self.target_node_id is None:
             raise ValueError("Start or Target node ID is None after reset.")
             
        # Initial check for path existence (optional but recommended)
        # try:
        #     if not nx.has_path(self.graph, self.current_node_id, self.target_node_id):
        #         print(f"Warning: No path from {self.current_node_id} to {self.target_node_id}")
        # except nx.NetworkXNoPath:
        #     print("No path found.")

        return self._get_current_state()

    def _get_current_state(self) -> Tuple[int, int, float]:
        """Helper to return the current state tuple."""
        if self.current_node_id is None or self.target_node_id is None:
            raise ValueError("Environment state is invalid (node IDs are None).")
        return (self.current_node_id, self.target_node_id, self.current_soc_fraction)

    def _calculate_energy_kwh(self, u: int, v: int) -> float:
        """
        Calculates the estimated energy consumption (kWh) for traversing edge (u, v, key=0).
        Uses a simplified model based on edge attributes.
        """
        # Note: Using key=0 for the default MultiDiGraph edge (assuming one primary link)
        if not self.graph.has_edge(u, v):
            return float('inf') 

        edge_data = self.graph.get_edge_data(u, v)[0] # Assuming key=0 for simplicity/consistency
        params = self.vehicle_params['energy_model_params']
        
        # Edge Attributes from network_gen.py
        length_km = edge_data.get('length', 0.0) / 1000.0 # Convert meters to kilometers
        slope_deg = edge_data.get('slope_deg', 0.0) # Degrees
        congestion_factor = edge_data.get('congestion_factor', 0.5) # 0 to 1
        road_quality = edge_data.get('road_quality', 0.5) # 0 to 1
        
        # Base consumption (distance-based)
        base_consumption = params['base_consumption_rate'] * length_km
        
        # Slope influence (positive for uphill, negative for downhill)
        slope_consumption = params['slope_factor'] * slope_deg * length_km
        
        # Road quality and congestion penalty (always positive consumption/loss)
        penalty_consumption = length_km * (
            params['quality_factor'] * road_quality + 
            params['congestion_penalty'] * congestion_factor
        )
        
        # Total energy: ensure it's not negative (net regeneration is assumed minimal/capped)
        energy_kwh = base_consumption + slope_consumption + penalty_consumption
        return max(0.0, energy_kwh)


    def _calculate_reward(self, prev_node_id: int, action_node_id: int, next_soc: float, max_steps: int) -> float:
        """
        Calculates the reward for moving from prev_node_id to action_node_id.
        Reward is a combination of negative energy cost and progress toward the target.
        """
        current_node_pos = self.node_positions.get(prev_node_id)
        target_node_pos = self.node_positions.get(self.target_node_id) # type: ignore
        next_node_pos = self.node_positions.get(action_node_id)
        
        # 1. Energy Cost (Negative Reward)
        energy_cost_kwh = self._calculate_energy_kwh(prev_node_id, action_node_id)
        energy_reward = -1.0 * energy_cost_kwh 

        # 2. Progress Reward (Euclidean distance reduction)
        progress_reward = 0.0
        if current_node_pos and target_node_pos and next_node_pos:
            # Distance from previous node to target
            dist_prev_target = geodesic(current_node_pos, target_node_pos).meters
            # Distance from next node to target
            dist_next_target = geodesic(next_node_pos, target_node_pos).meters
            
            # Progress: how much closer we got (positive)
            distance_reduced = dist_prev_target - dist_next_target
            # Normalize progress by a large, stable factor (e.g., 10km)
            progress_reward = distance_reduced / 10000.0 
        
        # 3. Goal Reached Bonus/Penalty
        if action_node_id == self.target_node_id:
            # Large positive reward for reaching the goal
            goal_reward = 100.0 
        elif self.current_step >= max_steps:
             # Penalty for hitting max steps (or being "stuck")
             goal_reward = -50.0 
        else:
            goal_reward = 0.0
            
        # 4. Low SOC Penalty
        min_soc = self.vehicle_params['min_soc_fraction']
        soc_penalty = 0.0
        if next_soc < min_soc:
            # Severe penalty for driving battery too low
            soc_penalty = -20.0 
        
        # Final reward combination
        total_reward = energy_reward + progress_reward + goal_reward + soc_penalty
        return total_reward

    def step(self, action_node_id: int, max_steps: int) -> Tuple[Tuple[int, int, float], float, bool, Dict[str, Any]]:
        """
        Performs one step in the environment by moving to the action_node_id.
        Returns: (next_state, reward, done, info)
        """
        prev_node_id = self.current_node_id
        if prev_node_id is None or self.target_node_id is None:
             raise ValueError("Environment step called with invalid state.")
        
        # 1. Calculate energy cost and update SoC
        energy_cost_kwh = self._calculate_energy_kwh(prev_node_id, action_node_id)
        capacity = self.vehicle_params['battery_capacity_kwh']
        energy_cost_fraction = energy_cost_kwh / capacity
        
        next_soc = self.current_soc_fraction - energy_cost_fraction
        
        # If the move was not valid (should be filtered by get_reachable_actions but a safety check)
        if next_soc < self.vehicle_params['min_soc_fraction'] and action_node_id != prev_node_id:
             # Agent chose an invalid move - penalize and end episode early
             next_soc = self.vehicle_params['min_soc_fraction'] - 0.01 
             reward = -100.0 
             done = True
             self.current_node_id = action_node_id # Move to terminal low-battery state
        else:
            # 2. Update state variables
            self.current_node_id = action_node_id
            self.current_soc_fraction = next_soc
            self.total_energy_kwh += energy_cost_kwh
            self.current_step += 1
            self.path_taken.append(action_node_id)
            
            # 3. Check for termination
            done = (self.current_node_id == self.target_node_id) or (self.current_step >= max_steps)
            
            # 4. Calculate reward
            reward = self._calculate_reward(prev_node_id, action_node_id, next_soc, max_steps)


        next_state = self._get_current_state()
        info = {
            'step': self.current_step,
            'energy_kwh': energy_cost_kwh,
            'total_energy_kwh': self.total_energy_kwh,
            'path_length': len(self.path_taken) - 1,
            'reason': 'Goal Reached' if self.current_node_id == self.target_node_id else \
                      'Max Steps' if self.current_step >= max_steps else \
                      'Low Battery' if self.current_soc_fraction < self.vehicle_params['min_soc_fraction'] else \
                      'Continue'
        }
        
        return next_state, reward, done, info


    def get_reachable_actions(self, current_node_id: Optional[int] = None, current_soc: Optional[float] = None, debug: bool = False) -> List[int]:
        """
        Returns a list of reachable neighbor node IDs (osmid) from the current state.
        A neighbor is reachable if the current SoC can cover the edge energy cost 
        and still leave the battery above the minimum required SoC (min_soc).
        """
        node_id = current_node_id if current_node_id is not None else self.current_node_id
        soc = current_soc if current_soc is not None else self.current_soc_fraction
        
        if node_id is None or node_id not in self.graph:
             return [] 

        min_soc = self.vehicle_params['min_soc_fraction']
        capacity = self.vehicle_params['battery_capacity_kwh']

        # Include self-loop (staying at the current node) as a valid action
        # This is primarily for conceptual completeness in RL, though usually unused in routing
        # reachable_node_ids = [node_id] 
        reachable_node_ids = []

        neighbors = list(self.graph.neighbors(node_id))

        for neighbor_id in neighbors:
            if neighbor_id not in self.node_to_idx:
                continue 

            # Energy calculation relies on the edge (u, v) existence
            energy_cost_kwh = self._calculate_energy_kwh(node_id, neighbor_id)
            required_soc_fraction = energy_cost_kwh / capacity
            
            # A move is valid if the remaining SoC is >= min_soc
            can_reach = soc >= required_soc_fraction + min_soc

            if can_reach:
                reachable_node_ids.append(neighbor_id)
        
        # If the target node is a neighbor, it's always reachable if energy allows
        
        return reachable_node_ids
