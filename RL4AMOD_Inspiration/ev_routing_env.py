# ev_routing_env.py
import networkx as nx
import numpy as np
import random
from typing import Tuple, List, Dict, Any

class EVRoutingEnv:
    """
    Reinforcement Learning Environment for Energy-Optimized EV Routing.

    Uses a NetworkX graph where edges have energy-relevant attributes.
    """
    def __init__(self, graph: nx.MultiDiGraph, vehicle_params: Dict[str, Any], start_node_id=None, target_node_id=None):
        self.graph = graph
        self.vehicle_params = vehicle_params # e.g., {'battery_capacity_kwh': 50, 'min_soc_fraction': 0.1, ... energy model params ...}
        self.nodes = list(graph.nodes) # Node IDs (e.g., osmid)
        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.nodes)}
        self.idx_to_node = {i: node_id for node_id, i in self.node_to_idx.items()}
        self.num_nodes = len(self.nodes)

        # Ensure necessary vehicle params are present
        if 'battery_capacity_kwh' not in self.vehicle_params or 'min_soc_fraction' not in self.vehicle_params:
            raise ValueError("vehicle_params must include 'battery_capacity_kwh' and 'min_soc_fraction'")

        self.start_node_id = start_node_id
        self.target_node_id = target_node_id
        self.current_node_id = None
        self.current_soc_fraction = 1.0 # Start fully charged

        self.max_steps = self.num_nodes * 2 # Heuristic limit to prevent infinite loops
        self.current_step = 0

        # --- Rewards/Penalties (Make these configurable via vehicle_params or env_config) ---
        self.goal_reward = self.vehicle_params.get('goal_reward', 1000)
        self.fail_penalty_soc = self.vehicle_params.get('fail_penalty_soc', -1000)
        self.fail_penalty_steps = self.vehicle_params.get('fail_penalty_steps', -500)
        self.fail_penalty_invalid = self.vehicle_params.get('fail_penalty_invalid', -100) # Penalty for choosing non-neighbor
        self.step_penalty = self.vehicle_params.get('step_penalty', -1.0) # Penalty per step
        self.energy_reward_factor = self.vehicle_params.get('energy_reward_factor', -1.0) # Multiplier for kWh cost

        # Basic validation
        if not nx.is_strongly_connected(self.graph):
             print("Warning: Graph is not strongly connected. Some routes may be impossible.")


    def _calculate_energy_kwh(self, u_node_id, v_node_id) -> float:
        """
        Calculates the energy required to travel from node u to node v.
        Placeholder for a detailed energy model.
        Uses attributes added by network_gen.py.
        """
        if not self.graph.has_edge(u_node_id, v_node_id):
            return float('inf') # Should not happen if checking neighbors

        # Get edge data (handle MultiDiGraph potential multiple edges - take first/shortest?)
        # For simplicity, assume one relevant edge or use shortest length if multiple exist.
        edge_data = min(self.graph.get_edge_data(u_node_id, v_node_id).values(),
                        key=lambda x: x.get('length', float('inf')))

        distance_km = edge_data.get('length', 100.0) / 1000.0 # Convert meters to km
        slope_deg = edge_data.get('slope_deg', 0.0)
        # Use dynamic congestion if available, else base congestion
        congestion = edge_data.get('congestion_factor', edge_data.get('base_congestion', 1.0))
        speed_kph = edge_data.get('speed_kph', 30.0) # Use provided speed
        road_quality = edge_data.get('road_quality', 0.8)

        # --- Replace with your sophisticated energy model ---
        # Simple Placeholder Model:
        base_consumption = self.vehicle_params.get('base_kwh_per_km', 0.18)
        slope_effect = self.vehicle_params.get('slope_factor', 0.05) * slope_deg # kwh/km per degree
        congestion_effect = self.vehicle_params.get('congestion_factor', 0.1) * max(0, congestion - 1.0) # Extra kwh/km
        speed_effect = self.vehicle_params.get('speed_factor', 0.0001) * max(0, speed_kph - 50)**2 # Penalty for high speed

        consumption_kwh_km = base_consumption * (1/max(0.3, road_quality)) + slope_effect + congestion_effect + speed_effect

        energy_kwh = distance_km * max(0.05, consumption_kwh_km) # Ensure minimum consumption

        # Basic Regeneration (if slope is negative) - very simplistic
        if slope_deg < -1.0: # Only significant downhill
             regen_efficiency = self.vehicle_params.get('regen_efficiency', 0.6)
             potential_regen = abs(distance_km * slope_effect) # Energy potentially gained from gravity
             energy_kwh -= potential_regen * regen_efficiency
             energy_kwh = max(0, energy_kwh) # Cannot gain more than used moving flat

        # --- End Placeholder Model ---

        # Add stochastic stop energy cost
        stop_prob = edge_data.get('stop_prob', 0.1)
        if random.random() < stop_prob:
             energy_kwh += self.vehicle_params.get('stop_penalty_kwh', 0.05)

        return energy_kwh

    def _get_state(self) -> Tuple[int, int, float]:
        """ Returns the current state as (current_node_idx, target_node_idx, current_soc_fraction) """
        current_idx = self.node_to_idx[self.current_node_id]
        target_idx = self.node_to_idx[self.target_node_id]
        return (current_idx, target_idx, self.current_soc_fraction)

    def reset(self, start_node_id=None, target_node_id=None, start_soc=1.0) -> Tuple[int, int, float]:
        """ Resets the environment to a new starting state. """
        if start_node_id is not None and start_node_id in self.graph:
            self.start_node_id = start_node_id
        else:
            self.start_node_id = random.choice(self.nodes)

        if target_node_id is not None and target_node_id in self.graph:
             self.target_node_id = target_node_id
        else:
            possible_targets = [n for n in self.nodes if n != self.start_node_id]
            if not possible_targets:
                 self.target_node_id = self.start_node_id # Handle single-node graph
            else:
                 # Try to pick a target reachable within initial SOC (heuristic)
                 potential_targets = random.sample(possible_targets, min(len(possible_targets), 20))
                 self.target_node_id = random.choice(potential_targets) # Fallback if none seem reachable

        self.current_node_id = self.start_node_id
        self.current_soc_fraction = start_soc
        self.current_step = 0

        # print(f"Env Reset: Start={self.start_node_id}, Target={self.target_node_id}, SoC={self.current_soc_fraction:.2f}")
        return self._get_state()

    def step(self, action_node_idx: int) -> Tuple[Tuple[int, int, float], float, bool, Dict[str, Any]]:
        """ Executes one step in the environment based on the chosen action (neighbor index). """
        self.current_step += 1
        done = False
        reward = self.step_penalty
        info = {'energy_consumed_kwh': 0, 'soc_violation': False, 'max_steps_reached': False, 'invalid_action': False, 'path': [self.current_node_id]}

        chosen_node_id = self.idx_to_node.get(action_node_idx, None)

        # --- Action Validation ---
        if chosen_node_id is None or not self.graph.has_edge(self.current_node_id, chosen_node_id):
            reward += self.fail_penalty_invalid
            info['invalid_action'] = True
            # Keep state the same, maybe end episode? For now, just penalize.
            # done = True # Optionally end episode on invalid action
            return self._get_state(), reward, done, info

        # --- Energy Calculation ---
        energy_cost_kwh = self._calculate_energy_kwh(self.current_node_id, chosen_node_id)
        info['energy_consumed_kwh'] = energy_cost_kwh
        reward += energy_cost_kwh * self.energy_reward_factor # Add energy cost penalty

        required_soc_fraction = energy_cost_kwh / self.vehicle_params['battery_capacity_kwh']
        min_soc = self.vehicle_params['min_soc_fraction']

        # --- State Transition ---
        if self.current_soc_fraction >= required_soc_fraction + min_soc:
            # Move successful
            self.current_node_id = chosen_node_id
            self.current_soc_fraction -= required_soc_fraction
            info['path'].append(self.current_node_id) # Log path
        else:
            # Failed move - insufficient SoC
            reward += self.fail_penalty_soc
            done = True
            info['soc_violation'] = True
            # State remains unchanged

        # --- Check for Goal ---
        if self.current_node_id == self.target_node_id:
            reward += self.goal_reward
            done = True

        # --- Check for Max Steps ---
        if self.current_step >= self.max_steps and not done:
            reward += self.fail_penalty_steps
            done = True
            info['max_steps_reached'] = True

        return self._get_state(), reward, done, info

    def get_valid_actions_indices(self, current_node_id=None) -> List[int]:
        """ Returns indices of valid neighboring nodes. """
        node_id = current_node_id if current_node_id is not None else self.current_node_id
        if node_id not in self.graph: return [] # Should not happen
        neighbors = list(self.graph.neighbors(node_id))
        neighbor_indices = [self.node_to_idx[n] for n in neighbors if n in self.node_to_idx]
        return neighbor_indices

    def get_reachable_actions_indices(self, current_node_id=None, current_soc=None) -> List[int]:
        """ Returns indices of neighboring nodes reachable with the current SoC. """
        node_id = current_node_id if current_node_id is not None else self.current_node_id
        soc = current_soc if current_soc is not None else self.current_soc_fraction
        min_soc = self.vehicle_params['min_soc_fraction']
        capacity = self.vehicle_params['battery_capacity_kwh']

        reachable_indices = []
        for neighbor_id in self.graph.neighbors(node_id):
            if neighbor_id not in self.node_to_idx: continue # Skip if neighbor not mapped

            energy_cost_kwh = self._calculate_energy_kwh(node_id, neighbor_id)
            required_soc_fraction = energy_cost_kwh / capacity

            if soc >= required_soc_fraction + min_soc:
                reachable_indices.append(self.node_to_idx[neighbor_id])
        return reachable_indices