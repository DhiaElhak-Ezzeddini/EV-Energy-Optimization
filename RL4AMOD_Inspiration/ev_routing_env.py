# ev_routing_env.py
import networkx as nx
import numpy as np
import random
from typing import Tuple, List, Dict, Any
import math # For distance calculation if needed

# Optional: For reward shaping
from geopy.distance import geodesic

class EVRoutingEnv:
    """
    Reinforcement Learning Environment for Energy-Optimized EV Routing.
    Uses a NetworkX graph where edges have energy-relevant attributes.
    """
    def __init__(self, graph: nx.MultiDiGraph, vehicle_params: Dict[str, Any], start_node_id=None, target_node_id=None):
        self.graph = graph
        # --- Precompute node positions if available ---
        self.node_positions = {nid: (data.get('x'), data.get('y'))
                               for nid, data in graph.nodes(data=True)
                               if 'x' in data and 'y' in data}

        self.vehicle_params = vehicle_params # e.g., {'battery_capacity_kwh': 50, 'min_soc_fraction': 0.1, ... energy model params ...}
        self.nodes = list(graph.nodes) # Node IDs (e.g., osmid)
        # Ensure nodes are hashable and consistently ordered if needed elsewhere
        try:
            self.nodes.sort() # Sort for consistency if node IDs allow it
        except TypeError:
            print("Warning: Node IDs are not sortable, order might vary.")

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

        # Use max_steps from config, fallback to heuristic
        self.max_steps = self.vehicle_params.get('max_steps', self.num_nodes * 2)
        self.current_step = 0

        # --- Rewards/Penalties ---
        self.goal_reward = float(self.vehicle_params.get('goal_reward', 1000.0))
        self.fail_penalty_soc = float(self.vehicle_params.get('fail_penalty_soc', -1000.0))
        self.fail_penalty_steps = float(self.vehicle_params.get('fail_penalty_steps', -500.0))
        self.fail_penalty_invalid = float(self.vehicle_params.get('fail_penalty_invalid', -100.0))
        self.fail_penalty_stuck = float(self.vehicle_params.get('fail_penalty_stuck', -500.0))
        self.step_penalty = float(self.vehicle_params.get('step_penalty', -1.0))
        # Modify energy reward: normalize or scale
        self.energy_reward_scale = float(self.vehicle_params.get('energy_reward_scale', 5.0)) # How many 'steps' is 1kWh worth? (Negative)

        # Optional Reward Shaping
        self.use_distance_shaping = self.vehicle_params.get('use_distance_shaping', False)
        self.distance_shaping_factor = float(self.vehicle_params.get('distance_shaping_factor', 0.1))
        self.last_heuristic_dist = None


        # Basic validation
        # Only check connectivity if graph is small enough to avoid long computation
        if self.num_nodes < 1000 and not nx.is_strongly_connected(self.graph):
             print("Warning: Graph is not strongly connected. Some routes may be impossible.")


    def _calculate_energy_kwh(self, u_node_id, v_node_id) -> float:
        """
        Calculates the energy required to travel from node u to node v.
        Uses attributes added by network_gen.py.
        TODO: Replace with a more sophisticated energy model.
        """
        if not self.graph.has_edge(u_node_id, v_node_id):
            return float('inf')

        try:
            # Get edge data - Handle MultiDiGraph: find the best edge (e.g., shortest length or lowest base energy estimate)
            possible_edges = self.graph.get_edge_data(u_node_id, v_node_id)
            if not possible_edges: return float('inf') # Should be caught by has_edge, but safety check
            # Choose edge with minimum length as a heuristic if multiple exist
            edge_data = min(possible_edges.values(), key=lambda x: x.get('length', float('inf')))

            distance_m = edge_data.get('length', 100.0) # Keep in meters for potential physics model
            distance_km = distance_m / 1000.0
            slope_deg = edge_data.get('slope_deg', 0.0)
            congestion = edge_data.get('congestion_factor', edge_data.get('base_congestion', 1.0))
            speed_kph = edge_data.get('speed_kph', 30.0)
            road_quality = edge_data.get('road_quality', 0.8)
            travel_time_sec = edge_data.get('travel_time', distance_m / (speed_kph * 1000 / 3600) if speed_kph > 0 else 3600)

            # --- Simple Placeholder Energy Model ---
            # Based on VT-CPFM concept (simplified)
            # Ref: https://www.researchgate.net/publication/332610118_Prediction_of_Energy_Consumption_for_Electric_Vehicles_Based_on_Real-World_Data
            mass_kg = self.vehicle_params.get('mass_kg', 1800)
            gravity = 9.81
            air_density = self.vehicle_params.get('air_density', 1.225) # kg/m^3
            drag_coeff = self.vehicle_params.get('drag_coeff', 0.3)
            frontal_area = self.vehicle_params.get('frontal_area', 2.5) # m^2
            rolling_coeff = self.vehicle_params.get('rolling_coeff', 0.01) * (1 + 0.5 * (1 - road_quality)) # Increase on poor roads
            motor_efficiency = self.vehicle_params.get('motor_efficiency', 0.9)
            regen_efficiency = self.vehicle_params.get('regen_efficiency', 0.6)
            accessory_power_kw = self.vehicle_params.get('accessory_power_kw', 1.0) # HVAC etc.

            speed_mps = speed_kph * 1000 / 3600 / max(1.0, congestion) # Effective speed
            if speed_mps < 1.0: speed_mps = 1.0 # Avoid division by zero, assume minimum movement speed
            if travel_time_sec <= 0: travel_time_sec = distance_m / speed_mps if speed_mps > 0 else 3600 # Estimate time if needed

            # Forces
            F_rolling = rolling_coeff * mass_kg * gravity * math.cos(math.radians(slope_deg))
            F_drag = 0.5 * air_density * drag_coeff * frontal_area * (speed_mps ** 2)
            F_slope = mass_kg * gravity * math.sin(math.radians(slope_deg))
            # Assume constant speed for simplicity (no acceleration force)
            F_traction = F_rolling + F_drag + F_slope

            # Power at wheels (Watts)
            P_wheels = F_traction * speed_mps

            # Energy (kWh)
            if P_wheels >= 0: # Motoring
                energy_wh = (P_wheels / motor_efficiency) * (travel_time_sec / 3600)
            else: # Regenerating
                energy_wh = (P_wheels * regen_efficiency) * (travel_time_sec / 3600)

            # Add accessory energy
            energy_wh += (accessory_power_kw * 1000) * (travel_time_sec / 3600)

            energy_kwh = energy_wh / 1000.0

            # --- Stochastic Stop Penalty ---
            stop_prob = edge_data.get('stop_prob', 0.1)
            if random.random() < stop_prob:
                 energy_kwh += self.vehicle_params.get('stop_penalty_kwh', 0.05)

            # Ensure non-negative energy cost if not modeling net regeneration gain over a trip
            # return max(0.001, energy_kwh) # Minimum energy cost to move
            return energy_kwh # Allow negative for regeneration steps

        except Exception as e:
            print(f"Error calculating energy for edge ({u_node_id} -> {v_node_id}): {e}")
            return float('inf') # Treat as impassable on error

    def _get_state(self) -> Tuple[int, int, float]:
        """ Returns the current state as (current_node_idx, target_node_idx, current_soc_fraction) """
        # Handle cases where nodes might not be in the map (should not happen in normal operation)
        current_idx = self.node_to_idx.get(self.current_node_id, -1)
        target_idx = self.node_to_idx.get(self.target_node_id, -1)
        if current_idx == -1 or target_idx == -1:
             print(f"Error: Current ({self.current_node_id}) or Target ({self.target_node_id}) node ID not found in node_to_idx map.")
             # Fallback or raise error? For now, return invalid indices
             return (-1, -1, self.current_soc_fraction)
        return (current_idx, target_idx, self.current_soc_fraction)

    def _get_heuristic_distance(self, node1_id, node2_id):
         """ Estimate distance using Euclidean distance as fallback if geodesic fails """
         pos1 = self.node_positions.get(node1_id)
         pos2 = self.node_positions.get(node2_id)
         if pos1 and pos2 and pos1[0] is not None and pos2[0] is not None:
              try:
                   # Use geodesic distance if coordinates are valid lat/lon
                   # Ensure order is (lat, lon) for geodesic
                   lat1, lon1 = self.graph.nodes[node1_id]['y'], self.graph.nodes[node1_id]['x']
                   lat2, lon2 = self.graph.nodes[node2_id]['y'], self.graph.nodes[node2_id]['x']
                   return geodesic((lat1, lon1), (lat2, lon2)).km
              except Exception:
                   # Fallback to Euclidean on x, y assuming they are planar for this estimate
                   try:
                       return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                   except Exception:
                       return float('inf') # Cannot calculate distance
         return float('inf') # Cannot calculate distance if positions are missing


    def reset(self, start_node_id=None, target_node_id=None, start_soc=1.0) -> Tuple[int, int, float]:
        """ Resets the environment to a new starting state. """
        if start_node_id is not None and start_node_id in self.graph:
            self.start_node_id = start_node_id
        else:
            self.start_node_id = random.choice(self.nodes)

        # Ensure target is different and exists
        possible_targets = [n for n in self.nodes if n != self.start_node_id]
        if target_node_id is not None and target_node_id in possible_targets:
            self.target_node_id = target_node_id
        elif possible_targets:
            self.target_node_id = random.choice(possible_targets)
        else: # Handle single-node or disconnected graph case
             self.target_node_id = self.start_node_id

        self.current_node_id = self.start_node_id
        self.current_soc_fraction = float(start_soc) # Ensure float
        self.current_step = 0
        self.last_heuristic_dist = self._get_heuristic_distance(self.current_node_id, self.target_node_id)


        # print(f"Env Reset: Start={self.start_node_id}, Target={self.target_node_id}, SoC={self.current_soc_fraction:.2f}")
        return self._get_state()

    def step(self, action_node_idx: int) -> Tuple[Tuple[int, int, float], float, bool, Dict[str, Any]]:
        """ Executes one step in the environment based on the chosen action (neighbor index). """
        self.current_step += 1
        done = False
        reward = self.step_penalty # Base penalty for taking a step
        status = "InProgress"
        info = {
            'energy_consumed_kwh': 0.0,
            'soc_violation': False,
            'max_steps_reached': False,
            'invalid_action': False,
            'status': status
            # 'path': [self.current_node_id] # Path tracking removed for performance, handled in train loop
        }

        chosen_node_id = self.idx_to_node.get(action_node_idx, None)

        # --- Action Validation ---
        if chosen_node_id is None or not self.graph.has_edge(self.current_node_id, chosen_node_id):
            reward += self.fail_penalty_invalid
            info['invalid_action'] = True
            status = "Fail_InvalidAction"
            # Keep state the same, maybe end episode? For now, just penalize.
            done = True # End episode on invalid action to prevent loops
            info['status'] = status
            return self._get_state(), reward, done, info

        # --- Energy Calculation ---
        energy_cost_kwh = self._calculate_energy_kwh(self.current_node_id, chosen_node_id)
        info['energy_consumed_kwh'] = energy_cost_kwh
        # Apply energy cost penalty (scaled)
        reward += energy_cost_kwh * (self.energy_reward_scale / self.vehicle_params['battery_capacity_kwh']) # Scale relative to capacity


        required_soc_fraction = energy_cost_kwh / self.vehicle_params['battery_capacity_kwh']
        min_soc = self.vehicle_params['min_soc_fraction']

        # --- State Transition ---
        next_node_id = self.current_node_id # Default to staying if move fails
        next_soc_fraction = self.current_soc_fraction

        if self.current_soc_fraction >= required_soc_fraction + min_soc:
            # Move successful
            next_node_id = chosen_node_id
            next_soc_fraction = self.current_soc_fraction - required_soc_fraction
            # info['path'].append(next_node_id) # Track path in train loop if needed

            # --- Reward Shaping (Optional) ---
            if self.use_distance_shaping:
                 current_heuristic_dist = self._get_heuristic_distance(next_node_id, self.target_node_id)
                 if self.last_heuristic_dist is not None and current_heuristic_dist < float('inf'):
                      # Reward for reducing distance to target
                      distance_reduction = self.last_heuristic_dist - current_heuristic_dist
                      shaping_reward = distance_reduction * self.distance_shaping_factor
                      reward += shaping_reward
                 self.last_heuristic_dist = current_heuristic_dist if current_heuristic_dist < float('inf') else self.last_heuristic_dist

        else:
            # Failed move - insufficient SoC
            reward += self.fail_penalty_soc
            done = True
            info['soc_violation'] = True
            status = "Fail_SoC"
            # State remains unchanged (next_node_id, next_soc_fraction are already set to current)


        # Update internal state *after* all calculations based on current state
        self.current_node_id = next_node_id
        self.current_soc_fraction = next_soc_fraction


        # --- Check for Goal ---
        if self.current_node_id == self.target_node_id:
            if not done: # Avoid double rewarding if failed on the last step to target
                reward += self.goal_reward
                status = "Success"
            done = True

        # --- Check for Max Steps ---
        if self.current_step >= self.max_steps and not done:
            reward += self.fail_penalty_steps
            done = True
            info['max_steps_reached'] = True
            status = "Fail_MaxSteps"

        info['status'] = status
        return self._get_state(), reward, done, info

    def get_valid_actions_indices(self, current_node_id=None) -> List[int]:
        """ Returns indices of valid neighboring nodes. """
        node_id = current_node_id if current_node_id is not None else self.current_node_id
        if node_id not in self.graph: return []
        neighbors = list(self.graph.neighbors(node_id))
        # Filter out neighbors that might not be in the node_to_idx map (if graph subsetting occurs)
        neighbor_indices = [self.node_to_idx[n] for n in neighbors if n in self.node_to_idx]
        return neighbor_indices

    def get_reachable_actions_indices(self, current_node_id=None, current_soc=None, debug=False) -> List[int]:
        """ Returns indices of neighboring nodes reachable with the current SoC. """
        node_id = current_node_id if current_node_id is not None else self.current_node_id
        soc = current_soc if current_soc is not None else self.current_soc_fraction
        min_soc = self.vehicle_params['min_soc_fraction']
        capacity = self.vehicle_params['battery_capacity_kwh']

        if node_id not in self.graph:
             if debug: print(f"Node {node_id} not in graph for reachability check.")
             return [] # Node doesn't exist

        reachable_indices = []
        neighbors = list(self.graph.neighbors(node_id))
        if debug: print(f"Checking reachability from {node_id} (SoC: {soc:.3f}) -> Neighbors: {neighbors}")

        for neighbor_id in neighbors:
            if neighbor_id not in self.node_to_idx:
                if debug: print(f"  Neighbor {neighbor_id} not in node map. Skipping.")
                continue # Skip if neighbor not mapped

            energy_cost_kwh = self._calculate_energy_kwh(node_id, neighbor_id)
            required_soc_fraction = energy_cost_kwh / capacity
            can_reach = soc >= required_soc_fraction + min_soc

            if debug:
                print(f"  -> {neighbor_id}: Cost {energy_cost_kwh:.4f} kWh (Req SoC: {required_soc_fraction:.4f}). Reachable: {can_reach}")

            if can_reach:
                reachable_indices.append(self.node_to_idx[neighbor_id])

        if debug and not reachable_indices: print(f"  No neighbors reachable from {node_id}.")
        return reachable_indices
