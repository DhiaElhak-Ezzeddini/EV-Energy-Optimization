# graph_parser.py
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import Tuple, Dict, List, Optional
import numpy as np

class GraphParser:
    """
    Parses the environment state into a PyTorch Geometric Data object
    for the GNN Q-Network. Includes feature normalization.
    """
    def __init__(self, graph: nx.MultiDiGraph, device: torch.device):
        self.graph = graph
        self.nodes = list(graph.nodes) # Node IDs
        try:
            self.nodes.sort() # Sort for consistency if node IDs allow it
        except TypeError:
            print("Warning: Node IDs are not sortable, parser order might vary.")

        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.nodes)}
        self.device = device
        self.num_nodes = len(self.nodes)

        # --- Precompute edge_index, edge_attr, and normalization stats ---
        print("Building edge index, attributes, and normalization stats for parser...")
        edge_list_indices: List[List[int]] = []
        edge_attributes_raw: List[List[float]] = []
        self.edge_idx_map: Dict[Tuple[int, int], int] = {} # Map (u_idx, v_idx) -> index in edge_attr tensor

        edge_counter = 0
        for i, u_node_id in enumerate(self.nodes):
            # Check if node exists (robustness for potential graph issues)
            if u_node_id not in self.graph: continue

            for v_node_id in self.graph.neighbors(u_node_id):
                if v_node_id not in self.node_to_idx: continue # Skip neighbors not in main node list
                j = self.node_to_idx[v_node_id]

                # Get edge data - Handle MultiDiGraph: find edge with minimum length
                possible_edges = self.graph.get_edge_data(u_node_id, v_node_id)
                if not possible_edges: continue # Skip if edge somehow doesn't exist
                edge_data = min(possible_edges.values(), key=lambda x: x.get('length', float('inf')))

                # Define edge features RAW: [length_m, slope_deg, base_congestion, road_quality]
                # Ensure these keys exist from network_gen_wrapper
                features_raw = [
                    float(edge_data.get('length', 100.0)),
                    float(edge_data.get('slope_deg', 0.0)),
                    float(edge_data.get('base_congestion', 1.0)),
                    float(edge_data.get('road_quality', 0.8))
                ]
                edge_list_indices.append([i, j])
                edge_attributes_raw.append(features_raw)
                self.edge_idx_map[(i,j)] = edge_counter
                edge_counter += 1

        if not edge_list_indices:
             # Handle empty graph case
             self.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
             self.edge_attr_normalized = torch.empty((0, 4), dtype=torch.float, device=device)
             self.edge_feature_dim = 4
             self.edge_attr_mean = torch.zeros(4, device=device)
             self.edge_attr_std = torch.ones(4, device=device)
        else:
             self.edge_index = torch.tensor(edge_list_indices, dtype=torch.long, device=device).t().contiguous()
             edge_attr_raw_tensor = torch.tensor(edge_attributes_raw, dtype=torch.float, device=device)

             # --- Calculate Normalization Stats ---
             self.edge_attr_mean = torch.mean(edge_attr_raw_tensor, dim=0)
             self.edge_attr_std = torch.std(edge_attr_raw_tensor, dim=0)
             # Add epsilon to std to prevent division by zero for constant features
             self.edge_attr_std = torch.where(self.edge_attr_std == 0, torch.tensor(1.0, device=device), self.edge_attr_std)

             # --- Apply Normalization (Standardization) ---
             self.edge_attr_normalized = (edge_attr_raw_tensor - self.edge_attr_mean) / self.edge_attr_std
             self.edge_feature_dim = self.edge_attr_normalized.shape[1]

             print("Edge Attribute Normalization Stats:")
             print(f"  Mean: {self.edge_attr_mean.cpu().numpy()}")
             print(f"  Std Dev: {self.edge_attr_std.cpu().numpy()}")


        # Define node features: [is_current, is_target, current_soc]
        self.node_feature_dim = 3
        print(f"Parser ready: {self.num_nodes} nodes, {self.edge_index.shape[1]} edges.")
        print(f"Node feature dim: {self.node_feature_dim}, Edge feature dim: {self.edge_feature_dim}")


    def parse_obs(self, state: Tuple[int, int, float]) -> Optional[Data]:
        """ Converts state tuple to PyG Data object. Returns None if state is invalid."""
        current_idx, target_idx, current_soc = state

        # --- Validate indices ---
        if not (0 <= current_idx < self.num_nodes and 0 <= target_idx < self.num_nodes):
             print(f"Error in parse_obs: Invalid indices! current={current_idx}, target={target_idx}, num_nodes={self.num_nodes}")
             return None # Return None to indicate invalid state


        x = torch.zeros((self.num_nodes, self.node_feature_dim), dtype=torch.float, device=self.device)
        x[current_idx, 0] = 1.0
        x[target_idx, 1] = 1.0
        x[:, 2] = current_soc # Broadcast SoC to all nodes (SoC is already 0-1)

        # Create PyG Data object - Use precomputed edge_index and NORMALIZED edge_attr
        data = Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr_normalized)
        # Store edge map for the agent/model to use - IMPORTANT
        data.edge_idx_map = self.edge_idx_map
        return data
