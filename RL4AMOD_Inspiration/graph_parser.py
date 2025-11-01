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

        # --- Precompute edge_index, edge_attr, and normalization stats ---\
        print("Building edge index, attributes, and normalization stats for parser...")
        edge_list_indices: List[List[int]] = []
        edge_attributes_raw: List[List[float]] = []
        # Map (u_idx, v_idx, key) -> index in edge_attr tensor (Corrected for MultiDiGraph)
        self.edge_idx_map: Dict[Tuple[int, int, int], int] = {} 

        edge_counter = 0
        for i, u_node_id in enumerate(self.nodes):
            # NetworkX MultiDiGraph yields (u, v, key, data)
            for u_id_check, v_node_id, edge_key, data in graph.out_edges(u_node_id, keys=True, data=True):
                v_idx = self.node_to_idx.get(v_node_id)
                u_idx = i
                
                if v_idx is None:
                    print(f"Warning: Neighbor node {v_node_id} not in node list. Skipping edge ({u_node_id} -> {v_node_id}).")
                    continue
                
                # 1. Edge Index
                edge_list_indices.append([u_idx, v_idx])
                
                # 2. Raw Edge Attributes (Ensure order is consistent with network_gen)
                length = data.get('length', 1.0)
                travel_time = data.get('travel_time', 1.0)
                slope = data.get('slope_deg', 0.0)
                quality = data.get('road_quality', 0.5)
                congestion = data.get('congestion_factor', 1.0)

                edge_attributes_raw.append([
                    length,
                    travel_time,
                    slope,
                    quality,
                    congestion
                ])
                
                # 3. Edge Map
                self.edge_idx_map[(u_idx, v_idx, edge_key)] = edge_counter
                edge_counter += 1


        if not edge_list_indices:
             raise ValueError("No edges found in the graph after parsing.")

        # Convert to PyG Tensors
        self.edge_attributes_raw = torch.tensor(edge_attributes_raw, dtype=torch.float, device=self.device)
        self.edge_feature_dim = self.edge_attributes_raw.shape[1]
        self.edge_index = torch.tensor(edge_list_indices, dtype=torch.long, device=self.device).t().contiguous()

        # --- Normalization ---
        self.edge_attr_mean = self.edge_attributes_raw.mean(dim=0, keepdim=True)
        self.edge_attr_std = self.edge_attributes_raw.std(dim=0, keepdim=True)
        self.edge_attr_std[self.edge_attr_std == 0] = 1.0 

        # Normalize the attributes
        self.edge_attr = (self.edge_attributes_raw - self.edge_attr_mean) / self.edge_attr_std
        
        self.node_feature_dim = 3
        print(f"Parser ready: {self.num_nodes} nodes, {self.edge_index.shape[1]} edges.")
        print(f"Node feature dim: {self.node_feature_dim}, Edge feature dim: {self.edge_feature_dim}")


    def parse_obs(self, state: Tuple[int, int, float]) -> Optional[Data]:
        """ 
        Converts state tuple (current_node_ID, target_node_ID, current_soc) to PyG Data object. 
        Returns None if state is invalid.
        """
        current_node_id, target_node_id, current_soc = state

        # Convert Node IDs to PyG Indices (0-to-N)
        current_idx = self.node_to_idx.get(current_node_id)
        target_idx = self.node_to_idx.get(target_node_id)

        # --- Validate indices ---
        if current_idx is None or target_idx is None:
             print(f"Error in parse_obs: Invalid node ID! current_id={current_node_id}, target_id={target_node_id}. One or both IDs were not found in the graph map.")
             return None

        # Feature Tensor x
        x = torch.zeros((self.num_nodes, self.node_feature_dim), dtype=torch.float, device=self.device)
        x[current_idx, 0] = 1.0
        x[target_idx, 1] = 1.0
        x[:, 2] = current_soc # Broadcast SoC to all nodes

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_nodes,
            # CRITICAL FIX: Convert scalar indices (current_idx, target_idx) to 0D tensors
            current_node_idx=torch.tensor(current_idx, dtype=torch.long, device=self.device),
            target_node_idx=torch.tensor(target_idx, dtype=torch.long, device=self.device),
            # Pass map for Q-value lookup later
            original_node_to_idx=self.node_to_idx,
            original_idx_to_node=self.nodes,
            original_graph=self.graph
        ).to(self.device)
        
        return data

    def get_edge_attribute_normalized(self, u_idx: int, v_idx: int, edge_key: int) -> torch.Tensor:
        """
        Retrieves the normalized edge attribute vector for a specific edge key.
        This is primarily for debugging/testing.
        """
        map_key = (u_idx, v_idx, edge_key)
        
        if map_key in self.edge_idx_map:
            tensor_idx = self.edge_idx_map[map_key]
            return self.edge_attr[tensor_idx]
        else:
            raise IndexError(f"Edge ({u_idx}, {v_idx}, {edge_key}) not found in map.")