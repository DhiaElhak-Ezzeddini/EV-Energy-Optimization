# graph_parser.py
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import Tuple

class GraphParser:
    """
    Parses the environment state into a PyTorch Geometric Data object
    for the GNN Q-Network.
    """
    def __init__(self, graph: nx.MultiDiGraph, device: torch.device):
        self.graph = graph
        self.nodes = list(graph.nodes) # Node IDs
        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.nodes)}
        self.device = device
        self.num_nodes = len(self.nodes)

        # --- Precompute edge_index and edge_attr ---
        edge_list_indices = []
        edge_attributes = []
        self.edge_idx_map = {} # Map (u_idx, v_idx) -> index in edge_attr tensor

        print("Building edge index and attributes for parser...")
        edge_counter = 0
        for i, u_node_id in enumerate(self.nodes):
            for v_node_id in self.graph.neighbors(u_node_id):
                if v_node_id not in self.node_to_idx: continue # Skip neighbors not in main node list
                j = self.node_to_idx[v_node_id]

                # Take the first edge's data if multiple edges exist (or average/min length etc.)
                edge_data = min(self.graph.get_edge_data(u_node_id, v_node_id).values(),
                                key=lambda x: x.get('length', float('inf')))

                # Define edge features: [length, slope_deg, base_congestion, road_quality] - Normalize!
                # Ensure these keys exist from network_gen_wrapper
                features = [
                    edge_data.get('length', 100.0) / 1000.0, # Length in km
                    edge_data.get('slope_deg', 0.0) / 10.0,    # Normalize slope (e.g., divide by typical max)
                    edge_data.get('base_congestion', 1.0) - 1.0, # Normalize congestion (0 is free flow)
                    edge_data.get('road_quality', 0.8)
                ]
                edge_list_indices.append([i, j])
                edge_attributes.append(features)
                self.edge_idx_map[(i,j)] = edge_counter
                edge_counter += 1

        if not edge_list_indices:
             # Handle empty graph case
             self.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
             self.edge_attr = torch.empty((0, 4), dtype=torch.float, device=device) # Adjust feature count if needed
             self.edge_feature_dim = 4
        else:
             self.edge_index = torch.tensor(edge_list_indices, dtype=torch.long, device=device).t().contiguous()
             self.edge_attr = torch.tensor(edge_attributes, dtype=torch.float, device=device)
             self.edge_feature_dim = self.edge_attr.shape[1]

        # Define node features: [is_current, is_target, current_soc]
        self.node_feature_dim = 3
        print(f"Parser ready: {self.num_nodes} nodes, {self.edge_index.shape[1]} edges.")
        print(f"Node feature dim: {self.node_feature_dim}, Edge feature dim: {self.edge_feature_dim}")


    def parse_obs(self, state: Tuple[int, int, float]) -> Data:
        """ Converts state tuple to PyG Data object. """
        current_idx, target_idx, current_soc = state

        x = torch.zeros((self.num_nodes, self.node_feature_dim), dtype=torch.float, device=self.device)
        if 0 <= current_idx < self.num_nodes:
             x[current_idx, 0] = 1.0
        if 0 <= target_idx < self.num_nodes:
             x[target_idx, 1] = 1.0
        x[:, 2] = current_soc # Broadcast SoC to all nodes

        # Create PyG Data object - Use precomputed edge_index and edge_attr
        data = Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)
        # Store edge map for the agent/model to use
        data.edge_idx_map = self.edge_idx_map
        return data