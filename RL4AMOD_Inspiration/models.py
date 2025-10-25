# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool # Or ChebConv, GATConv etc.
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Union

class GNN_QNetwork(nn.Module):
    """
    Graph Neural Network to estimate Q-values for EV routing.
    Outputs Q-values for *reachable neighbor* nodes.
    """
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Graph Convolution Layers
        # Consider making edge features optional or handled differently if GCNConv doesn't use them well
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, hidden_dim) # Optional third layer

        # MLP for Q-value estimation per reachable neighbor
        # Input features: embedding(current_node), embedding(neighbor_node), edge_features(current->neighbor)
        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim * 2),
            nn.ReLU(),
            # nn.Linear(hidden_dim * 2, hidden_dim), # Optional intermediate layer
            # nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, data: Union[Data , Batch], current_node_indices: List[int], reachable_neighbor_indices: List[List[int]]) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Processes a batch of graph states.

        Args:
            data: A PyG Data or Batch object containing graph structures.
            current_node_indices: List of current node indices for each graph in the batch.
            reachable_neighbor_indices: List of lists. Outer list corresponds to batch items.
                                        Inner list contains indices of *reachable* neighbors for that graph.

        Returns:
            A tuple containing:
            - q_values_list: A list where each element is a tensor of Q-values for the
                             reachable neighbors of the corresponding graph in the batch.
            - actual_indices_list: A list of lists, mirroring reachable_neighbor_indices,
                                   confirming the order of the returned Q-values.
        """
        x, edge_index, edge_attr, batch_map = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply GNN layers
        h = F.relu(self.conv1(x, edge_index)) # Add edge_attr if layer supports it
        h = F.relu(self.conv2(h, edge_index))
        # h = F.relu(self.conv3(h, edge_index)) # Optional

        q_values_list = []
        actual_indices_list = []

        # Process each graph in the batch
        for i in range(data.num_graphs):
            current_node_idx_global = current_node_indices[i] # This needs to be the index in the flattened batch 'h'
            neighbors = reachable_neighbor_indices[i] # List of reachable neighbor indices (also global)

            if not neighbors:
                q_values_list.append(torch.tensor([], device=h.device))
                actual_indices_list.append([])
                continue

            # Gather features for MLP input
            h_current = h[current_node_idx_global].expand(len(neighbors), -1) # Repeat current embedding
            h_neighbors = h[neighbors] # Gather neighbor embeddings

            # --- Gather edge attributes ---
            # This is the tricky part with batching. Need edge_index relative to the batch.
            # A precomputed map might be complex with batching.
            # Alternative: Iterate neighbors (less efficient but simpler for now)
            edge_attrs_list = []
            valid_neighbor_list_for_mlp = [] # Track neighbors for which we found edges
            for neighbor_idx_global in neighbors:
                 # Find edge index in the original *unbatched* graph's edge_idx_map
                 # Requires knowing the original indices before batching, or searching in the batch edge_index
                 # Let's assume a simplified lookup for now (inefficient search)
                 # Find row in edge_index where source is current_node_idx_global and target is neighbor_idx_global
                 mask = (edge_index[0] == current_node_idx_global) & (edge_index[1] == neighbor_idx_global)
                 edge_attr_indices = mask.nonzero(as_tuple=True)[0]

                 if len(edge_attr_indices) > 0:
                      # If multiple edges, take the first one found
                      edge_attr_for_pair = edge_attr[edge_attr_indices[0]]
                      edge_attrs_list.append(edge_attr_for_pair)
                      valid_neighbor_list_for_mlp.append(neighbor_idx_global)
                 # else: Edge not found in batch? Skip this neighbor.

            if not edge_attrs_list: # No valid edges found for neighbors?
                 q_values_list.append(torch.tensor([], device=h.device))
                 actual_indices_list.append([])
                 continue

            edge_attrs_tensor = torch.stack(edge_attrs_list)

            # Adjust h_current and h_neighbors if some neighbors were skipped
            if len(valid_neighbor_list_for_mlp) != len(neighbors):
                h_neighbors = h[valid_neighbor_list_for_mlp]
                h_current = h[current_node_idx_global].expand(len(valid_neighbor_list_for_mlp), -1)


            # Concatenate features: [h_current, h_neighbor, edge_attr]
            mlp_input = torch.cat([h_current, h_neighbors, edge_attrs_tensor], dim=-1)

            # Calculate Q-values for reachable neighbors
            q_values_for_neighbors = self.q_mlp(mlp_input).squeeze(-1) # Shape: [num_reachable_neighbors]

            q_values_list.append(q_values_for_neighbors)
            actual_indices_list.append(valid_neighbor_list_for_mlp) # Store indices corresponding to Qs

        return q_values_list, actual_indices_list