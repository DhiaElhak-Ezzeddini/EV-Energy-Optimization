# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool # Or ChebConv, GATConv etc.
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Union, Optional

class GNN_QNetwork(nn.Module):
    """
    Graph Neural Network to estimate Q-values for EV routing.
    Outputs Q-values for *reachable neighbor* nodes.
    Uses precomputed edge_idx_map for efficiency.
    """
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Graph Convolution Layers
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv3 = GCNConv(hidden_dim, hidden_dim) # Optional third layer

        # MLP for Q-value estimation per reachable neighbor
        # Input features: embedding(current_node), embedding(neighbor_node), edge_features(current->neighbor)
        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim * 2),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dim * 2), # Optional normalization
            nn.Linear(hidden_dim * 2, hidden_dim), # Optional intermediate layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,
                data: Union[Data, Batch],
                current_node_indices: List[int],
                reachable_neighbor_indices: List[List[int]],
                edge_idx_map: Dict[Tuple[int, int], int] # Pass the map here
                ) -> Tuple[List[Optional[torch.Tensor]], List[List[int]]]:
        """
        Processes a batch of graph states using precomputed edge map.

        Args:
            data: A PyG Data or Batch object.
            current_node_indices: List of current node indices (global within the batch).
            reachable_neighbor_indices: List of lists of reachable neighbor indices (global).
            edge_idx_map: Dictionary mapping (source_idx, target_idx) -> index in original edge_attr.

        Returns:
            Tuple: (q_values_list, actual_indices_list)
                   q_values_list contains tensors of Q-values for reachable neighbors for each batch item, or None if error.
                   actual_indices_list confirms the neighbor indices corresponding to the Q-values.
        """
        x, edge_index, edge_attr, batch_map = data.x, data.edge_index, data.edge_attr, data.batch
        is_batch = isinstance(data, Batch)

        # Apply GNN layers
        # GCNConv doesn't directly use edge attributes in its message passing formula,
        # but edge_attr is needed later for the Q-value MLP.
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        # h = F.relu(self.conv3(h, edge_index)) # Optional

        q_values_list: List[Optional[torch.Tensor]] = []
        actual_indices_list: List[List[int]] = []

        num_graphs = data.num_graphs if is_batch else 1

        # Process each graph in the batch
        for i in range(num_graphs):
            # Determine the slice of nodes belonging to this graph if batching
            if is_batch:
                node_slice = torch.where(batch_map == i)[0]
                if node_slice.numel() == 0: # Should not happen
                    q_values_list.append(None)
                    actual_indices_list.append([])
                    continue
                # Important: current_node_indices contains GLOBAL indices within the flattened batch
                current_node_idx_global = current_node_indices[i]
                neighbors_global = reachable_neighbor_indices[i]
            else: # Single Data object
                current_node_idx_global = current_node_indices[0] # Index within this single graph
                neighbors_global = reachable_neighbor_indices[0]

            # --- Handle case with no reachable neighbors ---
            if not neighbors_global:
                q_values_list.append(torch.tensor([], device=h.device)) # Empty tensor
                actual_indices_list.append([])
                continue

            # Gather features for MLP input
            try:
                h_current = h[current_node_idx_global].expand(len(neighbors_global), -1) # Repeat current embedding
                h_neighbors = h[neighbors_global] # Gather neighbor embeddings (these are global indices)

                # --- Efficiently Gather edge attributes using the map ---
                edge_attrs_list = []
                valid_neighbor_list_for_mlp = [] # Track neighbors for which we found edges

                # --- Determine Node Index Offset for Edge Map Lookup ---
                # The edge_idx_map uses original 0..N-1 indices.
                # If batching, we need the original index for the current node.
                if is_batch:
                    # Find the graph-local index of the current node
                    current_node_idx_local = (current_node_idx_global - node_slice[0]).item()
                    # Map global neighbor indices back to local indices relative to the original graph
                    neighbor_nodes_local_map = {global_idx: (global_idx - node_slice[0]).item() for global_idx in neighbors_global}
                else: # No batching, indices are already local
                    current_node_idx_local = current_node_idx_global
                    neighbor_nodes_local_map = {global_idx: global_idx for global_idx in neighbors_global}


                for neighbor_idx_global in neighbors_global:
                    neighbor_idx_local = neighbor_nodes_local_map[neighbor_idx_global]
                    edge_tuple_local = (current_node_idx_local, neighbor_idx_local)

                    # --- Lookup in edge_idx_map using LOCAL indices ---
                    original_edge_attr_index = edge_idx_map.get(edge_tuple_local)

                    if original_edge_attr_index is not None:
                        # Find where this edge appears in the BATCH edge_index/edge_attr
                        # This still requires mapping the original_edge_attr_index to the batch index,
                        # OR searching the batch edge_index for the GLOBAL pair.
                        # Searching is simpler to implement here, though less ideal.
                        mask = (edge_index[0] == current_node_idx_global) & (edge_index[1] == neighbor_idx_global)
                        batch_edge_attr_indices = mask.nonzero(as_tuple=True)[0]

                        if len(batch_edge_attr_indices) > 0:
                            edge_attr_for_pair = edge_attr[batch_edge_attr_indices[0]] # Take first found in batch
                            edge_attrs_list.append(edge_attr_for_pair)
                            valid_neighbor_list_for_mlp.append(neighbor_idx_global) # Store global index
                        # else: Edge exists in map but not found in batch edge_index? Should not happen.
                    # else: Edge tuple not in precomputed map? Graph mismatch?


                # --- Check if any valid edges were found ---
                if not edge_attrs_list:
                    print(f"Warning: No valid edge attributes found for neighbors of node {current_node_idx_global}")
                    q_values_list.append(torch.tensor([], device=h.device))
                    actual_indices_list.append([])
                    continue

                edge_attrs_tensor = torch.stack(edge_attrs_list)

                # Adjust h_current and h_neighbors if some neighbors were skipped
                if len(valid_neighbor_list_for_mlp) != len(neighbors_global):
                    h_neighbors = h[valid_neighbor_list_for_mlp] # Use filtered global indices
                    h_current = h[current_node_idx_global].expand(len(valid_neighbor_list_for_mlp), -1)

                # Concatenate features: [h_current, h_neighbor, edge_attr]
                mlp_input = torch.cat([h_current, h_neighbors, edge_attrs_tensor], dim=-1)

                # Calculate Q-values for the valid reachable neighbors
                q_values_for_neighbors = self.q_mlp(mlp_input).squeeze(-1) # Shape: [num_valid_neighbors]

                q_values_list.append(q_values_for_neighbors)
                actual_indices_list.append(valid_neighbor_list_for_mlp) # Store global indices corresponding to calculated Qs

            except IndexError as e:
                 print(f"Error processing graph {i} in batch: IndexError {e}")
                 print(f"  current_node_idx_global: {current_node_idx_global}")
                 print(f"  neighbors_global: {neighbors_global}")
                 print(f"  h shape: {h.shape}")
                 q_values_list.append(None) # Indicate error
                 actual_indices_list.append([])
            except Exception as e: # Catch other potential errors
                 print(f"Error processing graph {i} in batch: {type(e).__name__} {e}")
                 q_values_list.append(None)
                 actual_indices_list.append([])


        return q_values_list, actual_indices_list
