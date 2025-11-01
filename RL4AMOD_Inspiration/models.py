# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from typing import List, Dict, Tuple, Union, Optional
from graph_parser import GraphParser

class GNN_QNetwork(nn.Module):
    """
    Graph Neural Network to estimate Q-values for EV routing (Stage 2).
    It processes a combined feature vector (Dynamic + Static Embedding) 
    using a light GCN layer, and then uses an MLP to estimate Q-values 
    for the reachable neighbors.
    """
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # node_feature_dim = (3 dynamic features) + (hidden_dim static features)
        # Graph Convolution Layers - processes the combined features
        self.conv1 = GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim) 
        
        # MLP for Q-value estimation per reachable neighbor
        # Input features: embedding(current_node), embedding(neighbor_node), edge_features(current->neighbor)
        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1) # Output a single Q-value
        )
        self.hidden_dim = hidden_dim


    def forward(self, data: Union[Data, Batch], env_parser: GraphParser, reachable_indices_list: Optional[List[List[int]]] = None) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Performs GNN message passing and calculates Q-values for selected neighbors.

        Args:
            data: A PyG Data or Batch object containing the graph(s).
            env_parser: The GraphParser instance to access the edge index map.
            reachable_indices_list: A list of lists, where each inner list contains 
                                    the global node indices of reachable neighbors 
                                    for the current node in the corresponding graph.
                                    If None (for training/Q-value computation), it 
                                    assumes a Batch object is passed.

        Returns:
            A tuple: (List of Q-value tensors, List of corresponding action node indices)
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # --- GNN Message Passing ---
        # The input x already contains the static embedding from Stage 1.
        h = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr=edge_attr))
        # The final node embedding (h) shape: [num_nodes_in_batch, hidden_dim]

        # --- Q-Value Calculation (Per Graph/State) ---
        q_values_list: List[torch.Tensor] = []
        actual_indices_list: List[List[int]] = []
        
        # Split the batch back into individual graph representations
        # ptr is a tensor of indices that define the start of each graph in the batch
        ptr = data.ptr.cpu().tolist() if data.ptr is not None else [0, h.shape[0]] 
        
        num_graphs = len(ptr) - 1
        
        # current_node_idx is stored in the Data/Batch object
        current_node_idx_batch = data.current_node_idx.cpu().tolist()
        
        for i in range(num_graphs):
            start_idx = ptr[i]
            end_idx = ptr[i+1]
            
            # Global index of the current node in the batch's tensor
            current_node_idx_global = current_node_idx_batch[i]
            
            # Get the list of reachable neighbor global indices for this graph (state)
            # This is provided by the agent's select_action or learn loop
            if reachable_indices_list is None:
                # Should not happen in typical DQN but safety check
                raise ValueError("Reachable indices list must be provided during forward pass.")
            
            neighbors_global = reachable_indices_list[i]
            
            try:
                # Get edge attributes for only the current node's outgoing edges to neighbors
                # Iterate through the global index of the current node and its neighbors
                
                # Filter neighbors to only those present in the batch slice (optional if batching is perfect)
                valid_neighbor_list_for_mlp = [n_idx for n_idx in neighbors_global if start_idx <= n_idx < end_idx]

                if not valid_neighbor_list_for_mlp:
                    # If no valid neighbors, return empty lists for this state
                    q_values_list.append(torch.tensor([], device=x.device))
                    actual_indices_list.append([])
                    continue

                # 1. Edge Attributes (Normalized)
                # Fetch edge attributes from the parser's map using (current_idx, neighbor_idx)
                
                # The indices stored in the Data/Batch object (edge_attr) are local to the graph/batch. 
                # We need to map global indices to the local edge tensor index.
                # Since the current node idx is global, we map (global_current, global_neighbor)
                
                edge_attr_indices = []
                for neighbor_idx_global in valid_neighbor_list_for_mlp:
                    # Map global (u, v) back to the index in the original edge_attr_norm tensor
                    u_idx_local = current_node_idx_global - start_idx
                    v_idx_local = neighbor_idx_global - start_idx
                    
                    # Need to find the correct index in the batched edge_attr tensor
                    # This relies on the GNNConv operation respecting batched edge order, which is non-trivial.
                    
                    # --- SIMPLIFICATION: ---
                    # The most reliable way for DQN is to fetch the original, un-batched, normalized edge attributes 
                    # from the parser, which holds the entire graph's features in `edge_attr_norm`.
                    
                    # We use the full parser map (global index to full edge_attr index)
                    u_idx_full = current_node_idx_global # This is the global index in the full graph!
                    v_idx_full = neighbor_idx_global
                    
                    # Look up the index in the parser's full edge attribute map
                    full_edge_attr_index = env_parser.edge_idx_map.get((u_idx_full, v_idx_full))
                    if full_edge_attr_index is not None:
                        edge_attr_indices.append(full_edge_attr_index)
                    else:
                        # Should not happen if parser is correctly built
                        raise LookupError(f"Edge attribute index missing for edge ({u_idx_full}, {v_idx_full})")
                
                edge_attrs_tensor = env_parser.edge_attr_norm[edge_attr_indices] # Shape: [num_valid_neighbors, edge_feature_dim]

                # 2. Node Embeddings
                # h is the entire batch embedding. Select embeddings using global indices.
                h_neighbors = h[valid_neighbor_list_for_mlp] # Shape: [num_valid_neighbors, hidden_dim]
                # Current node embedding, expanded for concatenation
                h_current = h[current_node_idx_global].expand(len(valid_neighbor_list_for_mlp), -1) # Shape: [num_valid_neighbors, hidden_dim]

                # 3. Concatenate features: [h_current, h_neighbor, edge_attr]
                mlp_input = torch.cat([h_current, h_neighbors, edge_attrs_tensor], dim=-1)

                # 4. Calculate Q-values
                q_values_for_neighbors = self.q_mlp(mlp_input).squeeze(-1) # Shape: [num_valid_neighbors]

                q_values_list.append(q_values_for_neighbors)
                actual_indices_list.append(valid_neighbor_list_for_mlp) # Store global indices corresponding to calculated Qs

            except Exception as e: 
                 print(f"Error processing graph {i} in batch: {type(e).__name__} {e}")
                 # Append empty lists/tensors to maintain structure
                 q_values_list.append(torch.tensor([], device=x.device))
                 actual_indices_list.append([])
                 
        return q_values_list, actual_indices_list
