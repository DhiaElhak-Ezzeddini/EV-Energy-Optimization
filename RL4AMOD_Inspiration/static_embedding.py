# stage_1_precompute_embeddings.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import hydra
from omegaconf import DictConfig, OmegaConf
import time
import numpy as np
import os
import sys

# Assume the following utility file exists and is importable
from network_gen_wrapper import load_or_generate_graph
from graph_parser import GraphParser # Using the parser for node/edge indexing

# Define a simple GNN for embedding generation
class StructuralGNN(nn.Module):
    """
    A simple GCN to generate structural embeddings based on static node features (x, y, elev).
    This GNN is NOT trained; it's used only for a fast, single forward pass to capture 
    local neighborhood structure into a dense embedding for each node.
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim) # Final embedding layer
    
    def forward(self, x, edge_index):
        # Edge attributes are ignored as we only care about structural features (connectivity)
        # and geographical features (x, y, elev) captured in x.
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h_static = self.conv3(h, edge_index) # No final activation, linear projection
        return h_static

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("--- Stage 1: Pre-computing Static Node Embeddings ---")
    
    device = torch.device(cfg.agent.device if torch.cuda.is_available() and cfg.agent.device.lower()=='cuda' else "cpu")
    print(f"Using device: {device}")

    # --- Load Graph ---
    print("Loading graph...")
    try:
        graph = load_or_generate_graph(cfg.environment.db_path)
    except Exception as e:
        print(f"FATAL: Error loading/generating graph: {e}")
        return
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    # --- Get Graph Structure from Parser ---
    # We initialize the parser only to get its pre-computed node/edge indexing
    print("Initializing parser to get graph structure and indexing...")
    # NOTE: We pass a dummy tensor for static embeddings since the parser requires it, 
    # but the parser's logic isn't fully utilized yet in this script.
    dummy_embeddings = torch.zeros((len(graph.nodes), cfg.agent.hidden_dim))
    parser = GraphParser(graph, device) 
    edge_index = parser.edge_index.to(device)
    num_nodes = parser.num_nodes
    print("Graph structure (edge_index) obtained from parser.")
    
    # --- Create Static Node Features (Input to StructuralGNN) ---
    print("Calculating static node features (x, y, elevation)...")
    
    raw_features = []
    
    # Store min/max for normalization
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_elev, max_elev = float('inf'), float('-inf')
    
    # Iterate in the parser's sorted order to ensure consistency
    for node_id in parser.nodes: 
        node_data = graph.nodes.get(node_id, {})
        # Use .get with a default value to handle missing data gracefully
        x = node_data.get('x', 0.0) 
        y = node_data.get('y', 0.0)
        elev = node_data.get('elevation', 0.0)
        
        raw_features.append([x, y, elev])
        
        # Update bounds
        min_x = min(min_x, x); max_x = max(max_x, x)
        min_y = min(min_y, y); max_y = max(max_y, y)
        min_elev = min(min_elev, elev); max_elev = max(max_elev, elev)

    # Normalize features
    raw_features_np = np.array(raw_features, dtype=np.float32)
    
    # Apply Min-Max normalization (avoid division by zero)
    def safe_normalize(arr, min_val, max_val):
        range_val = max_val - min_val
        if range_val > 1e-6:
            return (arr - min_val) / range_val
        return arr # Return original if range is negligible

    raw_features_np[:, 0] = safe_normalize(raw_features_np[:, 0], min_x, max_x)
    raw_features_np[:, 1] = safe_normalize(raw_features_np[:, 1], min_y, max_y)
    raw_features_np[:, 2] = safe_normalize(raw_features_np[:, 2], min_elev, max_elev)


    x_static = torch.from_numpy(raw_features_np).to(device) # Shape: [num_nodes, 3]
    in_dim = 3
    
    print(f"Static node features created. Shape: {x_static.shape}")

    # --- Initialize Model ---
    hidden_dim = cfg.agent.hidden_dim
    embedding_dim = cfg.agent.hidden_dim 
    save_path = cfg.training.get("static_embedding_path", "static_node_embeddings.pt")

    model = StructuralGNN(in_dim, hidden_dim, embedding_dim).to(device)
    model.eval() # Not trained, only used for feature extraction
    
    print(f"Initialized StructuralGNN: {in_dim} -> {hidden_dim} -> {embedding_dim} output.")

    # --- Generate Embeddings ---
    print("Generating static embeddings (single forward pass)...")
    start_time = time.time()
    with torch.no_grad():
        # Edge attributes are None as the GCNConv will ignore them when not provided
        static_embeddings = model(x_static, edge_index).cpu() 
    end_time = time.time()
    
    print(f"Embeddings generated. Shape: {static_embeddings.shape}")
    print(f"Generation time: {end_time - start_time:.2f} seconds.")

    # --- Save Embeddings ---
    try:
        save_dir = os.path.dirname(save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
            
        torch.save(static_embeddings, save_path)
        print(f"Successfully saved static embeddings to: {save_path}")
    except Exception as e:
        print(f"Error saving embeddings to {save_path}: {e}")

    print("--- Stage 1 Complete ---")

if __name__ == "__main__":
    # Ensure Hydra is installed and setup correctly to run
    main()
