# network_gen_wrapper.py
import networkx as nx
import os
import sys

# --- Adjust this path based on your project structure ---
# Assuming network_gen.py is in Graph_Generation directory sibling to EV_RL_Routing
sys.path.append("/kaggle/input/tutor/Tutoré/EV-Energy-Optimization/Graph_Generation")
# Or adjust if network_gen is installed or located elsewhere
try:
    from Graph_Generation.src.network_gen import NetworkDatabase # type: ignore
    from Graph_Generation.config.logging_config import * # type: ignore
    from Graph_Generation.utils.logger import info # type: ignore
except ImportError as e:
    print(f"Error importing from network_gen: {e}")
    print("Please ensure network_gen.py and its dependencies are in the correct path.")
    sys.exit(1)
# --- End Path Adjustment ---

DB_PATH = "/kaggle/input/tutor/Tutoré/EV-Energy-Optimization/Graph_Generation/src/New_York_network_enhanced_attributes.pkl/New_York_network_enhanced_attributes.pkl"
def load_or_generate_graph(db_path: str = DB_PATH) -> nx.MultiDiGraph:
    """
    Loads the New York road network from the database if it exists,
    otherwise generates, saves, and returns it. Includes energy attributes.
    """
    info(f"Using network database path: {db_path}", 'wrapper')
    db = NetworkDatabase(db_path=db_path)

    # Use the enhanced load_or_create_network from your class
    network = db.load_or_create_network()

    # Ensure energy attributes are present (they should be added during creation)
    # Check a sample edge
    sample_edge_data = next(iter(network.edges(data=True)))[2]
    required_attrs = ['slope_deg', 'congestion_factor', 'length', 'travel_time']
    if not all(attr in sample_edge_data for attr in required_attrs):
        info("Augmenting existing graph with energy attributes (might be redundant)...", 'wrapper')
        # Re-apply augmentation if somehow missing (seed ensures consistency if re-run)
        network = db._augment_edges_with_energy_attributes(network, seed=12345)
        # Optionally re-save if attributes were missing
        # db.save_network(network, db.metadata)

    info(f"Graph ready: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges.", 'wrapper')
    return network

if __name__ == '__main__':
    # Example usage: Generate/Load the graph
    print("Attempting to load or generate the graph...")
    graph = load_or_generate_graph()
    print(f"Successfully obtained graph with {graph.number_of_nodes()} nodes.")
    # You can add more checks here, like printing sample node/edge data
    # print("\nSample Node Data:")
    # print(list(graph.nodes(data=True))[0])
    # print("\nSample Edge Data:")
    # print(list(graph.edges(data=True))[0])