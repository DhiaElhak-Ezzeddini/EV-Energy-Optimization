# network_gen_wrapper.py
import networkx as nx
import os
import sys

sys.path.append("C:\\Users\\YODA\\Documents\\Dhia_Salem_Tutoré_AIM_25-26\\New_repo_clone\\EV-Energy-Optimization\\Graph_Generation")
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

DB_PATH = "C:\\Users\\YODA\\Documents\\Dhia_Salem_Tutoré_AIM_25-26\\New_repo_clone\\EV-Energy-Optimization\\Graph_Generation\\src\\New_York_network_enhanced_attributes.pkl.gz"



def load_or_generate_graph(db_path: str = DB_PATH) -> nx.MultiDiGraph:
    """
    Loads the New York road network from the database if it exists,
    otherwise generates, saves, and returns it. Includes energy attributes.
    """
    # Placeholder for info logger if not imported successfully
    def info(msg, *args): 
        print(f"[INFO] {msg}")

    info(f"Database path: {db_path}", 'wrapper')
    
    try:     
        db = NetworkDatabase(db_path=db_path) # type: ignore
        
        network = db.load_or_create_network() # type: ignore

        # Ensure energy attributes are present (they should be added during creation)
        sample_edge_data = next(iter(network.edges(data=True)))[2]
        required_attrs = ['slope_deg', 'congestion_factor', 'length', 'road_quality']
        if not all(attr in sample_edge_data for attr in required_attrs):
            info("Augmenting existing graph with energy attributes (might be redundant)...", 'wrapper')
            # Re-apply augmentation if somehow missing (seed ensures consistency if re-run)
            network = db._augment_edges_with_energy_attributes(network, seed=12345) # type: ignore
        
        info(f"Graph ready: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges.", 'wrapper')
        return network
        
    except ImportError as e:
        print(f"CRITICAL: Failed to initialize NetworkDatabase. Please check path/imports. Error: {e}")
        # In a real environment, you might stop here or return a mock graph
        raise e
        
    except Exception as e:
        print(f"An unexpected error occurred during graph loading: {e}")
        raise e
