"""
Professional Streamlit Dashboard for Large-Scale EV Routing
Interactive web interface for EV route optimization with real-time visualization

Uses Real Street Routing: Dijkstra shortest paths on actual road network
via OSRM API, then DRL model for optimal tour selection.
"""
import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import time
import torch
from datetime import datetime
import requests
import polyline

# Import the new real streets pipeline
from real_streets_pipeline import RealStreetsPipeline, NYC_CHARGING_STATIONS, load_customers_from_csv


# Page configuration
st.set_page_config(
    page_title="EV Route Optimizer",
    page_icon="🚗⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #1a1a2e !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        border: 1px solid #2d2d44;
    }
    .stMetric > div {
        background-color: transparent !important;
    }
    .stMetric label {
        color: #4ECDC4 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #98D8C8 !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 25px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    h1 {
        color: #1f77b4;
        font-weight: bold;
    }
    h2 {
        color: #2c3e50;
    }
    .info-box {
        background-color: #e3f2fd !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
        color: #1a1a2e;
    }
    .info-box h3, .info-box h4 {
        color: #0d47a1;
    }
    .info-box p, .info-box li {
        color: #1a1a2e;
    }
    .success-box {
        background-color: #e8f5e9 !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
        color: #1a1a2e;
    }
    .warning-box {
        background-color: #fff3e0 !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 10px 0;
        color: #1a1a2e;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'customer_coords' not in st.session_state:
    st.session_state.customer_coords = None
if 'depot_coords' not in st.session_state:
    st.session_state.depot_coords = None


@st.cache_resource
def initialize_pipeline(depot_coords):
    """Initialize the Real Streets Pipeline with 50 NYC charging stations"""
    if st.session_state.pipeline is None:
        with st.spinner("🔌 Initializing pipeline with 50 NYC charging stations..."):
            st.session_state.pipeline = RealStreetsPipeline(
                station_coords=NYC_CHARGING_STATIONS,
                depot_coords=depot_coords
            )
        st.success("✅ Real Streets Pipeline initialized!")


def _hash_results(results):
    """Create a simple hash key from results for caching"""
    return f"{results.get('n_groups', 0)}_{results.get('total_cost', 0):.4f}_{results.get('total_distance_km', 0):.4f}"


@st.cache_data(ttl=300, show_spinner=False, hash_funcs={dict: _hash_results})
def create_map_visualization(results, show_groups=True):
    """
    Create an interactive map visualization using Plotly with real street routing
    
    The routes follow actual streets as computed by the pipeline.
    
    Args:
        results: Results dictionary from inference pipeline
        show_groups: Whether to color-code by groups
    """
    fig = go.Figure()
    
    # Get depot coordinates
    depot_lat, depot_lon = results['depot_coords']
    
    # Color scheme for groups - using bold, distinct colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788']
    
    # Plot depot
    fig.add_trace(go.Scattermapbox(
        lat=[depot_lat],
        lon=[depot_lon],
        mode='markers',
        marker=dict(size=20, color='#DC143C', symbol='square'),
        text=['Depot (Start/End)'],
        name='Depot',
        hoverinfo='text',
        showlegend=True
    ))
    
    # Plot each group's solution with real street paths
    for group_result in results['group_results']:
        group_id = group_result['group_id']
        group_info = group_result['group_info']
        color = colors[group_id % len(colors)]
        
        # Get coordinates
        customer_coords = np.array(group_info['customer_coords'])
        station_coords = np.array(group_info['station_coords'])
        
        # Get real street paths (already computed by pipeline)
        real_paths = group_result.get('real_paths', [])
        
        # Draw real street routes from pre-computed paths
        for path_idx, path_coords in enumerate(real_paths):
            if len(path_coords) > 1:
                path_lats = [c[0] for c in path_coords]
                path_lons = [c[1] for c in path_coords]
                
                fig.add_trace(go.Scattermapbox(
                    lat=path_lats,
                    lon=path_lons,
                    mode='lines',
                    line=dict(width=3, color=color),
                    name=f'Group {group_id+1} Route' if path_idx == 0 else None,
                    hoverinfo='skip',
                    showlegend=(path_idx == 0)
                ))
        
        # Plot customers
        fig.add_trace(go.Scattermapbox(
            lat=customer_coords[:, 0],
            lon=customer_coords[:, 1],
            mode='markers',
            marker=dict(size=10, color=color, opacity=0.9),
            text=[f'Customer {i+1} (Group {group_id+1})' for i in range(len(customer_coords))],
            name=f'Group {group_id+1} Customers',
            hoverinfo='text',
            showlegend=False
        ))
        
        # Plot assigned stations
        fig.add_trace(go.Scattermapbox(
            lat=station_coords[:, 0],
            lon=station_coords[:, 1],
            mode='markers',
            marker=dict(size=14, color=color, symbol='triangle', opacity=0.9),
            text=[f'Station {i+1} (Group {group_id+1})' for i in range(len(station_coords))],
            name=f'Group {group_id+1} Stations',
            hoverinfo='text',
            showlegend=False
        ))
    
    # Update map layout
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=depot_lat, lon=depot_lon),
            zoom=11
        ),
        height=700,
        title={
            'text': f'EV Route Optimization - {results["n_groups"]} Groups × 10 Customers (Real Street Routing)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f77b4', 'family': 'Arial Black'}
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


@st.cache_data(ttl=300, show_spinner=False, hash_funcs={dict: _hash_results})
def create_cost_chart(results):
    """Create a bar chart showing cost per group"""
    group_data = []
    for r in results['group_results']:
        group_data.append({
            'Group': f"Group {r['group_id'] + 1}",
            'Cost': r['route_cost'],
            'Customers': r['n_customers'],
            'Stations': r['n_stations']
        })
    
    df = pd.DataFrame(group_data)
    
    fig = px.bar(df, x='Group', y='Cost', 
                 color='Customers',
                 hover_data=['Customers', 'Stations'],
                 labels={'Cost': 'Energy Cost', 'Customers': 'Number of Customers'},
                 color_continuous_scale='Turbo')
    
    fig.update_layout(
        height=400,
        title={
            'text': 'Energy Cost per Customer Group',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#4ECDC4', 'family': 'Arial Black'}
        },
        xaxis_title='Customer Group',
        yaxis_title='Energy Cost',
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#ffffff', size=12),
        xaxis=dict(showgrid=True, gridcolor='#2d2d44', title_font=dict(color='#4ECDC4')),
        yaxis=dict(showgrid=True, gridcolor='#2d2d44', title_font=dict(color='#4ECDC4'))
    )
    
    fig.update_traces(textposition='outside', textfont=dict(color='#ffffff', size=11))
    
    return fig


@st.cache_data(ttl=300, show_spinner=False, hash_funcs={dict: _hash_results})
def create_performance_metrics(results):
    """Create performance metrics visualization"""
    group_data = []
    for r in results['group_results']:
        group_data.append({
            'Group': f"Group {r['group_id'] + 1}",
            'Inference Time (s)': r['inference_time'],
            'Customers': r['n_customers']
        })
    
    df = pd.DataFrame(group_data)
    
    fig = px.bar(df, x='Group', y='Inference Time (s)',
                 color='Customers',
                 color_continuous_scale='Tealgrn')
    
    fig.update_layout(
        height=400,
        title={
            'text': 'Inference Time per Group',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#4ECDC4', 'family': 'Arial Black'}
        },
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#ffffff', size=12),
        xaxis=dict(showgrid=True, gridcolor='#2d2d44', title='Customer Group', title_font=dict(color='#4ECDC4')),
        yaxis=dict(showgrid=True, gridcolor='#2d2d44', title='Time (seconds)', title_font=dict(color='#4ECDC4'))
    )
    
    fig.update_traces(textposition='outside', textfont=dict(color='#ffffff', size=11))
    
    return fig


@st.cache_data(ttl=600, show_spinner=False)
def create_station_coverage_map():
    """Create a map showing all 50 fixed charging stations across NYC"""
    fig = go.Figure()
    
    # Use the NYC_CHARGING_STATIONS from real_streets_pipeline
    stations = NYC_CHARGING_STATIONS
    
    fig.add_trace(go.Scattermapbox(
        lat=stations[:, 0],
        lon=stations[:, 1],
        mode='markers',
        marker=dict(size=12, color='green', symbol='circle'),
        text=[f'Station {i+1}' for i in range(len(stations))],
        name='Charging Stations',
        hoverinfo='text'
    ))
    
    center_lat = float(np.mean(stations[:, 0]))
    center_lon = float(np.mean(stations[:, 1]))
    
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
        height=600,
        title={
            'text': 'NYC Fixed Charging Station Network (50 Stations)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2c3e50'}
        },
        showlegend=True,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown("<h1 style='text-align: center;'>🚗⚡ EV Route Optimization Dashboard</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>"
                "Deep Reinforcement Learning with Real Street Routing (OSRM + Dijkstra)</p>", 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar for inputs
    with st.sidebar:
        st.image("dashboard_image.png", 
                use_container_width=True)
        
        st.markdown("## 📍 Configuration")
        
        # Depot configuration
        st.markdown("### Depot Location")
        col1, col2 = st.columns(2)
        with col1:
            depot_lat = st.number_input("Latitude", value=40.7589, format="%.6f", 
                                       help="Depot latitude coordinate")
        with col2:
            depot_lon = st.number_input("Longitude", value=-73.9851, format="%.6f",
                                       help="Depot longitude coordinate")
        
        st.session_state.depot_coords = (depot_lat, depot_lon)
        
        # Customer configuration
        st.markdown("### Customer Configuration")
        st.info("⚠️ Customers must be in multiples of 10 (each group = 10 customers + 4 stations)")
        n_customers = st.slider("Number of Customers", min_value=10, max_value=100, 
                               value=30, step=10,
                               help="Total customers (must be multiple of 10)")
        
        # Option to upload or generate customers
        customer_input_method = st.radio("Customer Input Method", 
                                        ["Generate Random", "Upload CSV"],
                                        help="Choose how to specify customer locations")
        
        if customer_input_method == "Generate Random":
            spread = st.slider("Geographic Spread (km)", min_value=1, max_value=10, 
                             value=5, help="How spread out are customers from depot")
            
            if st.button("🎲 Generate Customers", use_container_width=True):
                np.random.seed(42)
                # Convert km spread to approximate lat/lon degrees
                spread_deg = spread / 111.0  # 1 degree ≈ 111 km
                depot = st.session_state.depot_coords
                base = np.array([[depot[0], depot[1]]])
                st.session_state.customer_coords = np.random.randn(n_customers, 2) * spread_deg + base
                st.success(f"✅ Generated {n_customers} customer locations!")
        else:
            uploaded_file = st.file_uploader("Upload Customer CSV", type=['csv'],
                                           help="CSV with columns: latitude, longitude")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                if 'latitude' in df.columns and 'longitude' in df.columns:
                    st.session_state.customer_coords = df[['latitude', 'longitude']].values
                    st.success(f"✅ Loaded {len(st.session_state.customer_coords)} customers!")
                else:
                    st.error("❌ CSV must have 'latitude' and 'longitude' columns")
        
        st.markdown("---")
        
        # Run optimization button
        run_button = st.button("🚀 Run Optimization (Real Streets)", use_container_width=True, 
                              type="primary")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Solution Map", "📊 Analytics", 
                                       "🔌 Station Network", "📄 Detailed Results"])
    
    # Run optimization
    if run_button:
        if st.session_state.customer_coords is None:
            st.error("❌ Please generate or upload customer locations first!")
        elif len(st.session_state.customer_coords) < 10:
            st.error("❌ Need at least 10 customers!")
        else:
            n_usable = (len(st.session_state.customer_coords) // 10) * 10
            if n_usable < len(st.session_state.customer_coords):
                st.warning(f"⚠️ Using {n_usable} customers (groups of 10). "
                          f"{len(st.session_state.customer_coords) - n_usable} customers will be excluded.")
            
            with st.spinner("⚡ Running real-street optimization with Dijkstra + DRL..."):
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Initializing Real Streets Pipeline...")
                    progress_bar.progress(10)
                    
                    # Initialize pipeline with depot and stations
                    pipeline = RealStreetsPipeline(
                        station_coords=NYC_CHARGING_STATIONS,
                        depot_coords=st.session_state.depot_coords
                    )
                    st.session_state.pipeline = pipeline
                    
                    progress_bar.progress(20)
                    status_text.text("Grouping customers (10 per group) and finding nearest stations...")
                    
                    # Define progress callback
                    def update_progress(msg):
                        status_text.text(msg)
                    
                    progress_bar.progress(30)
                    
                    # Run complete pipeline
                    results = pipeline.run_pipeline(
                        customer_coords=st.session_state.customer_coords,
                        progress_callback=update_progress
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Optimization complete with real street routing!")
                    
                    st.session_state.results = results
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"🎉 Successfully optimized routes for {results['n_customers_total']} customers "
                             f"in {results['n_groups']} groups! Total distance: {results['total_distance_km']:.2f} km")
                    
                except Exception as e:
                    st.error(f"❌ Error during optimization: {str(e)}")
                    st.exception(e)
    
    # Display results
    if st.session_state.results is not None:
        results = st.session_state.results
        
        # Key metrics at the top
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Customers", results.get('n_customers_total', 0), 
                     help="Number of customer locations serviced (10 per group)",)
        with col2:
            st.metric("Customer Groups", results.get('n_groups', 0),
                     help="Number of groups created (exactly 10 customers each)")
        with col3:
            st.metric("Total Energy Cost", f"{results.get('total_cost', 0):.2f}",
                     help="Cumulative energy consumption across all routes")
        with col4:
            st.metric("Total Distance", f"{results.get('total_distance_km', 0):.2f} km",
                     help="Total real street distance across all routes")
        with col5:
            st.metric("Total Time", f"{results.get('total_time', 0):.2f}s",
                     help="Total computation time for optimization")
        
        # Tab 1: Solution Map
        with tab1:
            st.markdown("### 🗺️ Optimized Route Visualization (Real Streets)")
            st.markdown('<div class="info-box">Routes follow actual streets via OSRM routing API. '
                       'Each color represents a different customer group with its 4 assigned charging stations. '
                       'The DRL model optimizes the visit sequence, then routes are projected onto real roads.</div>',
                       unsafe_allow_html=True)
            
            try:
                fig_map = create_map_visualization(results)
                st.plotly_chart(fig_map, use_container_width=True, key="solution_map")
            except Exception as e:
                st.error(f"Error creating map: {str(e)}")
            
            # Group summary table
            st.markdown("### 📋 Group Summary")
            group_summary = []
            for r in results['group_results']:
                group_summary.append({
                    'Group': r['group_id'] + 1,
                    'Customers': r['n_customers'],
                    'Stations': r['n_stations'],
                    'Energy Cost': f"{r['route_cost']:.2f}",
                    'Distance (km)': f"{r.get('total_distance_m', 0)/1000:.2f}",
                    'Time (s)': f"{r['inference_time']:.3f}"
                })
            
            df_summary = pd.DataFrame(group_summary)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        # Tab 2: Analytics
        with tab2:
            st.markdown("### 📊 Performance Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_cost_chart(results), use_container_width=True, key="cost_chart")
            
            with col2:
                st.plotly_chart(create_performance_metrics(results), use_container_width=True, key="perf_chart")
            
            # Statistics
            st.markdown("### 📈 Statistical Summary")
            costs = [r['route_cost'] for r in results['group_results']]
            times = [r['inference_time'] for r in results['group_results']]
            distances = [r.get('total_distance_m', 0)/1000 for r in results['group_results']]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Cost", f"{min(costs):.2f}")
                st.metric("Max Cost", f"{max(costs):.2f}")
            with col2:
                st.metric("Avg Distance", f"{np.mean(distances):.2f} km")
                st.metric("Total Distance", f"{sum(distances):.2f} km")
            with col3:
                st.metric("Min Time", f"{min(times):.3f}s")
                st.metric("Max Time", f"{max(times):.3f}s")
        
        # Tab 3: Station Network
        with tab3:
            st.markdown("### 🔌 Fixed Charging Station Network (50 NYC Stations)")
            st.markdown('<div class="info-box">NYC area has 50 permanently fixed charging stations. '
                       'The algorithm assigns the nearest 4 stations to each customer group based on centroid proximity.</div>',
                       unsafe_allow_html=True)
            
            fig_stations = create_station_coverage_map()
            if fig_stations:
                st.plotly_chart(fig_stations, use_container_width=True, key="station_map_tab3")
            
            # Station statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Stations", 50)
            with col2:
                st.metric("Stations per Group", 4)
            with col3:
                st.metric("Customers per Group", 10)
        
        # Tab 4: Detailed Results
        with tab4:
            st.markdown("### 📄 Detailed Results")
            
            # Convert numpy arrays for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(i) for i in obj]
                return obj
            
            # Download button for results
            results_json = json.dumps(convert_for_json(results), indent=2, default=str)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=results_json,
                file_name=f"ev_routing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Detailed group information
            for group_result in results['group_results']:
                with st.expander(f"Group {group_result['group_id'] + 1} Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Customers:** {group_result['n_customers']}")
                        st.write(f"**Stations:** {group_result['n_stations']}")
                        st.write(f"**Route Cost:** {group_result['route_cost']:.2f}")
                        st.write(f"**Distance:** {group_result.get('total_distance_m', 0)/1000:.2f} km")
                    
                    with col2:
                        st.write(f"**Inference Time:** {group_result['inference_time']:.3f}s")
                        st.write(f"**Tour Length:** {len(group_result['tour_indices'])} stops")
                    
                    # Show tour sequence
                    st.write(f"**Tour Indices:** {group_result['tour_indices']}")
    
    else:
        # Initial state - show instructions
        st.markdown('<div class="info-box">'
                   '<h3>👋 Welcome to the EV Route Optimizer!</h3>'
                   '<p>This dashboard helps you optimize electric vehicle routes for large-scale delivery operations.</p>'
                   '<h4>How to use:</h4>'
                   '<ol>'
                   '<li>Set your depot location in the sidebar</li>'
                   '<li>Generate or upload customer locations (must be multiples of 10)</li>'
                   '<li>Click "Run Optimization" to compute optimal routes</li>'
                   '</ol>'
                   '<p><strong>Features:</strong></p>'
                   '<ul>'
                   '<li>✅ Groups customers into exactly 10 per group (K-means clustering)</li>'
                   '<li>✅ Uses 50 fixed charging stations across NYC</li>'
                   '<li>✅ Assigns nearest 4 stations to each group centroid</li>'
                   '<li>✅ Builds distance matrix via real roads (OSRM Dijkstra)</li>'
                   '<li>✅ Deep RL model predicts optimal visit sequence</li>'
                   '<li>✅ Routes projected onto actual streets</li>'
                   '</ul>'
                   '</div>', unsafe_allow_html=True)
        
        # Show station map
        st.markdown("### 🔌 Available Charging Station Network")
        fig_stations = create_station_coverage_map()
        if fig_stations:
            st.plotly_chart(fig_stations, use_container_width=True, key="station_map_welcome")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; padding: 20px;'>"
        "EV Route Optimizer v2.0 | Real Street Routing with Deep Reinforcement Learning | © 2026"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()