# 🚗⚡ EV-Energy-Optimization

## Deep Reinforcement Learning for Electric Vehicle Routing Problem (EM-EVRP)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end solution for **Energy-Minimizing Electric Vehicle Routing Problem (EM-EVRP)** using Deep Reinforcement Learning with an Attention-based model architecture. Features a professional **Streamlit dashboard** with real-time street routing via OSRM API.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training the DRL Model](#-training-the-drl-model)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Results](#-results)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project addresses the **Electric Vehicle Routing Problem (EVRP)** with the objective of minimizing total energy consumption while considering:

- ⚡ Battery State of Charge (SOC) constraints
- 🔌 Charging station visits for recharging
- 📦 Customer demand satisfaction
- ⏱️ Time window constraints
- 🛣️ Real street network routing

The solution combines:
1. **Deep Reinforcement Learning** - Attention-based encoder-decoder architecture
2. **REINFORCE Algorithm** - Policy gradient with rollout baseline
3. **Real Street Routing** - OSRM API for actual road distances
4. **Interactive Visualization** - Streamlit dashboard with Plotly maps

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **DRL Model** | Attention-based Transformer encoder with autoregressive decoder |
| 🔄 **Scalable Inference** | Groups customers into batches of 10 for large-scale problems |
| 🗺️ **Real Routing** | OSRM Table API for actual street distances (single API call) |
| 📊 **Interactive Dashboard** | Professional Streamlit UI with real-time optimization |
| 🔌 **50 NYC Stations** | Pre-configured charging station network |
| 📈 **Analytics** | Cost breakdown, performance metrics, and statistical summaries |
| 💾 **Export Results** | Download optimized routes as JSON |

---

## 📁 Project Structure

```
EV-Energy-Optimization/
│
├── EM-EVRP/                          # Main project directory
│   │
│   ├── 🎯 CORE APPLICATION
│   ├── dashboard.py                  # Streamlit web application
│   ├── real_streets_pipeline.py      # Complete inference pipeline with OSRM
│   ├── sample_customers.csv          # Sample customer locations (30 customers)
│   ├── sample_customers_2.csv        # Alternative test dataset
│   │
│   ├── 🧠 DRL MODEL TRAINING
│   ├── run.py                        # Main entry point for training/testing
│   ├── trainer.py                    # Training loop and optimization logic
│   │
│   ├── 📦 NEURAL NETWORK MODULES
│   ├── nets/
│   │   ├── DRLModel.py               # Main Attention Model architecture
│   │   ├── AM.py                     # Attention mechanism components
│   │   ├── GraphEncoder.py           # Graph neural network encoder
│   │   └── PointNetwork.py           # Pointer network for decoding
│   │
│   ├── 🔧 PROBLEM DEFINITION
│   ├── problems/
│   │   └── EVRP.py                   # EVRP environment and dataset generation
│   │
│   ├── 🛠️ UTILITIES
│   ├── utils/
│   │   ├── functions.py              # Helper functions and checkpoint loading
│   │   ├── reinforce_baselines.py    # Rollout baseline implementation
│   │   ├── beam_search.py            # Beam search decoding
│   │   ├── data_utils.py             # Data processing utilities
│   │   └── plot_delivery_graph.py    # Visualization helpers
│   │
│   ├── 💾 DATA & CHECKPOINTS
│   ├── ExperimentalData/
│   │   ├── train_data/               # Generated training instances
│   │   └── CVRPlib/                  # Benchmark instances (CVRPLIB)
│   │
│   ├── ExperimentalLog/
│   │   └── train/10/rollout/         # Trained model checkpoints
│   │       └── C10_02_42_53.467564/
│   │           └── best.pt           # Best performing model weights
│   │
│   ├── 📋 CONFIGURATION
│   ├── requirements_dashboard.txt    # Python dependencies
│   └── run_em_training.ps1           # PowerShell training script
│
└── README.md                         # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/EV-Energy-Optimization.git
cd EV-Energy-Optimization
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
cd EM-EVRP
pip install -r requirements_dashboard.txt
```

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥1.8.0 | Deep learning framework |
| `streamlit` | ≥1.28.0 | Web dashboard |
| `plotly` | ≥5.14.0 | Interactive visualizations |
| `numpy` | ≥1.19.0 | Numerical computing |
| `pandas` | ≥1.3.0 | Data manipulation |
| `scikit-learn` | ≥0.24.0 | K-means clustering |
| `requests` | ≥2.25.0 | OSRM API calls |
| `polyline` | ≥1.4.0 | Route geometry decoding |

---

## ⚡ Quick Start

### Run the Dashboard (Recommended)

```bash
cd EM-EVRP
streamlit run dashboard.py
```

Then open your browser to `http://localhost:8501`

### Quick Test via Command Line

```python
from real_streets_pipeline import RealStreetsPipeline, NYC_CHARGING_STATIONS, load_customers_from_csv

# Load customers
customers = load_customers_from_csv('sample_customers.csv')

# Initialize pipeline
pipeline = RealStreetsPipeline(
    station_coords=NYC_CHARGING_STATIONS,
    depot_coords=(40.7589, -73.9851)  # Times Square
)

# Run optimization
results = pipeline.run_pipeline(customer_coords=customers)

print(f"Total Cost: {results['total_cost']:.2f}")
print(f"Total Distance: {results['total_distance_km']:.2f} km")
```

---

## 🧠 Training the DRL Model

### Basic Training Command

```bash
cd EM-EVRP
python run.py --nodes 10 --iterations 100 --batch_size 1024
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--nodes` | 20 | Number of customers per instance |
| `--charging_num` | 5 | Number of charging stations |
| `--iterations` | 100 | Training epochs |
| `--batch_size` | 1024 | Batch size for training |
| `--Start_SOC` | 80 | Initial State of Charge (%) |
| `--t_limit` | 10 | Time limit (hours) |
| `--velocity` | 50 | Vehicle speed (km/h) |
| `--baselines` | rollout | Baseline type: 'rollout', 'critic', 'exponential' |

### Full Training Example

```bash
python run.py \
    --nodes 10 \
    --charging_num 4 \
    --iterations 100 \
    --batch_size 1024 \
    --Start_SOC 80 \
    --embedding_dim 128 \
    --hidden 128 \
    --n_encode_layers 3 \
    --actor_lr 1e-4 \
    --baselines rollout
```

### Using PowerShell Script

```powershell
.\run_em_training.ps1
```

### Training Output

Checkpoints are saved to:
```
ExperimentalLog/train/{nodes}/rollout/{timestamp}/
├── best.pt              # Best model weights
├── checkpoints/         # Periodic checkpoints
└── {epoch}/             # Per-epoch results
```

### Testing a Trained Model

```bash
python run.py --test --nodes 10 --test_size 256 --decode_strategy greedy
```

---

## 🖥️ Streamlit Dashboard

### Launch the Dashboard

```bash
cd EM-EVRP
streamlit run dashboard.py
```

### Dashboard Features

#### 1️⃣ Configuration Panel (Sidebar)
- Set depot location (latitude/longitude)
- Choose number of customers (10-100, multiples of 10)
- Generate random customers or upload CSV

#### 2️⃣ Solution Map Tab
- Interactive Plotly map with real street routes
- Color-coded customer groups
- Depot, customers, and charging stations visualization

#### 3️⃣ Analytics Tab
- Energy cost per group (bar chart)
- Inference time comparison
- Statistical summary (min/max/avg)

#### 4️⃣ Station Network Tab
- View all 50 NYC charging stations
- Station coverage visualization

#### 5️⃣ Detailed Results Tab
- Download results as JSON
- Per-group breakdown
- Tour sequence details

### CSV Format for Customer Upload

```csv
latitude,longitude
40.7580,-73.9855
40.7614,-73.9776
40.7550,-73.9710
...
```

---

## ⚙️ Configuration

### Model Parameters (real_streets_pipeline.py)

```python
model_params = {
    'embedding_dim': 128,      # Transformer embedding dimension
    'hidden_dim': 128,         # Hidden layer dimension
    'n_encode_layers': 3,      # Number of encoder layers
    'normalization': 'batch',  # Normalization type
    'tanh_clipping': 10.0,     # Output clipping
    'start_soc': 95.0,         # Initial battery SOC (%)
    't_limit': 10.0,           # Time limit (hours)
    'max_load': 4.0,           # Max vehicle load
    'velocity': 50.0,          # Speed (km/h)
    'num_nodes': 10,           # Customers per group
    'charging_num': 4          # Stations per group
}
```

### NYC Charging Stations

The system uses 50 pre-configured charging stations across NYC boroughs:
- **Manhattan**: 15 stations
- **Brooklyn**: 12 stations
- **Queens**: 10 stations
- **Bronx**: 8 stations
- **Staten Island**: 5 stations

---

## 📡 API Reference

### RealStreetsPipeline

```python
class RealStreetsPipeline:
    """Complete pipeline for EV routing with real street network"""
    
    def __init__(self, station_coords, depot_coords, device=None, checkpoint_path=None):
        """
        Args:
            station_coords: (N, 2) array of charging station coordinates
            depot_coords: (lat, lon) tuple for depot location
            device: 'cuda' or 'cpu' (auto-detected if None)
            checkpoint_path: Path to trained model weights
        """
    
    def run_pipeline(self, customer_coords, progress_callback=None) -> dict:
        """
        Run complete optimization pipeline
        
        Args:
            customer_coords: (M, 2) array of customer locations
            progress_callback: Optional function for progress updates
            
        Returns:
            dict with keys: 'n_groups', 'total_cost', 'total_distance_km',
                           'group_results', 'depot_coords', etc.
        """
```

### OSRMRouter

```python
class OSRMRouter:
    """Router using OSRM API for real street routing"""
    
    def build_distance_matrix(self, coords) -> np.ndarray:
        """Build N×N distance matrix using OSRM Table API (single request)"""
    
    def get_route(self, start, end) -> Tuple[List, float]:
        """Get route geometry and distance between two points"""
```

---

## 📊 Results

### Performance Benchmarks

| Metric | Value |
|--------|-------|
| Inference Time (10 customers) | ~0.5s |
| Distance Matrix (16×16) | ~1-2s (OSRM Table API) |
| Full Pipeline (30 customers) | ~5-10s |

### Model Performance

- Training on 10 customers: **100 epochs, ~2 hours** (GPU)
- Rollout baseline provides stable learning
- Greedy decoding achieves near-optimal solutions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
│  📍 Depot Coords    👥 Customer Coords    🔌 50 Fixed Stations  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                           │
│  K-Means Clustering → Groups of 10 → Nearest 4 Stations         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DISTANCE MATRIX (OSRM)                        │
│  Single API Call → 16×16 Real Road Distances                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DRL MODEL INFERENCE                         │
│  Attention Encoder → Autoregressive Decoder → Optimal Tour      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                               │
│  🗺️ Route Visualization    📊 Analytics    💾 JSON Export       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475) - Kool et al., 2019
- [OSRM - Open Source Routing Machine](http://project-osrm.org/)
- [Electric Vehicle Routing Problem: A Survey](https://doi.org/10.1016/j.apenergy.2023.121872)

---

## 👨‍💻 Author

**Dhia Salem** - *Deep Reinforcement Learning for EV Routing*

---

<p align="center">
  <b>🚗⚡ Optimizing the Future of Electric Vehicle Logistics ⚡🚗</b>
</p>
