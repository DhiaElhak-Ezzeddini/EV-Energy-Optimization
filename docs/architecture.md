# Architecture Overview

## Project Structure

```
EV-Energy-Optimization/
├── src/                        # Core source code
│   ├── models/                 # Neural network architectures
│   │   ├── drl_model.py       # Main attention-based DRL model
│   │   ├── attention_model.py # Alternative AM variant
│   │   ├── pointer_network.py # Pointer network variant
│   │   └── graph_encoder.py   # Shared encoder backbone
│   │
│   ├── problem/               # Problem definition
│   │   └── evrp_environment.py # EVRP dataset & dynamics
│   │
│   ├── training/              # Training pipeline
│   │   ├── trainer.py        # Main training loop
│   │   ├── baselines.py      # REINFORCE baselines
│   │   └── run_training.py   # CLI entry point
│   │
│   ├── inference/             # Inference & decoding
│   │   └── beam_search.py    # Beam search utilities
│   │
│   └── utils/                 # Shared utilities
│       ├── tensor_utils.py
│       ├── data_utils.py
│       └── functions.py
│
├── data_generation/           # Real-world data processing
│   ├── graph_extraction/      # Road network tools
│   └── config/               # EV & physics configuration
│
├── dashboard/                 # Streamlit web application
├── data/                      # Data files
├── checkpoints/              # Trained models
└── outputs/                  # Generated outputs
```

## Model Architecture

### DRL Model (Attention-based)
1. **Encoder**: GraphAttentionEncoder with multi-head attention
2. **Decoder**: Autoregressive with route context embedding
3. **Policy**: Softmax over masked action space

### Training
- Algorithm: REINFORCE with baseline
- Baselines: Rollout, Critic, Exponential
- Optimizer: Adam with gradient clipping

### Energy Model
Physics-based energy consumption:
- Drag force
- Rolling resistance
- Gravitational effects
- Regenerative braking
