"""
Professional Diagram Generator for EV Route Optimization Project
================================================================
Creates high-quality, publication-ready diagrams describing:
1. System Architecture
2. DRL Model Architecture  
3. Data Flow Pipeline
4. User Workflow
5. Component Overview

Author: Dhia Salem
Date: January 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Professional color palette
COLORS = {
    'primary': '#2563EB',      # Blue
    'secondary': '#7C3AED',    # Purple
    'success': '#059669',      # Green
    'warning': '#D97706',      # Orange
    'danger': '#DC2626',       # Red
    'info': '#0891B2',         # Cyan
    'dark': '#1F2937',         # Dark gray
    'light': '#F3F4F6',        # Light gray
    'white': '#FFFFFF',
}

# Pastel versions for backgrounds
PASTEL = {
    'blue': '#DBEAFE',
    'purple': '#EDE9FE',
    'green': '#D1FAE5',
    'orange': '#FEF3C7',
    'red': '#FEE2E2',
    'cyan': '#CFFAFE',
    'gray': '#F9FAFB',
}


def add_gradient_box(ax, x, y, width, height, color1, color2, label, sublabel=None, 
                     icon=None, border_color=None):
    """Add a professional gradient-style box with shadow effect"""
    
    # Shadow
    shadow = FancyBboxPatch((x + 0.05, y - 0.05), width, height, 
                            boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor='#00000020', edgecolor='none')
    ax.add_patch(shadow)
    
    # Main box
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color1, 
                         edgecolor=border_color or color2, 
                         linewidth=2.5)
    ax.add_patch(box)
    
    # Icon and label
    if icon:
        ax.text(x + width/2, y + height*0.65, icon, ha='center', va='center', 
               fontsize=16)
        ax.text(x + width/2, y + height*0.35, label, ha='center', va='center',
               fontsize=10, weight='bold', color=COLORS['dark'])
    else:
        ax.text(x + width/2, y + height*0.6, label, ha='center', va='center',
               fontsize=11, weight='bold', color=COLORS['dark'])
    
    if sublabel:
        ax.text(x + width/2, y + height*0.25, sublabel, ha='center', va='center',
               fontsize=8, color='#6B7280')


def add_arrow(ax, start, end, color=None, style='simple', curved=False):
    """Add a professional arrow between points"""
    if color is None:
        color = COLORS['dark']
    if curved:
        connectionstyle = "arc3,rad=0.2"
    else:
        connectionstyle = "arc3,rad=0"
    
    if style == 'simple':
        arrow = FancyArrowPatch(start, end, arrowstyle='-|>', 
                               connectionstyle=connectionstyle,
                               lw=2, color=color, mutation_scale=15)
    else:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', 
                               connectionstyle=connectionstyle,
                               lw=2.5, color=color, mutation_scale=20)
    ax.add_patch(arrow)


def create_system_architecture():
    """Create a modern system architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Title with modern styling
    ax.text(8, 10.5, '🚗⚡ EV Route Optimization System Architecture', 
            ha='center', va='center', fontsize=22, weight='bold', 
            color=COLORS['dark'], family='sans-serif')
    ax.text(8, 10.0, 'Deep Reinforcement Learning with Real Street Routing',
            ha='center', va='center', fontsize=12, color='#6B7280', style='italic')
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 1: INPUT SOURCES
    # ═══════════════════════════════════════════════════════════════
    
    # Input layer background
    input_bg = FancyBboxPatch((0.5, 8.0), 15, 1.5, boxstyle="round,pad=0.05",
                              facecolor=PASTEL['green'], edgecolor=COLORS['success'],
                              linewidth=3, alpha=0.7)
    ax.add_patch(input_bg)
    ax.text(0.8, 9.2, '📥 INPUT LAYER', fontsize=12, weight='bold', color=COLORS['success'])
    
    # Input boxes
    add_gradient_box(ax, 1.5, 8.2, 2.2, 1.0, COLORS['white'], COLORS['success'],
                    'Depot Location', '(lat, lon)', '📍')
    add_gradient_box(ax, 4.5, 8.2, 2.2, 1.0, COLORS['white'], COLORS['success'],
                    'Customer CSV', 'N × [lat, lon]', '👥')
    add_gradient_box(ax, 7.5, 8.2, 2.2, 1.0, COLORS['white'], COLORS['success'],
                    '50 NYC Stations', 'Fixed Network', '🔌')
    add_gradient_box(ax, 10.5, 8.2, 2.2, 1.0, COLORS['white'], COLORS['success'],
                    'DRL Model', 'Trained Weights', '🧠')
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 2: PREPROCESSING
    # ═══════════════════════════════════════════════════════════════
    
    # Arrows from input to preprocessing
    for x in [2.6, 5.6, 8.6, 11.6]:
        add_arrow(ax, (x, 8.2), (x, 7.0), COLORS['dark'])
    
    # Preprocessing background
    proc_bg = FancyBboxPatch((0.5, 5.5), 15, 1.5, boxstyle="round,pad=0.05",
                             facecolor=PASTEL['blue'], edgecolor=COLORS['primary'],
                             linewidth=3, alpha=0.7)
    ax.add_patch(proc_bg)
    ax.text(0.8, 6.7, '⚙️ PREPROCESSING LAYER', fontsize=12, weight='bold', color=COLORS['primary'])
    
    # Preprocessing boxes
    add_gradient_box(ax, 1.5, 5.7, 2.8, 1.0, COLORS['white'], COLORS['primary'],
                    'K-Means Clustering', 'Group by proximity', '🔄')
    add_gradient_box(ax, 5.0, 5.7, 2.8, 1.0, COLORS['white'], COLORS['primary'],
                    'Station Assignment', 'Nearest 4 per group', '📊')
    add_gradient_box(ax, 8.5, 5.7, 2.8, 1.0, COLORS['white'], COLORS['primary'],
                    'OSRM Table API', '16×16 distance matrix', '🛣️')
    add_gradient_box(ax, 12.0, 5.7, 2.8, 1.0, COLORS['white'], COLORS['primary'],
                    'Tensor Preparation', 'Static + Dynamic', '📦')
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 3: DRL INFERENCE
    # ═══════════════════════════════════════════════════════════════
    
    # Arrow to DRL layer
    add_arrow(ax, (8, 5.5), (8, 4.5), COLORS['dark'])
    
    # DRL background
    drl_bg = FancyBboxPatch((0.5, 2.8), 15, 1.7, boxstyle="round,pad=0.05",
                            facecolor=PASTEL['purple'], edgecolor=COLORS['secondary'],
                            linewidth=3, alpha=0.7)
    ax.add_patch(drl_bg)
    ax.text(0.8, 4.2, '🤖 DRL INFERENCE ENGINE', fontsize=12, weight='bold', color=COLORS['secondary'])
    
    # DRL architecture flow
    drl_components = [
        ('Graph\nEncoder', 2.0, '🔷'),
        ('Multi-Head\nAttention', 5.0, '🎯'),
        ('Context\nEmbedding', 8.0, '📐'),
        ('Pointer\nDecoder', 11.0, '👆'),
        ('Tour\nSequence', 14.0, '🛤️')
    ]
    
    for i, (label, x, icon) in enumerate(drl_components):
        add_gradient_box(ax, x - 1, 3.0, 2.0, 1.2, COLORS['white'], COLORS['secondary'],
                        label, None, icon)
        if i < len(drl_components) - 1:
            add_arrow(ax, (x + 1.0, 3.6), (drl_components[i+1][1] - 1, 3.6), 
                     COLORS['secondary'], style='bold')
    
    # ═══════════════════════════════════════════════════════════════
    # LAYER 4: OUTPUT
    # ═══════════════════════════════════════════════════════════════
    
    # Arrow to output
    add_arrow(ax, (8, 2.8), (8, 1.8), COLORS['dark'])
    
    # Output background
    out_bg = FancyBboxPatch((0.5, 0.3), 15, 1.5, boxstyle="round,pad=0.05",
                            facecolor=PASTEL['orange'], edgecolor=COLORS['warning'],
                            linewidth=3, alpha=0.7)
    ax.add_patch(out_bg)
    ax.text(0.8, 1.5, '📤 OUTPUT LAYER', fontsize=12, weight='bold', color=COLORS['warning'])
    
    # Output boxes
    add_gradient_box(ax, 1.5, 0.5, 2.5, 1.0, COLORS['white'], COLORS['warning'],
                    'Interactive Map', 'Plotly + OSRM', '🗺️')
    add_gradient_box(ax, 4.8, 0.5, 2.5, 1.0, COLORS['white'], COLORS['warning'],
                    'Analytics Charts', 'Cost & Time', '📊')
    add_gradient_box(ax, 8.1, 0.5, 2.5, 1.0, COLORS['white'], COLORS['warning'],
                    'Route Details', 'Per-group info', '📋')
    add_gradient_box(ax, 11.4, 0.5, 2.5, 1.0, COLORS['white'], COLORS['warning'],
                    'JSON Export', 'Full results', '💾')
    
    plt.tight_layout()
    return fig


def create_drl_model_architecture():
    """Create detailed DRL model architecture diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, '🧠 Attention-Based DRL Model Architecture', 
            ha='center', va='center', fontsize=20, weight='bold', color=COLORS['dark'])
    ax.text(8, 9.0, 'Encoder-Decoder with Multi-Head Attention for EVRP',
            ha='center', va='center', fontsize=11, color='#6B7280', style='italic')
    
    # ═══════════════════════════════════════════════════════════════
    # INPUT SECTION
    # ═══════════════════════════════════════════════════════════════
    
    ax.text(1.5, 8.2, 'INPUT', fontsize=11, weight='bold', color=COLORS['primary'])
    
    # Static features box
    static_box = FancyBboxPatch((0.5, 6.8), 3, 1.2, boxstyle="round,pad=0.03",
                                facecolor=PASTEL['blue'], edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(static_box)
    ax.text(2, 7.7, 'Static Features', ha='center', fontsize=10, weight='bold')
    ax.text(2, 7.4, '[x, y] coordinates', ha='center', fontsize=9, color='#6B7280')
    ax.text(2, 7.1, 'Shape: (B, 2, N)', ha='center', fontsize=8, color='#9CA3AF')
    
    # Dynamic features box
    dynamic_box = FancyBboxPatch((0.5, 5.3), 3, 1.2, boxstyle="round,pad=0.03",
                                 facecolor=PASTEL['green'], edgecolor=COLORS['success'], linewidth=2)
    ax.add_patch(dynamic_box)
    ax.text(2, 6.2, 'Dynamic Features', ha='center', fontsize=10, weight='bold')
    ax.text(2, 5.85, 'Load, Demand, SOC, Time', ha='center', fontsize=9, color='#6B7280')
    ax.text(2, 5.55, 'Shape: (B, 4, N)', ha='center', fontsize=8, color='#9CA3AF')
    
    # ═══════════════════════════════════════════════════════════════
    # ENCODER SECTION
    # ═══════════════════════════════════════════════════════════════
    
    ax.text(5.5, 8.2, 'ENCODER', fontsize=11, weight='bold', color=COLORS['secondary'])
    
    # Initial embedding
    add_arrow(ax, (3.5, 7.0), (4.5, 7.0), COLORS['dark'])
    embed_box = FancyBboxPatch((4.5, 6.5), 2.5, 1.0, boxstyle="round,pad=0.03",
                               facecolor=PASTEL['purple'], edgecolor=COLORS['secondary'], linewidth=2)
    ax.add_patch(embed_box)
    ax.text(5.75, 7.2, 'Linear Embedding', ha='center', fontsize=9, weight='bold')
    ax.text(5.75, 6.85, 'dim: 128', ha='center', fontsize=8, color='#6B7280')
    
    # Encoder layers (stacked)
    add_arrow(ax, (7.0, 7.0), (8.0, 7.0), COLORS['dark'])
    
    for i, y_offset in enumerate([0, -0.9, -1.8]):
        y = 6.5 + y_offset
        layer_box = FancyBboxPatch((8.0, y), 3.5, 0.8, boxstyle="round,pad=0.03",
                                   facecolor='#F3E8FF' if i % 2 == 0 else '#E9D5FF',
                                   edgecolor=COLORS['secondary'], linewidth=1.5)
        ax.add_patch(layer_box)
        ax.text(9.75, y + 0.55, f'Encoder Layer {i+1}', ha='center', fontsize=9, weight='bold')
        ax.text(9.75, y + 0.25, 'MHA → FFN → LayerNorm', ha='center', fontsize=7, color='#6B7280')
    
    ax.text(9.75, 4.3, '× 3 layers', ha='center', fontsize=9, style='italic', color='#7C3AED')
    
    # ═══════════════════════════════════════════════════════════════
    # DECODER SECTION
    # ═══════════════════════════════════════════════════════════════
    
    ax.text(13.5, 8.2, 'DECODER', fontsize=11, weight='bold', color=COLORS['danger'])
    
    # Context embedding
    add_arrow(ax, (11.5, 6.0), (12.5, 6.0), COLORS['dark'])
    context_box = FancyBboxPatch((12.5, 5.5), 2.5, 1.0, boxstyle="round,pad=0.03",
                                 facecolor=PASTEL['red'], edgecolor=COLORS['danger'], linewidth=2)
    ax.add_patch(context_box)
    ax.text(13.75, 6.2, 'Context Vector', ha='center', fontsize=9, weight='bold')
    ax.text(13.75, 5.85, 'Graph + Current', ha='center', fontsize=8, color='#6B7280')
    
    # Attention mechanism
    add_arrow(ax, (13.75, 5.5), (13.75, 4.5), COLORS['dark'])
    attn_box = FancyBboxPatch((12.5, 3.8), 2.5, 0.7, boxstyle="round,pad=0.03",
                              facecolor='#FEE2E2', edgecolor=COLORS['danger'], linewidth=2)
    ax.add_patch(attn_box)
    ax.text(13.75, 4.25, 'Attention Scores', ha='center', fontsize=9, weight='bold')
    
    # Softmax + selection
    add_arrow(ax, (13.75, 3.8), (13.75, 3.0), COLORS['dark'])
    softmax_box = FancyBboxPatch((12.5, 2.3), 2.5, 0.7, boxstyle="round,pad=0.03",
                                 facecolor='#FECACA', edgecolor=COLORS['danger'], linewidth=2)
    ax.add_patch(softmax_box)
    ax.text(13.75, 2.75, 'Softmax → π(a|s)', ha='center', fontsize=9, weight='bold')
    
    # Output
    add_arrow(ax, (13.75, 2.3), (13.75, 1.5), COLORS['dark'])
    out_box = FancyBboxPatch((12.0, 0.8), 3.5, 0.7, boxstyle="round,pad=0.03",
                             facecolor=PASTEL['orange'], edgecolor=COLORS['warning'], linewidth=2)
    ax.add_patch(out_box)
    ax.text(13.75, 1.25, '🎯 Next Node Selection', ha='center', fontsize=10, weight='bold')
    
    # ═══════════════════════════════════════════════════════════════
    # TRAINING INFO BOX
    # ═══════════════════════════════════════════════════════════════
    
    info_box = FancyBboxPatch((0.5, 0.5), 5.5, 3.5, boxstyle="round,pad=0.05",
                              facecolor='#FEF9C3', edgecolor='#CA8A04', linewidth=2)
    ax.add_patch(info_box)
    ax.text(3.25, 3.7, '⚡ Training Configuration', ha='center', fontsize=11, weight='bold', color='#92400E')
    
    training_info = [
        '• Algorithm: REINFORCE',
        '• Baseline: Rollout (greedy)',
        '• Embedding dim: 128',
        '• Hidden dim: 128', 
        '• Encoder layers: 3',
        '• Batch size: 1024',
        '• Learning rate: 1e-4',
        '• Optimizer: Adam'
    ]
    
    for i, text in enumerate(training_info):
        ax.text(1.0, 3.3 - i*0.35, text, fontsize=9, color='#78350F')
    
    # Autoregressive loop arrow
    loop_arrow = FancyArrowPatch((15.2, 1.2), (15.2, 6.0), arrowstyle='->', 
                                  connectionstyle='arc3,rad=-0.3',
                                  lw=2, color=COLORS['danger'], linestyle='--')
    ax.add_patch(loop_arrow)
    ax.text(15.5, 3.5, 'Autoregressive\nDecoding', ha='center', fontsize=8, 
           color=COLORS['danger'], rotation=90)
    
    plt.tight_layout()
    return fig


def create_data_flow_diagram():
    """Create a data flow pipeline diagram"""
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(8, 7.5, '🔄 Data Flow Pipeline', 
            ha='center', va='center', fontsize=20, weight='bold', color=COLORS['dark'])
    
    # Pipeline stages
    stages = [
        ('📍', 'Customer\nCSV', '30 locations\n(lat, lon)', PASTEL['blue'], COLORS['primary']),
        ('🔄', 'K-Means\nGrouping', '3 groups\n× 10 each', PASTEL['green'], COLORS['success']),
        ('🔌', 'Station\nAssignment', '4 nearest\nper group', PASTEL['purple'], COLORS['secondary']),
        ('🛣️', 'OSRM\nTable API', '16×16 matrix\n1 API call', PASTEL['cyan'], COLORS['info']),
        ('📊', 'Tensor\nBuild', 'Static +\nDynamic', PASTEL['orange'], COLORS['warning']),
        ('🧠', 'DRL\nInference', 'Attention\nModel', PASTEL['red'], COLORS['danger']),
        ('🗺️', 'Route\nVisualization', 'Interactive\nMap', PASTEL['green'], COLORS['success']),
    ]
    
    # Draw pipeline
    start_x = 1.0
    box_width = 1.8
    spacing = 0.3
    
    for i, (icon, title, desc, bg_color, border_color) in enumerate(stages):
        x = start_x + i * (box_width + spacing)
        
        # Box with shadow
        shadow = FancyBboxPatch((x + 0.05, 3.45), box_width, 2.5,
                                boxstyle="round,pad=0.05", facecolor='#00000015', edgecolor='none')
        ax.add_patch(shadow)
        
        box = FancyBboxPatch((x, 3.5), box_width, 2.5,
                             boxstyle="round,pad=0.05", facecolor=bg_color, 
                             edgecolor=border_color, linewidth=2.5)
        ax.add_patch(box)
        
        # Content
        ax.text(x + box_width/2, 5.5, icon, ha='center', fontsize=20)
        ax.text(x + box_width/2, 4.8, title, ha='center', fontsize=10, weight='bold', color=COLORS['dark'])
        ax.text(x + box_width/2, 4.0, desc, ha='center', fontsize=8, color='#6B7280')
        
        # Arrow to next
        if i < len(stages) - 1:
            arrow_start = (x + box_width, 4.75)
            arrow_end = (x + box_width + spacing, 4.75)
            add_arrow(ax, arrow_start, arrow_end, border_color, style='bold')
    
    # Timing annotations
    ax.text(8, 2.5, '⏱️ Processing Time Breakdown', ha='center', fontsize=12, weight='bold', color=COLORS['dark'])
    
    times = [
        ('Load CSV', '< 0.1s', 1.9, True),
        ('Clustering', '< 0.1s', 4.0, True),
        ('Assignment', '< 0.1s', 6.1, True),
        ('OSRM API', '~1-2s', 8.2, False),
        ('Tensor', '< 0.1s', 10.3, True),
        ('DRL', '~0.5s', 12.4, True),
        ('Render', '~0.5s', 14.5, True),
    ]
    
    for label, time_val, x, is_fast in times:
        ax.text(x, 2.0, time_val, ha='center', fontsize=9, weight='bold', 
               color=COLORS['success'] if is_fast else COLORS['warning'])
        ax.text(x, 1.6, label, ha='center', fontsize=8, color='#6B7280')
    
    # Total time box
    total_box = FancyBboxPatch((6, 0.5), 4, 0.8, boxstyle="round,pad=0.05",
                               facecolor=PASTEL['green'], edgecolor=COLORS['success'], linewidth=2)
    ax.add_patch(total_box)
    ax.text(8, 0.9, '✅ Total: ~3-5 seconds for 30 customers', ha='center', 
           fontsize=11, weight='bold', color=COLORS['success'])
    
    plt.tight_layout()
    return fig


def create_component_overview():
    """Create a component overview diagram showing all project files"""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(7, 11.5, '📁 Project Components Overview', 
            ha='center', va='center', fontsize=20, weight='bold', color=COLORS['dark'])
    
    # ═══════════════════════════════════════════════════════════════
    # CORE APPLICATION (Left side)
    # ═══════════════════════════════════════════════════════════════
    
    core_bg = FancyBboxPatch((0.3, 6.5), 4.2, 4.5, boxstyle="round,pad=0.05",
                             facecolor=PASTEL['blue'], edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(core_bg)
    ax.text(2.4, 10.7, '🎯 Core Application', ha='center', fontsize=12, weight='bold', color=COLORS['primary'])
    
    core_files = [
        ('dashboard.py', 'Streamlit Web UI', '🖥️'),
        ('real_streets_pipeline.py', 'Main Inference Pipeline', '🔄'),
        ('sample_customers.csv', 'Test Data (30 customers)', '📊'),
        ('requirements_dashboard.txt', 'Dependencies', '📋'),
    ]
    
    for i, (filename, desc, icon) in enumerate(core_files):
        y = 9.8 - i * 0.85
        ax.text(0.6, y, icon, fontsize=12)
        ax.text(1.0, y, filename, fontsize=9, weight='bold', color=COLORS['dark'], family='monospace')
        ax.text(1.0, y - 0.3, desc, fontsize=8, color='#6B7280')
    
    # ═══════════════════════════════════════════════════════════════
    # DRL TRAINING (Right side)
    # ═══════════════════════════════════════════════════════════════
    
    train_bg = FancyBboxPatch((4.8, 6.5), 4.2, 4.5, boxstyle="round,pad=0.05",
                              facecolor=PASTEL['purple'], edgecolor=COLORS['secondary'], linewidth=2)
    ax.add_patch(train_bg)
    ax.text(6.9, 10.7, '🧠 DRL Training', ha='center', fontsize=12, weight='bold', color=COLORS['secondary'])
    
    train_files = [
        ('run.py', 'Training Entry Point', '🚀'),
        ('trainer.py', 'Training Loop', '🔁'),
        ('run_em_training.ps1', 'PowerShell Script', '⚡'),
        ('plot_reward.py', 'Reward Visualization', '📈'),
    ]
    
    for i, (filename, desc, icon) in enumerate(train_files):
        y = 9.8 - i * 0.85
        ax.text(5.1, y, icon, fontsize=12)
        ax.text(5.5, y, filename, fontsize=9, weight='bold', color=COLORS['dark'], family='monospace')
        ax.text(5.5, y - 0.3, desc, fontsize=8, color='#6B7280')
    
    # ═══════════════════════════════════════════════════════════════
    # NEURAL NETWORKS (Bottom left)
    # ═══════════════════════════════════════════════════════════════
    
    nets_bg = FancyBboxPatch((0.3, 2.0), 4.2, 4.0, boxstyle="round,pad=0.05",
                             facecolor=PASTEL['green'], edgecolor=COLORS['success'], linewidth=2)
    ax.add_patch(nets_bg)
    ax.text(2.4, 5.7, '🔷 nets/ - Neural Networks', ha='center', fontsize=11, weight='bold', color=COLORS['success'])
    
    nets_files = [
        ('DRLModel.py', 'Main Attention Model'),
        ('AM.py', 'Attention Mechanisms'),
        ('GraphEncoder.py', 'Graph Neural Encoder'),
        ('PointNetwork.py', 'Pointer Network Decoder'),
    ]
    
    for i, (filename, desc) in enumerate(nets_files):
        y = 5.2 - i * 0.75
        ax.text(0.8, y, '•', fontsize=14, color=COLORS['success'])
        ax.text(1.1, y, filename, fontsize=9, weight='bold', color=COLORS['dark'], family='monospace')
        ax.text(2.8, y, f'→ {desc}', fontsize=8, color='#6B7280')
    
    # ═══════════════════════════════════════════════════════════════
    # PROBLEM DEFINITION (Bottom center)
    # ═══════════════════════════════════════════════════════════════
    
    prob_bg = FancyBboxPatch((4.8, 3.5), 4.2, 2.5, boxstyle="round,pad=0.05",
                             facecolor=PASTEL['orange'], edgecolor=COLORS['warning'], linewidth=2)
    ax.add_patch(prob_bg)
    ax.text(6.9, 5.7, '📦 problems/ - EVRP Definition', ha='center', fontsize=11, weight='bold', color=COLORS['warning'])
    
    ax.text(5.1, 5.0, '•', fontsize=14, color=COLORS['warning'])
    ax.text(5.4, 5.0, 'EVRP.py', fontsize=9, weight='bold', color=COLORS['dark'], family='monospace')
    ax.text(5.4, 4.6, '→ VehicleRoutingDataset class', fontsize=8, color='#6B7280')
    ax.text(5.4, 4.3, '→ State transitions & rewards', fontsize=8, color='#6B7280')
    ax.text(5.4, 4.0, '→ Mask generation logic', fontsize=8, color='#6B7280')
    
    # ═══════════════════════════════════════════════════════════════
    # UTILITIES (Bottom right)
    # ═══════════════════════════════════════════════════════════════
    
    utils_bg = FancyBboxPatch((9.3, 2.0), 4.4, 4.0, boxstyle="round,pad=0.05",
                              facecolor=PASTEL['cyan'], edgecolor=COLORS['info'], linewidth=2)
    ax.add_patch(utils_bg)
    ax.text(11.5, 5.7, '🛠️ utils/ - Utilities', ha='center', fontsize=11, weight='bold', color=COLORS['info'])
    
    utils_files = [
        ('functions.py', 'Helper functions'),
        ('reinforce_baselines.py', 'Rollout baseline'),
        ('beam_search.py', 'Beam search decoding'),
        ('data_utils.py', 'Data processing'),
    ]
    
    for i, (filename, desc) in enumerate(utils_files):
        y = 5.2 - i * 0.75
        ax.text(9.6, y, '•', fontsize=14, color=COLORS['info'])
        ax.text(9.9, y, filename, fontsize=9, weight='bold', color=COLORS['dark'], family='monospace')
        ax.text(12.0, y, f'→ {desc}', fontsize=8, color='#6B7280')
    
    # ═══════════════════════════════════════════════════════════════
    # DATA & CHECKPOINTS (Far right)
    # ═══════════════════════════════════════════════════════════════
    
    data_bg = FancyBboxPatch((9.3, 6.5), 4.4, 4.5, boxstyle="round,pad=0.05",
                             facecolor=PASTEL['red'], edgecolor=COLORS['danger'], linewidth=2)
    ax.add_patch(data_bg)
    ax.text(11.5, 10.7, '💾 Data & Checkpoints', ha='center', fontsize=11, weight='bold', color=COLORS['danger'])
    
    data_items = [
        ('ExperimentalLog/', 'Trained model weights'),
        ('  └─ train/10/rollout/', 'Training checkpoints'),
        ('      └─ best.pt', '🏆 Best model'),
        ('ExperimentalData/', 'Training instances'),
        ('  └─ CVRPlib/', 'Benchmark data'),
    ]
    
    for i, (path, desc) in enumerate(data_items):
        y = 9.8 - i * 0.75
        ax.text(9.6, y, path, fontsize=8, color=COLORS['dark'], family='monospace')
        if '🏆' in desc:
            ax.text(12.5, y, desc, fontsize=8, color=COLORS['danger'], weight='bold')
        else:
            ax.text(12.5, y, desc, fontsize=7, color='#6B7280')
    
    # Connection arrows showing dependencies
    ax.annotate('', xy=(4.5, 8.5), xytext=(4.8, 8.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    ax.annotate('', xy=(4.5, 4.5), xytext=(4.8, 4.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    
    plt.tight_layout()
    return fig


def create_user_workflow():
    """Create a step-by-step user workflow diagram"""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, '👤 User Workflow Guide', 
            ha='center', va='center', fontsize=20, weight='bold', color=COLORS['dark'])
    ax.text(7, 9.0, 'From Launch to Results in 8 Simple Steps',
            ha='center', va='center', fontsize=11, color='#6B7280', style='italic')
    
    # Steps
    steps = [
        ('1', 'Launch Dashboard', 'streamlit run dashboard.py', '🚀', COLORS['primary']),
        ('2', 'Set Depot Location', 'Enter lat/lon coordinates', '📍', COLORS['success']),
        ('3', 'Load Customers', 'Upload CSV or Generate Random', '👥', COLORS['secondary']),
        ('4', 'Configure Options', 'Adjust parameters (optional)', '⚙️', COLORS['info']),
        ('5', 'Run Optimization', 'Click the green button', '▶️', COLORS['warning']),
        ('6', 'View Route Map', 'Interactive Plotly visualization', '🗺️', COLORS['danger']),
        ('7', 'Analyze Results', 'Charts, metrics, statistics', '📊', COLORS['primary']),
        ('8', 'Export Data', 'Download JSON results', '💾', COLORS['success']),
    ]
    
    # Draw steps in a flowing pattern
    positions = [
        (2, 7.5), (5, 7.5), (8, 7.5), (11, 7.5),
        (11, 4.5), (8, 4.5), (5, 4.5), (2, 4.5)
    ]
    
    for i, ((num, title, desc, icon, color), (x, y)) in enumerate(zip(steps, positions)):
        # Circle with number
        circle = Circle((x, y + 0.8), 0.35, facecolor=color, edgecolor='white', linewidth=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.8, num, ha='center', va='center', fontsize=14, 
               weight='bold', color='white')
        
        # Box
        box = FancyBboxPatch((x - 1.2, y - 0.6), 2.4, 1.2, boxstyle="round,pad=0.05",
                             facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(box)
        
        # Content
        ax.text(x, y + 0.3, f'{icon} {title}', ha='center', fontsize=10, weight='bold', color=COLORS['dark'])
        ax.text(x, y - 0.15, desc, ha='center', fontsize=8, color='#6B7280')
        
        # Arrows
        if i < len(steps) - 1:
            next_x, next_y = positions[i + 1]
            if i == 3:  # Turn down
                ax.annotate('', xy=(next_x + 1.2, next_y + 0.3), xytext=(x, y - 0.6),
                           arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2,
                                          connectionstyle='arc3,rad=-0.3'))
            elif y == next_y:  # Horizontal
                direction = 1 if next_x > x else -1
                ax.annotate('', xy=(next_x - direction * 1.2, next_y + 0.3), 
                           xytext=(x + direction * 1.2, y + 0.3),
                           arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    
    # Tips box
    tips_box = FancyBboxPatch((0.5, 0.5), 13, 2.5, boxstyle="round,pad=0.05",
                              facecolor='#FEF3C7', edgecolor='#D97706', linewidth=2)
    ax.add_patch(tips_box)
    ax.text(7, 2.7, '💡 Pro Tips', ha='center', fontsize=12, weight='bold', color='#92400E')
    
    tips = [
        '• Customer locations must be in multiples of 10 (each group = 10 customers + 4 stations)',
        '• Use the provided sample_customers.csv for testing, or upload your own CSV with latitude,longitude columns',
        '• The OSRM Table API provides real road distances - requires internet connection',
        '• Results can be exported as JSON for integration with other systems',
    ]
    
    for i, tip in enumerate(tips):
        ax.text(1.0, 2.2 - i * 0.45, tip, fontsize=9, color='#78350F')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("=" * 60)
    print("🎨 Generating Professional Project Diagrams")
    print("=" * 60)
    
    diagrams = [
        ('system_architecture.png', create_system_architecture, 'System Architecture'),
        ('drl_model_architecture.png', create_drl_model_architecture, 'DRL Model Architecture'),
        ('data_flow_pipeline.png', create_data_flow_diagram, 'Data Flow Pipeline'),
        ('component_overview.png', create_component_overview, 'Component Overview'),
        ('user_workflow.png', create_user_workflow, 'User Workflow Guide'),
    ]
    
    for filename, func, desc in diagrams:
        print(f"\n📊 Creating {desc}...")
        fig = func()
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', 
                   edgecolor='none', pad_inches=0.1)
        plt.close(fig)
        print(f"   ✅ Saved: {filename}")
    
    print("\n" + "=" * 60)
    print("✅ All diagrams generated successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    for filename, _, desc in diagrams:
        print(f"   📄 {filename:30} - {desc}")
    print("\nUse these diagrams in your documentation and presentations!")
