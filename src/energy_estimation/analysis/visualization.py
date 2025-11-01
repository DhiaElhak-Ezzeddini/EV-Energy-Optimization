"""
Visualization Functions for Model Analysis.

Provides plotting functions for error metrics, timing, and predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_error_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['rmse', 'mae', 'r2', 'mape'],
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot comparison of error metrics across models.
    
    Args:
        results: Dictionary mapping model names to their metrics
        metrics: List of metrics to plot
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    models = list(results.keys())
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        values = [results[model].get(metric, 0) for model in models]
        colors = sns.color_palette("husl", len(models))
        
        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Highlight best model
        if metric in ['rmse', 'mae', 'mse', 'mape']:
            best_idx = np.argmin(values)
            best_label = '↓ Lower is better'
        else:  # r2
            best_idx = np.argmax(values)
            best_label = '↑ Higher is better'
        
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
        
        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison\n{best_label}', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved error metrics plot to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_inference_time_comparison(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot inference time and throughput comparison.
    
    Args:
        results: Dictionary mapping model names to their metrics
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = list(results.keys())
    colors = sns.color_palette("Set2", len(models))
    
    # Plot 1: Inference Time
    inference_times = [results[model].get('inference_time_seconds', 0) * 1000 for model in models]
    bars1 = ax1.barh(models, inference_times, color=colors, alpha=0.7, edgecolor='black')
    
    # Highlight fastest
    fastest_idx = np.argmin(inference_times)
    bars1[fastest_idx].set_edgecolor('green')
    bars1[fastest_idx].set_linewidth(3)
    
    for i, (bar, time) in enumerate(zip(bars1, inference_times)):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f' {time:.2f} ms',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Inference Time (milliseconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Inference Time per 1000 Samples\n← Faster is better', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Throughput (samples/second)
    throughput = [results[model].get('samples_per_second', 0) for model in models]
    bars2 = ax2.barh(models, throughput, color=colors, alpha=0.7, edgecolor='black')
    
    # Highlight highest throughput
    fastest_idx = np.argmax(throughput)
    bars2[fastest_idx].set_edgecolor('green')
    bars2[fastest_idx].set_linewidth(3)
    
    for i, (bar, tput) in enumerate(zip(bars2, throughput)):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f' {tput:,.0f} samp/s',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Throughput (samples/second)', fontsize=12, fontweight='bold')
    ax2.set_title('Inference Throughput\n→ Higher is better', fontsize=12)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved inference time plot to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_residuals_analysis(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot residuals analysis for multiple models.
    
    Args:
        y_true: True values
        predictions: Dictionary mapping model names to predictions
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        residuals = y_true - y_pred
        
        # Plot 1: Residuals vs Predicted
        ax1 = axes[0, idx]
        ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values (kWh)', fontsize=10)
        ax1.set_ylabel('Residuals (kWh)', fontsize=10)
        ax1.set_title(f'{model_name}\nResiduals vs Predicted', fontsize=11, fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Add statistics
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax1.text(0.05, 0.95, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
        
        # Plot 2: Residuals Distribution
        ax2 = axes[1, idx]
        ax2.hist(residuals, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Residuals (kWh)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title(f'{model_name}\nResiduals Distribution', fontsize=11, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved residuals analysis to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_prediction_scatter(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot actual vs predicted scatter plots for multiple models.
    
    Args:
        y_true: True values
        predictions: Dictionary mapping model names to predictions
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    n_models = len(predictions)
    cols = min(n_models, 2)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))
    
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, label='Predictions')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        ax.set_xlabel('Actual Energy Consumption (kWh)', fontsize=11)
        ax.set_ylabel('Predicted Energy Consumption (kWh)', fontsize=11)
        ax.set_title(f'{model_name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(alpha=0.3)
        
        # Add diagonal reference lines
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved prediction scatter plots to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_feature_importance(
    importance_dict: Dict[str, Dict[str, float]],
    top_n: int = 10,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot feature importance for multiple models.
    
    Args:
        importance_dict: Dictionary mapping model names to feature importance dicts
        top_n: Number of top features to show
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    n_models = len(importance_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, importance) in enumerate(importance_dict.items()):
        ax = axes[idx]
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, values = zip(*sorted_features)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        colors = sns.color_palette("viridis", len(features))
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name}\nTop {top_n} Features', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {width:.1f}',
                   ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance plot to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_learning_curves(
    train_sizes: np.ndarray,
    train_scores: Dict[str, np.ndarray],
    val_scores: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot learning curves for multiple models.
    
    Args:
        train_sizes: Array of training set sizes
        train_scores: Dictionary mapping model names to training scores
        val_scores: Dictionary mapping model names to validation scores
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = sns.color_palette("Set1", len(train_scores))
    
    for idx, model_name in enumerate(train_scores.keys()):
        color = colors[idx]
        
        # Plot training score
        ax.plot(train_sizes, train_scores[model_name], 
               marker='o', linestyle='-', linewidth=2, 
               color=color, alpha=0.7, label=f'{model_name} (train)')
        
        # Plot validation score
        ax.plot(train_sizes, val_scores[model_name],
               marker='s', linestyle='--', linewidth=2,
               color=color, alpha=0.9, label=f'{model_name} (val)')
    
    ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curves: Model Performance vs Training Size', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved learning curves to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot error distribution comparison across models.
    
    Args:
        y_true: True values
        predictions: Dictionary mapping model names to predictions
        save_path: Path to save the figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Absolute Error Distribution
    for model_name, y_pred in predictions.items():
        abs_error = np.abs(y_true - y_pred)
        ax1.hist(abs_error, bins=30, alpha=0.5, label=model_name, edgecolor='black')
    
    ax1.set_xlabel('Absolute Error (kWh)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Error Distribution', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Percentage Error Distribution
    for model_name, y_pred in predictions.items():
        pct_error = np.abs((y_true - y_pred) / y_true) * 100
        # Remove outliers for better visualization
        pct_error_clipped = np.clip(pct_error, 0, 50)
        ax2.hist(pct_error_clipped, bins=30, alpha=0.5, label=model_name, edgecolor='black')
    
    ax2.set_xlabel('Percentage Error (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Percentage Error Distribution (clipped at 50%)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved error distribution plot to: {save_path}")
    
    if show:
        plt.show()
    
    return fig
