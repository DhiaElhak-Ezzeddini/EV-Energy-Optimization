"""
Training Utilities

This module provides utility functions for training, including metrics,
callbacks, and helper functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        prefix (str): Prefix for metric names
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    metrics = {
        f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        f'{prefix}mae': mean_absolute_error(y_true, y_pred),
        f'{prefix}mse': mean_squared_error(y_true, y_pred),
        f'{prefix}r2': r2_score(y_true, y_pred),
        f'{prefix}mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
    }
    
    # Additional metrics
    residuals = y_true - y_pred
    metrics[f'{prefix}mean_residual'] = np.mean(residuals)
    metrics[f'{prefix}std_residual'] = np.std(residuals)
    metrics[f'{prefix}max_error'] = np.max(np.abs(residuals))
    metrics[f'{prefix}min_error'] = np.min(np.abs(residuals))
    
    # Percentage of predictions within thresholds
    within_5_percent = np.sum(np.abs(residuals / y_true) <= 0.05) / len(y_true) * 100
    within_10_percent = np.sum(np.abs(residuals / y_true) <= 0.10) / len(y_true) * 100
    within_20_percent = np.sum(np.abs(residuals / y_true) <= 0.20) / len(y_true) * 100
    
    metrics[f'{prefix}within_5pct'] = within_5_percent
    metrics[f'{prefix}within_10pct'] = within_10_percent
    metrics[f'{prefix}within_20pct'] = within_20_percent
    
    return metrics


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot predicted vs actual values.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
        show (bool): Whether to show the plot
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Labels and title
    ax.set_xlabel('Actual Energy Consumption (kWh)', fontsize=12)
    ax.set_ylabel('Predicted Energy Consumption (kWh)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metrics text
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    text_str = f'RMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}'
    ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot residuals.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
        show (bool): Whether to show the plot
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=12)
    axes[0].set_ylabel('Residuals', fontsize=12)
    axes[0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residuals', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics
    text_str = f'Mean: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}'
    axes[1].text(0.05, 0.95, text_str, transform=axes[1].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
        top_n (int): Number of top features to show
        title (str): Plot title
        save_path (Optional[str]): Path to save the plot
        show (bool): Whether to show the plot
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Get top N features
    plot_df = importance_df.head(top_n).copy()
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    # Horizontal bar plot
    ax.barh(range(len(plot_df)), plot_df['importance'])
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['feature'])
    ax.invert_yaxis()
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Compare multiple models' performance.
    
    Args:
        results_dict (Dict[str, Dict[str, float]]): Dict mapping model names to their metrics
        metrics (Optional[List[str]]): List of metrics to compare
        save_path (Optional[str]): Path to save the plot
        show (bool): Whether to show the plot
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    if metrics is None:
        metrics = ['rmse', 'mae', 'r2', 'mape']
    
    # Prepare data
    comparison_data = []
    for model_name, model_metrics in results_dict.items():
        for metric in metrics:
            if metric in model_metrics:
                comparison_data.append({
                    'Model': model_name,
                    'Metric': metric.upper(),
                    'Value': model_metrics[metric]
                })
    
    df = pd.DataFrame(comparison_data)
    
    # Create subplots for each metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        metric_data = df[df['Metric'] == metric.upper()]
        
        axes[idx].bar(range(len(metric_data)), metric_data['Value'])
        axes[idx].set_xticks(range(len(metric_data)))
        axes[idx].set_xticklabels(metric_data['Model'], rotation=45, ha='right')
        axes[idx].set_ylabel('Value', fontsize=11)
        axes[idx].set_title(metric.upper(), fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def print_metrics_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Print and return a comparison table of model metrics.
    
    Args:
        results_dict (Dict[str, Dict[str, float]]): Dict mapping model names to their metrics
        metrics (Optional[List[str]]): List of metrics to include
        
    Returns:
        pd.DataFrame: Comparison table
    """
    if metrics is None:
        # Get all available metrics from first model
        first_model = list(results_dict.values())[0]
        metrics = list(first_model.keys())
    
    # Create comparison DataFrame
    comparison_data = {}
    for model_name, model_metrics in results_dict.items():
        comparison_data[model_name] = {
            metric: model_metrics.get(metric, np.nan)
            for metric in metrics
        }
    
    comparison_df = pd.DataFrame(comparison_data).T
    
    print("\n" + "="*70)
    print("Model Performance Comparison")
    print("="*70)
    print(comparison_df.to_string())
    print("="*70 + "\n")
    
    return comparison_df


class EarlyStopping:
    """
    Early stopping callback to stop training when metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' for metrics to minimize, 'max' for metrics to maximize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score (float): Current metric score
            
        Returns:
            bool: True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
