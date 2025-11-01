"""
Trainer Package

This package contains training utilities and orchestration for model training.
"""

from .trainer import ModelTrainer
from .utils import (
    calculate_regression_metrics,
    plot_predictions,
    plot_residuals,
    plot_feature_importance,
    compare_models,
    print_metrics_comparison,
    EarlyStopping
)

__all__ = [
    'ModelTrainer',
    'calculate_regression_metrics',
    'plot_predictions',
    'plot_residuals',
    'plot_feature_importance',
    'compare_models',
    'print_metrics_comparison',
    'EarlyStopping'
]
