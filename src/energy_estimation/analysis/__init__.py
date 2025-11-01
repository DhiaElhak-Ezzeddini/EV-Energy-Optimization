"""
Analysis Package for Energy Estimation Models.

Tools for visualizing and analyzing model performance.
"""

from .performance_analyzer import PerformanceAnalyzer
from .visualization import (
    plot_error_metrics_comparison,
    plot_inference_time_comparison,
    plot_residuals_analysis,
    plot_prediction_scatter,
    plot_feature_importance,
    plot_learning_curves,
    plot_error_distribution
)

__all__ = [
    'PerformanceAnalyzer',
    'plot_error_metrics_comparison',
    'plot_inference_time_comparison',
    'plot_residuals_analysis',
    'plot_prediction_scatter',
    'plot_feature_importance',
    'plot_learning_curves',
    'plot_error_distribution'
]
