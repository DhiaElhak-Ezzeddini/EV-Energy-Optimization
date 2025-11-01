"""
Training Configuration Settings.

Contains configurations for the training process, Optuna optimization,
and preprocessing pipeline.
"""

from typing import Dict, Any

# ============================================================================
# Data Split Configuration
# ============================================================================

DATA_SPLIT_CONFIG = {
    'test_size': 0.2,           # 20% for test set
    'val_size': 0.2,            # 20% of remaining for validation (16% total)
    'random_state': 42,
    'shuffle': True,
    'stratify': False           # Set to True for classification
}

# ============================================================================
# Preprocessing Configuration
# ============================================================================

PREPROCESSING_CONFIG = {
    'scale_target': True,       # Whether to scale the target variable
    'scaler_type': 'standard',  # 'standard', 'minmax', 'robust'
    'handle_outliers': False,   # Whether to clip outliers
    'outlier_std': 3.0,        # Number of std for outlier detection
    'save_preprocessor': True   # Save preprocessor with model
}

# ============================================================================
# Training Configuration
# ============================================================================

TRAINING_CONFIG = {
    'use_optuna': False,        # Enable Optuna hyperparameter optimization
    'n_trials': 50,             # Number of Optuna trials
    'timeout': 3600,            # Timeout in seconds (1 hour)
    'save_model': True,         # Save model after training
    'save_study': True,         # Save Optuna study
    'log_level': 'INFO',        # Logging level: DEBUG, INFO, WARNING, ERROR
    'show_progress': True,      # Show progress bars during training
    'n_jobs': -1               # Number of parallel jobs (-1 = all cores)
}

# ============================================================================
# Optuna Configuration
# ============================================================================

OPTUNA_CONFIG = {
    'direction': 'minimize',    # 'minimize' for RMSE, 'maximize' for R²
    'sampler': 'TPE',          # 'TPE', 'Random', 'Grid', 'CmaEs'
    'pruner': 'MedianPruner',  # 'MedianPruner', 'HyperbandPruner', 'NopPruner'
    'n_startup_trials': 10,    # Random trials before TPE sampling
    'n_warmup_steps': 10,      # Steps before pruning starts
    'study_name_prefix': 'ev_energy',  # Prefix for study names
    'storage': None,           # Database URL for persistent storage (e.g., 'sqlite:///optuna.db')
    'load_if_exists': False    # Load existing study if available
}

# ============================================================================
# Evaluation Metrics Configuration
# ============================================================================

METRICS_CONFIG = {
    'primary_metric': 'rmse',   # Primary metric for model selection
    'metrics': [                # All metrics to compute
        'rmse',
        'mae',
        'mse',
        'r2',
        'mape',
        'mean_residual',
        'std_residual',
        'max_error'
    ],
    'calculate_inference_time': True,  # Measure prediction speed
    'inverse_transform': True          # Report metrics in original scale
}

# ============================================================================
# Model Saving Configuration
# ============================================================================

SAVE_CONFIG = {
    'save_dir': 'saved_models',     # Base directory for saved models
    'create_experiment_dir': True,   # Create subdirectory per experiment
    'save_metadata': True,           # Save model metadata JSON
    'save_predictions': True,        # Save test predictions
    'save_training_history': True,   # Save training history
    'save_summary': True,            # Save text summary
    'compression': 3                 # Joblib compression level (0-9)
}

# ============================================================================
# Cross-Validation Configuration
# ============================================================================

CV_CONFIG = {
    'n_folds': 5,               # Number of CV folds
    'shuffle': True,            # Shuffle before splitting
    'random_state': 42,
    'stratified': False         # Stratified CV for classification
}

# ============================================================================
# Early Stopping Configuration
# ============================================================================

EARLY_STOPPING_CONFIG = {
    'enabled': True,            # Enable early stopping
    'rounds': 50,               # Rounds without improvement before stopping
    'min_delta': 0.0001,       # Minimum change to qualify as improvement
    'verbose': False            # Print early stopping messages
}

# ============================================================================
# Logging Configuration
# ============================================================================

LOGGING_CONFIG = {
    'log_to_file': True,        # Write logs to file
    'log_to_console': True,     # Print logs to console
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file_level': 'INFO',       # File logging level
    'console_level': 'INFO',    # Console logging level
    'max_file_size': 10485760,  # 10 MB
    'backup_count': 5           # Number of backup log files
}

# ============================================================================
# Resource Management Configuration
# ============================================================================

RESOURCE_CONFIG = {
    'max_memory_gb': None,      # Maximum memory usage (None = no limit)
    'gpu_enabled': False,       # Enable GPU acceleration (if available)
    'gpu_device_id': 0,         # GPU device ID
    'thread_count': -1          # Number of threads (-1 = auto)
}

# ============================================================================
# Experiment Tracking Configuration
# ============================================================================

EXPERIMENT_CONFIG = {
    'track_experiments': True,  # Enable experiment tracking
    'experiment_name_format': '{model}_{timestamp}',  # Format for experiment names
    'include_timestamp': True,  # Add timestamp to experiment name
    'tags': [],                # Tags for experiment organization
    'description': ''          # Experiment description
}

# ============================================================================
# Utility Functions
# ============================================================================

def get_training_config(config_name: str = 'default') -> Dict[str, Any]:
    """
    Get a complete training configuration.
    
    Args:
        config_name: Name of configuration preset
                    'default' - Balanced settings
                    'fast' - Quick training for testing
                    'production' - Full optimization for production
                    
    Returns:
        Complete configuration dictionary
    """
    configs = {
        'default': {
            'data_split': DATA_SPLIT_CONFIG.copy(),
            'preprocessing': PREPROCESSING_CONFIG.copy(),
            'training': TRAINING_CONFIG.copy(),
            'optuna': OPTUNA_CONFIG.copy(),
            'metrics': METRICS_CONFIG.copy(),
            'save': SAVE_CONFIG.copy(),
            'cv': CV_CONFIG.copy(),
            'early_stopping': EARLY_STOPPING_CONFIG.copy(),
            'logging': LOGGING_CONFIG.copy(),
            'resource': RESOURCE_CONFIG.copy(),
            'experiment': EXPERIMENT_CONFIG.copy()
        },
        'fast': {
            'data_split': {**DATA_SPLIT_CONFIG, 'test_size': 0.3},
            'preprocessing': {**PREPROCESSING_CONFIG, 'scale_target': False},
            'training': {**TRAINING_CONFIG, 'use_optuna': False, 'n_trials': 10},
            'optuna': {**OPTUNA_CONFIG, 'n_startup_trials': 5},
            'metrics': METRICS_CONFIG.copy(),
            'save': {**SAVE_CONFIG, 'save_predictions': False},
            'cv': {**CV_CONFIG, 'n_folds': 3},
            'early_stopping': {**EARLY_STOPPING_CONFIG, 'rounds': 20},
            'logging': {**LOGGING_CONFIG, 'log_to_file': False},
            'resource': RESOURCE_CONFIG.copy(),
            'experiment': EXPERIMENT_CONFIG.copy()
        },
        'production': {
            'data_split': DATA_SPLIT_CONFIG.copy(),
            'preprocessing': PREPROCESSING_CONFIG.copy(),
            'training': {**TRAINING_CONFIG, 'use_optuna': True, 'n_trials': 100, 'timeout': 7200},
            'optuna': {**OPTUNA_CONFIG, 'n_startup_trials': 20, 'storage': 'sqlite:///optuna_production.db'},
            'metrics': METRICS_CONFIG.copy(),
            'save': SAVE_CONFIG.copy(),
            'cv': {**CV_CONFIG, 'n_folds': 10},
            'early_stopping': {**EARLY_STOPPING_CONFIG, 'rounds': 100},
            'logging': LOGGING_CONFIG.copy(),
            'resource': RESOURCE_CONFIG.copy(),
            'experiment': {**EXPERIMENT_CONFIG, 'track_experiments': True}
        }
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(configs.keys())}")
    
    return configs[config_name]


def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a configuration dictionary with new values.
    
    Args:
        base_config: Base configuration dictionary
        updates: Dictionary with updates to apply
        
    Returns:
        Updated configuration dictionary
    """
    import copy
    config = copy.deepcopy(base_config)
    
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            config[key].update(value)
        else:
            config[key] = value
    
    return config
