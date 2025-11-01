"""
Model Configuration Settings.

Contains default hyperparameters and Optuna search spaces for all models.
"""

from typing import Dict, Any, Callable
import optuna

# ============================================================================
# LightGBM Configuration
# ============================================================================

LIGHTGBM_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': -1,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'random_state': 42,
    'verbose': -1
}

def LIGHTGBM_OPTUNA_SPACE(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define Optuna search space for LightGBM hyperparameters.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of hyperparameters for the trial
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'random_state': 42,
        'verbose': -1
    }


# ============================================================================
# CatBoost Configuration
# ============================================================================

CATBOOST_DEFAULT_PARAMS = {
    'iterations': 100,
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'colsample_bylevel': 0.8,
    'random_state': 42,
    'verbose': False
}

def CATBOOST_OPTUNA_SPACE(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define Optuna search space for CatBoost hyperparameters.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of hyperparameters for the trial
    """
    return {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_state': 42,
        'verbose': False
    }


# ============================================================================
# XGBoost Configuration
# ============================================================================

XGBOOST_DEFAULT_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'early_stopping_rounds': 50,
    'random_state': 42,
    'verbosity': 0
}

def XGBOOST_OPTUNA_SPACE(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define Optuna search space for XGBoost hyperparameters.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of hyperparameters for the trial
    """
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'early_stopping_rounds': 50,
        'random_state': 42,
        'verbosity': 0
    }


# ============================================================================
# Stacking Ensemble Configuration
# ============================================================================

STACKING_DEFAULT_PARAMS = {
    'use_passthrough': False,
    'final_estimator_alpha': 1.0,
    'cv_folds': 5,
    'lightgbm_params': LIGHTGBM_DEFAULT_PARAMS.copy(),
    'catboost_params': CATBOOST_DEFAULT_PARAMS.copy(),
    'xgboost_params': XGBOOST_DEFAULT_PARAMS.copy()
}

def STACKING_OPTUNA_SPACE(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define Optuna search space for Stacking Ensemble hyperparameters.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Dictionary of hyperparameters for the trial
    """
    # Stacking-specific parameters
    use_passthrough = trial.suggest_categorical('use_passthrough', [True, False])
    final_alpha = trial.suggest_float('final_estimator_alpha', 0.1, 10.0, log=True)
    cv_folds = trial.suggest_int('cv_folds', 3, 10)
    
    # LightGBM parameters with prefix
    lightgbm_params = {
        'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
        'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 100),
        'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    
    # CatBoost parameters with prefix
    catboost_params = {
        'iterations': trial.suggest_int('cb_iterations', 50, 300),
        'learning_rate': trial.suggest_float('cb_learning_rate', 0.01, 0.3, log=True),
        'depth': trial.suggest_int('cb_depth', 3, 10),
        'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1, 10),
        'subsample': trial.suggest_float('cb_subsample', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('cb_colsample_bylevel', 0.6, 1.0),
        'random_state': 42,
        'verbose': False
    }
    
    # XGBoost parameters with prefix
    xgboost_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('xgb_gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbosity': 0
    }
    
    return {
        'use_passthrough': use_passthrough,
        'final_estimator_alpha': final_alpha,
        'cv_folds': cv_folds,
        'lightgbm_params': lightgbm_params,
        'catboost_params': catboost_params,
        'xgboost_params': xgboost_params
    }


# ============================================================================
# Utility Functions
# ============================================================================

def get_model_config(model_name: str, config_type: str = 'default') -> Dict[str, Any]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model ('lightgbm', 'catboost', 'xgboost', 'stacking')
        config_type: Type of config ('default' or 'optuna')
        
    Returns:
        Configuration dictionary or Optuna space function
        
    Raises:
        ValueError: If model_name or config_type is invalid
    """
    model_name = model_name.lower()
    config_type = config_type.lower()
    
    configs = {
        'lightgbm': {
            'default': LIGHTGBM_DEFAULT_PARAMS,
            'optuna': LIGHTGBM_OPTUNA_SPACE
        },
        'catboost': {
            'default': CATBOOST_DEFAULT_PARAMS,
            'optuna': CATBOOST_OPTUNA_SPACE
        },
        'xgboost': {
            'default': XGBOOST_DEFAULT_PARAMS,
            'optuna': XGBOOST_OPTUNA_SPACE
        },
        'stacking': {
            'default': STACKING_DEFAULT_PARAMS,
            'optuna': STACKING_OPTUNA_SPACE
        }
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(configs.keys())}")
    
    if config_type not in configs[model_name]:
        raise ValueError(f"Unknown config type: {config_type}. Choose from: ['default', 'optuna']")
    
    return configs[model_name][config_type]


# ============================================================================
# Production-Ready Configurations (Optimized)
# ============================================================================

# These are example optimized configurations from Optuna runs
# Replace with your own optimized values after running hyperparameter search

LIGHTGBM_OPTIMIZED_PARAMS = {
    'n_estimators': 250,
    'learning_rate': 0.05,
    'max_depth': 8,
    'num_leaves': 80,
    'subsample': 0.85,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 20,
    'random_state': 42,
    'verbose': -1
}

CATBOOST_OPTIMIZED_PARAMS = {
    'iterations': 300,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 5,
    'subsample': 0.9,
    'colsample_bylevel': 0.85,
    'min_data_in_leaf': 10,
    'random_state': 42,
    'verbose': False
}

XGBOOST_OPTIMIZED_PARAMS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 7,
    'min_child_weight': 3,
    'subsample': 0.85,
    'colsample_bytree': 0.9,
    'gamma': 0.1,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'early_stopping_rounds': 50,
    'random_state': 42,
    'verbosity': 0
}
