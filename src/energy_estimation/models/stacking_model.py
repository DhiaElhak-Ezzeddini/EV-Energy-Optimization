"""
Stacking Ensemble Model for Energy Consumption Prediction.

Combines LightGBM, CatBoost, and XGBoost using StackingRegressor.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
import json
import optuna

from .base_model import BaseModel
from .lightgbm_model import LightGBMModel
from .catboost_model import CatBoostModel
from .xgboost_model import XGBoostModel


class StackingEnsembleModel(BaseModel):
    """
    Stacking Ensemble combining LightGBM, CatBoost, and XGBoost.
    
    Uses StackingRegressor with Ridge regression as meta-learner.
    """
    
    def __init__(
        self,
        model_name: str = "StackingEnsemble",
        use_passthrough: bool = False,
        final_estimator_alpha: float = 1.0,
        cv_folds: int = 5
    ):
        """
        Initialize Stacking Ensemble Model.
        
        Args:
            model_name: Name of the model
            use_passthrough: Whether to pass original features to meta-learner
            final_estimator_alpha: Ridge regularization strength
            cv_folds: Number of cross-validation folds for stacking
        """
        super().__init__(model_name)
        self.use_passthrough = use_passthrough
        self.final_estimator_alpha = final_estimator_alpha
        self.cv_folds = cv_folds
        
        # Base models
        self.lightgbm_model = None
        self.catboost_model = None
        self.xgboost_model = None
        
        # Stacking ensemble
        self.model = None
        
    def build(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Build stacking ensemble with base models and meta-learner.
        
        Args:
            params: Dictionary containing:
                - lightgbm_params: Parameters for LightGBM
                - catboost_params: Parameters for CatBoost
                - xgboost_params: Parameters for XGBoost
                - final_estimator_alpha: Ridge alpha (optional)
                - use_passthrough: Whether to use passthrough (optional)
                - cv_folds: Number of CV folds (optional)
        """
        if params is None:
            params = {}
        
        # Extract base model parameters
        lightgbm_params = params.get('lightgbm_params', None)
        catboost_params = params.get('catboost_params', None)
        xgboost_params = params.get('xgboost_params', None)
        
        # Update stacking parameters if provided
        self.final_estimator_alpha = params.get('final_estimator_alpha', self.final_estimator_alpha)
        self.use_passthrough = params.get('use_passthrough', self.use_passthrough)
        self.cv_folds = params.get('cv_folds', self.cv_folds)
        
        # Initialize base models
        self.lightgbm_model = LightGBMModel(model_name="LightGBM_Base")
        self.catboost_model = CatBoostModel(model_name="CatBoost_Base")
        self.xgboost_model = XGBoostModel(model_name="XGBoost_Base")
        
        # Build base models with their parameters
        self.lightgbm_model.build(lightgbm_params)
        self.catboost_model.build(catboost_params)
        self.xgboost_model.build(xgboost_params)
        
        # Create estimators list for stacking
        estimators = [
            ('lightgbm', self.lightgbm_model.model),
            ('catboost', self.catboost_model.model),
            ('xgboost', self.xgboost_model.model)
        ]
        
        # Create final estimator (meta-learner)
        final_estimator = Ridge(alpha=self.final_estimator_alpha)
        
        # Create stacking regressor
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=self.cv_folds,
            passthrough=self.use_passthrough,
            n_jobs=-1,
            verbose=0
        )
        
        self.is_trained = False
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        use_optuna: bool = False,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the stacking ensemble.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (not used in stacking, but kept for API consistency)
            y_val: Validation target (not used in stacking, but kept for API consistency)
            use_optuna: Whether to use Optuna for hyperparameter optimization
            n_trials: Number of Optuna trials
            timeout: Timeout for Optuna optimization
            **kwargs: Additional training parameters
        
        Returns:
            Dictionary with training results
        """
        if use_optuna:
            return self._train_with_optuna(X_train, y_train, X_val, y_val, n_trials, timeout)
        else:
            return self._train_simple(X_train, y_train, X_val, y_val, **kwargs)
    
    def _train_simple(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train without Optuna optimization."""
        params = kwargs.get('params', None)
        self.build(params)
        
        # Stacking regressor handles cross-validation internally
        # No need to manually pass validation set
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        self.training_history['stacking_params'] = {
            'cv_folds': self.cv_folds,
            'use_passthrough': self.use_passthrough,
            'final_estimator_alpha': self.final_estimator_alpha
        }
        
        return {
            'cv_folds': self.cv_folds,
            'use_passthrough': self.use_passthrough,
            'final_estimator_alpha': self.final_estimator_alpha
        }
    
    def _train_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int,
        timeout: Optional[int]
    ) -> Dict[str, Any]:
        """
        Train with Optuna hyperparameter optimization.
        
        Optimizes:
        - Base model hyperparameters
        - Meta-learner regularization
        - Passthrough feature usage
        """
        def objective(trial):
            # Suggest stacking parameters
            use_passthrough = trial.suggest_categorical('use_passthrough', [True, False])
            final_alpha = trial.suggest_float('final_estimator_alpha', 0.1, 10.0, log=True)
            cv_folds = trial.suggest_int('cv_folds', 3, 10)
            
            # Suggest LightGBM parameters
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
            
            # Suggest CatBoost parameters
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
            
            # Suggest XGBoost parameters
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
            
            # Build and train model
            params = {
                'lightgbm_params': lightgbm_params,
                'catboost_params': catboost_params,
                'xgboost_params': xgboost_params,
                'final_estimator_alpha': final_alpha,
                'use_passthrough': use_passthrough,
                'cv_folds': cv_folds
            }
            
            self.build(params)
            self.model.fit(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            
            return rmse
        
        # Create and run study
        self.study = optuna.create_study(direction='minimize', study_name=f"{self.model_name}_study")
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        # Train final model with best parameters
        self.best_params = self.study.best_params
        
        # Reconstruct parameters from best trial
        lightgbm_params = {k.replace('lgb_', ''): v for k, v in self.best_params.items() if k.startswith('lgb_')}
        catboost_params = {k.replace('cb_', ''): v for k, v in self.best_params.items() if k.startswith('cb_')}
        xgboost_params = {k.replace('xgb_', ''): v for k, v in self.best_params.items() if k.startswith('xgb_')}
        
        final_params = {
            'lightgbm_params': lightgbm_params,
            'catboost_params': catboost_params,
            'xgboost_params': xgboost_params,
            'final_estimator_alpha': self.best_params['final_estimator_alpha'],
            'use_passthrough': self.best_params['use_passthrough'],
            'cv_folds': self.best_params['cv_folds']
        }
        
        self.build(final_params)
        self.model.fit(X_train, y_train)
        
        self.is_trained = True
        self.training_history['best_params'] = self.best_params
        self.training_history['best_score'] = self.study.best_value
        self.training_history['n_trials'] = len(self.study.trials)
        
        return {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from base models.
        
        Returns average feature importance across all base models.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance.")
        
        # Get importance from each fitted base estimator
        importances_list = []
        
        if hasattr(self.model, 'named_estimators_'):
            for name, estimator in self.model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    importance = estimator.feature_importances_
                    importances_list.append(importance)
        
        if not importances_list:
            return {}
        
        # Average importance across models
        avg_importance = np.mean(importances_list, axis=0)
        
        # Get feature names (assume same order as input X)
        # Use generic feature names since we don't have original feature names here
        feature_names = [f"feature_{i}" for i in range(len(avg_importance))]
        
        return dict(zip(feature_names, avg_importance))
    
    def get_base_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from each base model.
        
        Useful for analyzing individual model contributions.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        # Get predictions from fitted base estimators in the stacking regressor
        # estimators_ is a list of fitted estimators (no names)
        # named_estimators_ is a dictionary mapping names to estimators
        base_preds = {}
        
        if hasattr(self.model, 'named_estimators_'):
            for name, estimator in self.model.named_estimators_.items():
                base_preds[name] = estimator.predict(X)
        
        # Add ensemble prediction
        base_preds['ensemble'] = self.predict(X)
        
        return base_preds
    
    def get_meta_learner_weights(self) -> Dict[str, float]:
        """
        Get the coefficients/weights learned by the meta-learner.
        
        Returns the Ridge regression coefficients for each base model.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting weights.")
        
        weights = self.model.final_estimator_.coef_
        
        weight_dict = {
            'lightgbm': weights[0],
            'catboost': weights[1],
            'xgboost': weights[2]
        }
        
        if self.use_passthrough:
            weight_dict['passthrough_features'] = weights[3:]
        
        return weight_dict
    
    def save_model(
        self,
        save_dir: str,
        include_study: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save the stacking ensemble model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for unique filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.model_name}_{timestamp}.pkl"
        filepath = save_path / model_filename
        
        # Save entire stacking model
        joblib.dump(self.model, filepath)
        
        # Save metadata
        model_metadata = {
            'model_name': self.model_name,
            'model_type': 'StackingEnsemble',
            'cv_folds': self.cv_folds,
            'use_passthrough': self.use_passthrough,
            'final_estimator_alpha': self.final_estimator_alpha,
            'training_history': self.training_history,
            'meta_learner_weights': self.get_meta_learner_weights()
        }
        
        # Merge with additional metadata if provided
        if metadata:
            model_metadata.update(metadata)
        
        metadata_path = save_path / f"{self.model_name}_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2, default=str)
        
        # Save Optuna study if exists and requested
        if include_study and self.study is not None:
            study_path = save_path / f"{self.model_name}_{timestamp}_study.pkl"
            joblib.dump(self.study, study_path)
        
        return str(filepath)
    
    def load_model(self, filepath: Path) -> None:
        """Load a saved stacking ensemble model."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model
        self.model = joblib.load(filepath)
        self.is_trained = True
        
        # Load metadata
        metadata_path = filepath.parent / f"{filepath.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_name = metadata.get('model_name', self.model_name)
                self.cv_folds = metadata.get('cv_folds', self.cv_folds)
                self.use_passthrough = metadata.get('use_passthrough', self.use_passthrough)
                self.final_estimator_alpha = metadata.get('final_estimator_alpha', self.final_estimator_alpha)
                self.training_history = metadata.get('training_history', {})
        
        # Base models are already part of the stacking regressor, no need to extract them
