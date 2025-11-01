"""
XGBoost Model for Energy Estimation

This module implements an XGBoost regressor for EV energy consumption prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import xgboost as xgb
import optuna

from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost model implementation for energy estimation.
    
    Uses extreme gradient boosting with regularization for
    high performance and prevention of overfitting.
    """
    
    def __init__(self, model_name: str = "XGBoost"):
        super().__init__(model_name)
        
    def build(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Build XGBoost model with given parameters."""
        if params is None:
            params = self._get_default_params()
        
        self.best_params = params
        self.model = xgb.XGBRegressor(**params)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        }
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        use_optuna: bool = True,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the XGBoost model."""
        self._validate_data(X_train, y_train)
        
        if use_optuna and X_val is not None and y_val is not None:
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
        """Train without Optuna."""
        params = kwargs.get('params', None)
        self.build(params)
        
        if X_val is not None and y_val is not None:
            # XGBoost 2.x sklearn wrapper: use early_stopping_rounds in set_params
            self.model.set_params(early_stopping_rounds=50)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)
        
        self.training_history['n_estimators'] = self.model.n_estimators
        
        return {'status': 'completed'}
    
    def _train_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int,
        timeout: Optional[int]
    ) -> Dict[str, Any]:
        """Train with Optuna hyperparameter optimization."""
        
        def objective(trial):
            params = {
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
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            
            return rmse
        
        self.study = optuna.create_study(direction='minimize', study_name=f"{self.model_name}_study")
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        self.best_params = self.study.best_params
        self.build(self.best_params)
        
        self.model.set_params(early_stopping_rounds=50)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        self._validate_data(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance."""
        if self.model is None:
            return None
        
        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })
        
        return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
