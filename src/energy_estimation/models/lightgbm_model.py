"""
LightGBM Model for Energy Estimation

This module implements a LightGBM regressor for EV energy consumption prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import lightgbm as lgb
import optuna

from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM model implementation for energy estimation.
    
    Uses gradient boosting with leaf-wise tree growth for fast training
    and high accuracy on tabular data.
    """
    
    def __init__(self, model_name: str = "LightGBM"):
        super().__init__(model_name)
        
    def build(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Build LightGBM model with given parameters."""
        if params is None:
            params = self._get_default_params()
        
        self.best_params = params
        self.model = lgb.LGBMRegressor(**params)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': -1,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42,
            'verbose': -1
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
        """Train the LightGBM model."""
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
        
        eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)] if eval_set else None
        )
        
        self.training_history['n_estimators'] = self.model.n_estimators_
        
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
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            
            return rmse
        
        self.study = optuna.create_study(direction='minimize', study_name=f"{self.model_name}_study")
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        
        self.best_params = self.study.best_params
        self.build(self.best_params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
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
        
        importance_df = pd.DataFrame({
            'feature': self.model.feature_name_,
            'importance': self.model.feature_importances_
        })
        
        return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
