"""
CatBoost Model for Energy Estimation

This module implements a CatBoost regressor for EV energy consumption prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from catboost import CatBoostRegressor, Pool
import optuna

from .base_model import BaseModel


class CatBoostModel(BaseModel):
    """
    CatBoost model implementation for energy estimation.
    
    Uses ordered boosting and efficient categorical feature handling
    for robust predictions on mixed data types.
    """
    
    def __init__(self, model_name: str = "CatBoost"):
        super().__init__(model_name)
        
    def build(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Build CatBoost model with given parameters."""
        if params is None:
            params = self._get_default_params()
        
        self.best_params = params
        self.model = CatBoostRegressor(**params)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'random_state': 42,
            'verbose': False
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
        """Train the CatBoost model."""
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
        
        eval_set = Pool(X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50 if eval_set else None,
            verbose=False
        )
        
        self.training_history['n_iterations'] = self.model.tree_count_
        
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
                'iterations': trial.suggest_int('iterations', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                'random_state': 42,
                'verbose': False
            }
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=Pool(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
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
            eval_set=Pool(X_val, y_val),
            early_stopping_rounds=50,
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
        
        importance_df = pd.DataFrame({
            'feature': self.model.feature_names_,
            'importance': self.model.feature_importances_
        })
        
        return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
