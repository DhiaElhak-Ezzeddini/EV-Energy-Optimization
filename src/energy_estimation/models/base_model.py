"""
Base Model Abstract Class for Energy Estimation Models

This module defines the abstract base class that all energy estimation models
must inherit from. It provides a common interface for building, training,
predicting, and saving models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
import optuna


class BaseModel(ABC):
    """
    Abstract base class for all energy estimation models.
    
    This class defines the common interface that all models (LightGBM, CatBoost,
    XGBoost, and Ensemble models) must implement.
    
    Attributes:
        model_name (str): Name identifier for the model
        model (Any): The underlying model object
        best_params (Dict): Best hyperparameters found during optimization
        study (optuna.Study): Optuna study object for hyperparameter optimization
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the base model.
        
        Args:
            model_name (str): Name identifier for the model
        """
        self.model_name = model_name
        self.model: Optional[Any] = None
        self.best_params: Dict[str, Any] = {}
        self.study: Optional[optuna.Study] = None
        self.training_history: Dict[str, Any] = {}
        
    @abstractmethod
    def build(self, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Build the model with given parameters.
        
        Args:
            params (Optional[Dict[str, Any]]): Model hyperparameters.
                If None, use default parameters.
        """
        pass
    
    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        use_optuna: bool = True,
        n_trials: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (Optional[pd.DataFrame]): Validation features
            y_val (Optional[pd.Series]): Validation target
            use_optuna (bool): Whether to use Optuna for hyperparameter optimization
            n_trials (int): Number of Optuna trials
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, Any]: Training metrics and information
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X (pd.DataFrame): Features to predict on
            
        Returns:
            np.ndarray: Predictions
            
        Raises:
            ValueError: If model has not been trained yet
        """
        pass
    
    def save_model(
        self,
        save_dir: str,
        include_study: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save the trained model to disk.
        
        Args:
            save_dir (str): Directory to save the model
            include_study (bool): Whether to save the Optuna study
            metadata (Optional[Dict[str, Any]]): Additional metadata to save
            
        Returns:
            str: Path where the model was saved
            
        Raises:
            ValueError: If model has not been trained yet
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Cannot save.")
        
        # Create save directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.model_name}_{timestamp}"
        
        # Save the model
        model_path = save_path / f"{model_filename}.pkl"
        joblib.dump(self.model, model_path)
        
        # Prepare metadata
        save_metadata = {
            "model_name": self.model_name,
            "timestamp": timestamp,
            "best_params": self.best_params,
            "training_history": self.training_history,
        }
        
        if metadata:
            save_metadata.update(metadata)
        
        # Save metadata
        metadata_path = save_path / f"{model_filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(save_metadata, f, indent=4, default=str)
        
        # Save Optuna study if requested
        if include_study and self.study is not None:
            study_path = save_path / f"{model_filename}_study.pkl"
            joblib.dump(self.study, study_path)
        
        print(f"Model saved successfully to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str, load_metadata: bool = True) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model file
            load_metadata (bool): Whether to load metadata
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_file = Path(model_path)
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model
        self.model = joblib.load(model_file)
        
        # Load metadata if requested
        if load_metadata:
            metadata_path = model_file.parent / f"{model_file.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.best_params = metadata.get("best_params", {})
                    self.training_history = metadata.get("training_history", {})
            
            # Try to load study
            study_path = model_file.parent / f"{model_file.stem}_study.pkl"
            if study_path.exists():
                self.study = joblib.load(study_path)
        
        print(f"Model loaded successfully from: {model_path}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with feature names and importance scores,
                or None if not available
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # This will be overridden by child classes to provide specific implementations
        return None
    
    def _validate_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Validate input data format and types.
        
        Args:
            X (pd.DataFrame): Features
            y (Optional[pd.Series]): Target
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Validated data
            
        Raises:
            ValueError: If data format is invalid
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("X cannot be empty")
        
        if y is not None:
            if not isinstance(y, (pd.Series, np.ndarray)):
                raise ValueError("y must be a pandas Series or numpy array")
            
            if len(X) != len(y):
                raise ValueError("X and y must have the same length")
        
        return X, y
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (f"{self.__class__.__name__}(model_name='{self.model_name}', "
                f"trained={self.model is not None})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "trained" if self.model is not None else "not trained"
        return f"{self.model_name} ({status})"
