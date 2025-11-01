"""
Data Preprocessor for EV Energy Consumption Dataset

This module provides preprocessing functionality for the EV energy consumption data,
including encoding categorical features and scaling numerical features.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.CONFIG.energy_features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    IDENTIFIER_FEATURES,
    ENERGY_CONSUMPTION_KWH
)


class Preprocessor:
    """
    Preprocessor for EV energy consumption data.
    
    Handles:
    - Categorical feature encoding (Label Encoding)
    - Numerical feature scaling (Standard Scaling)
    - Missing value handling
    - Feature validation
    
    Attributes:
        categorical_features (List[str]): List of categorical feature names
        numerical_features (List[str]): List of numerical feature names
        target_feature (str): Target variable name
        scaler (StandardScaler): Scaler for numerical features
        label_encoders (Dict[str, LabelEncoder]): Label encoders for categorical features
        is_fitted (bool): Whether the preprocessor has been fitted
    """
    
    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        target_feature: str = ENERGY_CONSUMPTION_KWH,
        scale_target: bool = False
    ):
        """
        Initialize the preprocessor.
        
        Args:
            categorical_features (Optional[List[str]]): Categorical features to encode.
                If None, uses default from energy_features.py
            numerical_features (Optional[List[str]]): Numerical features to scale.
                If None, uses default from energy_features.py (excluding target and identifiers)
            target_feature (str): Name of the target variable
            scale_target (bool): Whether to scale the target variable
        """
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES.copy()
        
        # Use all numerical features except target and identifiers
        if numerical_features is None:
            self.numerical_features = [
                f for f in NUMERICAL_FEATURES 
                if f not in IDENTIFIER_FEATURES and f != target_feature
            ]
        else:
            self.numerical_features = numerical_features.copy()
        
        self.target_feature = target_feature
        self.scale_target = scale_target
        
        # Initialize scalers and encoders
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler() if scale_target else None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Store encoding mappings for interpretability
        self.encoding_mappings: Dict[str, Dict] = {}
        
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Preprocessor':
        """
        Fit the preprocessor on the training data.
        
        Args:
            X (pd.DataFrame): Features DataFrame
            y (Optional[pd.Series]): Target variable
            
        Returns:
            Preprocessor: Self for method chaining
        """
        X = X.copy()
        
        # Validate features
        self._validate_features(X)
        
        # Fit label encoders for categorical features
        for feature in self.categorical_features:
            if feature in X.columns:
                encoder = LabelEncoder()
                encoder.fit(X[feature].astype(str))
                self.label_encoders[feature] = encoder
                
                # Store mapping for interpretability
                self.encoding_mappings[feature] = {
                    str(val): int(code) 
                    for val, code in zip(encoder.classes_, encoder.transform(encoder.classes_))
                }
        
        # Fit scaler for numerical features
        numerical_cols = [col for col in self.numerical_features if col in X.columns]
        if numerical_cols:
            self.scaler.fit(X[numerical_cols])
        
        # Fit target scaler if needed
        if self.scale_target and y is not None:
            self.target_scaler.fit(y.values.reshape(-1, 1))
        
        self.is_fitted = True
        print(f"Preprocessor fitted successfully!")
        print(f"  - Categorical features encoded: {len(self.label_encoders)}")
        print(f"  - Numerical features scaled: {len(numerical_cols)}")
        
        return self
    
    def transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform the data using fitted preprocessor.
        
        Args:
            X (pd.DataFrame): Features DataFrame
            y (Optional[pd.Series]): Target variable
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Transformed features and target
            
        Raises:
            ValueError: If preprocessor hasn't been fitted
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit() first.")
        
        X = X.copy()
        
        # Encode categorical features
        for feature in self.categorical_features:
            if feature in X.columns:
                # Handle unseen categories by using the most frequent class
                X[feature] = X[feature].astype(str)
                unknown_mask = ~X[feature].isin(self.label_encoders[feature].classes_)
                
                if unknown_mask.any():
                    # Use the first class as default for unknown values
                    X.loc[unknown_mask, feature] = self.label_encoders[feature].classes_[0]
                
                X[feature] = self.label_encoders[feature].transform(X[feature])
        
        # Scale numerical features
        numerical_cols = [col for col in self.numerical_features if col in X.columns]
        if numerical_cols:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        # Scale target if needed
        if y is not None and self.scale_target:
            y = pd.Series(
                self.target_scaler.transform(y.values.reshape(-1, 1)).flatten(),
                index=y.index,
                name=y.name
            )
        
        return X, y
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Fit the preprocessor and transform the data in one step.
        
        Args:
            X (pd.DataFrame): Features DataFrame
            y (Optional[pd.Series]): Target variable
            
        Returns:
            Tuple[pd.DataFrame, Optional[pd.Series]]: Transformed features and target
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform the target variable (useful for predictions).
        
        Args:
            y (np.ndarray): Scaled target values
            
        Returns:
            np.ndarray: Original scale target values
        """
        if self.scale_target and self.target_scaler is not None:
            return self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten()
        return y
    
    def _validate_features(self, X: pd.DataFrame) -> None:
        """
        Validate that required features are present in the DataFrame.
        
        Args:
            X (pd.DataFrame): Features DataFrame
            
        Raises:
            ValueError: If required features are missing
        """
        missing_categorical = [f for f in self.categorical_features if f not in X.columns]
        missing_numerical = [f for f in self.numerical_features if f not in X.columns]
        
        if missing_categorical:
            print(f"Warning: Missing categorical features: {missing_categorical}")
        
        if missing_numerical:
            print(f"Warning: Missing numerical features: {missing_numerical}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of all features that will be processed.
        
        Returns:
            List[str]: List of feature names
        """
        return self.categorical_features + self.numerical_features
    
    def save(self, save_path: str) -> None:
        """
        Save the preprocessor to disk.
        
        Args:
            save_path (str): Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor. Call fit() first.")
        
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        preprocessor_data = {
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'label_encoders': self.label_encoders,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'target_feature': self.target_feature,
            'scale_target': self.scale_target,
            'encoding_mappings': self.encoding_mappings,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(preprocessor_data, save_path)
        
        # Save encoding mappings as JSON for easy reference
        mappings_path = Path(save_path).parent / f"{Path(save_path).stem}_mappings.json"
        with open(mappings_path, 'w') as f:
            json.dump(self.encoding_mappings, f, indent=4)
        
        print(f"Preprocessor saved to: {save_path}")
        print(f"Encoding mappings saved to: {mappings_path}")
    
    def load(self, load_path: str) -> 'Preprocessor':
        """
        Load a preprocessor from disk.
        
        Args:
            load_path (str): Path to the saved preprocessor
            
        Returns:
            Preprocessor: Self for method chaining
        """
        if not Path(load_path).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {load_path}")
        
        preprocessor_data = joblib.load(load_path)
        
        self.scaler = preprocessor_data['scaler']
        self.target_scaler = preprocessor_data['target_scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.categorical_features = preprocessor_data['categorical_features']
        self.numerical_features = preprocessor_data['numerical_features']
        self.target_feature = preprocessor_data['target_feature']
        self.scale_target = preprocessor_data['scale_target']
        self.encoding_mappings = preprocessor_data['encoding_mappings']
        self.is_fitted = preprocessor_data['is_fitted']
        
        print(f"Preprocessor loaded from: {load_path}")
        
        return self
    
    def get_encoding_mapping(self, feature: str) -> Optional[Dict]:
        """
        Get the encoding mapping for a categorical feature.
        
        Args:
            feature (str): Feature name
            
        Returns:
            Optional[Dict]: Mapping dictionary or None if not found
        """
        return self.encoding_mappings.get(feature)
    
    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"Preprocessor(categorical={len(self.categorical_features)}, "
                f"numerical={len(self.numerical_features)}, {status})")
