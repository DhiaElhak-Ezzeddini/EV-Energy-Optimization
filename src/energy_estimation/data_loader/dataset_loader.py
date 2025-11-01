"""
Dataset Loader for EV Energy Consumption

This module provides functionality to load, split, and prepare the EV energy
consumption dataset for model training and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.CONFIG.energy_features import (
    ENERGY_CONSUMPTION_KWH,
    IDENTIFIER_FEATURES,
    ALL_INPUT_FEATURES
)
from src.energy_estimation.data_utils.preprocessor import Preprocessor


class DatasetLoader:
    """
    Dataset loader for EV energy consumption data.
    
    Handles:
    - Loading CSV data
    - Train/validation/test splitting
    - Feature and target separation
    - Data preprocessing integration
    - Data validation
    
    Attributes:
        data_path (str): Path to the CSV dataset
        target_feature (str): Name of the target variable
        test_size (float): Proportion of data for test set
        val_size (float): Proportion of training data for validation set
        random_state (int): Random seed for reproducibility
        preprocessor (Optional[Preprocessor]): Data preprocessor
    """
    
    def __init__(
        self,
        data_path: str,
        target_feature: str = ENERGY_CONSUMPTION_KWH,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
        use_preprocessor: bool = True
    ):
        """
        Initialize the dataset loader.
        
        Args:
            data_path (str): Path to the CSV dataset
            target_feature (str): Name of the target variable
            test_size (float): Proportion of data for test set (0.0 to 1.0)
            val_size (float): Proportion of training data for validation set (0.0 to 1.0)
            random_state (int): Random seed for reproducibility
            use_preprocessor (bool): Whether to use the preprocessor
        """
        self.data_path = Path(data_path)
        self.target_feature = target_feature
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.use_preprocessor = use_preprocessor
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {data_path}")
        
        self.preprocessor: Optional[Preprocessor] = None
        self.data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        
    def load_data(self, drop_identifiers: bool = True) -> pd.DataFrame:
        """
        Load the dataset from CSV.
        
        Args:
            drop_identifiers (bool): Whether to drop identifier columns (e.g., Vehicle_ID)
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"Loading data from: {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        
        print(f"Dataset loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Drop identifier columns if requested
        if drop_identifiers:
            identifier_cols = [col for col in IDENTIFIER_FEATURES if col in self.data.columns]
            if identifier_cols:
                self.data = self.data.drop(columns=identifier_cols)
                print(f"Dropped identifier columns: {identifier_cols}")
        
        # Validate target feature
        if self.target_feature not in self.data.columns:
            raise ValueError(f"Target feature '{self.target_feature}' not found in dataset")
        
        # Store feature columns (all except target)
        self.feature_columns = [col for col in self.data.columns if col != self.target_feature]
        
        # Basic data info
        print("\nData Info:")
        print(f"  - Features: {len(self.feature_columns)}")
        print(f"  - Target: {self.target_feature}")
        print(f"  - Missing values: {self.data.isnull().sum().sum()}")
        
        return self.data
    
    def split_data(
        self,
        data: Optional[pd.DataFrame] = None,
        stratify: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data (Optional[pd.DataFrame]): Dataset to split. If None, uses self.data
            stratify (bool): Whether to use stratified splitting (for classification)
            
        Returns:
            Tuple containing:
                - X_train (pd.DataFrame): Training features
                - X_val (pd.DataFrame): Validation features
                - X_test (pd.DataFrame): Test features
                - y_train (pd.Series): Training target
                - y_val (pd.Series): Validation target
                - y_test (pd.Series): Test target
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first.")
            data = self.data
        
        # Separate features and target
        X = data[self.feature_columns]
        y = data[self.target_feature]
        
        # First split: train+val vs test
        stratify_train = y if stratify else None
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_train
        )
        
        # Second split: train vs val
        stratify_val = y_train_val if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=stratify_val
        )
        
        print(f"\nData split:")
        print(f"  - Train: {X_train.shape[0]} samples ({X_train.shape[0]/len(data)*100:.1f}%)")
        print(f"  - Val:   {X_val.shape[0]} samples ({X_val.shape[0]/len(data)*100:.1f}%)")
        print(f"  - Test:  {X_test.shape[0]} samples ({X_test.shape[0]/len(data)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data(
        self,
        apply_preprocessing: bool = True,
        scale_target: bool = False
    ) -> Dict[str, any]:
        """
        Load, split, and preprocess the data in one go.
        
        Args:
            apply_preprocessing (bool): Whether to apply preprocessing
            scale_target (bool): Whether to scale the target variable
            
        Returns:
            Dict containing:
                - 'X_train', 'X_val', 'X_test': Feature DataFrames
                - 'y_train', 'y_val', 'y_test': Target Series
                - 'preprocessor': Fitted preprocessor (if used)
                - 'raw_data': Original unprocessed data splits (if preprocessing applied)
        """
        # Load data
        self.load_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        
        result = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        # Apply preprocessing if requested
        if apply_preprocessing and self.use_preprocessor:
            print("\nApplying preprocessing...")
            
            # Store raw data before preprocessing
            result['raw_data'] = {
                'X_train': X_train.copy(),
                'X_val': X_val.copy(),
                'X_test': X_test.copy(),
                'y_train': y_train.copy(),
                'y_val': y_val.copy(),
                'y_test': y_test.copy()
            }
            
            # Initialize and fit preprocessor on training data
            self.preprocessor = Preprocessor(scale_target=scale_target)
            X_train, y_train = self.preprocessor.fit_transform(X_train, y_train)
            
            # Transform validation and test data
            X_val, y_val = self.preprocessor.transform(X_val, y_val)
            X_test, y_test = self.preprocessor.transform(X_test, y_test)
            
            result['X_train'] = X_train
            result['X_val'] = X_val
            result['X_test'] = X_test
            result['y_train'] = y_train
            result['y_val'] = y_val
            result['y_test'] = y_test
            result['preprocessor'] = self.preprocessor
            
            print("Preprocessing complete!")
        
        return result
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of the loaded data.
        
        Returns:
            Dict: Summary statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        summary = {
            'shape': self.data.shape,
            'features': self.feature_columns,
            'target': self.target_feature,
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': self.data.dtypes.to_dict(),
            'numerical_stats': self.data.describe().to_dict(),
            'target_stats': {
                'mean': self.data[self.target_feature].mean(),
                'std': self.data[self.target_feature].std(),
                'min': self.data[self.target_feature].min(),
                'max': self.data[self.target_feature].max(),
                'median': self.data[self.target_feature].median()
            }
        }
        
        return summary
    
    def check_data_quality(self) -> Dict[str, any]:
        """
        Perform data quality checks.
        
        Returns:
            Dict: Data quality report
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        quality_report = {
            'missing_values': {},
            'duplicates': 0,
            'infinite_values': {},
            'constant_features': []
        }
        
        # Check missing values
        missing = self.data.isnull().sum()
        quality_report['missing_values'] = {col: int(count) for col, count in missing.items() if count > 0}
        
        # Check duplicates
        quality_report['duplicates'] = int(self.data.duplicated().sum())
        
        # Check for infinite values in numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            inf_count = np.isinf(self.data[col]).sum()
            if inf_count > 0:
                quality_report['infinite_values'][col] = int(inf_count)
        
        # Check for constant features
        for col in self.data.columns:
            if self.data[col].nunique() == 1:
                quality_report['constant_features'].append(col)
        
        # Print summary
        print("\n=== Data Quality Report ===")
        print(f"Missing values: {sum(quality_report['missing_values'].values())} total")
        print(f"Duplicate rows: {quality_report['duplicates']}")
        print(f"Infinite values: {sum(quality_report['infinite_values'].values())} total")
        print(f"Constant features: {len(quality_report['constant_features'])}")
        
        return quality_report
    
    def get_preprocessor(self) -> Optional[Preprocessor]:
        """
        Get the fitted preprocessor.
        
        Returns:
            Optional[Preprocessor]: Fitted preprocessor or None
        """
        return self.preprocessor
    
    def __repr__(self) -> str:
        """String representation of the dataset loader."""
        data_status = f"loaded ({self.data.shape})" if self.data is not None else "not loaded"
        return (f"DatasetLoader(data_path='{self.data_path.name}', "
                f"data={data_status}, test_size={self.test_size}, val_size={self.val_size})")
