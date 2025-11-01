"""
Data Validators

This module provides validation functions for data quality checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Check for missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        Dict[str, int]: Dictionary mapping column names to missing value counts
    """
    missing = df.isnull().sum()
    return {col: int(count) for col, count in missing.items() if count > 0}


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Check for duplicate rows in the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        int: Number of duplicate rows
    """
    return int(df.duplicated().sum())


def check_data_types(df: pd.DataFrame, expected_types: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Check data types of columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        expected_types (Optional[Dict[str, str]]): Expected data types
        
    Returns:
        Dict[str, str]: Actual data types of columns
    """
    actual_types = df.dtypes.astype(str).to_dict()
    
    if expected_types:
        mismatches = []
        for col, expected in expected_types.items():
            if col in actual_types and actual_types[col] != expected:
                mismatches.append(f"{col}: expected {expected}, got {actual_types[col]}")
        
        if mismatches:
            print("Data type mismatches found:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
    
    return actual_types


def check_value_ranges(
    df: pd.DataFrame,
    column: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> Tuple[bool, List[int]]:
    """
    Check if values in a column are within expected range.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check
        min_val (Optional[float]): Minimum expected value
        max_val (Optional[float]): Maximum expected value
        
    Returns:
        Tuple[bool, List[int]]: (all_valid, list of invalid indices)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    invalid_indices = []
    
    if min_val is not None:
        invalid_indices.extend(df[df[column] < min_val].index.tolist())
    
    if max_val is not None:
        invalid_indices.extend(df[df[column] > max_val].index.tolist())
    
    invalid_indices = list(set(invalid_indices))  # Remove duplicates
    
    return len(invalid_indices) == 0, invalid_indices


def check_categorical_values(
    df: pd.DataFrame,
    column: str,
    expected_values: Optional[List] = None
) -> Dict[str, any]:
    """
    Check categorical column for unexpected values.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Column name to check
        expected_values (Optional[List]): List of expected categorical values
        
    Returns:
        Dict: Information about categorical values
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    unique_values = df[column].unique().tolist()
    value_counts = df[column].value_counts().to_dict()
    
    result = {
        'unique_values': unique_values,
        'num_unique': len(unique_values),
        'value_counts': value_counts
    }
    
    if expected_values is not None:
        unexpected = set(unique_values) - set(expected_values)
        missing = set(expected_values) - set(unique_values)
        
        result['unexpected_values'] = list(unexpected)
        result['missing_expected_values'] = list(missing)
        result['all_valid'] = len(unexpected) == 0
    
    return result


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    check_empty: bool = True
) -> Dict[str, any]:
    """
    Comprehensive DataFrame validation.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        required_columns (Optional[List[str]]): List of required column names
        check_empty (bool): Whether to check for empty DataFrame
        
    Returns:
        Dict: Validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if DataFrame is empty
    if check_empty and df.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append("DataFrame is empty")
        return validation_result
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Missing required columns: {list(missing_cols)}")
    
    # Check for missing values
    missing = check_missing_values(df)
    if missing:
        validation_result['warnings'].append(f"Found missing values in columns: {list(missing.keys())}")
    
    # Check for duplicates
    duplicates = check_duplicates(df)
    if duplicates > 0:
        validation_result['warnings'].append(f"Found {duplicates} duplicate rows")
    
    # Check for infinite values in numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            validation_result['warnings'].append(f"Found {inf_count} infinite values in column '{col}'")
    
    return validation_result


def print_data_quality_report(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print a comprehensive data quality report.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        name (str): Name of the dataset for the report
    """
    print(f"\n{'='*60}")
    print(f"Data Quality Report: {name}")
    print(f"{'='*60}")
    
    print(f"\nBasic Info:")
    print(f"  - Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nMissing Values:")
    missing = check_missing_values(df)
    if missing:
        for col, count in sorted(missing.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(df) * 100
            print(f"  - {col}: {count} ({pct:.2f}%)")
    else:
        print("  - No missing values found ✓")
    
    print(f"\nDuplicates:")
    duplicates = check_duplicates(df)
    if duplicates > 0:
        print(f"  - Found {duplicates} duplicate rows ({duplicates/len(df)*100:.2f}%)")
    else:
        print("  - No duplicates found ✓")
    
    print(f"\nData Types:")
    dtypes = df.dtypes.value_counts()
    for dtype, count in dtypes.items():
        print(f"  - {dtype}: {count} columns")
    
    print(f"\n{'='*60}\n")
