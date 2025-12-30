"""
Data loading utilities for credit risk prediction project.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv()
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if path.stat().st_size == 0:
        raise ValueError(f"Data file is empty: {file_path}")
    
    try:
        df = pd.read_csv(file_path, **kwargs)
        
        if df.empty:
            raise ValueError(f"No data loaded from file: {file_path}")
            
        print(f"Data loaded successfully from {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {str(e)}")


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    return info


def preview_data(df: pd.DataFrame, n_rows: int = 5) -> pd.DataFrame:
    """
    Preview first n rows of the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        n_rows (int): Number of rows to preview
        
    Returns:
        pd.DataFrame: Preview of the data
    """
    return df.head(n_rows)
