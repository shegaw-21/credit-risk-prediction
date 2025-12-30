"""
Data cleaning utilities for credit risk prediction project.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class DataCleaner:
    """
    A class for cleaning and preprocessing data for credit risk prediction.
    """
    
    def __init__(self):
        self.cleaning_log = []
        
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with duplicates removed
        """
        initial_shape = df.shape
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_shape[0] - df_cleaned.shape[0]
        
        self.cleaning_log.append(f"Removed {duplicates_removed} duplicate rows")
        
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy for handling missing values
                          ('auto', 'mean', 'median', 'mode', 'drop')
                          
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        df_cleaned = df.copy()
        missing_info = df.isnull().sum()
        
        for column in df.columns:
            missing_count = missing_info[column]
            
            if missing_count > 0:
                if strategy == 'auto':
                    # Choose strategy based on data type and missing percentage
                    missing_percent = (missing_count / len(df)) * 100
                    
                    if missing_percent > 50:
                        # Drop column if more than 50% missing
                        df_cleaned = df_cleaned.drop(column, axis=1)
                        self.cleaning_log.append(f"Dropped column {column} ({missing_percent:.1f}% missing)")
                        
                    elif df[column].dtype in ['int64', 'float64']:
                        # Use median for numerical columns
                        median_val = df[column].median()
                        df_cleaned[column] = df_cleaned[column].fillna(median_val)
                        self.cleaning_log.append(f"Filled {missing_count} missing values in {column} with median")
                        
                    else:
                        # Use mode for categorical columns
                        mode_val = df[column].mode()[0]
                        df_cleaned[column] = df_cleaned[column].fillna(mode_val)
                        self.cleaning_log.append(f"Filled {missing_count} missing values in {column} with mode")
                        
                elif strategy == 'mean':
                    mean_val = df[column].mean()
                    df_cleaned[column] = df_cleaned[column].fillna(mean_val)
                    self.cleaning_log.append(f"Filled {missing_count} missing values in {column} with mean")
                    
                elif strategy == 'median':
                    median_val = df[column].median()
                    df_cleaned[column] = df_cleaned[column].fillna(median_val)
                    self.cleaning_log.append(f"Filled {missing_count} missing values in {column} with median")
                    
                elif strategy == 'mode':
                    mode_val = df[column].mode()[0]
                    df_cleaned[column] = df_cleaned[column].fillna(mode_val)
                    self.cleaning_log.append(f"Filled {missing_count} missing values in {column} with mode")
                    
                elif strategy == 'drop':
                    df_cleaned = df_cleaned.dropna(subset=[column])
                    self.cleaning_log.append(f"Dropped {missing_count} rows with missing values in {column}")
        
        return df_cleaned
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle outliers in numerical columns.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Method for outlier detection ('iqr', 'zscore', 'capping')
            columns (List[str]): Columns to process (None for all numerical)
            
        Returns:
            pd.DataFrame: Dataframe with outliers handled
        """
        df_cleaned = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in columns:
            if column not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                
                if outliers > 0:
                    # Cap outliers
                    df_cleaned[column] = np.where(df_cleaned[column] < lower_bound, lower_bound, df_cleaned[column])
                    df_cleaned[column] = np.where(df_cleaned[column] > upper_bound, upper_bound, df_cleaned[column])
                    self.cleaning_log.append(f"Capped {outliers} outliers in {column} using IQR method")
                    
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = (z_scores > 3).sum()
                
                if outliers > 0:
                    # Cap outliers at 3 standard deviations
                    lower_bound = df[column].mean() - 3 * df[column].std()
                    upper_bound = df[column].mean() + 3 * df[column].std()
                    
                    df_cleaned[column] = np.where(df_cleaned[column] < lower_bound, lower_bound, df_cleaned[column])
                    df_cleaned[column] = np.where(df_cleaned[column] > upper_bound, upper_bound, df_cleaned[column])
                    self.cleaning_log.append(f"Capped {outliers} outliers in {column} using Z-score method")
                    
            elif method == 'capping':
                # Custom capping based on domain knowledge
                if column == 'person_age':
                    df_cleaned[column] = np.where(df_cleaned[column] > 100, 100, df_cleaned[column])
                    self.cleaning_log.append(f"Capped person_age at 100")
                elif column == 'person_emp_length':
                    df_cleaned[column] = np.where(df_cleaned[column] > 50, 50, df_cleaned[column])
                    self.cleaning_log.append(f"Capped person_emp_length at 50")
                elif column == 'loan_percent_income':
                    df_cleaned[column] = np.where(df_cleaned[column] > 1.0, 1.0, df_cleaned[column])
                    self.cleaning_log.append(f"Capped loan_percent_income at 1.0")
        
        return df_cleaned
    
    def validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and correct data types.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with corrected data types
        """
        df_cleaned = df.copy()
        
        # Convert categorical columns to object type if they're numeric
        categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        
        for col in categorical_columns:
            if col in df.columns and df[col].dtype != 'object':
                df_cleaned[col] = df[col].astype('object')
                self.cleaning_log.append(f"Converted {col} to object type")
        
        # Ensure target variable is integer
        if 'loan_status' in df.columns:
            df_cleaned['loan_status'] = df_cleaned['loan_status'].astype('int64')
            self.cleaning_log.append("Ensured loan_status is integer type")
        
        return df_cleaned
    
    def get_cleaning_summary(self) -> List[str]:
        """
        Get a summary of all cleaning operations performed.
        
        Returns:
            List[str]: List of cleaning operations
        """
        return self.cleaning_log.copy()
    
    def clean_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Perform complete data cleaning pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            **kwargs: Additional arguments for cleaning methods
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        print("Starting data cleaning...")
        initial_shape = df.shape
        
        # Remove duplicates
        df_cleaned = self.remove_duplicates(df)
        
        # Handle missing values
        missing_strategy = kwargs.get('missing_strategy', 'auto')
        df_cleaned = self.handle_missing_values(df_cleaned, strategy=missing_strategy)
        
        # Handle outliers
        outlier_method = kwargs.get('outlier_method', 'capping')
        df_cleaned = self.handle_outliers(df_cleaned, method=outlier_method)
        
        # Validate data types
        df_cleaned = self.validate_data_types(df_cleaned)
        
        final_shape = df_cleaned.shape
        
        print(f"Data cleaning completed!")
        print(f"Initial shape: {initial_shape}")
        print(f"Final shape: {final_shape}")
        print(f"Rows removed: {initial_shape[0] - final_shape[0]}")
        
        return df_cleaned
