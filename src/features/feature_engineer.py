"""
Feature engineering utilities for credit risk prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from typing import Dict, List, Tuple, Optional
import joblib


class FeatureEngineer:
    """
    A class for feature engineering and transformation.
    """
    
    def __init__(self):
        self.encoders = {}
        self.scalers = {}
        self.feature_log = []
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between existing variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with interaction features
        """
        df_engineered = df.copy()
        
        # Debt-to-income ratio (already calculated in preprocessing)
        if 'loan_amnt' in df.columns and 'person_income' in df.columns:
            df_engineered['debt_to_income'] = df['loan_amnt'] / df['person_income']
            self.feature_log.append("Created debt_to_income feature")
        
        # Loan-to-income ratio
        if 'loan_amnt' in df.columns and 'person_income' in df.columns:
            df_engineered['loan_to_income'] = df['loan_amnt'] / df['person_income']
            self.feature_log.append("Created loan_to_income feature")
        
        # Interest burden
        if 'loan_int_rate' in df.columns and 'loan_amnt' in df.columns:
            df_engineered['interest_burden'] = (df['loan_int_rate'] / 100) * df['loan_amnt']
            self.feature_log.append("Created interest_burden feature")
        
        # Employment stability score
        if 'person_emp_length' in df.columns and 'person_age' in df.columns:
            df_engineered['emp_stability'] = df['person_emp_length'] / df['person_age']
            self.feature_log.append("Created emp_stability feature")
        
        return df_engineered
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create categorical features from numerical variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with categorical features
        """
        df_engineered = df.copy()
        
        # Age groups
        if 'person_age' in df.columns:
            df_engineered['age_group'] = pd.cut(
                df['person_age'], 
                bins=[0, 25, 35, 45, 55, 100], 
                labels=['18-25', '26-35', '36-45', '46-55', '56+']
            )
            self.feature_log.append("Created age_group feature")
        
        # Income groups
        if 'person_income' in df.columns:
            df_engineered['income_group'] = pd.cut(
                df['person_income'], 
                bins=[0, 30000, 60000, 100000, 200000, np.inf], 
                labels=['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
            )
            self.feature_log.append("Created income_group feature")
        
        # Loan amount groups
        if 'loan_amnt' in df.columns:
            df_engineered['loan_amount_group'] = pd.cut(
                df['loan_amnt'], 
                bins=[0, 5000, 10000, 15000, 20000, np.inf], 
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
            self.feature_log.append("Created loan_amount_group feature")
        
        # Interest rate groups
        if 'loan_int_rate' in df.columns:
            df_engineered['interest_rate_group'] = pd.cut(
                df['loan_int_rate'], 
                bins=[0, 8, 12, 16, 20, np.inf], 
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )
            self.feature_log.append("Created interest_rate_group feature")
        
        return df_engineered
    
    def encode_categorical_features(self, df: pd.DataFrame, encoding_type: str = 'mixed') -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            encoding_type (str): Type of encoding ('label', 'onehot', 'mixed')
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target variable if present
        if 'loan_status' in categorical_columns:
            categorical_columns.remove('loan_status')
        
        for column in categorical_columns:
            if encoding_type == 'label':
                # Label encoding for all categorical features
                if column not in self.encoders:
                    self.encoders[column] = LabelEncoder()
                    df_encoded[f'{column}_encoded'] = self.encoders[column].fit_transform(df[column].astype(str))
                else:
                    df_encoded[f'{column}_encoded'] = self.encoders[column].transform(df[column].astype(str))
                
                self.feature_log.append(f"Label encoded {column}")
                
            elif encoding_type == 'onehot':
                # One-hot encoding for all categorical features
                dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(column, axis=1)
                
                self.feature_log.append(f"One-hot encoded {column}")
                
            elif encoding_type == 'mixed':
                # Mixed encoding based on feature characteristics
                if column == 'loan_grade':
                    # Label encoding for ordinal features
                    if column not in self.encoders:
                        self.encoders[column] = LabelEncoder()
                        df_encoded[f'{column}_encoded'] = self.encoders[column].fit_transform(df[column])
                    else:
                        df_encoded[f'{column}_encoded'] = self.encoders[column].transform(df[column])
                    
                    self.feature_log.append(f"Label encoded ordinal feature {column}")
                    
                elif column in ['cb_person_default_on_file']:
                    # Label encoding for binary features
                    if column not in self.encoders:
                        self.encoders[column] = LabelEncoder()
                        df_encoded[f'{column}_encoded'] = self.encoders[column].fit_transform(df[column])
                    else:
                        df_encoded[f'{column}_encoded'] = self.encoders[column].transform(df[column])
                    
                    self.feature_log.append(f"Label encoded binary feature {column}")
                    
                else:
                    # One-hot encoding for nominal features
                    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded = df_encoded.drop(column, axis=1)
                    
                    self.feature_log.append(f"One-hot encoded nominal feature {column}")
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, method: str = 'standard', columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Scaling method ('standard', 'minmax', 'robust')
            columns (List[str]): Columns to scale (None for all numerical)
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        df_scaled = df.copy()
        
        if columns is None:
            # Identify numerical columns (excluding encoded categorical features)
            numerical_columns = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] and not col.endswith('_encoded'):
                    numerical_columns.append(col)
        else:
            numerical_columns = columns
        
        if method == 'standard':
            scaler = StandardScaler()
            
            if 'standard' not in self.scalers:
                self.scalers['standard'] = scaler
                df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            else:
                df_scaled[numerical_columns] = self.scalers['standard'].transform(df[numerical_columns])
                
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            if 'minmax' not in self.scalers:
                self.scalers['minmax'] = scaler
                df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            else:
                df_scaled[numerical_columns] = self.scalers['minmax'].transform(df[numerical_columns])
                
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            
            if 'robust' not in self.scalers:
                self.scalers['robust'] = scaler
                df_scaled[numerical_columns] = scaler.fit_transform(df[numerical_columns])
            else:
                df_scaled[numerical_columns] = self.scalers['robust'].transform(df[numerical_columns])
        
        self.feature_log.append(f"Scaled {len(numerical_columns)} numerical features using {method} method")
        
        return df_scaled
    
    def save_encoders(self, filepath: str):
        """
        Save encoders to file.
        
        Args:
            filepath (str): Path to save encoders
        """
        joblib.dump(self.encoders, filepath)
        self.feature_log.append(f"Saved encoders to {filepath}")
    
    def load_encoders(self, filepath: str):
        """
        Load encoders from file.
        
        Args:
            filepath (str): Path to load encoders from
        """
        self.encoders = joblib.load(filepath)
        self.feature_log.append(f"Loaded encoders from {filepath}")
    
    def save_scalers(self, filepath: str):
        """
        Save scalers to file.
        
        Args:
            filepath (str): Path to save scalers
        """
        joblib.dump(self.scalers, filepath)
        self.feature_log.append(f"Saved scalers to {filepath}")
    
    def load_scalers(self, filepath: str):
        """
        Load scalers from file.
        
        Args:
            filepath (str): Path to load scalers from
        """
        self.scalers = joblib.load(filepath)
        self.feature_log.append(f"Loaded scalers from {filepath}")
    
    def get_feature_log(self) -> List[str]:
        """
        Get a log of all feature engineering operations.
        
        Returns:
            List[str]: List of feature engineering operations
        """
        return self.feature_log.copy()
    
    def engineer_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Perform complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            **kwargs: Additional arguments for feature engineering methods
            
        Returns:
            pd.DataFrame: Engineered dataframe
        """
        print("Starting feature engineering...")
        initial_shape = df.shape
        
        # Create interaction features
        df_engineered = self.create_interaction_features(df)
        
        # Create categorical features
        df_engineered = self.create_categorical_features(df_engineered)
        
        # Encode categorical features
        encoding_type = kwargs.get('encoding_type', 'mixed')
        df_engineered = self.encode_categorical_features(df_engineered, encoding_type=encoding_type)
        
        # Scale numerical features
        scaling_method = kwargs.get('scaling_method', 'standard')
        df_engineered = self.scale_numerical_features(df_engineered, method=scaling_method)
        
        final_shape = df_engineered.shape
        
        print(f"Feature engineering completed!")
        print(f"Initial shape: {initial_shape}")
        print(f"Final shape: {final_shape}")
        print(f"Features created: {final_shape[1] - initial_shape[1]}")
        
        return df_engineered
