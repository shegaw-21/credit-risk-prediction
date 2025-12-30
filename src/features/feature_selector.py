"""
Feature selection utilities for credit risk prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (SelectKBest, f_classif, chi2, RFE, 
                                     SelectFromModel, mutual_info_classif)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    A class for feature selection and analysis.
    """
    
    def __init__(self):
        self.selected_features = {}
        self.feature_importance = {}
        self.selection_log = []
        
    def statistical_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                    method: str = 'f_classif', k: int = 20) -> Tuple[List[str], pd.DataFrame]:
        """
        Perform statistical feature selection.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            method (str): Selection method ('f_classif', 'chi2', 'mutual_info')
            k (int): Number of features to select
            
        Returns:
            Tuple[List[str], pd.DataFrame]: Selected features and scores
        """
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'chi2':
            # Ensure all values are non-negative for chi2
            X_positive = X - X.min().min() + 1
            selector = SelectKBest(score_func=chi2, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create scores dataframe
        scores_df = pd.DataFrame({
            'feature': X.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        self.selection_log.append(f"Statistical selection ({method}): Selected {len(selected_features)} features")
        
        return selected_features, scores_df
    
    def correlation_based_selection(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            X (pd.DataFrame): Features
            threshold (float): Correlation threshold
            
        Returns:
            List[str]: Features to keep
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Features to keep
        features_to_keep = [col for col in X.columns if col not in to_drop]
        
        self.selection_log.append(f"Correlation selection: Removed {len(to_drop)} highly correlated features")
        
        return features_to_keep
    
    def variance_based_selection(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        Remove low variance features.
        
        Args:
            X (pd.DataFrame): Features
            threshold (float): Variance threshold
            
        Returns:
            List[str]: Features to keep
        """
        # Calculate variance for each feature
        variances = X.var()
        
        # Features to keep (above threshold)
        features_to_keep = variances[variances > threshold].index.tolist()
        
        # Features to drop
        to_drop = variances[variances <= threshold].index.tolist()
        
        self.selection_log.append(f"Variance selection: Removed {len(to_drop)} low variance features")
        
        return features_to_keep
    
    def model_based_selection(self, X: pd.DataFrame, y: pd.Series,
                             model_type: str = 'random_forest', 
                             threshold: Optional[float] = None,
                             k: Optional[int] = None) -> List[str]:
        """
        Perform model-based feature selection.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            model_type (str): Type of model ('random_forest', 'logistic')
            threshold (Optional[float]): Importance threshold
            k (Optional[int]: Number of features to select
            
        Returns:
            List[str]: Selected features
        """
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X, y)
            importances = np.abs(model.coef_[0])
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Store feature importance
        self.feature_importance[model_type] = dict(zip(X.columns, importances))
        
        # Select features
        if threshold is not None:
            selected_features = [feature for feature, importance in zip(X.columns, importances) 
                               if importance >= threshold]
        elif k is not None:
            # Get top k features
            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            selected_features = feature_importance_df.head(k)['feature'].tolist()
        else:
            # Use median as threshold
            median_importance = np.median(importances)
            selected_features = [feature for feature, importance in zip(X.columns, importances) 
                               if importance > median_importance]
        
        self.selection_log.append(f"Model-based selection ({model_type}): Selected {len(selected_features)} features")
        
        return selected_features
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series,
                                    estimator: Any = None, n_features: int = 20) -> List[str]:
        """
        Perform Recursive Feature Elimination.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            estimator: Base estimator (default: RandomForest)
            n_features (int): Number of features to select
            
        Returns:
            List[str]: Selected features
        """
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        selected_features = X.columns[rfe.support_].tolist()
        
        self.selection_log.append(f"RFE: Selected {len(selected_features)} features")
        
        return selected_features
    
    def cross_validation_selection(self, X: pd.DataFrame, y: pd.Series,
                                 features: List[str], cv: int = 5,
                                 scoring: str = 'roc_auc') -> Dict[str, float]:
        """
        Evaluate feature subsets using cross-validation.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            features (List[str]): Features to evaluate
            cv (int): Number of CV folds
            scoring (str): Scoring metric
            
        Returns:
            Dict[str, float]: CV scores for each feature
        """
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        feature_scores = {}
        
        for feature in features:
            X_subset = X[[feature]]
            scores = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring)
            feature_scores[feature] = scores.mean()
        
        # Sort features by score
        feature_scores = dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))
        
        self.selection_log.append(f"CV selection: Evaluated {len(features)} features")
        
        return feature_scores
    
    def plot_feature_importance(self, model_type: str = 'random_forest',
                               top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance from model-based selection.
        
        Args:
            model_type (str): Model type for importance
            top_n (int): Number of top features to show
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if model_type not in self.feature_importance:
            print(f"No feature importance available for {model_type}")
            return None
        
        importance_dict = self.feature_importance[model_type]
        
        # Create dataframe and sort
        importance_df = pd.DataFrame(list(importance_dict.items()), 
                                   columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
        ax.set_title(f'Top {top_n} Feature Importance - {model_type.title()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_statistical_scores(self, scores_df: pd.DataFrame, method: str = 'f_classif',
                               top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot statistical feature selection scores.
        
        Args:
            scores_df (pd.DataFrame): Scores dataframe
            method (str): Selection method
            top_n (int): Number of top features to show
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        top_scores_df = scores_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.barplot(x='score', y='feature', data=top_scores_df, ax=ax)
        ax.set_title(f'Top {top_n} Features - {method.upper()} Scores', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def compare_selection_methods(self, X: pd.DataFrame, y: pd.Series,
                                 methods: List[str] = None, k: int = 20) -> pd.DataFrame:
        """
        Compare different feature selection methods.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            methods (List[str]): Methods to compare
            k (int): Number of features to select
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if methods is None:
            methods = ['f_classif', 'mutual_info', 'random_forest', 'logistic']
        
        comparison_results = {}
        
        for method in methods:
            try:
                if method in ['f_classif', 'chi2', 'mutual_info']:
                    selected_features, _ = self.statistical_feature_selection(X, y, method=method, k=k)
                elif method in ['random_forest', 'logistic']:
                    selected_features = self.model_based_selection(X, y, model_type=method, k=k)
                
                comparison_results[method] = selected_features
                
            except Exception as e:
                print(f"Error with method {method}: {str(e)}")
                comparison_results[method] = []
        
        # Create comparison dataframe
        all_features = set()
        for features in comparison_results.values():
            all_features.update(features)
        
        comparison_df = pd.DataFrame(index=list(all_features))
        
        for method, features in comparison_results.items():
            comparison_df[method] = [1 if feature in features else 0 for feature in comparison_df.index]
        
        # Add summary column
        comparison_df['selected_by'] = comparison_df.sum(axis=1)
        comparison_df = comparison_df.sort_values('selected_by', ascending=False)
        
        return comparison_df
    
    def select_features_pipeline(self, X: pd.DataFrame, y: pd.Series,
                               methods: List[str] = None, k: int = 20) -> List[str]:
        """
        Complete feature selection pipeline.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            methods (List[str]): Methods to use in pipeline
            k (int): Number of features to select
            
        Returns:
            List[str]: Final selected features
        """
        print("Starting feature selection pipeline...")
        initial_features = X.columns.tolist()
        
        # Step 1: Remove low variance features
        features_variance = self.variance_based_selection(X)
        X_var = X[features_variance]
        
        # Step 2: Remove highly correlated features
        features_corr = self.correlation_based_selection(X_var)
        X_corr = X_var[features_corr]
        
        # Step 3: Statistical selection
        features_stat, _ = self.statistical_feature_selection(X_corr, y, method='f_classif', k=k*2)
        X_stat = X_corr[features_stat]
        
        # Step 4: Model-based selection
        features_model = self.model_based_selection(X_stat, y, model_type='random_forest', k=k)
        
        final_features = features_model
        
        print(f"Feature selection completed!")
        print(f"Initial features: {len(initial_features)}")
        print(f"After variance filtering: {len(features_variance)}")
        print(f"After correlation filtering: {len(features_corr)}")
        print(f"After statistical selection: {len(features_stat)}")
        print(f"Final features: {len(final_features)}")
        
        self.selected_features['final'] = final_features
        
        return final_features
    
    def get_selection_log(self) -> List[str]:
        """
        Get a log of all selection operations.
        
        Returns:
            List[str]: List of selection operations
        """
        return self.selection_log.copy()
    
    def get_selected_features(self, method: str = 'final') -> List[str]:
        """
        Get selected features for a specific method.
        
        Args:
            method (str): Selection method
            
        Returns:
            List[str]: Selected features
        """
        return self.selected_features.get(method, [])
    
    def get_feature_importance(self, model_type: str) -> Dict[str, float]:
        """
        Get feature importance for a specific model type.
        
        Args:
            model_type (str): Model type
            
        Returns:
            Dict[str, float]: Feature importance dictionary
        """
        return self.feature_importance.get(model_type, {})
