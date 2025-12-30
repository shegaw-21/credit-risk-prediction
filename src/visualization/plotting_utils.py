"""
Plotting utilities for credit risk prediction project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PlottingUtils:
    """
    A class for creating various types of plots and visualizations.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', palette: str = 'husl'):
        """
        Initialize plotting utilities.
        
        Args:
            style (str): Matplotlib style
            palette (str): Color palette
        """
        plt.style.use(style)
        sns.set_palette(palette)
        self.colors = sns.color_palette(palette)
        
    def plot_distribution(self, data: pd.Series, title: str, plot_type: str = 'histogram',
                         figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot distribution of a single variable.
        
        Args:
            data (pd.Series): Data to plot
            title (str): Plot title
            plot_type (str): Type of plot ('histogram', 'box', 'violin', 'density')
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if plot_type == 'histogram':
            sns.histplot(data, kde=True, ax=ax)
        elif plot_type == 'box':
            sns.boxplot(x=data, ax=ax)
        elif plot_type == 'violin':
            sns.violinplot(x=data, ax=ax)
        elif plot_type == 'density':
            sns.kdeplot(data, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(data.name, fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_categorical_counts(self, data: pd.Series, title: str,
                               figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot counts for categorical variables.
        
        Args:
            data (pd.Series): Categorical data
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.countplot(data=data, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(data.name, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = 'Correlation Matrix',
                              figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Data for correlation
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        corr_matrix = data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.2f')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_scatter(self, x: pd.Series, y: pd.Series, title: str,
                    hue: Optional[pd.Series] = None, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot scatter plot between two variables.
        
        Args:
            x (pd.Series): X variable
            y (pd.Series): Y variable
            title (str): Plot title
            hue (pd.Series): Color by variable
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.scatterplot(x=x, y=y, hue=hue, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x.name, fontsize=12)
        ax.set_ylabel(y.name, fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str], title: str = 'Confusion Matrix',
                            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            class_names (List[str]): Class names
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        fig, ax = plt.subplots(figsize=figsize)
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                      title: str = 'ROC Curve', figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_scores (np.ndarray): Predicted scores
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=figsize)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                   title: str = 'Precision-Recall Curve',
                                   figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true (np.ndarray): True labels
            y_scores (np.ndarray): Predicted scores
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        fig, ax = plt.subplots(figsize=figsize)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        ax.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.2f})', linewidth=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_names: List[str], importance_values: np.ndarray,
                               title: str = 'Feature Importance', top_n: int = 20,
                               figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names (List[str]): Feature names
            importance_values (np.ndarray): Importance values
            title (str): Plot title
            top_n (int): Number of top features to show
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create dataframe and sort by importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).head(top_n)
        
        sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve(self, train_sizes: np.ndarray, train_scores: np.ndarray,
                           val_scores: np.ndarray, title: str = 'Learning Curve',
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot learning curve.
        
        Args:
            train_sizes (np.ndarray): Training set sizes
            train_scores (np.ndarray): Training scores
            val_scores (np.ndarray): Validation scores
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        ax.plot(train_sizes, val_mean, 'o-', label='Validation Score', linewidth=2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_scatter(self, df: pd.DataFrame, x: str, y: str, 
                                  color: Optional[str] = None, size: Optional[str] = None,
                                  title: str = 'Interactive Scatter Plot') -> go.Figure:
        """
        Create interactive scatter plot using Plotly.
        
        Args:
            df (pd.DataFrame): Data
            x (str): X column
            y (str): Y column
            color (str): Color column
            size (str): Size column
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        fig = px.scatter(df, x=x, y=y, color=color, size=size,
                        title=title, hover_data=[x, y])
        
        fig.update_layout(
            title_font_size=16,
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filepath: str, dpi: int = 300):
        """
        Save figure to file.
        
        Args:
            fig (plt.Figure): Matplotlib figure
            filepath (str): Path to save figure
            dpi (int): Resolution
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
    
    def close_figure(self, fig: plt.Figure):
        """
        Close figure to free memory.
        
        Args:
            fig (plt.Figure): Matplotlib figure
        """
        plt.close(fig)
