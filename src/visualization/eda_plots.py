"""
EDA-specific plotting utilities for credit risk prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
from .plotting_utils import PlottingUtils


class EDAPlots(PlottingUtils):
    """
    Extended plotting utilities specifically for Exploratory Data Analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', palette: str = 'husl'):
        """
        Initialize EDA plotting utilities.
        
        Args:
            style (str): Matplotlib style
            palette (str): Color palette
        """
        super().__init__(style, palette)
    
    def plot_target_distribution(self, y: pd.Series, title: str = 'Target Variable Distribution',
                                figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot distribution of target variable with percentages.
        
        Args:
            y (pd.Series): Target variable
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Count plot
        value_counts = y.value_counts()
        sns.countplot(x=y, ax=ax1)
        ax1.set_title('Count Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Loan Status', fontsize=10)
        ax1.set_ylabel('Count', fontsize=10)
        
        # Add count labels
        for i, count in enumerate(value_counts):
            ax1.text(i, count + max(value_counts) * 0.01, str(count), 
                    ha='center', va='bottom', fontweight='bold')
        
        # Pie chart with percentages
        percentages = y.value_counts(normalize=True) * 100
        labels = [f'No Default ({percentages[0]:.1f}%)', 
                 f'Default ({percentages[1]:.1f}%)']
        colors = [self.colors[0], self.colors[1]]
        
        ax2.pie(percentages, labels=labels, colors=colors, autopct='', startangle=90)
        ax2.set_title('Percentage Distribution', fontsize=12, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_missing_values(self, df: pd.DataFrame, title: str = 'Missing Values Analysis',
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot missing values analysis.
        
        Args:
            df (pd.DataFrame): Data to analyze
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Missing values heatmap
        missing_data = df.isnull()
        sns.heatmap(missing_data, cbar=False, cmap='viridis', yticklabels=False, ax=ax1)
        ax1.set_title('Missing Values Heatmap', fontsize=12, fontweight='bold')
        
        # Missing values count and percentage
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        missing_df = pd.DataFrame({'Count': missing_count, 'Percentage': missing_percent})
        missing_df = missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False)
        
        if not missing_df.empty:
            # Bar plot for missing values
            bars = ax2.bar(range(len(missing_df)), missing_df['Count'], color=self.colors[0])
            ax2.set_xlabel('Features', fontsize=10)
            ax2.set_ylabel('Missing Count', fontsize=10)
            ax2.set_title('Missing Values Count by Feature', fontsize=12, fontweight='bold')
            ax2.set_xticks(range(len(missing_df)))
            ax2.set_xticklabels(missing_df.index, rotation=45)
            
            # Add percentage labels on bars
            for i, (bar, percent) in enumerate(zip(bars, missing_df['Percentage'])):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(missing_df['Count']) * 0.01,
                        f'{percent:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax2.text(0.5, 0.5, 'No missing values found', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Missing Values Count by Feature', fontsize=12, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_outlier_analysis(self, df: pd.DataFrame, numerical_cols: List[str],
                             title: str = 'Outlier Analysis', figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot outlier analysis for numerical features.
        
        Args:
            df (pd.DataFrame): Data to analyze
            numerical_cols (List[str]): Numerical columns
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        n_cols = len(numerical_cols)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                # Box plot
                sns.boxplot(x=df[col], ax=axes[i])
                axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
                axes[i].set_xlabel('', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_categorical_vs_target(self, df: pd.DataFrame, categorical_cols: List[str],
                                  target_col: str = 'loan_status',
                                  figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Plot categorical features vs target variable.
        
        Args:
            df (pd.DataFrame): Data to analyze
            categorical_cols (List[str]): Categorical columns
            target_col (str): Target column name
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        n_cols = len(categorical_cols)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                # Cross-tabulation and plot
                cross_tab = pd.crosstab(df[col], df[target_col], normalize='index')
                cross_tab.plot(kind='bar', ax=axes[i], color=[self.colors[0], self.colors[1]])
                
                axes[i].set_title(f'{col} vs {target_col}', fontsize=10, fontweight='bold')
                axes[i].set_xlabel(col, fontsize=8)
                axes[i].set_ylabel('Proportion', fontsize=8)
                axes[i].legend(title=target_col, labels=['No Default', 'Default'])
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(categorical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Categorical Features vs {target_col}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_numerical_vs_target(self, df: pd.DataFrame, numerical_cols: List[str],
                                target_col: str = 'loan_status',
                                figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot numerical features vs target variable.
        
        Args:
            df (pd.DataFrame): Data to analyze
            numerical_cols (List[str]): Numerical columns
            target_col (str): Target column name
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        n_cols = len(numerical_cols)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                # Box plot by target
                sns.boxplot(x=target_col, y=col, data=df, ax=axes[i])
                axes[i].set_title(f'{col} by {target_col}', fontsize=10, fontweight='bold')
                axes[i].set_xlabel(target_col, fontsize=8)
                axes[i].set_ylabel(col, fontsize=8)
        
        # Hide unused subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Numerical Features vs {target_col}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_pairplot(self, df: pd.DataFrame, features: List[str], 
                       target_col: str = 'loan_status', 
                       title: str = 'Pair Plot of Selected Features') -> sns.PairGrid:
        """
        Create pair plot for selected features.
        
        Args:
            df (pd.DataFrame): Data to plot
            features (List[str]): Features to include
            target_col (str): Target column
            title (str): Plot title
            
        Returns:
            sns.PairGrid: Pair plot
        """
        plot_df = df[features + [target_col]]
        
        g = sns.pairplot(plot_df, hue=target_col, diag_kind='kde', 
                        plot_kws={'alpha': 0.6}, diag_kws={'alpha': 0.7})
        
        g.fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        
        return g
    
    def create_3d_scatter(self, df: pd.DataFrame, x: str, y: str, z: str,
                         color: str = 'loan_status',
                         title: str = '3D Scatter Plot') -> go.Figure:
        """
        Create 3D scatter plot using Plotly.
        
        Args:
            df (pd.DataFrame): Data to plot
            x (str): X column
            y (str): Y column
            z (str): Z column
            color (str): Color column
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly 3D scatter plot
        """
        fig = px.scatter_3d(df, x=x, y=y, z=z, color=color,
                          title=title, opacity=0.7)
        
        fig.update_layout(
            scene=dict(
                xaxis_title=x,
                yaxis_title=y,
                zaxis_title=z
            ),
            title_font_size=16,
            width=800,
            height=600
        )
        
        return fig
    
    def plot_correlation_with_target(self, df: pd.DataFrame, target_col: str = 'loan_status',
                                    title: str = 'Correlation with Target Variable',
                                    figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot correlation of all features with target variable.
        
        Args:
            df (pd.DataFrame): Data to analyze
            target_col (str): Target column
            title (str): Plot title
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Calculate correlations with target
        correlations = df.corr()[target_col].sort_values(ascending=True)
        correlations = correlations.drop(target_col)  # Remove self-correlation
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(correlations)), correlations.values)
        
        # Color bars based on correlation direction
        for i, bar in enumerate(bars):
            if correlations.values[i] < 0:
                bar.set_color(self.colors[0])  # Negative correlation
            else:
                bar.set_color(self.colors[1])  # Positive correlation
        
        ax.set_yticks(range(len(correlations)))
        ax.set_yticklabels(correlations.index)
        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add correlation values as text
        for i, (bar, value) in enumerate(zip(bars, correlations.values)):
            ax.text(value + 0.01 if value >= 0 else value - 0.01, i, 
                   f'{value:.3f}', ha='left' if value >= 0 else 'right', 
                   va='center', fontsize=8)
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        return fig
    
    def create_summary_dashboard(self, df: pd.DataFrame, target_col: str = 'loan_status',
                               save_path: Optional[str] = None) -> go.Figure:
        """
        Create a comprehensive EDA dashboard using Plotly.
        
        Args:
            df (pd.DataFrame): Data to analyze
            target_col (str): Target column
            save_path (Optional[str]): Path to save the dashboard
            
        Returns:
            go.Figure: Plotly dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Target Distribution', 'Missing Values', 
                          'Numerical Features Distribution', 'Categorical Features'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Target distribution
        target_counts = df[target_col].value_counts()
        fig.add_trace(
            go.Pie(labels=['No Default', 'Default'], values=target_counts.values,
                  name="Target Distribution"),
            row=1, col=1
        )
        
        # Missing values
        missing_counts = df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if not missing_counts.empty:
            fig.add_trace(
                go.Bar(x=missing_counts.index, y=missing_counts.values,
                      name="Missing Values"),
                row=1, col=2
            )
        
        # Numerical feature distribution (example with first numerical column)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols and target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        if numerical_cols:
            fig.add_trace(
                go.Histogram(x=df[numerical_cols[0]], name=numerical_cols[0]),
                row=2, col=1
            )
        
        # Categorical feature distribution (example with first categorical column)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            cat_counts = df[categorical_cols[0]].value_counts()
            fig.add_trace(
                go.Bar(x=cat_counts.index, y=cat_counts.values,
                      name=categorical_cols[0]),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="EDA Dashboard",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Dashboard saved to {save_path}")
        
        return fig
