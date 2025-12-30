"""
Model evaluation utilities for credit risk prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                           precision_recall_curve, roc_auc_score, accuracy_score,
                           precision_score, recall_score, f1_score)
from sklearn.model_selection import learning_curve, validation_curve
import shap
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    A class for comprehensive model evaluation and interpretation.
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.shap_values = {}
        
    def evaluate_classification(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                               model_name: str = 'model') -> Dict[str, Any]:
        """
        Comprehensive classification model evaluation.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        
        # Precision-Recall curve data
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Store results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
        }
        
        self.evaluation_results[model_name] = results
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models based on their metrics.
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Dictionary of model results
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC AUC': metrics['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by ROC AUC (descending)
        comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)
        
        return comparison_df
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot model comparison.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                bars = axes[i].bar(range(len(comparison_df)), comparison_df[metric])
                axes[i].set_title(metric, fontsize=12, fontweight='bold')
                axes[i].set_xticks(range(len(comparison_df)))
                axes[i].set_xticklabels(comparison_df['Model'], rotation=45)
                axes[i].set_ylabel('Score', fontsize=10)
                
                # Add value labels on bars
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide last subplot if unused
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def calculate_shap_values(self, model: Any, X_test: pd.DataFrame, 
                            model_name: str = 'model') -> np.ndarray:
        """
        Calculate SHAP values for model interpretation.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            model_name (str): Name of the model
            
        Returns:
            np.ndarray: SHAP values
        """
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
            
            self.shap_values[model_name] = shap_values
            
            return shap_values
            
        except Exception as e:
            print(f"Error calculating SHAP values: {str(e)}")
            return None
    
    def plot_shap_summary(self, shap_values: np.ndarray, X_test: pd.DataFrame,
                         model_name: str = 'model', max_display: int = 20) -> plt.Figure:
        """
        Plot SHAP summary plot.
        
        Args:
            shap_values (np.ndarray): SHAP values
            X_test (pd.DataFrame): Test features
            model_name (str): Name of the model
            max_display (int): Maximum number of features to display
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if shap_values is None:
            print("No SHAP values available for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        shap.summary_plot(shap_values, X_test, max_display=max_display, 
                        plot_type="bar", show=False)
        
        ax.set_title(f'SHAP Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_shap_waterfall(self, shap_values: np.ndarray, X_test: pd.DataFrame,
                          instance_idx: int = 0, model_name: str = 'model') -> plt.Figure:
        """
        Plot SHAP waterfall plot for a single instance.
        
        Args:
            shap_values (np.ndarray): SHAP values
            X_test (pd.DataFrame): Test features
            instance_idx (int): Index of the instance to explain
            model_name (str): Name of the model
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        if shap_values is None:
            print("No SHAP values available for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        shap.waterfall_plot(shap_values[instance_idx], max_display=20, show=False)
        
        ax.set_title(f'SHAP Waterfall Plot - {model_name} (Instance {instance_idx})', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, model: Any, X: pd.DataFrame, y: pd.Series,
                           cv: int = 5, scoring: str = 'roc_auc',
                           figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot learning curves for model.
        
        Args:
            model: Model to evaluate
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
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
        ax.set_ylabel(f'{scoring.upper()} Score', fontsize=12)
        ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_validation_curves(self, model: Any, X: pd.DataFrame, y: pd.Series,
                            param_name: str, param_range: List[Any],
                            cv: int = 5, scoring: str = 'roc_auc',
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot validation curves for hyperparameter analysis.
        
        Args:
            model: Model to evaluate
            X (pd.DataFrame): Features
            y (pd.Series): Target
            param_name (str): Parameter name to vary
            param_range (List[Any]): Parameter values to test
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            figsize (Tuple[int, int]): Figure size
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(param_range, train_mean, 'o-', label='Training Score', linewidth=2)
        ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        ax.plot(param_range, val_mean, 'o-', label='Validation Score', linewidth=2)
        ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel(f'{scoring.upper()} Score', fontsize=12)
        ax.set_title(f'Validation Curves - {param_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_prediction_errors(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                                model_name: str = 'model') -> pd.DataFrame:
        """
        Analyze prediction errors in detail.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: Error analysis dataframe
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Create error analysis dataframe
        error_df = X_test.copy()
        error_df['actual'] = y_test.values
        error_df['predicted'] = y_pred
        error_df['probability'] = y_pred_proba
        error_df['is_correct'] = (y_pred == y_test.values)
        error_df['error_type'] = error_df.apply(
            lambda row: 'True Positive' if row['actual'] == 1 and row['predicted'] == 1
            else 'True Negative' if row['actual'] == 0 and row['predicted'] == 0
            else 'False Positive' if row['actual'] == 0 and row['predicted'] == 1
            else 'False Negative', axis=1
        )
        
        # Separate errors
        false_positives = error_df[error_df['error_type'] == 'False Positive']
        false_negatives = error_df[error_df['error_type'] == 'False Negative']
        
        print(f"Error Analysis for {model_name}:")
        print(f"Total predictions: {len(error_df)}")
        print(f"Correct predictions: {error_df['is_correct'].sum()} ({error_df['is_correct'].mean():.2%})")
        print(f"False Positives: {len(false_positives)} ({len(false_positives)/len(error_df):.2%})")
        print(f"False Negatives: {len(false_negatives)} ({len(false_negatives)/len(error_df):.2%})")
        
        return error_df
    
    def generate_evaluation_report(self, model_results: Dict[str, Dict[str, Any]],
                                 save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            model_results (Dict[str, Dict[str, Any]]): Model evaluation results
            save_path (Optional[str]): Path to save the report
            
        Returns:
            str: Evaluation report
        """
        report = []
        report.append("# Model Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Model comparison
        comparison_df = self.compare_models(model_results)
        report.append("## Model Performance Comparison")
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Detailed results for each model
        for model_name, results in model_results.items():
            report.append(f"## {model_name.title()} Model")
            report.append("-" * 30)
            
            metrics = results['metrics']
            report.append("### Performance Metrics")
            for metric, value in metrics.items():
                report.append(f"- {metric.title()}: {value:.4f}")
            report.append("")
            
            # Classification report
            report.append("### Classification Report")
            class_report = results['classification_report']
            report.append(f"Precision (Class 0): {class_report['0']['precision']:.4f}")
            report.append(f"Recall (Class 0): {class_report['0']['recall']:.4f}")
            report.append(f"F1-Score (Class 0): {class_report['0']['f1-score']:.4f}")
            report.append(f"Precision (Class 1): {class_report['1']['precision']:.4f}")
            report.append(f"Recall (Class 1): {class_report['1']['recall']:.4f}")
            report.append(f"F1-Score (Class 1): {class_report['1']['f1-score']:.4f}")
            report.append("")
            
            # Confusion matrix
            report.append("### Confusion Matrix")
            cm = results['confusion_matrix']
            report.append(f"True Negatives: {cm[0,0]}")
            report.append(f"False Positives: {cm[0,1]}")
            report.append(f"False Negatives: {cm[1,0]}")
            report.append(f"True Positives: {cm[1,1]}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")
        
        return report_text
    
    def get_evaluation_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get evaluation results for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Optional[Dict[str, Any]]: Evaluation results
        """
        return self.evaluation_results.get(model_name)
    
    def get_shap_values(self, model_name: str) -> Optional[np.ndarray]:
        """
        Get SHAP values for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Optional[np.ndarray]: SHAP values
        """
        return self.shap_values.get(model_name)
