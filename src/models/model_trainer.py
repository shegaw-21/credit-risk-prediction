"""
Model training utilities for credit risk prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    A class for training and tuning machine learning models.
    """
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.training_log = []
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                    random_state: int = 42, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
            stratify (bool): Whether to stratify the split
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        self.training_log.append(f"Data split: Train={X_train.shape}, Test={X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def compute_class_weights(self, y_train: pd.Series, method: str = 'balanced') -> Dict[int, float]:
        """
        Compute class weights for imbalanced datasets.
        
        Args:
            y_train (pd.Series): Training target variable
            method (str): Method for computing weights ('balanced', 'custom')
            
        Returns:
            Dict[int, float]: Class weights dictionary
        """
        if method == 'balanced':
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        else:
            # Custom weights based on domain knowledge
            class_counts = y_train.value_counts()
            total_samples = len(y_train)
            class_weight_dict = {
                0: total_samples / (2 * class_counts[0]),  # Non-default
                1: total_samples / (2 * class_counts[1])   # Default
            }
        
        self.training_log.append(f"Computed class weights: {class_weight_dict}")
        
        return class_weight_dict
    
    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           class_weights: Optional[Dict[int, float]] = None) -> LogisticRegression:
        """
        Train a baseline logistic regression model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            class_weights (Dict[int, float]): Class weights for imbalanced data
            
        Returns:
            LogisticRegression: Trained baseline model
        """
        if class_weights:
            baseline_model = LogisticRegression(
                random_state=42, 
                class_weight=class_weights, 
                max_iter=1000
            )
        else:
            baseline_model = LogisticRegression(random_state=42, max_iter=1000)
        
        baseline_model.fit(X_train, y_train)
        
        self.models['baseline'] = baseline_model
        self.training_log.append("Trained baseline logistic regression model")
        
        return baseline_model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                          class_weights: Optional[Dict[int, float]] = None,
                          tune_hyperparameters: bool = False) -> RandomForestClassifier:
        """
        Train a Random Forest classifier.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            class_weights (Dict[int, float]): Class weights for imbalanced data
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            RandomForestClassifier: Trained Random Forest model
        """
        if class_weights:
            rf_model = RandomForestClassifier(
                random_state=42, 
                class_weight=class_weights,
                n_jobs=-1
            )
        else:
            rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                rf_model, 
                param_grid, 
                cv=3, 
                scoring='roc_auc', 
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_rf = grid_search.best_estimator_
            self.training_log.append(f"Random Forest hyperparameter tuning completed. Best params: {grid_search.best_params_}")
            
        else:
            # Default parameters
            rf_model.set_params(n_estimators=200, max_depth=20, min_samples_split=2)
            rf_model.fit(X_train, y_train)
            best_rf = rf_model
            self.training_log.append("Trained Random Forest with default parameters")
        
        self.models['random_forest'] = best_rf
        self.best_models['random_forest'] = best_rf
        
        return best_rf
    
    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                             tune_hyperparameters: bool = False) -> GradientBoostingClassifier:
        """
        Train a Gradient Boosting classifier.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            GradientBoostingClassifier: Trained Gradient Boosting model
        """
        gb_model = GradientBoostingClassifier(random_state=42)
        
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
            
            grid_search = GridSearchCV(
                gb_model, 
                param_grid, 
                cv=3, 
                scoring='roc_auc', 
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_gb = grid_search.best_estimator_
            self.training_log.append(f"Gradient Boosting hyperparameter tuning completed. Best params: {grid_search.best_params_}")
            
        else:
            # Default parameters
            gb_model.set_params(n_estimators=200, learning_rate=0.1, max_depth=5)
            gb_model.fit(X_train, y_train)
            best_gb = gb_model
            self.training_log.append("Trained Gradient Boosting with default parameters")
        
        self.models['gradient_boosting'] = best_gb
        self.best_models['gradient_boosting'] = best_gb
        
        return best_gb
    
    def train_support_vector_machine(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   class_weights: Optional[Dict[int, float]] = None) -> SVC:
        """
        Train a Support Vector Machine classifier.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            class_weights (Dict[int, float]): Class weights for imbalanced data
            
        Returns:
            SVC: Trained SVM model
        """
        if class_weights:
            svm_model = SVC(
                random_state=42, 
                class_weight=class_weights,
                probability=True
            )
        else:
            svm_model = SVC(random_state=42, probability=True)
        
        # Default parameters for SVM
        svm_model.set_params(C=1.0, kernel='rbf')
        svm_model.fit(X_train, y_train)
        
        self.models['svm'] = svm_model
        self.training_log.append("Trained Support Vector Machine")
        
        return svm_model
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                           cv: int = 5, scoring: str = 'roc_auc') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            
        Returns:
            Dict[str, float]: Cross-validation results
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        self.training_log.append(f"Cross-validation completed. Mean {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_results
    
    def save_model(self, model: Any, filepath: str, model_name: str):
        """
        Save a trained model to file.
        
        Args:
            model: Trained model
            filepath (str): Path to save the model
            model_name (str): Name of the model
        """
        joblib.dump(model, filepath)
        self.training_log.append(f"Saved {model_name} model to {filepath}")
    
    def load_model(self, filepath: str, model_name: str) -> Any:
        """
        Load a trained model from file.
        
        Args:
            filepath (str): Path to load the model from
            model_name (str): Name of the model
            
        Returns:
            Any: Loaded model
        """
        model = joblib.load(filepath)
        self.models[model_name] = model
        self.training_log.append(f"Loaded {model_name} model from {filepath}")
        
        return model
    
    def get_training_log(self) -> List[str]:
        """
        Get a log of all training operations.
        
        Returns:
            List[str]: List of training operations
        """
        return self.training_log.copy()
    
    def get_best_model(self, model_name: str) -> Any:
        """
        Get the best trained model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            Any: Best trained model
        """
        return self.best_models.get(model_name)
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series,
                        tune_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train multiple models and compare their performance.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            tune_hyperparameters (bool): Whether to tune hyperparameters
            
        Returns:
            Dict[str, Any]: Dictionary of trained models and their performance
        """
        print("Starting model training pipeline...")
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_train)
        
        # Train baseline model
        baseline_model = self.train_baseline_model(X_train, y_train, class_weights)
        baseline_metrics = self.evaluate_model(baseline_model, X_test, y_test)
        
        # Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train, class_weights, tune_hyperparameters)
        rf_metrics = self.evaluate_model(rf_model, X_test, y_test)
        
        # Train Gradient Boosting
        gb_model = self.train_gradient_boosting(X_train, y_train, tune_hyperparameters)
        gb_metrics = self.evaluate_model(gb_model, X_test, y_test)
        
        # Train SVM
        svm_model = self.train_support_vector_machine(X_train, y_train, class_weights)
        svm_metrics = self.evaluate_model(svm_model, X_test, y_test)
        
        # Compare models
        model_comparison = {
            'baseline': {'model': baseline_model, 'metrics': baseline_metrics},
            'random_forest': {'model': rf_model, 'metrics': rf_metrics},
            'gradient_boosting': {'model': gb_model, 'metrics': gb_metrics},
            'svm': {'model': svm_model, 'metrics': svm_metrics}
        }
        
        # Find best model based on ROC AUC
        best_model_name = max(model_comparison.keys(), 
                            key=lambda x: model_comparison[x]['metrics']['roc_auc'])
        
        print(f"Model training completed!")
        print(f"Best model: {best_model_name} (ROC AUC: {model_comparison[best_model_name]['metrics']['roc_auc']:.4f})")
        
        return model_comparison
