import joblib
import pandas as pd
from pathlib import Path

# Load the current model and check what features it expects
PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_path = PROJECT_ROOT / 'models' / 'best_model.pkl'

try:
    model = joblib.load(model_path)
    
    # Check if model has feature_names_in_ attribute (sklearn >= 1.0)
    if hasattr(model, 'feature_names_in_'):
        print("Model features (from feature_names_in_):")
        features = list(model.feature_names_in_)
        for i, feature in enumerate(features):
            print(f"{i+1:2d}. {feature}")
        
        # Create the exact EXPECTED_MODEL_COLUMNS list
        print(f"\nCopy this list to streamlit_app.py:")
        print("EXPECTED_MODEL_COLUMNS = [")
        for feature in features:
            print(f"    '{feature}',")
        print("]")
        
        # Also create NUMERICAL_FEATURES by filtering numerical ones
        numerical_features = [f for f in features if any(num in f for num in ['age', 'income', 'length', 'amnt', 'rate', 'percent', 'hist', 'log', 'per_age', 'to_income'])]
        print(f"\nNUMERICAL_FEATURES = [")
        for feature in numerical_features:
            print(f"    '{feature}',")
        print("]")
        
    else:
        print("Model doesn't have feature_names_in_ attribute")
        
except Exception as e:
    print(f"Error loading model: {e}")
    # Try to get features from training data as fallback
    X_train_path = PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv'
    if X_train_path.exists():
        X_train = pd.read_csv(X_train_path)
        print("\nFeatures from X_train.csv:")
        for i, feature in enumerate(X_train.columns):
            print(f"{i+1:2d}. {feature}")
    else:
        print("X_train.csv not found")
