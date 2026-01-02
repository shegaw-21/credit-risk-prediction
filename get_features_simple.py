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
        features = list(model.feature_names_in_)
        print("Model features count:", len(features))
        print("First 10 features:", features[:10])
        print("Last 10 features:", features[-10:])
        
        # Save to a file for easy copying
        with open('model_features.txt', 'w') as f:
            f.write("EXPECTED_MODEL_COLUMNS = [\n")
            for feature in features:
                f.write(f"    '{feature}',\n")
            f.write("]\n")
        
        # Also create NUMERICAL_FEATURES
        numerical_features = [f for f in features if any(num in f for num in ['age', 'income', 'length', 'amnt', 'rate', 'percent', 'hist', 'log', 'per_age', 'to_income'])]
        with open('numerical_features.txt', 'w') as f:
            f.write("NUMERICAL_FEATURES = [\n")
            for feature in numerical_features:
                f.write(f"    '{feature}',\n")
            f.write("]\n")
        
        print("Features saved to model_features.txt and numerical_features.txt")
        
    else:
        print("Model doesn't have feature_names_in_ attribute")
        
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
