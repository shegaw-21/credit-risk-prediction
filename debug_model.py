import joblib
import pandas as pd
from pathlib import Path

# Load the current model
PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_path = PROJECT_ROOT / 'models' / 'best_model.pkl'

print("Loading model from:", model_path)

try:
    model = joblib.load(model_path)
    print(f"Model type: {type(model).__name__}")
    
    # Check if model has feature_names_in_ attribute
    if hasattr(model, 'feature_names_in_'):
        features = list(model.feature_names_in_)
        print(f"\nModel has {len(features)} features:")
        for i, feature in enumerate(features):
            print(f"{i+1:2d}. {feature}")
    else:
        print("Model doesn't have feature_names_in_ attribute")
        
        # Try to get features from training data
        X_train_path = PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv'
        if X_train_path.exists():
            X_train = pd.read_csv(X_train_path)
            print(f"\nFeatures from X_train.csv ({len(X_train.columns)} features):")
            for i, feature in enumerate(X_train.columns):
                print(f"{i+1:2d}. {feature}")
        else:
            print("X_train.csv not found")
    
    # Try to make a test prediction to see what the model expects
    print("\n" + "="*50)
    print("Testing model with sample data...")
    
    # Load test data
    X_test = pd.read_csv(str(PROJECT_ROOT / 'data' / 'processed' / 'X_test.csv'))
    print(f"X_test shape: {X_test.shape}")
    print(f"X_test columns: {list(X_test.columns)}")
    
    # Convert boolean columns to int
    bool_cols = X_test.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"Converting boolean columns: {list(bool_cols)}")
        X_test[bool_cols] = X_test[bool_cols].astype(int)
    
    # Try prediction
    try:
        y_pred = model.predict(X_test.head(1))
        print(f"✓ Prediction successful: {y_pred}")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
