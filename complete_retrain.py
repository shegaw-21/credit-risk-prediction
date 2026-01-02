"""
Complete Model Retraining Script
Run this script to retrain the model with consistent features for the Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path
import sys

def main():
    print("🚀 Starting model retraining...")
    
    # Set paths - adjust if needed
    try:
        # Try to find the project root
        current_path = Path(__file__).resolve()
        if current_path.name == 'src':
            PROJECT_ROOT = current_path.parent
        elif current_path.name == 'models':
            PROJECT_ROOT = current_path.parent.parent
        else:
            PROJECT_ROOT = current_path
            
        DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
        MODELS_DIR = PROJECT_ROOT / 'models'
        
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Models directory: {MODELS_DIR}")
        
    except Exception as e:
        print(f"❌ Path setup error: {e}")
        return False

    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(exist_ok=True)

    print("\n📁 Loading processed data...")
    
    # Load the processed data
    try:
        X_train = pd.read_csv(DATA_DIR / 'X_train.csv')
        X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
        y_train = pd.read_csv(DATA_DIR / 'y_train.csv', index_col=0).iloc[:, 0]
        y_test = pd.read_csv(DATA_DIR / 'y_test.csv', index_col=0).iloc[:, 0]
        
        print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        print(f"✅ y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("Please run the preprocessing notebook first!")
        return False

    # Define the features we want to keep (matching Streamlit app)
    FEATURES_TO_KEEP = [
        'person_age',
        'person_income', 
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'person_income_log',
        'income_per_age',
        'debt_to_income',
        'loan_grade_encoded',
        'cb_person_default_on_file_encoded',
        'person_home_ownership_OTHER',
        'person_home_ownership_OWN',
        'person_home_ownership_RENT',
        'loan_intent_EDUCATION',
        'loan_intent_HOMEIMPROVEMENT',
        'loan_intent_MEDICAL',
        'loan_intent_PERSONAL',
        'loan_intent_VENTURE',
        'age_group_26-35',
        'age_group_36-45',
        'age_group_46-55',
        'age_group_56+',
        'income_group_Lower-Middle',
        'income_group_Middle',
        'income_group_Upper-Middle',
        'income_group_High',
    ]

    print(f"\n🔧 Keeping {len(FEATURES_TO_KEEP)} features")

    # Filter features
    X_train_filtered = X_train[FEATURES_TO_KEEP]
    X_test_filtered = X_test[FEATURES_TO_KEEP]

    print(f"✅ Filtered train shape: {X_train_filtered.shape}")
    print(f"✅ Filtered test shape: {X_test_filtered.shape}")

    # Define numerical features for scaling
    NUMERICAL_FEATURES = [
        'person_age',
        'person_income',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'person_income_log',
        'income_per_age',
        'debt_to_income',
    ]

    print("\n⚙️ Scaling numerical features...")

    # Scale numerical features
    scaler = RobustScaler()
    X_train_scaled = X_train_filtered.copy()
    X_test_scaled = X_test_filtered.copy()

    X_train_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(X_train_filtered[NUMERICAL_FEATURES])
    X_test_scaled[NUMERICAL_FEATURES] = scaler.transform(X_test_filtered[NUMERICAL_FEATURES])

    print("\n🌲 Training Random Forest model...")

    # Train Random Forest model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )

    rf.fit(X_train_scaled, y_train)

    print("\n🔍 Making predictions...")

    # Make predictions
    y_pred = rf.predict(X_test_scaled)
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]

    print("\n📊 Model Performance:")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n💾 Saving model and preprocessing objects...")

    # Save model and preprocessing objects
    joblib.dump(rf, MODELS_DIR / 'best_model.pkl')
    joblib.dump(scaler, MODELS_DIR / 'robust_scaler.pkl')

    # Create dummy label encoders for Streamlit app compatibility
    le_loan_grade = LabelEncoder()
    le_loan_grade.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    joblib.dump(le_loan_grade, MODELS_DIR / 'le_loan_grade.pkl')

    le_default = LabelEncoder()
    le_default.fit(['N', 'Y'])
    joblib.dump(le_default, MODELS_DIR / 'le_default.pkl')

    # Save the filtered data for consistency
    X_train_scaled.to_csv(DATA_DIR / 'X_train_final.csv', index=False)
    X_test_scaled.to_csv(DATA_DIR / 'X_test_final.csv', index=False)
    y_train.to_csv(DATA_DIR / 'y_train_final.csv')
    y_test.to_csv(DATA_DIR / 'y_test_final.csv')

    print("\n✅ Model retraining completed successfully!")
    print(f"📁 Model saved to: {MODELS_DIR / 'best_model.pkl'}")
    print(f"📁 Scaler saved to: {MODELS_DIR / 'robust_scaler.pkl'}")
    print(f"🔢 Features used: {len(FEATURES_TO_KEEP)}")
    print("\n📋 Feature order:")
    for i, feature in enumerate(FEATURES_TO_KEEP):
        print(f"{i+1:2d}. {feature}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Retraining completed! Your Streamlit app should now work correctly.")
    else:
        print("\n❌ Retraining failed. Please check the error messages above.")
    
    input("\nPress Enter to exit...")
