import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler, LabelEncoder
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import sys
import traceback

# Configure warnings and error handling
warnings.filterwarnings('ignore')

# Global exception handler to prevent crashes
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    st.error(f"An unexpected error occurred: {exc_value}")
    print("Error details:", ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

sys.excepthook = handle_exception

# -------------------------------------------------------------------
# Feature configuration shared between single and batch prediction
# -------------------------------------------------------------------
NUMERICAL_FEATURES = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'person_income_log',
    'income_per_age',
    'debt_to_income',
]

EXPECTED_MODEL_COLUMNS = [
    'person_age',
    'person_income',
    'person_emp_length',
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

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Load model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        model = joblib.load(str(PROJECT_ROOT / 'models' / 'best_model.pkl'))
        scaler = joblib.load(str(PROJECT_ROOT / 'models' / 'robust_scaler.pkl'))
        le_loan_grade = joblib.load(str(PROJECT_ROOT / 'models' / 'le_loan_grade.pkl'))
        le_default = joblib.load(str(PROJECT_ROOT / 'models' / 'le_default.pkl'))
        return model, scaler, le_loan_grade, le_default
    except FileNotFoundError:
        st.error("Model files not found. Please run the preprocessing and modeling notebooks first.")
        return None, None, None, None

# Load data for EDA
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(str(PROJECT_ROOT / 'data' / 'raw' / 'credit_risk_dataset.csv'))
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure data/raw/credit_risk_dataset.csv exists.")
        return pd.DataFrame()

model, scaler, le_loan_grade, le_default = load_model()
df = load_data()

if model is None or df.empty:
    st.stop()

st.title("Credit Risk Prediction Dashboard")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Batch Prediction", "EDA", "Model Insights", "Data Summary", "Model Performance", "Feature Selection", "About"])

if page == "Prediction":
    st.header("Loan Default Prediction")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        person_age = st.slider("Person Age", 18, 100, 30)
        person_income = st.number_input("Annual Income", 0, 10000000, 50000)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_percent_income = st.slider("Loan Percent Income", 0.0, 1.0, 0.1)
    
    with col2:
        loan_amnt = st.number_input("Loan Amount", 0, 1000000, 10000)
        loan_int_rate = st.slider("Interest Rate (%)", 0.0, 50.0, 10.0)
        person_emp_length = st.slider("Employment Length (years)", 0, 50, 5)
        cb_person_default_on_file = st.selectbox("Historical Default", ["N", "Y"])
        cb_person_cred_hist_length = st.slider("Credit History Length (years)", 0, 50, 5)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔴 Apply High Risk Test Values"):
            st.session_state.test_high_risk = True
            st.session_state.test_low_risk = False
    with col2:
        if st.button("🟢 Apply Low Risk Test Values"):
            st.session_state.test_low_risk = True
            st.session_state.test_high_risk = False
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.test_high_risk = False
            st.session_state.test_low_risk = False
    
    # Apply test values if buttons clicked
    if st.session_state.get('test_high_risk', False):
        st.info("🔴 High Risk Test Values Applied")
        person_age = 22
        person_income = 15000
        person_home_ownership = "RENT"
        loan_intent = "PERSONAL"
        loan_grade = "F"
        loan_percent_income = 0.5
        loan_amnt = 75000
        loan_int_rate = 25.0
        person_emp_length = 0.5
        cb_person_default_on_file = "Y"
        cb_person_cred_hist_length = 1
    elif st.session_state.get('test_low_risk', False):
        st.info("🟢 Low Risk Test Values Applied")
        person_age = 45
        person_income = 150000
        person_home_ownership = "OWN"
        loan_intent = "HOMEIMPROVEMENT"
        loan_grade = "A"
        loan_percent_income = 0.1
        loan_amnt = 10000
        loan_int_rate = 5.0
        person_emp_length = 15
        cb_person_default_on_file = "N"
        cb_person_cred_hist_length = 20
    
    if st.button("Predict"):
        try:
            # Create input DataFrame from user inputs
            input_data = pd.DataFrame({
                'person_age': [person_age],
                'person_income': [person_income],
                'person_home_ownership': [person_home_ownership],
                'loan_intent': [loan_intent],
                'loan_grade': [loan_grade],
                'loan_amnt': [loan_amnt],
                'loan_int_rate': [loan_int_rate],
                'person_emp_length': [person_emp_length],
                'cb_person_default_on_file': [cb_person_default_on_file],
                'cb_person_cred_hist_length': [cb_person_cred_hist_length],
                'loan_percent_income': [loan_percent_income]
            })
            
            # Create derived features
            input_data['person_income_log'] = np.log1p(input_data['person_income'])
            input_data['income_per_age'] = input_data['person_income'] / input_data['person_age']
            input_data['emp_stability'] = input_data['person_emp_length'] / input_data['person_age']
            input_data['debt_to_income'] = input_data['loan_amnt'] / input_data['person_income']
            
            # Create categorical groups
            input_data['age_group'] = pd.cut(
                input_data['person_age'], 
                bins=[0, 25, 35, 45, 55, 100], 
                labels=['18-25', '26-35', '36-45', '46-55', '56+']
            )
            input_data['income_group'] = pd.cut(
                input_data['person_income'], 
                bins=[0, 30000, 60000, 100000, 200000, np.inf], 
                labels=['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
            )
            
            # Encode categorical features
            input_data['loan_grade_encoded'] = le_loan_grade.transform(input_data['loan_grade'])
            input_data['cb_person_default_on_file_encoded'] = le_default.transform(input_data['cb_person_default_on_file'])
            
            # One-hot encode nominal features
            input_data = pd.get_dummies(input_data, columns=['person_home_ownership', 'loan_intent', 'age_group', 'income_group'], drop_first=True)
            
            # Ensure all expected columns exist (add missing ones with 0)
            expected_columns = [
                'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
                'loan_percent_income', 'cb_person_cred_hist_length', 'person_income_log', 'income_per_age',
                'emp_stability', 'debt_to_income', 'loan_grade_encoded', 'cb_person_default_on_file_encoded',
                'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
                'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
                'loan_intent_PERSONAL', 'loan_intent_VENTURE', 'age_group_26-35', 'age_group_36-45',
                'age_group_46-55', 'age_group_56+', 'income_group_Lower-Middle', 'income_group_Middle',
                'income_group_Upper-Middle', 'income_group_High'
            ]
            
            for col in expected_columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Reorder columns to match model expectations
            input_data = input_data[expected_columns]
            
            # Scale numerical features
            numerical_cols = [
                'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
                'loan_percent_income', 'cb_person_cred_hist_length', 'debt_to_income', 'person_income_log', 'income_per_age'
            ]
            input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
            
            # Try prediction
            st.write("🔍 Debug - Input Values:")
            st.write(f"Age: {person_age}, Income: {person_income}, Loan: {loan_amnt}")
            st.write(f"Interest: {loan_int_rate}%, Employment: {person_emp_length} years")
            st.write(f"Credit History: {cb_person_cred_hist_length} years, Grade: {loan_grade}")
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            st.write("🔍 Debug - Model Output:")
            st.write(f"Raw prediction: {prediction}")
            st.write(f"Probability: {probability:.4f} ({probability:.2%})")
            
            # Dynamic risk classification based on probability thresholds
            if probability >= 0.7:
                st.error(f"🔴 Very High Risk of Default (Probability: {probability:.2%})")
                risk_level = "Very High"
            elif probability >= 0.5:
                st.error(f"🔴 High Risk of Default (Probability: {probability:.2%})")
                risk_level = "High"
            elif probability >= 0.3:
                st.warning(f"🟡 Moderate Risk of Default (Probability: {probability:.2%})")
                risk_level = "Moderate"
            elif probability >= 0.15:
                st.success(f"🟢 Low Risk of Default (Probability: {probability:.2%})")
                risk_level = "Low"
            else:
                st.success(f"🟢 Very Low Risk of Default (Probability: {probability:.2%})")
                risk_level = "Very Low"
            
            # Show model prediction vs threshold explanation
            st.info(f"💡 Model Prediction: {prediction} (1=Default, 0=No Default) | Probability: {probability:.2%}")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Debug info:")
            st.write(f"Model type: {type(model).__name__}")
            import traceback
            st.write(traceback.format_exc())

elif page == "Batch Prediction":
    st.header("Batch Loan Default Prediction")
    
    uploaded_file = st.file_uploader("Upload a CSV file with loan applicant data", type="csv")
    predict_all_clicked = st.button("Predict for All", key="predict_all_batch")

    # No file uploaded yet
    if uploaded_file is None:
        st.info("Please upload a CSV file to run batch predictions.")

    # File uploaded: show preview and optionally run predictions
    else:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(batch_df.head())

        if predict_all_clicked:
            try:
                # Basic schema validation for uploaded CSV
                required_input_cols = [
                    'person_age',
                    'person_income',
                    'person_home_ownership',
                    'loan_intent',
                    'loan_grade',
                    'loan_amnt',
                    'loan_int_rate',
                    'person_emp_length',
                    'cb_person_default_on_file',
                    'cb_person_cred_hist_length',
                ]
                missing_base = [c for c in required_input_cols if c not in batch_df.columns]
                if missing_base:
                    raise ValueError(
                        f"Missing required columns in uploaded CSV: {', '.join(missing_base)}"
                    )

                # Handle missing values as in preprocessing
                batch_df['person_emp_length'] = batch_df['person_emp_length'].fillna(batch_df['person_emp_length'].median())
                batch_df['loan_int_rate'] = batch_df['loan_int_rate'].fillna(batch_df['loan_int_rate'].mean())
                
                # Cap outliers
                batch_df['person_age'] = np.where(batch_df['person_age'] > 100, 100, batch_df['person_age'])
                
                # Process each row
                results = []
                st.write(f"Running batch prediction on {len(batch_df)} rows...")
                for idx, row in batch_df.iterrows():
                    # Extract values
                    person_age = row['person_age']
                    person_income = row['person_income']
                    person_home_ownership = row['person_home_ownership']
                    loan_intent = row['loan_intent']
                    loan_grade = row['loan_grade']
                    loan_amnt = row['loan_amnt']
                    loan_int_rate = row['loan_int_rate']
                    person_emp_length = row['person_emp_length']
                    cb_person_default_on_file = row['cb_person_default_on_file']
                    cb_person_cred_hist_length = row['cb_person_cred_hist_length']
                    loan_percent_income = row.get('loan_percent_income', loan_amnt / person_income if person_income > 0 else 0)
                    
                    # Compute additional features
                    debt_to_income = loan_amnt / person_income if person_income > 0 else 0
                    person_income_log = np.log1p(person_income)  # log(1 + x) to handle zero
                    income_per_age = person_income / person_age if person_age > 0 else 0
                    age_group = pd.cut([person_age], bins=[0, 25, 35, 45, 55, 100], labels=['18-25', '26-35', '36-45', '46-55', '56+'])[0]
                    income_group = pd.cut([person_income], bins=[0, 30000, 60000, 100000, 200000, np.inf], labels=['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High'])[0]
                    
                    # Create input data
                    input_data = pd.DataFrame({
                        'person_age': [person_age],
                        'person_income': [person_income],
                        'person_home_ownership': [person_home_ownership],
                        'loan_intent': [loan_intent],
                        'loan_grade': [loan_grade],
                        'loan_amnt': [loan_amnt],
                        'loan_int_rate': [loan_int_rate],
                        'person_emp_length': [person_emp_length],
                        'cb_person_default_on_file': [cb_person_default_on_file],
                        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
                        'loan_percent_income': [loan_percent_income],
                        'debt_to_income': [debt_to_income],
                        'person_income_log': [person_income_log],
                        'income_per_age': [income_per_age],
                        'age_group': [age_group],
                        'income_group': [income_group]
                    })
                    
                    # Encode and process as above
                    input_data['loan_grade_encoded'] = le_loan_grade.transform([loan_grade])[0]
                    input_data['cb_person_default_on_file_encoded'] = le_default.transform([cb_person_default_on_file])[0]
                    
                    categorical_cols = ['person_home_ownership', 'loan_intent', 'age_group', 'income_group']
                    input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
                    input_encoded = input_encoded.drop(['loan_grade', 'cb_person_default_on_file'], axis=1)
                    
                    input_encoded[NUMERICAL_FEATURES] = scaler.transform(input_encoded[NUMERICAL_FEATURES])
                    
                    for col in EXPECTED_MODEL_COLUMNS:
                        if col not in input_encoded.columns:
                            input_encoded[col] = 0
                    
                    input_encoded = input_encoded[EXPECTED_MODEL_COLUMNS]
                    
                    # Predict
                    prediction = model.predict(input_encoded)[0]
                    probability = model.predict_proba(input_encoded)[0][1]
                    
                    results.append({
                        'Index': idx,
                        'Prediction': 'High Risk' if prediction == 1 else 'Low Risk',
                        'Probability': probability
                    })
                
                results_df = pd.DataFrame(results)
                st.write("Prediction Results:")
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button("Download Results", csv, "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Error processing batch predictions: {e}")
                st.write("Ensure the CSV has the required columns: person_age, person_income, person_home_ownership, loan_intent, loan_grade, loan_amnt, loan_int_rate, person_emp_length, cb_person_default_on_file, cb_person_cred_hist_length, loan_percent_income (optional)")

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        selected_grade = st.multiselect("Filter by Loan Grade", df['loan_grade'].unique(), default=df['loan_grade'].unique())
        selected_status = st.multiselect("Filter by Loan Status", df['loan_status'].unique(), default=df['loan_status'].unique())
    with col2:
        selected_home = st.multiselect("Filter by Home Ownership", df['person_home_ownership'].unique(), default=df['person_home_ownership'].unique())
        selected_intent = st.multiselect("Filter by Loan Intent", df['loan_intent'].unique(), default=df['loan_intent'].unique())
    
    filtered_df = df[
        (df['loan_grade'].isin(selected_grade)) &
        (df['loan_status'].isin(selected_status)) &
        (df['person_home_ownership'].isin(selected_home)) &
        (df['loan_intent'].isin(selected_intent))
    ]
    
    if filtered_df.empty:
        st.write("No data matches the selected filters. Please adjust the filters.")
    else:
        # Target distribution
        st.subheader("Loan Status Distribution")
        fig, ax = plt.subplots()
        filtered_df['loan_status'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel("Loan Status")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
        # Age distribution
        st.subheader("Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['person_age'], kde=True, ax=ax)
        st.pyplot(fig)
        
        # Income vs Loan Amount
        st.subheader("Income vs Loan Amount")
        fig, ax = plt.subplots()
        sns.scatterplot(data=filtered_df, x='person_income', y='loan_amnt', hue='loan_status', ax=ax)
        st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("Correlation Matrix")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        corr = filtered_df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Box plot for loan amount by grade
        st.subheader("Loan Amount by Grade")
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_df, x='loan_grade', y='loan_amnt', ax=ax)
        st.pyplot(fig)

elif page == "Model Insights":
    st.header("Model Insights")
    
    # Load test data for insights
    X_test = pd.read_csv(str(PROJECT_ROOT / 'data' / 'processed' / 'X_test.csv'))
    y_test = pd.read_csv(str(PROJECT_ROOT / 'data' / 'processed' / 'y_test.csv'))
    
    # Feature Importance
    X_test = pd.read_csv(str(PROJECT_ROOT / 'data' / 'processed' / 'X_test.csv'))
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20), ax=ax)
        ax.set_title('Top 20 Feature Importances')
        st.pyplot(fig)
    
    # SHAP Explanation - Temporarily disabled for stability
    st.subheader("Model Explainability")
    st.info("SHAP analysis temporarily disabled for app stability. Feature importance is shown above.")
    st.write("The basic feature importance plot (shown above) provides insights into which factors most influence loan default predictions.")

elif page == "Data Summary":
    st.header("Data Summary")
    
    st.subheader("Dataset Overview")
    st.write(f"Total records: {len(df)}")
    st.write(f"Features: {len(df.columns)}")
    st.write(f"Target distribution: {df['loan_status'].value_counts().to_dict()}")
    
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.write(missing)
    else:
        st.write("No missing values.")
    
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    st.subheader("Descriptive Statistics")
    desc = df.describe()
    st.dataframe(desc)

elif page == "Model Performance":
    st.header("Model Performance")
    
    try:
        X_test = pd.read_csv(str(PROJECT_ROOT / 'data' / 'processed' / 'X_test.csv'))
        y_test = pd.read_csv(str(PROJECT_ROOT / 'data' / 'processed' / 'y_test.csv'), index_col=0).iloc[:, 0]
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        # Fix PyArrow compatibility issues
        report_df = report_df.astype('float64')
        st.dataframe(report_df)
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        
        st.subheader("ROC AUC Score")
        auc = roc_auc_score(y_test, y_pred_proba)
        st.write(f"ROC AUC: {auc:.3f}")
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("Probability Distribution Analysis")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribution of predicted probabilities
        ax1.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Predicted Probability of Default')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Predicted Probabilities')
        ax1.grid(True, alpha=0.3)
        
        # Probability distribution by actual class
        ax2.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='No Default', edgecolor='black')
        ax2.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Default', edgecolor='black')
        ax2.set_xlabel('Predicted Probability of Default')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Probability Distribution by Actual Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        st.subheader("Threshold Analysis")
        thresholds = np.arange(0.1, 0.9, 0.05)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision_scores.append(precision_score(y_test, y_pred_thresh))
            recall_scores.append(recall_score(y_test, y_pred_thresh))
            f1_scores.append(f1_score(y_test, y_pred_thresh))
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, precision_scores, marker='o', label='Precision')
        ax.plot(thresholds, recall_scores, marker='s', label='Recall')
        ax.plot(thresholds, f1_scores, marker='^', label='F1 Score')
        ax.axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal: {optimal_threshold:.2f}')
        ax.set_xlabel('Classification Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics vs Classification Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.write(f"**Optimal threshold (max F1): {optimal_threshold:.2f}**")
        st.write(f"F1 Score at optimal threshold: {f1_scores[optimal_idx]:.3f}")
        
        st.subheader("PR Curve")
        from sklearn.metrics import precision_recall_curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    except FileNotFoundError:
        st.error("Test data not found. Please run the preprocessing and modeling notebooks.")
    except Exception as e:
        st.error(f"Error loading performance metrics: {e}")

elif page == "Feature Selection":
    st.header("Feature Selection Analysis")
    
    try:
        import sys
        sys.path.append(str(PROJECT_ROOT / 'src'))
        from features.feature_selector import FeatureSelector
        from visualization.eda_plots import EDAPlots
        
        # Load processed data
        X_train = pd.read_csv(str(PROJECT_ROOT / 'data' / 'processed' / 'X_train.csv'))
        y_train = pd.read_csv(str(PROJECT_ROOT / 'data' / 'processed' / 'y_train.csv')).iloc[:, 0]
        
        selector = FeatureSelector()
        
        st.subheader("Statistical Feature Selection")
        method = st.selectbox("Select method", ["f_classif", "mutual_info"])
        k = st.slider("Number of features to select", 5, 30, 20)
        
        if st.button("Run Statistical Selection"):
            selected_features, scores_df = selector.statistical_feature_selection(X_train, y_train, method=method, k=k)
            
            st.write(f"Selected {len(selected_features)} features:")
            st.write(selected_features)
            
            # Plot scores
            fig, ax = plt.subplots(figsize=(10, 6))
            scores_df.head(k).plot(kind='bar', x='feature', y='score', ax=ax)
            ax.set_title(f'Top {k} Features - {method.upper()} Scores')
            ax.set_xlabel('Features')
            ax.set_ylabel('Score')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        st.subheader("Model-Based Feature Selection")
        model_type = st.selectbox("Select model type", ["random_forest", "logistic"])
        
        if st.button("Run Model-Based Selection"):
            selected_features = selector.model_based_selection(X_train, y_train, model_type=model_type, k=k)
            
            st.write(f"Selected {len(selected_features)} features:")
            st.write(selected_features)
            
            # Plot feature importance
            fig = selector.plot_feature_importance(model_type=model_type, top_n=k)
            st.pyplot(fig)
        
        st.subheader("Correlation Analysis")
        threshold = st.slider("Correlation threshold", 0.8, 0.99, 0.95)
        
        if st.button("Remove Highly Correlated Features"):
            features_to_keep = selector.correlation_based_selection(X_train, threshold=threshold)
            st.write(f"Features to keep: {len(features_to_keep)}")
            st.write(f"Features removed: {len(X_train.columns) - len(features_to_keep)}")
            st.write(features_to_keep)
        
        st.subheader("Complete Feature Selection Pipeline")
        if st.button("Run Complete Pipeline"):
            final_features = selector.select_features_pipeline(X_train, y_train, k=k)
            st.write(f"Final selected features ({len(final_features)}):")
            st.write(final_features)
            
            # Show selection log
            st.subheader("Selection Process Log")
            log = selector.get_selection_log()
            for entry in log:
                st.write(f"• {entry}")
                
    except ImportError:
        st.error("Feature selection modules not available. Please ensure src/ modules are properly installed.")
    except Exception as e:
        st.error(f"Error in feature selection: {e}")

elif page == "About":
    st.header("About This Project")
    
    st.write("""
    This dashboard is part of a comprehensive credit risk prediction project.
    
    **Project Goals:**
    - Predict loan default risk using machine learning
    - Handle imbalanced classification (22% defaults)
    - Provide interpretable model insights
    
    **Technologies Used:**
    - Python, Pandas, Scikit-learn
    - Random Forest with class weights
    - SHAP for model explainability
    - Streamlit for the dashboard
    
    **Model Performance:**
    - ROC AUC: ~0.93-0.95
    - Random Forest with balanced class weights
    - Robust preprocessing with outlier handling
    - Focus on precision-recall balance for loan decisions
    
    For more details, see the project notebooks and README.
    """)
    
    st.subheader("Contact")
    st.write("This is an educational project for data science demonstration.")