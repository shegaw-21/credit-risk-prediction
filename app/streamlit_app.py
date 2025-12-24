import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap

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
        scaler = joblib.load(str(PROJECT_ROOT / 'models' / 'scaler.pkl'))
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
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Batch Prediction", "EDA", "Model Insights", "Data Summary", "Model Performance", "About"])

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
    
    if st.button("Predict"):
        try:
            # Compute additional features
            debt_to_income = loan_amnt / person_income if person_income > 0 else 0
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
                'age_group': [age_group],
                'income_group': [income_group]
            })
            
            # Encode categorical
            input_data['loan_grade_encoded'] = le_loan_grade.transform([loan_grade])[0]
            input_data['cb_person_default_on_file_encoded'] = le_default.transform([cb_person_default_on_file])[0]
            
            # One-hot encode
            categorical_cols = ['person_home_ownership', 'loan_intent', 'age_group', 'income_group']
            input_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
            
            # Drop original string columns
            input_encoded = input_encoded.drop(['loan_grade', 'cb_person_default_on_file'], axis=1)
            
            # Scale numerical features
            input_encoded[NUMERICAL_FEATURES] = scaler.transform(input_encoded[NUMERICAL_FEATURES])
            
            # Ensure all expected columns are present
            for col in EXPECTED_MODEL_COLUMNS:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[EXPECTED_MODEL_COLUMNS]
            
            # Predict
            prediction = model.predict(input_encoded)[0]
            probability = model.predict_proba(input_encoded)[0][1]
            
            if prediction == 1:
                st.error(f"High Risk of Default (Probability: {probability:.2%})")
            else:
                st.success(f"Low Risk of Default (Probability: {probability:.2%})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("This may be due to sklearn version incompatibility. Please retrain the model with the current sklearn version.")

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
    
    # SHAP Explanation
    st.subheader("Model Explainability")
    st.write("SHAP explanations are not available due to library compatibility issues. Feature importance is shown above for model interpretability.")

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
        y_test = pd.read_csv(str(PROJECT_ROOT / 'data' / 'processed' / 'y_test.csv'))
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
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
        
    except FileNotFoundError:
        st.error("Test data not found. Please run the preprocessing and modeling notebooks.")
    except Exception as e:
        st.error(f"Error loading performance metrics: {e}")

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
    - ROC AUC: ~0.95
    - Focus on recall for identifying high-risk loans
    
    For more details, see the project notebooks and README.
    """)
    
    st.subheader("Contact")
    st.write("This is an educational project for data science demonstration.")