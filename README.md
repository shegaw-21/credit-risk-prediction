# Credit Risk Prediction Project

## Overview
This project develops a machine learning model to predict credit risk using the credit_risk_dataset.csv. The dataset contains information about loan applicants, and the goal is to classify whether a loan will default (loan_status = 1) or not.

## Dataset
- **Source**: credit_risk_dataset.csv
- **Size**: ~32,581 rows, 12 columns
- **Target**: loan_status (binary: 0 = no default, 1 = default)
- **Features**: person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length

## Key Insights from EDA
- Imbalanced dataset: 78% non-defaults, 22% defaults
- Missing values: person_emp_length (2.7%), loan_int_rate (9.6%)
- 165 duplicate rows
- Outliers present in numerical features

## Project Structure
```
DS_project/
├── data/
│   ├── raw/           # Original dataset
│   ├── processed/     # Cleaned and preprocessed data
│   └── external/      # Additional data (if any)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb # Data Cleaning & Preprocessing
│   ├── 03_modeling.ipynb     # Model Training
│   └── 04_evaluation.ipynb   # Model Evaluation
├── models/            # Saved trained models
├── reports/           # Figures and evaluation results
├── app/               # Streamlit app (to be developed)
├── src/               # Modular code (optional)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Installation
1. Clone or download the project.
2. Install Python 3.8+.
3. Create a virtual environment: `python -m venv venv`
4. Activate: `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`

## Usage
1. Run notebooks in order: 01_eda.ipynb → 02_preprocessing.ipynb → 03_modeling.ipynb → 04_evaluation.ipynb
2. Each notebook contains detailed code and explanations.

## Dashboard
An interactive Streamlit app is available in the `app/` directory.
- **Prediction**: Input applicant data for real-time risk assessment
- **EDA**: Explore dataset distributions and correlations

To run: `streamlit run app/streamlit_app.py`

## Models
- Baseline: Logistic Regression with class weights
- Main: Random Forest with hyperparameter tuning
- Evaluation metrics focus on recall due to imbalance

## Results
- Best model: Tuned Random Forest
- ROC AUC: ~0.95 (estimated)
- Emphasizes recall for identifying potential defaults

## Future Work
- Implement SMOTE for oversampling
- Try deep learning models
- Add SHAP for interpretability

## License
This project is for educational purposes.