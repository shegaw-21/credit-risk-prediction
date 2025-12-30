# Credit Risk Prediction Project

## Table of Contents
- [Project Overview](#project-overview)
- [Learning Outcomes](#learning-outcomes)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Methodology](#models-and-methodology)
- [Results](#results)
- [Dashboard](#dashboard)
- [Key Findings and Insights](#key-findings-and-insights)
- [Future Work](#future-work)
- [License](#license)

## Project Overview
This comprehensive data science project develops a machine learning model to predict credit risk using the credit_risk_dataset.csv. The dataset contains information about loan applicants, and the goal is to classify whether a loan will default (loan_status = 1) or not.

The project follows a complete data science workflow from exploratory data analysis through model deployment, demonstrating proficiency in data cleaning, preprocessing, modeling, evaluation, and interpretation.

## Learning Outcomes

### Skills Developed
- **Data Cleaning & Preprocessing**: Handling missing values, outlier detection, data transformation, feature scaling, and data type conversions
- **Exploratory Data Analysis (EDA)**: Univariate, bivariate, and multivariate analysis, hypothesis testing, and data visualization to uncover insights and patterns
- **Machine Learning Modeling**: Implementing and training various ML algorithms, understanding model assumptions, and hyperparameter tuning
- **Model Evaluation & Selection**: Applying appropriate metrics for imbalanced classification, cross-validation, and model comparison
- **Feature Engineering and Selection**: Creating new features from existing ones to improve model performance

### Tools Utilized
- **Python**: Proficient use for data manipulation, analysis, and modeling
- **Pandas**: Data loading, cleaning, transformation, and analysis
- **Scikit-learn**: Implementing various machine learning algorithms, preprocessing techniques, and model selection utilities
- **Matplotlib/Seaborn/Plotly**: Creating compelling static and interactive data visualizations
- **Jupyter Notebooks**: Interactive development and documentation of data science workflows
- **Streamlit**: Building interactive web applications for model deployment
- **SHAP**: Model interpretability and explainability

### Knowledge Gained
- **Imbalanced Classification**: Strategies for handling imbalanced datasets using class weights and exploring SMOTE
- **Feature Encoding**: Applying various encoding techniques for categorical features (One-Hot Encoding, Label Encoding)
- **Model Interpretability**: Techniques to understand and explain model predictions using SHAP values
- **Deployment Concepts**: Deploying machine learning models using Streamlit web framework

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