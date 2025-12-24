# Credit Risk Prediction Dashboard

This Streamlit app provides an interactive interface for credit risk prediction and exploratory data analysis.

## Features

### Prediction Page
- Input loan applicant details
- Get real-time prediction of loan default risk
- View prediction probability

### EDA Page
- Visualize target variable distribution
- Explore age distribution
- Scatter plot of income vs loan amount
- Correlation matrix heatmap

## Setup

1. Ensure the model and preprocessing objects are trained and saved:
   - Run `notebooks/02_preprocessing.ipynb` to process data and save scalers/encoders
   - Run `notebooks/03_modeling.ipynb` to train and save the best model

2. Install dependencies:
   ```bash
   pip install streamlit
   ```

3. Run the app:
   ```bash
   streamlit run app/streamlit_app.py
   ```

## File Structure
- `app/streamlit_app.py`: Main Streamlit application
- `models/`: Contains saved model and preprocessing objects
- `data/raw/`: Raw dataset for EDA