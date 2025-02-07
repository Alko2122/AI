import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load trained model and column order
MODEL_PATH = "gbm_mlp_model.pkl"
COLUMNS_PATH = "columns.pkl"

# Function to safely load the model
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"Error: Model file '{MODEL_PATH}' not found!")
        return None

# Function to safely load columns.pkl
def load_columns():
    if os.path.exists(COLUMNS_PATH):
        return joblib.load(COLUMNS_PATH)
    else:
        st.error(f"Error: Column file '{COLUMNS_PATH}' not found!")
        return None

# Load model and columns
gbm_mlp_model = load_model()
expected_columns = load_columns()

if gbm_mlp_model is None or expected_columns is None:
    st.stop()  # Stop execution if model or columns are missing

# Function to preprocess user input
def preprocess_input(user_input):
    """Prepares input to match the trained model's feature set."""
    df = pd.DataFrame([user_input])  # Convert input to DataFrame

    # Apply one-hot encoding (same as training)
    df = pd.get_dummies(df)

    # Align columns with training data
    missing_cols = set(expected_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0  # Add missing columns with default 0

    df = df[expected_columns]  # Reorder columns

    return df

# Streamlit UI
st.title("Customer Churn Prediction")

# Input fields
user_input = {
    "feature_1": st.number_input("Feature 1", value=0.0),
    "feature_2": st.number_input("Feature 2", value=0.0),
    "feature_3": st.selectbox("Feature 3 Category", ["A", "B", "C"]),
}

# Prediction button
if st.button("Predict"):
    input_data = preprocess_input(user_input)

    # Check if model has predict method
    if hasattr(gbm_mlp_model, "predict"):
        prediction = gbm_mlp_model.predict(input_data)
    else:
        st.error("Error: The model does not support 'predict'.")
        st.stop()

    # Check if model has predict_proba method
    if hasattr(gbm_mlp_model, "predict_proba"):
        churn_probability = gbm_mlp_model.predict_proba(input_data)[:, 1]
    else:
        churn_probability = None  # Some models may not support probabilities

    # Display result
    result = "Churn" if prediction[0] == 1 else "Not Churn"
    st.write(f"**Prediction:** {result}")

    if churn_probability is not None:
        st.write(f"**Churn Probability:** {churn_probability[0]:.2%}")
