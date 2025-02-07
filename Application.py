import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

class CombinedModel:
    def __init__(self, gbm, mlp):
        self.gbm = gbm
        self.mlp = mlp

    def predict_proba(self, X):
        gbm_preds = self.gbm.predict_proba(X)[:, 1]
        mlp_preds = self.mlp.predict_proba(X)[:, 1]
        combined_preds = (gbm_preds + mlp_preds) / 2
        return np.column_stack([1 - combined_preds, combined_preds])

# ✅ Define CombinedModel before loading the model
class CombinedModel:
    def __init__(self, gbm, mlp):
        self.gbm = gbm
        self.mlp = mlp

    def predict(self, X):
        gbm_preds = self.gbm.predict(X)
        mlp_preds = self.mlp.predict(X)
        return (gbm_preds + mlp_preds) / 2  # Average of both predictions

    def predict_proba(self, X):
        gbm_preds = self.gbm.predict_proba(X)[:, 1]
        mlp_preds = self.mlp.predict_proba(X)[:, 1]
        combined_preds = (gbm_preds + mlp_preds) / 2
        return np.column_stack([1 - combined_preds, combined_preds])  # Convert to probability format

# File paths
MODEL_PATH = "lgbm_mlp_model.pkl"
COLUMNS_PATH = "columns.pkl"

# Function to safely load the model
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)  # ✅ Now it recognizes CombinedModel
    else:
        st.error(f"Error: Model file '{MODEL_PATH}' not found!")
        return None

# Function to safely load column names
def load_columns():
    if os.path.exists(COLUMNS_PATH):
        return joblib.load(COLUMNS_PATH)
    else:
        st.error(f"Error: Column file '{COLUMNS_PATH}' not found!")
        return None

# Load model and columns
lgbm_mlp_model = load_model()
expected_columns = load_columns()

if lgbm_mlp_model is None or expected_columns is None:
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
    if hasattr(lgbm_mlp_model, "predict"):
        prediction = lgbm_mlp_model.predict(input_data)
    else:
        st.error("Error: The model does not support 'predict'.")
        st.stop()

    # Check if model has predict_proba method
    if hasattr(lgbm_mlp_model, "predict_proba"):
        churn_probability = lgbm_mlp_model.predict_proba(input_data)[:, 1]
    else:
        churn_probability = None  # Some models may not support probabilities

    # Display result
    result = "Churn" if prediction[0] == 1 else "Not Churn"
    st.write(f"**Prediction:** {result}")

    if churn_probability is not None:
        st.write(f"**Churn Probability:** {churn_probability[0]:.2%}")
