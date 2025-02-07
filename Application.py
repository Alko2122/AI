import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
        try:
            return joblib.load(MODEL_PATH)  # ✅ Now it recognizes CombinedModel
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error(f"Error: Model file '{MODEL_PATH}' not found!")
        return None

# Function to safely load column names
def load_columns():
    if os.path.exists(COLUMNS_PATH):
        try:
            return joblib.load(COLUMNS_PATH)
        except Exception as e:
            st.error(f"Error loading columns: {e}")
            return None
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

    extra_cols = set(df.columns) - set(expected_columns)
    if extra_cols:
        df = df.drop(columns=extra_cols)  # Remove unexpected columns

    df = df[expected_columns]  # Reorder columns

    return df

# Streamlit UI
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")

# Input fields
st.subheader("🔹 Enter Customer Features:")
user_input = {
    "feature_1": st.number_input("Feature 1", value=0.0),
    "feature_2": st.number_input("Feature 2", value=0.0),
    "feature_3": st.selectbox("Feature 3 Category", ["A", "B", "C"]),
}

# Prediction button
if st.button("🔍 Predict"):
    input_data = preprocess_input(user_input)

    if lgbm_mlp_model is not None:
        try:
            # Ensure model supports prediction
            if hasattr(lgbm_mlp_model, "predict"):
                prediction = lgbm_mlp_model.predict(input_data)
                result = "Churn" if prediction[0] == 1 else "Not Churn"
                st.write(f"**Prediction:** {result}")

            else:
                st.error("Error: The model does not support 'predict'.")

            # Ensure model supports probability prediction
            if hasattr(lgbm_mlp_model, "predict_proba"):
                churn_probability = lgbm_mlp_model.predict_proba(input_data)[:, 1]
                st.write(f"**Churn Probability:** {churn_probability[0]:.2%}")

        except Exception as e:
            st.error(f"An error occurred while making a prediction: {e}")
