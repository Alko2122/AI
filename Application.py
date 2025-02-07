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

# Streamlit UI
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")

# 🔹 Define multiple input features
st.subheader("🔹 Enter Customer Features:")

user_input = {
    "age": st.number_input("Age", min_value=18, max_value=100, value=30, step=1),
    "monthly_charges": st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.1),
    "total_charges": st.number_input("Total Charges ($)", min_value=0.0, value=500.0, step=0.1),
    "contract_type": st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"]),
    "internet_service": st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No Internet"]),
    "payment_method": st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"]),
    "has_phone_service": st.radio("Phone Service", ["Yes", "No"]),
    "has_multiple_lines": st.radio("Multiple Lines", ["Yes", "No", "No Phone Service"]),
    "has_online_security": st.radio("Online Security", ["Yes", "No", "No Internet Service"]),
    "has_online_backup": st.radio("Online Backup", ["Yes", "No", "No Internet Service"]),
    "has_device_protection": st.radio("Device Protection", ["Yes", "No", "No Internet Service"]),
}

# Function to preprocess user input
def preprocess_input(user_input):
    """Prepares input to match the trained model's feature set."""
    df = pd.DataFrame([user_input])  # Convert input to DataFrame

    # ✅ One-hot encode categorical features
    categorical_features = ["contract_type", "internet_service", "payment_method", 
                            "has_phone_service", "has_multiple_lines", 
                            "has_online_security", "has_online_backup", 
                            "has_device_protection"]

    df = pd.get_dummies(df, columns=categorical_features, drop_first=False)  

    # ✅ Ensure all expected columns exist (adding missing ones with 0s)
    missing_cols = set(expected_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0  # Add missing columns with default 0

    # ✅ Ensure correct column order
    df = df[expected_columns]

    # ✅ Debugging: Print transformed DataFrame
    st.write("📌 **Debug Mode: Processed Input DataFrame**")
    st.write(df)

    return df


# Prediction button
if st.button("🔍 Predict"):
    input_data = preprocess_input(user_input)

    # ✅ DEBUGGING: Check input transformation
    st.write("📌 **DEBUG MODE: Checking Input Processing**")
    st.write("Raw User Input:", user_input)
    st.write("Processed Input DataFrame:", input_data)

    if lgbm_mlp_model is not None:
        try:
            # Ensure model supports prediction
            if hasattr(lgbm_mlp_model, "predict"):
                prediction = lgbm_mlp_model.predict(input_data)
                st.write("📌 **Raw Model Output:**", prediction)

                result = "Churn" if prediction[0] >= 0.5 else "Not Churn"  # Ensure correct thresholding
                st.write(f"**Final Prediction:** {result}")

            else:
                st.error("Error: The model does not support 'predict'.")

            # Ensure model supports probability prediction
            if hasattr(lgbm_mlp_model, "predict_proba"):
                churn_probability = lgbm_mlp_model.predict_proba(input_data)[:, 1]
                st.write("📌 **Predicted Churn Probability:**", churn_probability[0])
                st.write(f"**Churn Probability:** {churn_probability[0]:.2%}")

        except Exception as e:
            st.error(f"An error occurred while making a prediction: {e}")

