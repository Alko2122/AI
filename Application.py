import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ✅ Define the CombinedModel class
class CombinedModel:
    def __init__(self, gbm, mlp):
        self.gbm = gbm
        self.mlp = mlp

    def predict(self, X):
        gbm_preds = self.gbm.predict(X)
        mlp_preds = self.mlp.predict(X)
        return (gbm_preds + mlp_preds) / 2  # Average predictions

    def predict_proba(self, X):
        gbm_preds = self.gbm.predict_proba(X)[:, 1]
        mlp_preds = self.mlp.predict_proba(X)[:, 1]
        combined_preds = (gbm_preds + mlp_preds) / 2
        return np.column_stack([1 - combined_preds, combined_preds])  # Convert to probability format

# ✅ Define file paths
MODEL_PATH = "lgbm_mlp_model.pkl"
FEATURES_PATH = "feature_names.xlsx"

# ✅ Load model
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"Error: Model file '{MODEL_PATH}' not found!")
        return None

# ✅ Load feature names from Excel
def load_feature_names():
    try:
        feature_df = pd.read_excel(FEATURES_PATH)
        return list(feature_df["Feature_Names"])  # Extract feature names
    except FileNotFoundError:
        st.error("Error: 'feature_names.xlsx' not found! Ensure you saved feature names from X_train.")
        return None

# ✅ Load model & feature names
lgbm_mlp_model = load_model()
feature_names = load_feature_names()

if lgbm_mlp_model is None or feature_names is None:
    st.stop()  # Stop execution if model or features are missing

# ✅ Preprocess user input dynamically
def preprocess_input(user_input):
    """Prepares input to match the trained model's feature set."""
    df = pd.DataFrame([user_input])  # Convert input to DataFrame

    # Apply one-hot encoding (same as training)
    df = pd.get_dummies(df)

    # Align columns with training data
    missing_cols = set(feature_names) - set(df.columns)
    for col in missing_cols:
        df[col] = 0  # Add missing columns with default 0

    df = df[feature_names]  # Reorder columns

    return df

# ✅ Streamlit UI
st.title("📊 Customer Churn Prediction App")

st.write("Predict whether a customer is likely to churn based on their profile and service usage.")

# ✅ Input fields
user_input = {
    "age": st.number_input("Age", value=30, min_value=18, max_value=100),
    "monthly_charges": st.number_input("Monthly Charges", value=50.0, min_value=0.0),
    "total_charges": st.number_input("Total Charges", value=500.0, min_value=0.0),
    "contract_type": st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"]),
    "internet_service": st.selectbox("Internet Service", ["DSL", "Fiber Optic", "None"]),
    "payment_method": st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check"]),
    "has_phone_service": st.selectbox("Phone Service", ["Yes", "No"]),
    "has_multiple_lines": st.selectbox("Multiple Lines", ["Yes", "No"]),
    "has_online_security": st.selectbox("Online Security", ["Yes", "No"]),
    "has_online_backup": st.selectbox("Online Backup", ["Yes", "No"]),
    "has_device_protection": st.selectbox("Device Protection", ["Yes", "No"]),
}

# ✅ Predict button
if st.button("🔍 Predict"):
    input_data = preprocess_input(user_input)

    # Check if model supports prediction
    if hasattr(lgbm_mlp_model, "predict"):
        prediction = lgbm_mlp_model.predict(input_data)
    else:
        st.error("Error: The model does not support 'predict'.")
        st.stop()

    # Check if model supports probability prediction
    if hasattr(lgbm_mlp_model, "predict_proba"):
        churn_probability = lgbm_mlp_model.predict_proba(input_data)[:, 1]
    else:
        churn_probability = None  # Some models may not support probabilities

    # ✅ Display results
    result = "❌ Churn" if prediction[0] == 1 else "✅ Not Churn"
    st.subheader(f"**Prediction:** {result}")

    if churn_probability is not None:
        st.write(f"📌 **Churn Probability:** {churn_probability[0]:.2%}")
