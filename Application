import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# Load trained models
gbm_model = joblib.load("gbm_model.pkl")
mlp_model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")  # Ensure the same scaler is used

# Define input fields
st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict whether they will churn.")

# User Inputs
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=24)
monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "None"])
payment_method = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])
paperless_billing = st.radio("Paperless Billing?", ["Yes", "No"])

# Encode categorical variables
contract_map = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
internet_map = {"DSL": 0, "Fiber Optic": 1, "None": 2}
payment_map = {"Electronic Check": 0, "Mailed Check": 1, "Bank Transfer": 2, "Credit Card": 3}
paperless_map = {"Yes": 1, "No": 0}

# Convert input to DataFrame
user_data = pd.DataFrame({
    "tenure": [tenure],
    "monthly_charges": [monthly_charges],
    "contract": [contract_map[contract_type]],
    "internet_service": [internet_map[internet_service]],
    "payment_method": [payment_map[payment_method]],
    "paperless_billing": [paperless_map[paperless_billing]]
})

# Scale the inputs
user_data_scaled = scaler.transform(user_data)

# Predict using both models
gbm_pred = gbm_model.predict_proba(user_data_scaled)[:, 1]  # Probability of churn
mlp_pred = mlp_model.predict_proba(user_data_scaled)[:, 1]

# Ensemble Prediction (average of both models)
final_pred = (gbm_pred + mlp_pred) / 2

# Display Prediction
st.subheader("Prediction Result:")
if final_pred[0] > 0.5:
    st.error(f"🔴 High Churn Risk! (Probability: {final_pred[0]:.2f})")
else:
    st.success(f"🟢 Customer is likely to stay. (Probability: {final_pred[0]:.2f})")

# Feature Importance
if st.checkbox("Show Feature Importance"):
    gbm_importances = pd.Series(gbm_model.feature_importances_, index=user_data.columns).sort_values(ascending=False)
    mlp_importances = permutation_importance(mlp_model, user_data_scaled, [0], scoring="roc_auc", n_repeats=5).importances_mean
    mlp_importances = pd.Series(mlp_importances, index=user_data.columns).sort_values(ascending=False)

    # Combined importances
    combined_importance = (gbm_importances + mlp_importances) / 2
    st.bar_chart(combined_importance)
