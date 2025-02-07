import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the model and scaler

# --- Configuration ---
PAGE_TITLE = "Customer Churn Prediction"

# --- Functions ---
def load_artifacts():
    try:
        model = joblib.load("lgbm_mlp_model.pkl")
        scaler = joblib.load("scaler.pkl")
        columns = joblib.load("columns.pkl")
        return model, scaler, columns
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

# --- App Layout ---
st.set_page_config(page_title=PAGE_TITLE, layout="wide")

st.title(PAGE_TITLE)

# --- Load Artifacts ---
model, scaler, columns = load_artifacts()

if not all([model, scaler, columns]):
    st.stop()  # Halt if artifacts are missing

# --- Side Panel for User Input ---
st.sidebar.header("User Input")

# 1. Gather Input Features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=1000.0)
total_services = st.sidebar.slider("Total Services", min_value=0, max_value=6, value=3)  # Assuming a max of 6 based on notebook
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
tenure_group_established = st.sidebar.selectbox("Tenure Group Established", [0, 1])

# 2. Create a DataFrame from User Inputs
user_data = {
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "PaperlessBilling": [paperless_billing],
    "TotalCharges": [total_charges],
    "TotalServices": [total_services],
    "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
    "InternetService_No": [1 if internet_service == "No" else 0],
    "Contract_One year": [1 if contract == "One year" else 0],
    "Contract_Two year": [1 if contract == "Two year" else 0],
    "PaymentMethod_Credit card (automatic)": [1 if payment_method == "Credit card (automatic)" else 0],
    "PaymentMethod_Electronic check": [1 if payment_method == "Electronic check" else 0],
    "PaymentMethod_Mailed check": [1 if payment_method == "Mailed check" else 0],
    "MultipleLines_No phone service": [1 if multiple_lines == "No phone service" else 0],
    "MultipleLines_Yes": [1 if multiple_lines == "Yes" else 0],
    "Tenure_Group_Established": [tenure_group_established]
}
user_df = pd.DataFrame(user_data)

# 3. Ensure User DataFrame has same columns and order as trained model (Important!)
user_df = user_df.reindex(columns=columns, fill_value=0)

# 4. Preprocess the User Input

# Scale numeric features
user_df[["TotalCharges", "TotalServices"]] = scaler.transform(user_df[["TotalCharges", "TotalServices"]])

# Verify X_scaled has right columns before using model:
missing_cols = set(columns) - set(user_df.columns)
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
else:
    extra_cols = set(user_df.columns) - set(columns)
    if extra_cols:
        st.error(f"Extra columns: {extra_cols}")

# Make prediction
if st.button("Predict"):
    if missing_cols or extra_cols:
        st.error("Fix the column mismatch before predicting.")
    else:
        y_proba = model.predict_proba(user_df)[0, 1]  # Get churn probability

        st.write("Churn Probability:", y_proba)
        if y_proba > 0.5:
            st.warning("Customer is likely to churn.")
        else:
            st.success("Customer is unlikely to churn.")

st.write("Some additional info")

# Example
#     python -m streamlit run app.py
