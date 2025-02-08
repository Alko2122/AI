import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# --- Define the CombinedModel class here ---
class CombinedModel:
    def __init__(self, gbm, mlp):
        self.gbm = gbm
        self.mlp = mlp

    def predict_proba(self, X):
        gbm_preds = self.gbm.predict_proba(X)[:, 1]
        mlp_preds = self.mlp.predict_proba(X)[:, 1]
        combined_preds = (gbm_preds + mlp_preds) / 2
        return np.column_stack([1 - combined_preds, combined_preds])

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
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction")

# --- Load Artifacts ---
model, scaler, columns = load_artifacts()

if not all([model, scaler, columns]):
    st.stop()

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

# 2. Create a DataFrame from User Inputs (Unscaled)
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

# 3. Ensure User DataFrame has same columns as training data (Important!)
user_df = user_df.reindex(columns=columns, fill_value=0)

# 4. Now, create a *new* DataFrame with only the *numeric* columns, in the right order
numeric_cols = ["TotalCharges", "TotalServices"]
user_df_scaled = pd.DataFrame(columns=columns)

# Iterate over and transform.
user_df_scaled[numeric_cols] = scaler.transform(user_df[numeric_cols])
# 5. Now, the Model will work
# Make prediction
if st.button("Predict"):
    # Now, the transform was used on the data set to get only the selected columns
    y_proba = model.predict_proba(user_df_scaled)[0, 1]  # Get churn probability

    st.write("Churn Probability:", y_proba)
    if y_proba > 0.5:
        st.warning("Customer is likely to churn.")
    else:
        st.success("Customer is unlikely to churn.")

st.write("Some additional info")
