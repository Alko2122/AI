import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ✅ Define CombinedModel BEFORE loading the model
class CombinedModel:
    def __init__(self, gbm, mlp):
        self.gbm = gbm
        self.mlp = mlp

    def predict_proba(self, X):
        gbm_preds = self.gbm.predict_proba(X)[:, 1]  
        mlp_preds = self.mlp.predict_proba(X)[:, 1]  
        combined_preds = (gbm_preds + mlp_preds) / 2  
        return np.column_stack([1 - combined_preds, combined_preds])  

# ✅ Now load the model
try:
    gbm_mlp_model = joblib.load("lgbm_mlp_model.pkl")
    expected_columns = joblib.load("columns.pkl")  
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")


# Function to preprocess user input  
def preprocess_input(user_input):  
    """Prepares input to match the trained model's feature set."""  
    df = pd.DataFrame([user_input])  

    # Apply one-hot encoding (same as training)  
    df = pd.get_dummies(df)  

    # Ensure correct feature ordering  
    df = df.reindex(columns=expected_columns, fill_value=0)  

    return df  

# Streamlit UI  
st.title("Customer Churn Prediction App")  

# Get user input  
user_input = {  
    "TotalCharges": st.number_input("Total Charges", min_value=0.0, step=0.01),  
    "InternetService_Fiber optic": st.selectbox("Fiber Optic Internet", [0, 1]),  
    "Contract_Two year": st.selectbox("Two-Year Contract", [0, 1]),  
    "TotalServices": st.number_input("Total Services Used", min_value=0, step=1),  
    "Contract_One year": st.selectbox("One-Year Contract", [0, 1]),  
    "gender": st.selectbox("Gender (1=Male, 0=Female)", [0, 1]),  
    "PaperlessBilling": st.selectbox("Paperless Billing (1=Yes, 0=No)", [0, 1]),  
    "MultipleLines_Yes": st.selectbox("Multiple Lines (1=Yes, 0=No)", [0, 1]),  
    "PaymentMethod_Electronic check": st.selectbox("Electronic Check Payment", [0, 1]),  
    "Tenure_Group_Established": st.selectbox("Established Tenure Group (1=Yes, 0=No)", [0, 1]),  
    "Partner": st.selectbox("Partner (1=Yes, 0=No)", [0, 1]),  
    "Dependents": st.selectbox("Dependents (1=Yes, 0=No)", [0, 1]),  
    "SeniorCitizen": st.selectbox("Senior Citizen (1=Yes, 0=No)", [0, 1]),  
    "InternetService_No": st.selectbox("No Internet Service (1=Yes, 0=No)", [0, 1]),  
    "PaymentMethod_Credit card (automatic)": st.selectbox("Credit Card Payment", [0, 1]),  
    "PaymentMethod_Mailed check": st.selectbox("Mailed Check Payment", [0, 1]),  
    "MultipleLines_No phone service": st.selectbox("No Phone Service (1=Yes, 0=No)", [0, 1])  
}  

if st.button("Predict"):  
    # Process input  
    input_data = preprocess_input(user_input)  

    # Make prediction  
    prediction = gbm_mlp_model.predict(input_data)  
    churn_probability = gbm_mlp_model.predict_proba(input_data)[:, 1]  

    # Show result  
    st.write(f"**Prediction:** {'Churn' if prediction[0] == 1 else 'Not Churn'}")  
    st.write(f"**Churn Probability:** {churn_probability[0]:.2%}")  
