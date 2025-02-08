import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('churn_model.joblib')

model = load_model()

# Create the input form
st.subheader("Customer Information")

# Create form with exact matching fields from your data
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Basic Information
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        # Internet Service
        internet_service = st.selectbox(
            "Internet Service", 
            ["DSL", "Fiber optic", "No"]
        )
        
        # Contract
        contract = st.selectbox(
            "Contract", 
            ["Month-to-month", "One year", "Two year"]
        )
        
    with col2:
        # Services
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        
        # Billing
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        
        # Charges
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=50.0
        )
        
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Create DataFrame with user inputs
    data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'TotalCharges': monthly_charges * tenure,  # Approximation
        'TotalServices': sum([
            phone_service == "Yes",
            internet_service != "No",
            multiple_lines == "Yes"
        ]),
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'Tenure_Group_Established': 1 if tenure > 12 else 0  # Assuming established means >12 months
    }
    
    # Create DataFrame with exact column order
    input_df = pd.DataFrame([data])[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                                   'PaperlessBilling', 'TotalCharges', 'TotalServices',
                                   'InternetService_Fiber optic', 'InternetService_No',
                                   'Contract_One year', 'Contract_Two year',
                                   'PaymentMethod_Credit card (automatic)',
                                   'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
                                   'MultipleLines_No phone service', 'MultipleLines_Yes',
                                   'Tenure_Group_Established']]
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Display results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ High Risk of Churn")
        else:
            st.success("✅ Low Risk of Churn")
            
    with col2:
        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )
    
    # Display gauge chart for probability
    st.progress(probability)
    
    # Show feature values used
    with st.expander("View processed features"):
        st.dataframe(input_df)

# Add information about the model
with st.expander("Model Information"):
    st.write("""
    This model uses a voting ensemble of Neural Network and LightGBM classifiers.
    Performance metrics on test data:
    - Accuracy: 0.786
    - ROC-AUC: 0.829
    - F1-Score: 0.597
    """)
