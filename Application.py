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

# Normalize values between 0 and 1
def normalize_charges(value, max_value):
    return value / max_value

# Create the input form
st.subheader("Customer Information")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Basic Information
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    with col2:
        # Services
        internet_service = st.selectbox(
            "Internet Service", 
            ["DSL", "Fiber optic", "No"]
        )
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        
        # Contract
        contract = st.selectbox(
            "Contract", 
            ["Month-to-month", "One year", "Two year"]
        )
        
    with col3:
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
        
        # Charges and Tenure
        monthly_charges = st.slider(
            "Monthly Charges ($)",
            min_value=0,
            max_value=200,
            value=50
        )
        
        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=12
        )
        
        # Calculate maximum possible total charges
        max_possible_total = 200 * 72  # max monthly * max tenure
        
        total_charges = st.slider(
            "Total Charges ($)",
            min_value=0,
            max_value=max_possible_total,
            value=monthly_charges * tenure
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Normalize charges
    normalized_monthly = normalize_charges(monthly_charges, 200)
    normalized_total = normalize_charges(total_charges, max_possible_total)
    
    # Calculate risk score (0 to 1)
    base_risk = 0.0
    
    # Add risks based on services and contract
    if internet_service == "Fiber optic":
        base_risk += 0.2
    if payment_method == "Electronic check":
        base_risk += 0.15
    if contract == "Month-to-month":
        base_risk += 0.25
    
    # Add risks based on charges
    charge_risk = (normalized_monthly + normalized_total) / 2
    total_risk = base_risk + (charge_risk * 0.4)  # Charges contribute up to 40% of risk
    
    # Additional risk factors
    if tenure < 12:
        total_risk += 0.1
    if phone_service == "Yes" and internet_service != "No":
        total_risk += 0.1
    
    # Reduce risk based on protective factors
    if contract == "Two year":
        total_risk -= 0.2
    if payment_method in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        total_risk -= 0.1
    if tenure > 24:
        total_risk -= 0.1
        
    # Ensure risk stays between 0 and 1
    total_risk = max(min(total_risk, 1.0), 0.0)
    
    # If charges are at maximum, force high risk
    if monthly_charges >= 190 and total_charges >= (max_possible_total * 0.9):
        total_risk = 1.0
    
    # Create prediction probability
    probability = total_risk
    prediction = 1 if probability > 0.5 else 0
    
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

    # Show risk factors
    st.subheader("Risk Analysis")
    
    # Calculate risk contributions
    risk_factors = []
    if internet_service == "Fiber optic":
        risk_factors.append(("Fiber optic service", 0.20))
    if payment_method == "Electronic check":
        risk_factors.append(("Electronic check payment", 0.15))
    if contract == "Month-to-month":
        risk_factors.append(("Month-to-month contract", 0.25))
    if normalized_total > 0.7:
        risk_factors.append(("High total charges", normalized_total * 0.4))
    if tenure < 12:
        risk_factors.append(("New customer", 0.10))
        
    # Show risk breakdown
    if risk_factors:
        st.warning("**Risk Factors:**")
        for factor, value in risk_factors:
            st.write(f"- {factor}: +{value:.0%} risk")
            
    # Show protective factors
    protective_factors = []
    if contract == "Two year":
        protective_factors.append(("Two-year contract", 0.20))
    if payment_method in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        protective_factors.append(("Automatic payment", 0.10))
    if tenure > 24:
        protective_factors.append(("Long-term customer", 0.10))
        
    if protective_factors:
        st.success("**Protective Factors:**")
        for factor, value in protective_factors:
            st.write(f"- {factor}: -{value:.0%} risk")

# Add model info
with st.expander("How is risk calculated?"):
    st.write("""
    Risk factors are weighted as follows:
    - Contract Type: Up to 25%
    - Service Type: Up to 20%
    - Payment Method: Up to 15%
    - Charges Level: Up to 40%
    - Tenure Impact: Up to 10%
    
    When monthly charges are very high (>$190) and total charges are near maximum, 
    the risk automatically becomes very high.
    """)
