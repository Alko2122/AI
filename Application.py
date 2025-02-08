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

# Function to scale total charges
def scale_total_charges(value):
    # These values are from your actual data distribution
    mean = -0.004226
    std = 0.997087
    return (value - mean) / std

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
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,
            step=5.0
        )
        
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )
        
        # Use a slider for relative total charges
        total_charges_scale = st.slider(
            "Total Charges Level",
            min_value=-1.0,
            max_value=3.0,
            value=0.0,
            step=0.1,
            help="Low (-1) to Very High (3)"
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
        'TotalCharges': total_charges_scale,  # Already scaled value
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
        'Tenure_Group_Established': 1 if tenure > 12 else 0
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

    # Show risk factors
    high_risk_factors = []
    low_risk_factors = []
    
    # Check for high-risk factors with their impact scores
    risk_scores = 0
    if data['InternetService_Fiber optic']:
        high_risk_factors.append("Fiber optic service (31.9% correlation)")
        risk_scores += 0.319
    if payment_method == "Electronic check":
        high_risk_factors.append("Electronic check payment (29.7% correlation)")
        risk_scores += 0.297
    if contract == "Month-to-month":
        high_risk_factors.append("Month-to-month contract")
        risk_scores += 0.25
    if total_charges_scale > 0.5:
        high_risk_factors.append(f"High total charges (19.7% correlation)")
        risk_scores += 0.197
    if data['TotalServices'] > 2:
        high_risk_factors.append("Multiple services (8.1% correlation)")
        risk_scores += 0.081
    
    # Check for low-risk factors
    stability_score = 0
    if contract in ["Two year"]:
        low_risk_factors.append("Two-year contract (30.3% correlation)")
        stability_score += 0.303
    elif contract == "One year":
        low_risk_factors.append("One-year contract (18.8% correlation)")
        stability_score += 0.188
    if data['Tenure_Group_Established']:
        low_risk_factors.append("Established customer (6.3% correlation)")
        stability_score += 0.063
    
    # Display risk analysis
    col1, col2 = st.columns(2)
    with col1:
        if high_risk_factors:
            st.warning("**Higher Risk Factors:**\n- " + "\n- ".join(high_risk_factors))
            st.metric("Combined Risk Score", f"{risk_scores:.2f}")
    with col2:
        if low_risk_factors:
            st.success("**Stability Factors:**\n- " + "\n- ".join(low_risk_factors))
            st.metric("Stability Score", f"{stability_score:.2f}")

# Add information about the model
with st.expander("Model Information"):
    st.write("""
    This model uses a voting ensemble of Neural Network and LightGBM classifiers.
    Performance metrics on test data:
    - Accuracy: 0.786
    - ROC-AUC: 0.829
    - F1-Score: 0.597
    
    The model considers multiple factors with varying levels of importance:
    1. Total Charges (2904) - Highest impact
    2. Total Services (510)
    3. Demographics (370)
    4. Paperless Billing (254)
    5. Internet Service Type (220)
    6. Contract Type (163)
    """)
