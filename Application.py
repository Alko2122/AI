import streamlit as st
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
import json
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from sklearn.preprocessing import StandardScaler

# Load the churn prediction model
churn_model = joblib.load("churn_model.joblib")

df = pd.read_csv("Dataset.csv")  # Load dataset
scaler = StandardScaler()

# FastAPI for AI-driven recommendations
app = FastAPI()

class CustomerData(BaseModel):
    age: int
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    payment_method: str

def preprocess_customer_data(data: CustomerData):
    # Convert categorical to numerical if needed
    data_dict = data.dict()
    df_temp = pd.DataFrame([data_dict])
    df_temp = pd.get_dummies(df_temp)
    return df_temp

@app.post("/predict_churn/")
def predict_churn(data: CustomerData):
    processed_data = preprocess_customer_data(data)
    prediction = churn_model.predict(processed_data)
    return {"churn_probability": prediction.tolist()}

@app.post("/recommend_plan/")
def recommend_plan(data: CustomerData):
    # Use past trends to recommend plans
    suitable_plans = df[df['monthly_charges'] <= data.monthly_charges].nlargest(3, 'tenure')
    return {"recommended_plans": suitable_plans[['plan_name', 'monthly_charges']].to_dict(orient='records')}

# Streamlit chatbot interface
st.title("Telco AI Chatbot")

with st.sidebar:
    st.header("Chat with AI")
    user_input = st.text_input("Ask me anything about your telecom plan:")
    
    if user_input:
        if "churn" in user_input.lower():
            st.write("Would you like a churn prediction? Please enter your details.")
            age = st.number_input("Age", min_value=18, max_value=100)
            tenure = st.number_input("Tenure", min_value=0, max_value=100)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
            total_charges = st.number_input("Total Charges", min_value=0.0)
            contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
            payment_method = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Credit Card", "Bank Transfer"])
            
            if st.button("Predict Churn"):
                input_data = CustomerData(
                    age=int(age), tenure=int(tenure), monthly_charges=float(monthly_charges),
                    total_charges=float(total_charges), contract=contract, payment_method=payment_method
                )
                response = predict_churn(input_data)
                st.write(response)
        
        elif "recommend" in user_input.lower() or "plan" in user_input.lower():
            st.write("Tell me about your budget, and I'll suggest a plan.")
            monthly_budget = st.number_input("Enter your budget:", min_value=0.0)
            
            if st.button("Get Plan Recommendations"):
                input_data = CustomerData(
                    age=30, tenure=12, monthly_charges=float(monthly_budget),
                    total_charges=monthly_budget * 12, contract="Month-to-Month", payment_method="Electronic Check"
                )
                response = recommend_plan(input_data)
                st.write(response)
        
        else:
            st.write("I can help with churn prediction and plan recommendations. Try asking about those!")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Main app - Churn Prediction
st.title("Customer Churn Prediction")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('churn_model.joblib')

model = load_model()

# Create the input form
st.subheader("Customer Information")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    with col2:
        internet_service = st.selectbox(
            "Internet Service", 
            ["DSL", "Fiber optic", "No"]
        )
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        contract = st.selectbox(
            "Contract", 
            ["Month-to-month", "One year", "Two year"]
        )
        
    with col3:
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
        
        max_possible_total = 200 * 72
        total_charges = st.slider(
            "Total Charges ($)",
            min_value=0,
            max_value=max_possible_total,
            value=monthly_charges * tenure
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    def calculate_tenure_impact(tenure_months):
        if tenure_months <= 6:
            return 0.3
        elif tenure_months <= 12:
            return 0.2
        elif tenure_months <= 24:
            return 0.1
        elif tenure_months <= 48:
            return -0.1
        else:
            return -0.2

    def calculate_charges_risk(monthly, total, tenure):
        monthly_normalized = monthly / 200.0
        expected_total = monthly * tenure
        total_ratio = total / max(expected_total, 1)
        charge_risk = monthly_normalized * 0.4
        
        if total_ratio > 1.2:
            charge_risk += 0.2
        elif total_ratio < 0.8:
            charge_risk -= 0.1
            
        return min(charge_risk, 0.6)

    # Calculate base risk
    base_risk = 0.0
    
    if internet_service == "Fiber optic":
        base_risk += 0.15
    if payment_method == "Electronic check":
        base_risk += 0.15
    if contract == "Month-to-month":
        base_risk += 0.20
    elif contract == "One year":
        base_risk -= 0.15
    elif contract == "Two year":
        base_risk -= 0.30
    
    # Calculate risks
    tenure_impact = calculate_tenure_impact(tenure)
    charges_risk = calculate_charges_risk(monthly_charges, total_charges, tenure)
    
    # Combine all risk factors
    total_risk = base_risk + tenure_impact + charges_risk
    
    # Additional service-based adjustments
    if phone_service == "Yes" and internet_service != "No":
        total_risk += 0.05
    if paperless_billing == "Yes":
        total_risk += 0.05
    
    # Payment method adjustments
    if payment_method in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        total_risk -= 0.10
    
    # Special cases
    if monthly_charges >= 190 and tenure <= 6:
        total_risk += 0.3  # Very high risk for new customers with high charges
    elif monthly_charges >= 190 and tenure >= 48:
        total_risk -= 0.1  # Lower risk for long-term customers even with high charges
    
    # Ensure risk stays between 0 and 1
    total_risk = max(min(total_risk, 1.0), 0.0)
    
    # Convert to probability
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

    # Show detailed risk breakdown
    st.subheader("Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk Factors:**")
        if charges_risk > 0:
            st.write(f"- Charges Impact: +{charges_risk:.0%}")
        if tenure_impact > 0:
            st.write(f"- Tenure Risk (Customer Age): +{tenure_impact:.0%}")
        if internet_service == "Fiber optic":
            st.write("- Fiber Service: +15%")
        if payment_method == "Electronic check":
            st.write("- Electronic Check Payment: +15%")
        if contract == "Month-to-month":
            st.write("- Month-to-month Contract: +20%")
            
    with col2:
        st.write("**Protective Factors:**")
        if tenure_impact < 0:
            st.write(f"- Long Tenure Benefit: {tenure_impact:.0%}")
        if contract == "Two year":
            st.write("- Two-year Contract: -30%")
        elif contract == "One year":
            st.write("- One-year Contract: -15%")
        if payment_method in ["Bank transfer (automatic)", "Credit card (automatic)"]:
            st.write("- Automatic Payment: -10%")

    # Show relationship explanation
    st.write("\n**Key Relationships:**")
    st.info(f"""
    - Tenure {tenure} months: {'Very Stable' if tenure > 48 else 'Stable' if tenure > 24 else 'Moderate' if tenure > 12 else 'New Customer'}
    - Monthly Charges ${monthly_charges}: {'Very High' if monthly_charges >= 150 else 'High' if monthly_charges >= 100 else 'Moderate' if monthly_charges >= 50 else 'Low'}
    - Impact of Tenure on Charges Risk: {'Significantly Reduced' if tenure > 48 else 'Reduced' if tenure > 24 else 'Slightly Reduced' if tenure > 12 else 'No Reduction'}
    """)

# Add model info
with st.expander("How is risk calculated?"):
    st.write("""
    Risk factors are weighted as follows:
    - Contract Type: -30% to +20%
    - Tenure Impact: -20% to +30%
    - Charges Impact: up to 60%
    - Internet Service: up to 15%
    - Payment Method: -10% to +15%
    
    Special considerations:
    - New customers (≤6 months) with high charges have increased risk
    - Long-term customers (>48 months) have reduced risk even with high charges
    - Multiple services and paperless billing add small additional risk
    - Automatic payments provide risk reduction
    """)
