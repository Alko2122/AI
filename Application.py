import streamlit as st
import pandas as pd
import numpy as np
import joblib
import openai
from datetime import datetime

# Page config
st.set_page_config(page_title="Telco Customer Service", layout="wide")

# Configure OpenAI - Replace this with your API key
openai.api_key = "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"

# Package Information for AI
PACKAGE_INFO = """
1. Basic Package ($50/month):
   - DSL Internet (50 Mbps)
   - Email Support
   - Ideal for basic browsing and email

2. Standard Package ($85/month):
   - Fiber Optic Internet (200 Mbps)
   - Phone Service
   - 24/7 Technical Support
   - Great for streaming and gaming

3. Premium Package ($120/month):
   - Fiber Optic Internet (500 Mbps)
   - Phone Service
   - Premium 24/7 Support
   - Extra Features (Cloud Storage, Security Suite)
   - Perfect for heavy users and businesses
"""

def get_ai_response(history, new_message, package_info):
    messages = [
        {"role": "system", "content": f"""You are a helpful telecommunications service assistant. 
        Here are the available packages:
        {package_info}
        
        Your goal is to help customers choose the best package based on their needs.
        Be concise, friendly, and provide specific package recommendations when appropriate.
        Always maintain the conversation context and remember previous customer preferences.
        Stay focused on telecom packages and related services."""}
    ]
    
    for msg in history:
        messages.append({
            "role": "user" if msg[0] == "user" else "assistant",
            "content": msg[1]
        })
    
    messages.append({"role": "user", "content": new_message})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"I apologize, but I'm having trouble connecting. Please try again or contact our support team. Error: {str(e)}"

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    initial_ai_msg = get_ai_response([], "Start a new conversation to help choose a package", PACKAGE_INFO)
    st.session_state.chat_history.append(("assistant", initial_ai_msg))

# Sidebar for AI Assistant
st.sidebar.title("💬 AI Package Assistant")

# Chat interface
for message in st.session_state.chat_history:
    role, content = message
    if role == "user":
        st.sidebar.markdown(f"👤 **You:** {content}")
    else:
        st.sidebar.markdown(f"🤖 **Assistant:** {content}")

# User input
user_input = st.sidebar.text_input("Type your message here...")
if st.sidebar.button("Send"):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        ai_response = get_ai_response(st.session_state.chat_history[:-1], user_input, PACKAGE_INFO)
        st.session_state.chat_history.append(("assistant", ai_response))
        st.experimental_rerun()

# Quick action buttons
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions:**")
cols = st.sidebar.columns(2)
if cols[0].button("Compare Packages"):
    comparison_query = "Can you compare all available packages and their features?"
    st.session_state.chat_history.append(("user", comparison_query))
    ai_response = get_ai_response(st.session_state.chat_history[:-1], comparison_query, PACKAGE_INFO)
    st.session_state.chat_history.append(("assistant", ai_response))
    st.experimental_rerun()

if cols[1].button("Best Deals"):
    deals_query = "What are the current best deals or promotions?"
    st.session_state.chat_history.append(("user", deals_query))
    ai_response = get_ai_response(st.session_state.chat_history[:-1], deals_query, PACKAGE_INFO)
    st.session_state.chat_history.append(("assistant", ai_response))
    st.experimental_rerun()

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    initial_ai_msg = get_ai_response([], "Start a new conversation to help choose a package", PACKAGE_INFO)
    st.session_state.chat_history.append(("assistant", initial_ai_msg))
    st.experimental_rerun()

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
    total_risk = base_risk + tenure_impact + charges_risk
    
    # Additional adjustments
    if phone_service == "Yes" and internet_service != "No":
        total_risk += 0.05
    if paperless_billing == "Yes":
        total_risk += 0.05
    if payment_method in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        total_risk -= 0.10
    
    # Special cases
    if monthly_charges >= 190 and tenure <= 6:
        total_risk += 0.3
    elif monthly_charges >= 190 and tenure >= 48:
        total_risk -= 0.1
    
    total_risk = max(min(total_risk, 1.0), 0.0)
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
        st.metric("Churn Probability", f"{probability:.1%}")
    
    st.progress(probability)

    # Risk Analysis
    st.subheader("Risk Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Risk Factors:**")
        if charges_risk > 0:
            st.write(f"- Charges Impact: +{charges_risk:.0%}")
        if tenure_impact > 0:
            st.write(f"- Tenure Risk: +{tenure_impact:.0%}")
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

    # Relationship explanation
    st.write("\n**Key Relationships:**")
    st.info(f"""
    - Tenure {tenure} months: {'Very Stable' if tenure > 48 else 'Stable' if tenure > 24 else 'Moderate' if tenure > 12 else 'New Customer'}
    - Monthly Charges ${monthly_charges}: {'Very High' if monthly_charges >= 150 else 'High' if monthly_charges >= 100 else 'Moderate' if monthly_charges >= 50 else 'Low'}
    - Impact of Tenure on Charges Risk: {'Significantly Reduced' if tenure > 48 else 'Reduced' if tenure > 24 else 'Slightly Reduced' if tenure > 12 else 'No Reduction'}
    """)

# Model information
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
