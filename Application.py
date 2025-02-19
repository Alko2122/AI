import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Page config
st.set_page_config(page_title="Telco Customer Service", layout="wide")

def get_assistant_response(message, context=None):
    """
    Improved AI assistant for telecom package recommendations with better intent recognition
    """
    message = message.lower()
    
    # Package information
    packages = {
        'basic': {
            'name': 'Basic Package',
            'price': 50,
            'internet': 'DSL (50 Mbps)',
            'features': ['Email Support', 'Basic Internet'],
            'best_for': 'Basic browsing and email'
        },
        'standard': {
            'name': 'Standard Package',
            'price': 85,
            'internet': 'Fiber Optic (200 Mbps)',
            'features': ['24/7 Support', 'Phone Service', 'High-Speed Internet'],
            'best_for': 'Streaming and gaming'
        },
        'premium': {
            'name': 'Premium Package',
            'price': 120,
            'internet': 'Fiber Optic (500 Mbps)',
            'features': ['Premium Support', 'Phone Service', 'Ultra-Fast Internet', 'Cloud Storage', 'Security Suite'],
            'best_for': 'Heavy users and businesses'
        }
    }
    
    # Intent-based responses
    keywords = {
        "compare": ["compare", "difference", "vs"],
        "price": ["price", "cost", "cheap", "affordable"],
        "speed": ["speed", "internet speed", "fast", "slow"],
        "streaming": ["gaming", "stream", "netflix", "youtube", "4k"],
        "business": ["business", "work", "company", "office"],
        "recommend": ["recommend", "suggest", "best for me"],
        "thank": ["thank", "thanks"],
        "greeting": ["hi", "hello", "hey"],
        "cancellation": ["cancel", "end subscription", "terminate"],
        "discounts": ["discount", "promo", "offer", "deals"],
        "loyalty": ["reward", "loyalty", "bonus", "long-term"]
    }

    def check_keywords(category):
        return any(word in message for word in keywords[category])
    
    # Responses based on intent
    if check_keywords("compare"):
        return """Here's a package comparison:

📦 **Basic** ($50/month): 50 Mbps DSL – good for browsing  
📦 **Standard** ($85/month): 200 Mbps Fiber – great for streaming  
📦 **Premium** ($120/month): 500 Mbps Fiber – ideal for businesses  

Would you like help choosing the best one for you?"""

    elif check_keywords("price"):
        return """Our package pricing:
- **Basic:** $50/month
- **Standard:** $85/month
- **Premium:** $120/month  

Would you like a breakdown of features for each package?"""

    elif check_keywords("speed"):
        return """Our internet speeds:
- **Basic:** 50 Mbps (DSL)
- **Standard:** 200 Mbps (Fiber)
- **Premium:** 500 Mbps (Fiber)  

What do you usually use the internet for?"""

    elif check_keywords("streaming"):
        return """For streaming and gaming, I recommend **Standard (200 Mbps Fiber)**.  
It offers **smooth 4K streaming** and low-latency gaming.  

Would you like details on its features?"""

    elif check_keywords("business"):
        return """For businesses, our **Premium Package (500 Mbps Fiber)** is ideal.  
It includes **priority support, cloud storage, and security features**.  

Would you like a tailored business plan?"""

    elif check_keywords("recommend"):
        return """I can help you choose the best package!  
Let me know:
1️⃣ What do you mainly use the internet for?  
2️⃣ How many people will be using it?  
3️⃣ What’s your monthly budget?"""

    elif check_keywords("cancellation"):
        return """You can cancel your subscription anytime.  
**Things to know before canceling:**  
- Month-to-month plans can be canceled instantly.  
- One-year and two-year contracts may have **early termination fees**.  
Would you like help switching to a cheaper plan instead?"""

    elif check_keywords("discounts"):
        return """💰 **Current Offers:**  
🎉 Get **10% off for 6 months** on our Standard & Premium plans!  
🎯 New customers can get a **$25 sign-up bonus**.  
Would you like me to apply a promo for you?"""

    elif check_keywords("loyalty"):
        return """🎖️ **Loyalty Rewards:**  
- **6+ months customers**: Free speed upgrade  
- **12+ months customers**: 5% monthly discount  
- **24+ months customers**: Free Wi-Fi extender  

Would you like to check your eligibility?"""

    elif check_keywords("thank"):
        return "You're welcome! 😊 Let me know if you need anything else."

    elif check_keywords("greeting"):
        return """Hello! 👋 I'm here to help with:  
✅ Package recommendations  
✅ Speed comparisons  
✅ Price information  
✅ Discounts & loyalty rewards  

What would you like to know?"""

    # Specific package inquiries
    for package in packages:
        if package in message:
            pkg = packages[package]
            return f"""📦 **{pkg['name']}** (${pkg['price']}/month)  
🚀 Speed: {pkg['internet']}  
🔹 Features: {', '.join(pkg['features'])}  
💡 Best for: {pkg['best_for']}  

Would you like to compare it with other options?"""

    # Default fallback response
    return """I'm here to assist with telecom packages!  
You can ask about:  
- Package comparisons  
- Internet speeds  
- Prices & features  
- Discounts & offers  

What can I help you with?"""

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    # Add initial greeting
    st.session_state.chat_history.append(
        ("assistant", "Hello! 👋 I'm your telecom package assistant. How can I help you today?")
    )

# Main app - AI Assistant Sidebar
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
        # Add user message to history
        st.session_state.chat_history.append(("user", user_input))
        
        # Get assistant response
        ai_response = get_assistant_response(user_input)
        
        # Add assistant response to history
        st.session_state.chat_history.append(("assistant", ai_response))
        
        # Rerun to update chat display
        st.rerun()

# Quick action buttons
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Actions:**")
cols = st.sidebar.columns(2)
if cols[0].button("Compare Packages"):
    comparison_query = "Compare packages"
    st.session_state.chat_history.append(("user", comparison_query))
    ai_response = get_assistant_response(comparison_query)
    st.session_state.chat_history.append(("assistant", ai_response))
    st.rerun()

if cols[1].button("Best Deals"):
    deals_query = "What are your best deals?"
    st.session_state.chat_history.append(("user", deals_query))
    ai_response = get_assistant_response(deals_query)
    st.session_state.chat_history.append(("assistant", ai_response))
    st.rerun()

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.chat_history = []
    initial_ai_msg = "Hello! 👋 I'm your telecom package assistant. How can I help you today?"
    st.session_state.chat_history.append(("assistant", initial_ai_msg))
    st.rerun()

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
