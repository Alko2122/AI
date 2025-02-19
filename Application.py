import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests


def analyze_historical_data(df):
    if df is None:
        return {
            'avg_tenure': 24,
            'avg_monthly_charges': 70,
            'popular_internet': 'Fiber optic',
            'popular_contract': 'Month-to-month',
            'churn_rate_fiber': 0.4,
            'churn_rate_dsl': 0.2,
            'avg_tenure_churned': 12,
            'avg_tenure_stayed': 36
        }
    
    # Convert Yes/No to 1/0 for churn calculation
    df['Churn_Numeric'] = (df['Churn'] == 'Yes').astype(int)
    
    return {
        'avg_tenure': df['tenure'].mean(),
        'avg_monthly_charges': df['MonthlyCharges'].mean(),
        'popular_internet': df['InternetService'].mode()[0],
        'popular_contract': df['Contract'].mode()[0],
        'churn_rate_fiber': df[df['InternetService']=='Fiber optic']['Churn_Numeric'].mean(),
        'churn_rate_dsl': df[df['InternetService']=='DSL']['Churn_Numeric'].mean(),
        'avg_tenure_churned': df[df['Churn']=='Yes']['tenure'].mean(),
        'avg_tenure_stayed': df[df['Churn']=='No']['tenure'].mean()
    }

def get_data_driven_response(message, insights):
    """Generate responses using historical data insights"""
    message = message.lower()
    
    if 'average' in message and 'tenure' in message:
        return f"""Based on our customer data:
        • Average customer tenure: {insights['avg_tenure']:.1f} months
        • Satisfied customers stay for: {insights['avg_tenure_stayed']:.1f} months
        
        Would you like to know more about our long-term packages?"""
    
    elif 'popular' in message or 'most common' in message:
        return f"""Based on our customer preferences:
        • Most popular internet service: {insights['popular_internet']}
        • Most common contract type: {insights['popular_contract']}
        • Average monthly charges: ${insights['avg_monthly_charges']:.2f}
        
        Would you like to know more about any of these options?"""
    
    elif 'success rate' in message or 'satisfaction' in message:
        fiber_success = (1 - insights['churn_rate_fiber']) * 100
        dsl_success = (1 - insights['churn_rate_dsl']) * 100
        return f"""Our customer satisfaction rates:
        • Fiber optic service: {fiber_success:.1f}% satisfaction rate
        • DSL service: {dsl_success:.1f}% satisfaction rate
        
        Which service would you like to learn more about?"""
    
    return None

def get_rule_based_response(message, insights):
    """Default rule-based responses"""
    message = message.lower()
    
    if 'price' in message or 'cost' in message:
        return """Our packages range from:
        • Basic: $50/month
        • Standard: $85/month
        • Premium: $120/month
        
        Which price range interests you?"""
    
    elif 'internet' in message or 'speed' in message:
        return """We offer:
        • DSL: 50 Mbps
        • Fiber Optic: 200-500 Mbps
        
        What speed are you looking for?"""
    
    elif 'help' in message:
        return """I can help you with:
        • Package recommendations
        • Service comparisons
        • Pricing information
        • Technical support
        
        What would you like to know?"""
    
    return "How can I help you choose the right package? You can ask about prices, speeds, or features."

# Load historical data
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv('dataset.csv')
        return df
    except:
        st.warning("Historical dataset not found. Using default insights.")
        return None

# Initialize historical insights
historical_data = load_historical_data()
historical_insights = analyze_historical_data(historical_data)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append(
        ("assistant", "Hello! 👋 I'm your telecom package assistant. How can I help you today?")
    )
    
def get_huggingface_response(prompt, api_key):
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    # Add context about telecom packages
    full_prompt = f"""You are a telecom service assistant. Here are our packages:
    Basic Package ($50/month): DSL Internet (50 Mbps), Email Support
    Standard Package ($85/month): Fiber Optic (200 Mbps), Phone Service, 24/7 Support
    Premium Package ($120/month): Fiber Optic (500 Mbps), Premium Support, Extra Features

    Customer query: {prompt}
    """
    
    try:
        output = query({
            "inputs": full_prompt,
            "parameters": {"max_length": 150}
        })
        
        if isinstance(output, dict) and 'error' in output:
            return f"I apologize, but I'm having trouble connecting. Please try again or contact our support team. Error: {output['error']}"
            
        return output[0]['generated_text']
    except Exception as e:
        return f"I apologize, but I'm having trouble right now. Please try again later."

def get_assistant_response(message, insights, hf_api_key=None):
    """Enhanced assistant response function with HuggingFace integration"""
    
    # First try to get a data-driven response
    data_response = get_data_driven_response(message, insights)
    if data_response:
        return data_response
        
    # If no data-driven response and HuggingFace API is available, use it
    if hf_api_key:
        ai_response = get_huggingface_response(message, hf_api_key)
        if ai_response and "error" not in ai_response.lower():
            return ai_response
    
    # Fall back to rule-based responses
    return get_rule_based_response(message, insights)

# Add this to your Streamlit app:
huggingface_api_key = "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"  # Replace with your key

# Update the chat interface:
if st.sidebar.button("Send"):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        ai_response = get_assistant_response(user_input, historical_insights, huggingface_api_key)
        st.session_state.chat_history.append(("assistant", ai_response))
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
