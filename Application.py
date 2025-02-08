import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔄",
    layout="wide"
)

# Load the model and necessary files
@st.cache_resource
def load_model():
    model = joblib.load('churn_model.joblib')
    with open('optimal_threshold.txt', 'r') as f:
        threshold = float(f.read().strip())
    with open('feature_names.txt', 'r') as f:
        features = f.read().splitlines()
    return model, threshold, features

model, threshold, features = load_model()

# Title and description
st.title("🔄 Customer Churn Prediction")
st.markdown("""
This application predicts customer churn using a voting ensemble of Neural Network and LightGBM models.
Upload your data or input values manually to get predictions.
""")

# Sidebar
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Upload CSV", "Manual Input"]
)

def make_prediction(input_df):
    # Get probability predictions
    proba = model.predict_proba(input_df)[:, 1]
    # Apply threshold
    predictions = (proba >= threshold).astype(int)
    return predictions, proba

def format_prediction(prediction, probability):
    if prediction == 1:
        return f"❌ High Risk of Churn (Probability: {probability:.2%})"
    return f"✅ Low Risk of Churn (Probability: {probability:.2%})"

if input_method == "Upload CSV":
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        # Check if all required features are present
        missing_features = set(features) - set(input_df.columns)
        if missing_features:
            st.error(f"Missing features in uploaded file: {missing_features}")
        else:
            # Make predictions
            predictions, probas = make_prediction(input_df[features])
            
            # Display results
            st.subheader("Predictions")
            results_df = pd.DataFrame({
                'Prediction': predictions,
                'Churn Probability': probas
            })
            results_df['Status'] = results_df.apply(
                lambda x: format_prediction(x['Prediction'], x['Churn Probability']),
                axis=1
            )
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Total Customers",
                    len(predictions)
                )
            with col2:
                st.metric(
                    "Predicted Churns",
                    sum(predictions)
                )
            
            # Show detailed results
            st.dataframe(results_df)
            
            # Plot distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.hist(probas, bins=50, edgecolor='black')
            plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
            plt.xlabel('Churn Probability')
            plt.ylabel('Count')
            plt.title('Distribution of Churn Probabilities')
            plt.legend()
            st.pyplot(fig)

else:
    st.subheader("Manual Input")
    
    # Create input fields for each feature
    input_data = {}
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            input_data[feature] = st.number_input(
                f"Enter {feature}",
                value=0.0,
                format="%.4f"
            )
    
    if st.button("Predict"):
        # Create DataFrame from input
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction, proba = make_prediction(input_df)
        
        # Display result
        st.subheader("Prediction Result")
        st.markdown(f"### {format_prediction(prediction[0], proba[0])}")
        
        # Create gauge chart for probability
        fig, ax = plt.subplots(figsize=(10, 2))
        plt.barh([0], [proba[0]], color='lightblue')
        plt.barh([0], [1], color='lightgray', alpha=0.3)
        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
        plt.xlim(0, 1)
        plt.yticks([])
        plt.xlabel('Churn Probability')
        plt.title('Probability Gauge')
        plt.legend()
        st.pyplot(fig)

# Add model information in expander
with st.expander("Model Information"):
    st.markdown("""
    ### Model Details
    - **Model Type**: Voting Ensemble (Neural Network + LightGBM)
    - **Decision Threshold**: {:.3f}
    - **Base Model Performance**:
        - Accuracy: 0.786
        - ROC-AUC: 0.829
        - F1-Score: 0.597
    """.format(threshold))
