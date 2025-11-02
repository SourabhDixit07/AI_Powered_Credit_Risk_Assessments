import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model, scaler, encoders, and default values
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
default_values = joblib.load("default_values.pkl")  # Load default values

# Get expected feature names
expected_features = list(default_values.keys())

# Streamlit UI
st.title("🚀 AI-Powered Credit Risk Assessment")

# Create input fields dynamically based on expected features
input_data = {}

for feature in expected_features:
    if feature in label_encoders:
        # Categorical Feature (Dropdown)
        input_data[feature] = st.selectbox(f"🔹 {feature}", label_encoders[feature].classes_)
    else:
        # Numerical Feature (Number Input)
        input_data[feature] = st.number_input(f"🔢 {feature}", value=float(default_values[feature]))

# Predict Button
if st.button("🔍 Assess Risk"):
    try:
        # Encode categorical inputs
        for feature in label_encoders:
            input_data[feature] = label_encoders[feature].transform([input_data[feature]])[0]

        # Convert input into DataFrame and ensure correct ordering
        input_df = pd.DataFrame([input_data])
        input_df = input_df[expected_features]  # Arrange columns in correct order

        # Apply scaling
        input_scaled = scaler.transform(input_df)

        # Predict risk
        # prediction = model.predict(input_scaled)[0]
        # result = "✅ Low Risk" if prediction ==1 else "⚠ High Risk"
        # Predict probabilities
        probabilities = model.predict_proba(input_scaled)
        low_risk_prob = probabilities[0][0]  # Probability of Low Risk
        high_risk_prob = probabilities[0][1]  # Probability of High Risk

        st.write(f"🔍 Probability of Low Risk: {low_risk_prob:.2f}")
        st.write(f"⚠ Probability of High Risk: {high_risk_prob:.2f}")

        # Adjust the threshold for classification
        threshold = 0.45  # Lower the threshold slightly
        prediction = "✅ Low Risk" if low_risk_prob > threshold else "⚠ High Risk"

        st.success(f"### Prediction: {prediction}")

        st.write("🔍 Scaled Input Data:", input_scaled)


    except Exception as e:
        st.error(f"❌ Error: {e}")
