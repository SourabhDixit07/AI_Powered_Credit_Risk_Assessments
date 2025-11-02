from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model, scaler, and default values
model = joblib.load("credit_risk_model.pkl")
scaler = joblib.load("scaler.pkl")
default_values = joblib.load("default_values.pkl")  # Load mean values from training data

# Define the expected feature names (must match training data)
expected_features = list(default_values.keys())


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data as JSON
        data = request.json

        # Validate that input is a dictionary
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected a dictionary."}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure all expected features are present (fill missing ones with default values)
        for feature in expected_features:
            if feature not in input_df:
                input_df[feature] = default_values[feature]

        # Ensure correct feature ordering
        input_df = input_df[expected_features]

        # Apply scaling
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)

        # Convert output to human-readable format
        result = "Low Risk" if prediction[0] == 1 else "High Risk"

        return jsonify({"credit_risk": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
