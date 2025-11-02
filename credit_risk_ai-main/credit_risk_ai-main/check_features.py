import joblib

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Check expected number of features
print("Scaler expects:", scaler.n_features_in_)
