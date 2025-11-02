import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ["Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings",
           "Employment", "InstallmentRate", "PersonalStatus", "Debtors", "ResidenceYears",
           "Property", "Age", "OtherInstallment", "Housing", "ExistingCredits", "Job",
           "LiablePersons", "Telephone", "ForeignWorker", "CreditRisk"]

df = pd.read_csv(url, sep=" ", names=columns)

# Convert categorical variables
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for future use

# Define features & target
X = df.drop(columns=["CreditRisk"])  # Features
y = df["CreditRisk"]  # Target (1 = Good Credit, 2 = Bad Credit)

# Compute default values (mean for numerical features)
default_values = X.mean().to_dict()

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred)}")

# Save model, scaler, and additional data
joblib.dump(model, "credit_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")  # Save encoders
joblib.dump(default_values, "default_values.pkl")  # Save default values

print("✅ Model trained and saved successfully!")
