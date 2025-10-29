# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib, os, json

# Load dataset
file_path = "Dataset .csv"
data = pd.read_csv(file_path)

# Preprocessing
drop_cols = [
    'Restaurant ID', 'Restaurant Name', 'Address',
    'Locality Verbose', 'Rating color', 'Rating text'
]
df = data.drop(columns=drop_cols)
df = df.fillna(df.mode().iloc[0])

# Label encode object columns
encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split into features and target
X = df.drop(columns=['Aggregate rating'])
y = df['Aggregate rating']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- Random Forest Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse_rf:.4f}")
print(f"R-squared (R²): {r2_rf:.4f}")

# Save metrics to JSON
metrics = {
    "Mean Squared Error (MSE)": round(mse_rf, 4),
    "R-squared (R²)": round(r2_rf, 4)
}
with open("TASK_1/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Metrics saved to TASK_1/metrics.json")

# === Save Model and Encoders ===
os.makedirs("TASK_1", exist_ok=True)
joblib.dump(rf_model, "TASK_1/random_forest_model.pkl")
joblib.dump(encoders, "TASK_1/label_encoders.pkl")

print("✅ Model and encoders saved successfully in TASK_1/")
