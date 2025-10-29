# app.py
from flask import Blueprint, render_template, request
import numpy as np
import joblib
import os

task1_bp = Blueprint('task1', __name__, template_folder='templates', static_folder='static')

# Load model and encoders
model_path = os.path.join("TASK_1", "random_forest_model.pkl")
encoders_path = os.path.join("TASK_1", "label_encoders.pkl")

model = joblib.load(model_path)
encoders = joblib.load(encoders_path)

# Define the same columns used for training
columns = [
    'Country Code', 'City', 'Locality', 'Longitude', 'Latitude', 
    'Cuisines', 'Average Cost for two', 'Currency',
    'Has Table booking', 'Has Online delivery', 
    'Is delivering now', 'Switch to order menu', 'Price range', 'Votes'
]

@task1_bp.route('/')
def home():
    return render_template('index.html')

@task1_bp.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form input
        user_input = {col: request.form.get(col) for col in columns}

        # Convert numeric fields
        numeric_cols = [
            'Country Code', 'Longitude', 'Latitude', 
            'Average Cost for two', 'Price range', 'Votes'
        ]
        for col in numeric_cols:
            user_input[col] = float(user_input[col])

        # Encode categorical fields using saved encoders
        encoded_values = []
        for col in columns:
            val = user_input[col]
            if col in encoders:
                le = encoders[col]
                if val not in le.classes_:
                    # Handle unseen category gracefully
                    le.classes_ = np.append(le.classes_, val)
                val = le.transform([val])[0]
            encoded_values.append(val)

        # Predict
        data = np.array(encoded_values).reshape(1, -1)
        pred = model.predict(data)[0]

        return render_template('index.html', prediction_text=f"⭐ Predicted Aggregate Rating: {round(pred, 2)}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")


