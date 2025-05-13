# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load('models/rf_model.pkl')
encoders = joblib.load('models/encoders.pkl')

# Feature columns
feature_columns = [
    'heart_rate', 'calories_burned', 'sleep_hours', 'steps_walked', 'stress_level',
    'posture_score', 'joint_flexibility', 'repetition_count', 'motion_type',
    'shift_hours', 'night_shift', 'overtime_hours'
]

@app.route('/', methods=['GET'])
def home():
    return "✅ Occupational Health Risk Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate input
    missing_fields = [field for field in feature_columns if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

    # Create DataFrame for model
    input_data = pd.DataFrame([data])

    # Encode motion_type
    if 'motion_type' in encoders:
        input_data['motion_type'] = encoders['motion_type'].transform(input_data['motion_type'])

    # Predict
    prediction = model.predict(input_data)

    return jsonify({"risk_class": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
