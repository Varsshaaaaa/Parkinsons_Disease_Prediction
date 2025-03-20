import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

# Ensure UTF-8 Encoding
sys.stdout.reconfigure(encoding='utf-8')

# Initialize Flask App
app = Flask(__name__)

# Load Models
try:
    lstm_model = load_model('lstm_model.h5')
    rf_model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Required Features for the Model
REQUIRED_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

@app.route('/')
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles file upload and makes predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file format. Please upload a CSV file'}), 400

    try:
        # Load CSV Data
        data = pd.read_csv(file, encoding='utf-8')
        data.columns = data.columns.str.strip()  # Remove extra spaces

        # Debugging: Show detected column names
        detected_columns = list(data.columns)
        print(f"📊 Detected Columns: {detected_columns}")

        # Check for Missing Columns
        missing_features = [col for col in REQUIRED_FEATURES if col not in detected_columns]

        # Auto-Fill Missing Columns with 0
        for col in missing_features:
            print(f"⚠️ Missing column: {col}. Filling with 0.")
            data[col] = 0

        # Ensure Columns are in Correct Order
        data = data[REQUIRED_FEATURES]

        # Scale Data
        data_scaled = scaler.transform(data)
        data_reshaped = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))

        # LSTM Feature Extraction
        lstm_features = lstm_model.predict(data_reshaped)

        # Random Forest Prediction
        predictions = rf_model.predict(lstm_features)

        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
