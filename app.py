import joblib
from flask import Flask, request, jsonify, render_template
import numpy as np
import re
import os
from flask_cors import CORS # Import CORS

# Define file paths for the model and vectorizer
MODEL_PATH = 'model.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'

# Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("Error: Model or vectorizer files not found. Please run pipeline.py first.")
    exit()

# Load the trained model and vectorizer
try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model files: {e}")
    exit()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Preprocessing function, same as used for training
def preprocess_text(text):
    """Cleans and preprocesses the text for prediction."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = text.strip()
    return text

# New route to serve the HTML visualizer
@app.route('/')
def home():
    """Serves the HTML file for the visualizer."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to classify a text reply.
    Input: JSON with a 'text' string.
    Output: JSON with 'label' and 'confidence'.
    """
    try:
        data = request.get_json(force=True)
        if 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request body'}), 400
        
        input_text = data['text']
        
        # Preprocess and vectorize the input text
        cleaned_text = preprocess_text(input_text)
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Get prediction and probabilities
        prediction_label = model.predict(vectorized_text)[0]
        prediction_proba = model.predict_proba(vectorized_text)
        
        # Get the confidence score for the predicted label
        confidence = np.max(prediction_proba)
        
        response = {
            "label": prediction_label,
            "confidence": float(confidence)
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)