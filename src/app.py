from flask import Flask, request, jsonify
import joblib
import numpy as np
from waitress import serve  # Import the production server

app = Flask(__name__)
# Load the model you trained earlier [cite: 537, 550]
model = joblib.load("models/predictive_maintenance_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Prepare features: Temperature, Vibration, Current [cite: 524, 554]
    features = np.array([[data["temperature"], data["vibration"], data["current"]]])
    prediction = model.predict(features)
    
  # Standard if-else block is cleaner and follows PEP 8
    if prediction[0] == 1:
            result = "Machine failure predicted!"
    else:
            result = "Machine is running normally."

    return jsonify({"Prediction": result})
if __name__ == '__main__':
    print("🚀 Starting Production WSGI Server on http://0.0.0.0:5000")
    # This replaces app.run(debug=True) for industry-level deployment [cite: 262, 559]
    serve(app, host='0.0.0.0', port=5000)
