from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "models/churn_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Trained model not found. Run model.py first.")

# Load model once at startup
model = joblib.load(MODEL_PATH)

EXPECTED_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

def validate_input(data: dict):
    """
    Validate incoming JSON payload.
    """
    missing_features = [f for f in EXPECTED_FEATURES if f not in data]
    if missing_features:
        return False, f"Missing features: {missing_features}"

    return True, None

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400

    data = request.get_json()

    is_valid, error = validate_input(data)
    if not is_valid:
        return jsonify({"error": error}), 400

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        result = {
            "prediction": "Yes" if prediction == 1 else "No",
            "probability": round(float(probability), 4)
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
