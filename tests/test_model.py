import joblib
import pandas as pd
import os

MODEL_PATH = "models/churn_model.joblib"


def test_model_file_exists():
    """Check if trained model file exists"""
    assert os.path.exists(MODEL_PATH)


def test_model_can_be_loaded():
    """Check if model can be loaded"""
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_prediction_output():
    """Check if model returns a valid prediction"""
    model = joblib.load(MODEL_PATH)

    sample_input = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 250.0
    }])

    prediction = model.predict(sample_input)

    assert prediction[0] in [0, 1]
