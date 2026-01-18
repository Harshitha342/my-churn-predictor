import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(filepath="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Load the Telco Customer Churn dataset.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_dataframe(df: pd.DataFrame):
    """
    Clean the dataset and prepare features and target.
    """
    # Drop customer ID (not useful for prediction)
    df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric (it contains empty strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Handle missing values (created by coercion)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Separate target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    return X, y

def train_model(df: pd.DataFrame):
    """
    Train the churn prediction model, evaluate it, and save the pipeline.
    """
    X, y = preprocess_dataframe(df)

    # Identify column types
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns

    # Preprocessing pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Full pipeline: preprocessing + model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Model Evaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save entire pipeline (preprocessor + model)
    joblib.dump(model, "models/churn_model.joblib")

    return model

if __name__ == "__main__":
    df = load_data()
    train_model(df)
