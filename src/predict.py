import joblib
import numpy as np
import pandas as pd
import mlflow.xgboost
from src.config import MLFLOW_PARAMS


def load_model():
    mlflow.set_tracking_uri("http://localhost:5000")
    model = mlflow.xgboost.load_model(
        f"models:/{MLFLOW_PARAMS['model_name']}/Production"
    )
    scaler = joblib.load("models/scaler.pkl")
    encoder = joblib.load("models/encoder.pkl")
    return model, scaler, encoder


def predict(data: dict):
    model, scaler, encoder = load_model()

    df = pd.DataFrame([data])

    # Encode
    for col in ["Geography", "Gender"]:
        df[col] = encoder.transform(df[col])

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "label": "Churn" if prediction == 1 else "No Churn",
    }
