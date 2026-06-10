from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import joblib
import pandas as pd
import os
import logging
import time

app = FastAPI(
    title="Churn Prediction API",
    description="XGBoost model for customer churn prediction",
    version="1.0.0"
)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GEOGRAPHY_MAP = {"France": 0, "Germany": 1, "Spain": 2}
GENDER_MAP = {"Male": 0, "Female": 1}

model = None
scaler = None


@app.on_event("startup")
async def load_model():
    global model, scaler

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    model = mlflow.xgboost.load_model(
        "models:/ChurnModel/Production"
    )
    scaler = joblib.load("models/scaler.pkl")
    print("✅ Model loaded!")


class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.get("/")
def root():
    return {"message": "Churn Prediction API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: CustomerData):
    start = time.time()
    try:
        df = pd.DataFrame([data.dict()])

        df["Geography"] = df["Geography"].map(GEOGRAPHY_MAP)
        df["Gender"] = df["Gender"].map(GENDER_MAP)

        X_scaled = scaler.transform(df)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        latency = (time.time() - start) * 1000
        logger.info(
            f"prediction={prediction} "
            f"prob={probability:.4f} "
            f"latency={latency:.1f}ms"
        )

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "label": "Churn" if prediction == 1 else "No Churn",
            "latency_ms": round(latency, 2)
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))