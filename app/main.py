from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import joblib
import pandas as pd
import os
import logging
import time

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

app = FastAPI(
    title="Churn Prediction API",
    description="XGBoost model for customer churn prediction",
    version="1.0.0",
)
# Prometheus
Instrumentator().instrument(app).expose(app)

PREDICTIONS_TOTAL = Counter("churn_predictions_total", "Total predictions", ["label"])

PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Prediction latency",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Churn probability distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
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

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    model = mlflow.xgboost.load_model("models:/ChurnModel/Production")

    # Load scaler men MLflow artifacts
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("ChurnModel", stages=["Production"])
    run_id = versions[0].run_id
    scaler_path = client.download_artifacts(run_id, "scaler/scaler.pkl")
    scaler = joblib.load(scaler_path)

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


@app.get("/model-info")
def model_info():
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("ChurnModel", stages=["Production"])
    if versions:
        v = versions[0]
        return {
            "model_name": v.name,
            "version": v.version,
            "stage": v.current_stage,
            "run_id": v.run_id,
        }
    return {"error": "No production model found"}


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

        latency = time.time() - start
        label = "Churn" if prediction == 1 else "No Churn"

        # Prometheus metrics
        PREDICTIONS_TOTAL.labels(label=label).inc()
        PREDICTION_LATENCY.observe(latency)
        CHURN_PROBABILITY.observe(float(probability))
        logger.info(
            f"prediction={prediction} "
            f"prob={probability:.4f} "
            f"latency={latency:.1f}ms"
        )

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 4),
            "label": label,
            "latency_ms": round(latency * 1000, 2),
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
