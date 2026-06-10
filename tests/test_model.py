import pytest
import numpy as np
import pandas as pd
import joblib
import mlflow.xgboost
from src.config import MLFLOW_PARAMS
from src.evaluate import validate_data, check_bias, check_latency, check_data_integrity
from src.features import GEOGRAPHY_MAP, GENDER_MAP

# ─── Fixtures ─────────────────────────────────────────────


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "RowNumber": [1, 2],
            "CustomerId": [1001, 1002],
            "Surname": ["Smith", "Jones"],
            "CreditScore": [600, 700],
            "Geography": ["France", "Germany"],
            "Gender": ["Male", "Female"],
            "Age": [35, 45],
            "Tenure": [5, 3],
            "Balance": [0, 50000],
            "NumOfProducts": [1, 2],
            "HasCrCard": [1, 0],
            "IsActiveMember": [1, 1],
            "EstimatedSalary": [50000, 80000],
            "Exited": [0, 1],
        }
    )


@pytest.fixture
def model_and_scaler():
    mlflow.set_tracking_uri("http://localhost:5000")
    model = mlflow.xgboost.load_model(
        f"models:/{MLFLOW_PARAMS['model_name']}/Production"
    )
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler


# ─── Tests ────────────────────────────────────────────────


def test_data_integrity_pass(sample_df):
    """Data integrity test"""
    validate_data(sample_df)


def test_data_empty():
    """Empty dataset should fail"""
    with pytest.raises(AssertionError):
        validate_data(pd.DataFrame())


def test_data_missing_target(sample_df):
    """Missing target column should fail"""
    df = sample_df.drop(columns=["Exited"])
    with pytest.raises(AssertionError):
        validate_data(df)


def test_model_predict(model_and_scaler):
    """Model should return valid predictions"""
    model, scaler = model_and_scaler

    X = pd.DataFrame(
        [
            {
                "CreditScore": 600,
                "Geography": 0,
                "Gender": 0,
                "Age": 35,
                "Tenure": 5,
                "Balance": 0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000,
            }
        ]
    )

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled)

    assert pred[0] in [0, 1]
    assert 0 <= prob[0][1] <= 1


def test_model_accuracy(model_and_scaler):
    """Model accuracy should be > 0.80"""
    from src.data import load_data, split_data
    from src.features import preprocess
    from sklearn.metrics import accuracy_score

    model, scaler = model_and_scaler
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    assert accuracy >= 0.80, f"❌ Accuracy {accuracy:.2f} < 0.80"
    print(f"✅ Accuracy: {accuracy:.4f}")


def test_latency_inference(model_and_scaler):
    """Inference latency < 100ms for 100 samples"""
    model, scaler = model_and_scaler

    X = pd.DataFrame(
        [
            {
                "CreditScore": 600,
                "Geography": 0,
                "Gender": 0,
                "Age": 35,
                "Tenure": 5,
                "Balance": 0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000,
            }
        ]
        * 100
    )

    X_scaled = scaler.transform(X)
    check_latency(model, X_scaled, max_latency_ms=100)


def test_api_health():
    """API health endpoint should return 200"""
    from fastapi.testclient import TestClient
    from app.main import app
    import app.main as main_module

    # Mock model o scaler
    import joblib
    import mlflow.xgboost

    mlflow.set_tracking_uri("http://localhost:5000")
    main_module.model = mlflow.xgboost.load_model("models:/ChurnModel/Production")
    main_module.scaler = joblib.load("models/scaler.pkl")

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_api_predict():
    """API predict endpoint should return valid response"""
    from fastapi.testclient import TestClient
    from app.main import app
    import app.main as main_module

    # Mock model o scaler
    import joblib
    import mlflow.xgboost

    mlflow.set_tracking_uri("http://localhost:5000")
    main_module.model = mlflow.xgboost.load_model("models:/ChurnModel/Production")
    main_module.scaler = joblib.load("models/scaler.pkl")

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "CreditScore": 600,
            "Geography": "France",
            "Gender": "Male",
            "Age": 35,
            "Tenure": 5,
            "Balance": 0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 50000,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "label" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1
