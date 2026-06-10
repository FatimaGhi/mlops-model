import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import pandas as pd
import numpy as np

from src.config import MODEL_PARAMS, MLFLOW_PARAMS
from src.data import load_data, split_data
from src.features import preprocess


def train():
    # Load + split + preprocess
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled = preprocess(X_train, X_test)

    # MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(MLFLOW_PARAMS["experiment_name"])

    with mlflow.start_run():

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=MODEL_PARAMS["n_estimators"],
            max_depth=MODEL_PARAMS["max_depth"],
            learning_rate=MODEL_PARAMS["learning_rate"],
            random_state=MODEL_PARAMS["random_state"],
            eval_metric=MODEL_PARAMS["eval_metric"],
            use_label_encoder=False
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        f1        = f1_score(y_test, y_pred)
        auc       = roc_auc_score(y_test, y_prob)

        # Log params
        mlflow.log_params(MODEL_PARAMS)

        # Log metrics
        mlflow.log_metrics({
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "auc":       auc
        })

        # Validation — accuracy > 0.80
        if accuracy < 0.80:
            print(f"❌ Accuracy {accuracy:.2f} < 0.80 — model rejected!")
            return

        # Log model
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MLFLOW_PARAMS["model_name"]
        )

        print(f"✅ Model trained!")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1:        {f1:.4f}")
        print(f"   AUC:       {auc:.4f}")


if __name__ == "__main__":
    train()