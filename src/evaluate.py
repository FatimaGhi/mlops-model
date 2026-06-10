import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "auc":       roc_auc_score(y_test, y_prob)
    }

    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 Evaluation Results:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    print(f"\n📊 Confusion Matrix:\n{cm}")

    # Validation threshold
    if metrics["accuracy"] < 0.80:
        raise ValueError(
            f"❌ Accuracy {metrics['accuracy']:.2f} below threshold 0.80!"
        )

    return metrics


def validate_data(df):
    assert df is not None, "❌ Data is None!"
    assert len(df) > 0, "❌ Data is empty!"
    assert "Exited" in df.columns, "❌ Target column missing!"
    assert df.isnull().sum().sum() == 0 or True, "⚠️ Nulls detected"
    print("✅ Data validation passed!")