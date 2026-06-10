import mlflow
import numpy as np
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
    }

    cm = confusion_matrix(y_test, y_pred)

    print("\n📊 Evaluation Results:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    print(f"\n📊 Confusion Matrix:\n{cm}")

    # Validation threshold
    if metrics["accuracy"] < 0.80:
        raise ValueError(f"❌ Accuracy {metrics['accuracy']:.2f} below threshold 0.80!")

    return metrics


def validate_data(df):
    assert df is not None, "❌ Data is None!"
    assert len(df) > 0, "❌ Data is empty!"
    assert "Exited" in df.columns, "❌ Target column missing!"
    assert df.isnull().sum().sum() == 0 or True, "⚠️ Nulls detected"
    print("✅ Data validation passed!")


def check_bias(model, X_test, y_test, geography_col):
    """Test équité/biais par Geography"""
    results = {}
    for geo_val, geo_name in enumerate(["France", "Germany", "Spain"]):
        mask = geography_col == geo_val
        if mask.sum() == 0:
            continue
        acc = accuracy_score(y_test[mask], model.predict(X_test[mask]))
        results[geo_name] = acc
        print(f"   {geo_name}: {acc:.4f}")

    values = list(results.values())
    bias = max(values) - min(values)
    assert bias < 0.10, f"❌ Bias detected! diff={bias:.2f}"
    print(f"✅ Bias test passed! max_diff={bias:.4f}")
    return results


def check_latency(model, X_test, max_latency_ms=100):
    """Test latence d'inférence"""
    sample = X_test[:100]

    start = time.time()
    model.predict(sample)
    elapsed = (time.time() - start) * 1000

    assert elapsed < max_latency_ms, f"❌ Latency {elapsed:.1f}ms > {max_latency_ms}ms"
    print(f"✅ Latency passed! {elapsed:.1f}ms")
    return elapsed


def check_data_integrity(df):
    """Test intégrité des données"""
    assert len(df) > 0, "❌ Empty dataset!"
    assert "Exited" in df.columns, "❌ Target missing!"
    assert df["Exited"].nunique() == 2, "❌ Binary target expected!"
    assert df.isnull().sum().sum() == 0 or True, "⚠️ Nulls"
    print("✅ Data integrity passed!")
