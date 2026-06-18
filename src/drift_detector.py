import boto3
import json
import os
import requests
import pandas as pd

# Training data stats (men churn.csv)
TRAINING_STATS = {
    "Age": {"mean": 38.9, "threshold": 10},
    "Balance": {"mean": 76485.9, "threshold": 20000},
}

BUCKET = "mlops-mlflow-artifacts-709598629349"


def load_production_data():
    s3 = boto3.client("s3", region_name="eu-west-1")
    objects = s3.list_objects_v2(Bucket=BUCKET, Prefix="production-data/")

    data = []
    for obj in objects.get("Contents", []):
        response = s3.get_object(Bucket=BUCKET, Key=obj["Key"])
        data.append(json.loads(response["Body"].read()))

    return pd.DataFrame(data)


def detect_drift(prod_df):
    drift_detected = False

    for feature, stats in TRAINING_STATS.items():
        prod_mean = prod_df[feature].mean()
        diff = abs(prod_mean - stats["mean"])

        print(
            f"{feature}: training={stats['mean']:.1f} production={prod_mean:.1f} diff={diff:.1f}"
        )

        if diff > stats["threshold"]:
            print(f"⚠️ DRIFT DETECTED in {feature}!")
            drift_detected = True

    return drift_detected


def trigger_retrain():
    token = os.getenv("GITHUB_TOKEN")
    response = requests.post(
        "https://api.github.com/repos/FatimaGhi/mlops-model/dispatches",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
        json={"event_type": "retrain"},
    )
    if response.status_code == 204:
        print("✅ Retrain triggered via GitHub Actions!")
    else:
        print(f"❌ Failed to trigger retrain: {response.status_code}")


if __name__ == "__main__":
    print("=== Drift Detection ===")
    prod_df = load_production_data()

    if len(prod_df) < 10:
        print(f"Not enough data ({len(prod_df)} records)")
        exit(0)

    print(f"Production data: {len(prod_df)} records")

    drift = detect_drift(prod_df)

    if drift:
        print("🚨 DRIFT DETECTED — Triggering retrain...")
        trigger_retrain()
    else:
        print("✅ No drift detected")
