import boto3
import json
import os
import requests
import pandas as pd

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training data baseline statistics (computed from churn.csv)
# These represent the "normal" distribution of the training data
# Drift is detected when production data deviates significantly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING_STATS = {
    "Age": {"mean": 38.9, "threshold": 10},  # Alert if avg age shifts by >10 years
    "Balance": {
        "mean": 76485.9,
        "threshold": 20000,
    },  # Alert if avg balance shifts by >20k€
    "CreditScore": {
        "mean": 650.5,
        "threshold": 50,
    },  # Alert if avg credit score shifts by >50
    "NumOfProducts": {
        "mean": 1.53,
        "threshold": 0.5,
    },  # Alert if avg products shifts by >0.5
    "IsActiveMember": {
        "mean": 0.51,
        "threshold": 0.2,
    },  # Alert if active member ratio shifts by >20%
}

BUCKET = "mlops-mlflow-artifacts-709598629349"
TRAINING_DATA_KEY = "data/churn.csv"
PRODUCTION_DATA_PREFIX = "production-data/"
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")


def load_production_data():
    """
    Load all production predictions stored in S3.
    Each file contains one prediction with its input features.
    Returns a DataFrame with all production records.
    """
    s3 = boto3.client("s3", region_name="eu-west-1")
    objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=PRODUCTION_DATA_PREFIX)

    data = []
    for obj in objects.get("Contents", []):
        response = s3.get_object(Bucket=BUCKET, Key=obj["Key"])
        data.append(json.loads(response["Body"].read()))

    return pd.DataFrame(data)


def detect_drift(prod_df):
    """
    Compare production data statistics against training data baseline.
    For each feature, compute the absolute difference between means.
    If the difference exceeds the threshold → drift detected.
    Returns: (drift_detected: bool, drifted_features: list)
    """
    drift_detected = False
    drifted_features = []

    for feature, stats in TRAINING_STATS.items():
        if feature not in prod_df.columns:
            continue

        prod_mean = prod_df[feature].mean()
        diff = abs(prod_mean - stats["mean"])

        print(
            f"{feature}: training={stats['mean']:.2f} production={prod_mean:.2f} diff={diff:.2f}"
        )

        if diff > stats["threshold"]:
            print(f"⚠️ DRIFT DETECTED in {feature}!")
            drifted_features.append(feature)
            drift_detected = True

    return drift_detected, drifted_features


def merge_data(prod_df):
    """
    Merge production data with original training data.
    This enriches the training dataset with real production samples,
    allowing the retrained model to better handle the new data distribution.
    The merged dataset is uploaded back to S3 as the new training data.
    """
    s3 = boto3.client("s3", region_name="eu-west-1")

    # Load original training data from S3
    response = s3.get_object(Bucket=BUCKET, Key=TRAINING_DATA_KEY)
    training_df = pd.read_csv(response["Body"])

    # Keep only columns that exist in both datasets
    common_cols = [c for c in training_df.columns if c in prod_df.columns]
    prod_subset = prod_df[common_cols].copy()

    # Concatenate and remove duplicates
    merged_df = pd.concat([training_df, prod_subset], ignore_index=True)
    merged_df = merged_df.drop_duplicates()

    # Upload merged dataset back to S3 (overwrites original churn.csv)
    s3.put_object(
        Bucket=BUCKET, Key=TRAINING_DATA_KEY, Body=merged_df.to_csv(index=False)
    )
    print(
        f"✅ Merged data uploaded: {len(merged_df)} records "
        f"(training: {len(training_df)} + production: {len(prod_subset)})"
    )
    return merged_df


def send_slack_alert(drifted_features):
    """
    Send a Slack notification when drift is detected.
    Lists the features that drifted and confirms retraining was triggered.
    """
    if not SLACK_WEBHOOK:
        return
    message = {
        "text": (
            f"🚨 *DATA DRIFT DETECTED*\n"
            f"Features drifted: `{', '.join(drifted_features)}`\n"
            f"Action: Retraining triggered automatically 🔄"
        )
    }
    requests.post(SLACK_WEBHOOK, json=message)
    print("✅ Slack alert sent!")


def trigger_retrain():
    """
    Trigger the ML training pipeline via GitHub Actions repository_dispatch event.
    This starts the full CI/CD pipeline: lint → test → build → train → deploy.
    """
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
    print("=== Drift Detection Started ===")

    # Step 1: Load production data from S3
    prod_df = load_production_data()

    if len(prod_df) < 10:
        print(f"Not enough production data ({len(prod_df)} records). Skipping.")
        exit(0)

    print(f"Production data loaded: {len(prod_df)} records")

    # Step 2: Detect drift by comparing feature distributions
    drift, drifted_features = detect_drift(prod_df)

    if drift:
        print(f"\n🚨 DRIFT DETECTED in features: {drifted_features}")

        # Step 3: Send Slack alert
        send_slack_alert(drifted_features)

        # Step 4: Merge production data with training data
        print("\n=== Merging Production + Training Data ===")
        merge_data(prod_df)

        # Step 5: Trigger retraining pipeline
        print("\n=== Triggering Retraining Pipeline ===")
        trigger_retrain()

    else:
        print("✅ No drift detected. Model performance is stable.")
