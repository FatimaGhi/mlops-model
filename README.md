# MLOps Model

ML model code, FastAPI serving API, and data drift detection for the End-to-End MLOps Platform on Cloud.

## Overview

This repository contains:
- **XGBoost model** for customer churn prediction
- **FastAPI** REST API for model serving
- **Gradio UI** for interactive model testing
- **DVC** pipeline for reproducible training
- **MLflow** integration for experiment tracking
- **Data drift detection** with automated retraining trigger
- **CI/CD pipeline** with GitHub Actions (6 jobs)

## Repository Structure

```
mlops-model/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline (lint, test, security, build, train, deploy)
├── app/
│   ├── main.py                 # FastAPI application (predict, health, metrics)
│   └── gradio_app.py           # Gradio UI for interactive model testing
├── src/
│   ├── train.py                # XGBoost training script
│   ├── features.py             # Data preprocessing
│   ├── evaluate.py             # Model evaluation (accuracy, bias, latency)
│   ├── config.py               # Configuration
│   ├── data.py                 # Data loading utilities
│   ├── predict.py              # Prediction utilities
│   └── drift_detector.py       # Data drift detection + retraining trigger
├── data/
│   └── churn.csv               # Dataset (tracked by DVC)
├── models/
│   ├── scaler.pkl              # Feature scaler (DVC output)
│   └── encoder.pkl             # Label encoder (DVC output)
├── tests/
│   └── test_data.py            # Unit tests (pytest)
├── Dockerfile                  # Container image
├── dvc.yaml                    # DVC pipeline definition
├── dvc.lock                    # DVC pipeline lock file
├── params.yaml                 # Model hyperparameters
├── metrics.json                # Evaluation metrics
├── requirements.txt            # Python dependencies
└── train-job.yaml              # Kubernetes training job
```

## CI/CD Pipeline

Every push to `main` triggers 6 sequential jobs:

```
JOB 1: Lint (flake8 + black)
    → JOB 2: Tests (pytest + DVC pull)
        → JOB 3: Security scan (Trivy filesystem)
            → JOB 4: Docker Build + Push (ECR) + Trivy image scan
                → JOB 5: Train Model (Kubernetes Job on EKS)
                    → JOB 6: Update GitOps (mlops-gitops values.yaml)
```

The pipeline also supports **automated retraining** via `repository_dispatch` event:
```yaml
on:
  push:
    branches: [main, dev]
  repository_dispatch:
    types: [retrain]   # triggered by drift detector
```

Authentication to AWS uses **OIDC** — no static AWS credentials stored in GitHub Secrets.

## DVC Pipeline

```
data/churn.csv
    → preprocess (src/features.py) → data/processed/
        → train (src/train.py) → models/scaler.pkl + encoder.pkl
            → evaluate (src/evaluate.py) → metrics.json
```

```bash
# Run full pipeline
dvc repro

# Pull data from S3
dvc pull
```

## Model

- **Algorithm**: XGBoost Classifier
- **Dataset**: Churn Modelling (Kaggle) — 10,000 observations, 14 features
- **Split**: 80% train / 20% test
- **Hyperparameters** (`params.yaml`):

```yaml
model:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  random_state: 42
  eval_metric: logloss
```

## Model Validation

Before promotion to MLflow Production registry, the model must pass:

| Metric | Result | Threshold |
|---|---|---|
| Accuracy | 86.5% | ≥ 80% |
| Bias (demographic groups) | 7.94% | < 10% |
| Inference latency | 0.7ms | < 100ms |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status |
| `/health` | GET | Health check (liveness/readiness probe) |
| `/model-info` | GET | Current production model metadata |
| `/predict` | POST | Churn prediction |
| `/metrics` | GET | Prometheus metrics |

## Data Drift Detection

A Kubernetes CronJob (`drift-detector`) runs every hour to:
1. Load production data from S3 (`production-data/`)
2. Compare 5 feature distributions against training baseline (Age, Balance, CreditScore, NumOfProducts, IsActiveMember)
3. If drift detected → send Slack alert to `#mlops-alerts`
4. Merge production + training data → upload to S3
5. Trigger automated retraining via GitHub Actions `repository_dispatch`

## Prerequisites

- Python 3.11
- Docker
- AWS CLI configured
- kubectl (for training job)

## Local Setup

```bash
# 1. Clone repository
git clone https://github.com/FatimaGhi/mlops-model

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull data
dvc pull

# 5. Run pipeline
dvc repro

# 6. Run API locally (requires MLflow running)
uvicorn app.main:app --reload --port 8000
```

## Related Repositories

- [mlops-infrastructure](https://github.com/FatimaGhi/mlops-infrastructure) — AWS infrastructure (Terraform)
- [mlops-gitops](https://github.com/FatimaGhi/mlops-gitops) — Kubernetes manifests (GitOps)