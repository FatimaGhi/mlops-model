import yaml
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
SRC_DIR = ROOT_DIR / "src"

RAW_DATA_PATH = DATA_DIR / "churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed"

# Load params
with open(ROOT_DIR / "params.yaml") as f:
    params = yaml.safe_load(f)

DATA_PARAMS = params["data"]
FEATURE_PARAMS = params["features"]
MODEL_PARAMS = params["model"]
MLFLOW_PARAMS = params["mlflow"]