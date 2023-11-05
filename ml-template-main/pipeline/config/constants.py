"""Copyright (c) 2022, Liberty Mutual Group."""
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[2]

MODEL_ARTIFACT = Path(BASE_PATH) / "data" / "temp" / "artifacts" / "mlflow" / "model.pkl"
MODEL_METRICS_ARTIFACT = Path(BASE_PATH) / "data" / "temp" / "artifacts" / "mlflow" / "model_metrics.pkl"

DATABRICKS_GROUP_NAME = "DATABRICKS_GROUP_NAME"
DATABRICKS_EXPERIMENT_NAME = "DATABRICKS_EXPERIMENT_NAME"
DATABRICKS_REGISTERED_MODEL_NAME = "DATABRICKS_REGISTERED_MODEL_NAME"

ENFORCE_CLEAN_WORKSPACE = "ENFORCE_CLEAN_WORKSPACE"
