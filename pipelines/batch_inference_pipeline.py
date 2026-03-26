"""
Batch Inference Pipeline
==========================
Reads the joined feature table from the feature store, loads the registered
MLflow model, scores all records, and writes predictions back to the data
lake (simulated as a Parquet file here).

In production this pipeline runs as a scheduled Databricks Job (via the
Databricks Jobs API or Azure ML Pipeline), typically once per day.  The
output delta table is consumed by downstream BI / CRM tools and by the
monitoring job.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from models.model import FEATURE_COLUMNS, predict_batch

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store" / "joined_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "inference_results" / "batch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")
REGISTERED_MODEL_NAME = "customer_churn_model"
MODEL_VERSION = "1"


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_features(path: Path = FEATURE_STORE_PATH) -> pd.DataFrame:
    """Step 1 – read feature table."""
    df = pd.read_parquet(path)
    print(f"  ✓ Loaded {len(df):,} records from feature store")
    return df


def load_model() -> mlflow.pyfunc.PyFuncModel:
    """Step 2 – load model from MLflow Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"  ✓ Loaded model from registry: {model_uri}")
    except Exception:
        # Fallback for demo environments without a full Registry
        print(f"  ℹ Model registry unavailable; falling back to in-process model")
        model = None
    return model


def run_inference(df: pd.DataFrame, model: mlflow.pyfunc.PyFuncModel | None) -> pd.DataFrame:
    """Step 3 – score records."""
    feature_df = df[FEATURE_COLUMNS].astype(float)

    if model is not None:
        probs = model.predict(feature_df)
        df = df.copy()
        df["churn_probability"] = probs.values if hasattr(probs, "values") else probs
    else:
        df = predict_batch(df)

    df["inference_timestamp"] = datetime.now(timezone.utc).isoformat()
    df["model_name"] = REGISTERED_MODEL_NAME
    df["model_version"] = MODEL_VERSION
    print(f"  ✓ Scored {len(df):,} records")
    return df


def write_results(df: pd.DataFrame) -> Path:
    """Step 4 – persist predictions."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output_path = OUTPUT_DIR / f"predictions_{ts}.parquet"
    df[
        ["customer_id", "churn_probability", "inference_timestamp", "model_name", "model_version"]
    ].to_parquet(output_path, index=False)
    print(f"  ✓ Saved predictions → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_batch_inference() -> pd.DataFrame:
    """Execute the full batch inference pipeline."""
    print("=== Batch Inference Pipeline ===")
    features = load_features()
    model = load_model()
    results = run_inference(features, model)
    write_results(results)
    print("=== Done ===\n")

    # Summary stats
    print("Prediction summary:")
    print(results["churn_probability"].describe().round(4).to_string())
    high_risk = (results["churn_probability"] >= 0.7).sum()
    print(f"\nHigh-risk customers (p≥0.70): {high_risk:,} / {len(results):,}")
    return results


if __name__ == "__main__":
    run_batch_inference()
