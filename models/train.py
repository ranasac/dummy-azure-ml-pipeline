"""
Model Training & MLflow Registration
======================================
Simulates the Databricks ML training job that:
  1. Loads joined features from the feature store
  2. Trains the model (fake / random for this demo)
  3. Logs metrics, parameters, and the model artifact to MLflow
  4. Registers the model in the MLflow Model Registry

In production this script would be run as a Databricks Job or Azure ML
Pipeline step.  The MLflow tracking URI would point to the workspace-level
MLflow server (automatically configured in Databricks).
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from models.model import FEATURE_COLUMNS, FakeChurnModel

# ---------------------------------------------------------------------------
# Paths & MLflow config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store" / "joined_features.parquet"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

EXPERIMENT_NAME = "customer_churn_experiment"
REGISTERED_MODEL_NAME = "customer_churn_model"


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------

def train_and_register() -> str:
    """Train the model, log everything to MLflow, and register it.

    Returns the run ID.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_parquet(FEATURE_STORE_PATH)

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = FakeChurnModel(seed=42)

    with mlflow.start_run(run_name="fake_churn_model_v1") as run:
        run_id = run.info.run_id

        # ── Log hyper-parameters ──────────────────────────────────────────
        mlflow.log_params(
            {
                "model_type": "FakeChurnModel",
                "seed": 42,
                "n_features": len(FEATURE_COLUMNS),
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )

        # ── Evaluate (fake probabilities, so metrics are illustrative) ────
        ctx = None  # pyfunc context not needed for evaluation calls
        test_probs = model.predict(ctx, X_test).values
        train_probs = model.predict(ctx, X_train).values

        test_auc = roc_auc_score(y_test, test_probs)
        train_auc = roc_auc_score(y_train, train_probs)
        test_logloss = log_loss(y_test, test_probs)

        mlflow.log_metrics(
            {
                "train_roc_auc": round(train_auc, 4),
                "test_roc_auc": round(test_auc, 4),
                "test_log_loss": round(test_logloss, 4),
                "churn_rate_train": round(float(y_train.mean()), 4),
                "churn_rate_test": round(float(y_test.mean()), 4),
            }
        )

        # ── Log feature schema ────────────────────────────────────────────
        signature = mlflow.models.infer_signature(
            X_test, pd.Series(test_probs, name="churn_probability")
        )

        # ── Log model artifact ────────────────────────────────────────────
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            signature=signature,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(
            f"  Run ID   : {run_id}\n"
            f"  Train AUC: {train_auc:.4f}\n"
            f"  Test AUC : {test_auc:.4f}\n"
            f"  Log-loss : {test_logloss:.4f}"
        )

    print(
        f"\n✓ Model registered as '{REGISTERED_MODEL_NAME}' in MLflow "
        f"Model Registry at {MLFLOW_TRACKING_URI}"
    )
    return run_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting model training …")
    run_id = train_and_register()
    print(f"\nDone.  MLflow run: {run_id}")
