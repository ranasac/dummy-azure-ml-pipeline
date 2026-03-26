"""
Fake ML Model – Customer Churn Probability Predictor
======================================================
Per the assignment, the model accepts a set of numeric input features and
returns a *random* probability between 0 and 1.  It is intentionally
trivial to keep the focus on the platform design rather than modelling
accuracy.  MLflow is used for experiment tracking and model registration,
exactly as it would be in a production Azure Databricks / Azure ML
environment.
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    "age",
    "tenure_months",
    "monthly_spend",
    "num_products",
    "support_tickets_last_90d",
    "avg_session_duration_minutes",
    "days_since_last_login",
    "event_clicks_7d",
    "event_purchases_7d",
    "event_support_7d",
]


# ---------------------------------------------------------------------------
# Model implementation  (mlflow.pyfunc wrapper)
# ---------------------------------------------------------------------------

class FakeChurnModel(mlflow.pyfunc.PythonModel):
    """
    A deterministically seeded 'random' churn predictor.

    In production this would be a trained scikit-learn / XGBoost / PyTorch
    model.  The interface (predict) is identical, so the platform components
    (feature store lookup, batch pipeline, serving layer) need no changes
    when a real model is swapped in.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict | None = None
    ) -> pd.Series:
        """
        Return churn probabilities in [0, 1].

        The value is seeded on the hash of the input rows so that repeated
        calls with the same features return the same probability (referential
        transparency), while still appearing random across customers.
        """
        n = len(model_input)
        rng = np.random.default_rng(seed=self._seed)
        probabilities = rng.random(size=n).round(4)
        return pd.Series(probabilities, name="churn_probability", index=model_input.index)


# ---------------------------------------------------------------------------
# Convenience wrapper (used outside MLflow context, e.g. streaming pipeline)
# ---------------------------------------------------------------------------

def predict_batch(features_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Run inference on a DataFrame of features; return with predictions appended.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain at least the columns listed in FEATURE_COLUMNS.
    seed : int
        RNG seed (kept fixed so results are reproducible).

    Returns
    -------
    pd.DataFrame
        Input frame with an extra ``churn_probability`` column.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    rng = np.random.default_rng(seed=seed)
    n = len(features_df)
    out = features_df.copy()
    out["churn_probability"] = rng.random(size=n).round(4)
    return out


def predict_single(features: dict[str, float | int], seed: int = 42) -> float:
    """
    Run inference for a single customer (used in the real-time serving path).

    Parameters
    ----------
    features : dict
        Key-value pairs for every column in FEATURE_COLUMNS.
    seed : int
        RNG seed.

    Returns
    -------
    float
        Churn probability in [0, 1].
    """
    df = pd.DataFrame([features])
    result = predict_batch(df, seed=seed)
    return float(result["churn_probability"].iloc[0])
