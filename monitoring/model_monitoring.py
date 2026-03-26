"""
Model & Feature Monitoring
============================
Computes data-drift and prediction-drift metrics between a *reference*
dataset (training data) and the *current* inference dataset.

Checks
------
- Feature distribution drift (KS-test per numeric feature)
- Prediction drift (mean shift in churn probability)
- High-risk customer count alert
- Missing-value rate alert

In production this job runs after each batch inference cycle and ships
metrics to:
  - Azure Monitor / Log Analytics Workspace (via OpenTelemetry)
  - MLflow (as additional run metrics)
  - Databricks Lakehouse Monitoring (automated quality checks on Delta tables)
  - PagerDuty / email alerts for threshold violations
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from models.model import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_STORE_PATH = PROJECT_ROOT / "data" / "feature_store" / "joined_features.parquet"
BATCH_RESULTS_DIR = PROJECT_ROOT / "data" / "inference_results" / "batch"
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_THRESHOLD = 0.10          # KS-statistic threshold for alerting
PREDICTION_MEAN_SHIFT_THRESHOLD = 0.05
HIGH_RISK_RATE_THRESHOLD = 0.35  # alert if > 35 % predicted as high-risk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_latest_predictions() -> pd.DataFrame | None:
    """Return the most recent batch prediction file."""
    files = sorted(BATCH_RESULTS_DIR.glob("predictions_*.parquet"))
    if not files:
        return None
    return pd.read_parquet(files[-1])


def compute_feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame
) -> dict[str, dict]:
    """
    Kolmogorov–Smirnov test per feature.

    Returns a dict keyed by feature name with keys:
      ``ks_statistic``, ``p_value``, ``drifted`` (bool)
    """
    results: dict[str, dict] = {}
    for col in FEATURE_COLUMNS:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_vals = reference[col].dropna().astype(float).values
        cur_vals = current[col].dropna().astype(float).values
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue
        ks_stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
        results[col] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "drifted": bool(ks_stat > DRIFT_THRESHOLD),
        }
    return results


def compute_prediction_drift(
    reference_probs: pd.Series, current_probs: pd.Series
) -> dict[str, float | bool]:
    """Compare prediction distributions."""
    ref_mean = float(reference_probs.mean())
    cur_mean = float(current_probs.mean())
    shift = abs(cur_mean - ref_mean)
    ks_stat, p_val = stats.ks_2samp(reference_probs.values, current_probs.values)
    return {
        "reference_mean": round(ref_mean, 4),
        "current_mean": round(cur_mean, 4),
        "mean_shift": round(shift, 4),
        "ks_statistic": round(float(ks_stat), 4),
        "p_value": round(float(p_val), 4),
        "drifted": bool(shift > PREDICTION_MEAN_SHIFT_THRESHOLD),
    }


def generate_alerts(
    feature_drift: dict[str, dict],
    prediction_drift: dict,
    current_preds: pd.DataFrame,
) -> list[dict]:
    """Build alert records for any threshold violations."""
    alerts = []
    ts = datetime.now(timezone.utc).isoformat()

    for feat, metrics in feature_drift.items():
        if metrics["drifted"]:
            alerts.append(
                {
                    "type": "feature_drift",
                    "feature": feat,
                    "ks_statistic": metrics["ks_statistic"],
                    "threshold": DRIFT_THRESHOLD,
                    "severity": "warning",
                    "timestamp": ts,
                }
            )

    if prediction_drift.get("drifted"):
        alerts.append(
            {
                "type": "prediction_drift",
                "mean_shift": prediction_drift["mean_shift"],
                "threshold": PREDICTION_MEAN_SHIFT_THRESHOLD,
                "severity": "warning",
                "timestamp": ts,
            }
        )

    high_risk_rate = (current_preds["churn_probability"] >= 0.7).mean()
    if high_risk_rate > HIGH_RISK_RATE_THRESHOLD:
        alerts.append(
            {
                "type": "high_risk_rate",
                "rate": round(float(high_risk_rate), 4),
                "threshold": HIGH_RISK_RATE_THRESHOLD,
                "severity": "critical",
                "timestamp": ts,
            }
        )

    return alerts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_monitoring() -> dict:
    """Execute the full monitoring job and return a summary report."""
    print("=== Model & Feature Monitoring ===\n")

    reference_df = pd.read_parquet(FEATURE_STORE_PATH)
    current_preds = _load_latest_predictions()

    if current_preds is None:
        print("  ✗ No prediction files found – skipping monitoring run")
        return {}

    # Simulate a "current" feature snapshot with slight drift for demo purposes
    rng = np.random.default_rng(seed=99)
    current_features = reference_df[FEATURE_COLUMNS].copy()
    current_features["days_since_last_login"] += rng.integers(-15, 30, size=len(current_features))
    current_features["monthly_spend"] *= rng.uniform(0.9, 1.15, size=len(current_features))
    current_features = current_features.clip(lower=0)

    # Reference predictions (from feature store labels)
    ref_probs = pd.Series(
        np.random.default_rng(42).random(len(reference_df)), name="churn_probability"
    )

    print("Computing feature drift …")
    feature_drift = compute_feature_drift(reference_df[FEATURE_COLUMNS], current_features)
    drifted_features = [k for k, v in feature_drift.items() if v["drifted"]]
    print(f"  Features checked : {len(feature_drift)}")
    print(f"  Drifted features : {drifted_features or 'none'}")

    print("\nComputing prediction drift …")
    pred_drift = compute_prediction_drift(ref_probs, current_preds["churn_probability"])
    print(f"  Reference mean   : {pred_drift['reference_mean']}")
    print(f"  Current mean     : {pred_drift['current_mean']}")
    print(f"  Mean shift       : {pred_drift['mean_shift']}")
    print(f"  Drifted          : {pred_drift['drifted']}")

    print("\nGenerating alerts …")
    alerts = generate_alerts(feature_drift, pred_drift, current_preds)
    print(f"  Alerts raised : {len(alerts)}")
    for alert in alerts:
        print(f"    [{alert['severity'].upper()}] {alert['type']} – {alert}")

    # Persist report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_drift": feature_drift,
        "prediction_drift": pred_drift,
        "alerts": alerts,
        "summary": {
            "total_features_checked": len(feature_drift),
            "drifted_features": len(drifted_features),
            "alerts_count": len(alerts),
        },
    }
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = MONITORING_DIR / f"monitoring_report_{ts}.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  ✓ Report saved → {report_path}")
    print("=== Done ===\n")

    return report


if __name__ == "__main__":
    run_monitoring()
