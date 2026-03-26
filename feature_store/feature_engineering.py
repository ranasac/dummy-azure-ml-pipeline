"""
Feature Engineering & Feature Store
=====================================
Simulates the Unity Catalog / Delta-table-backed feature store that would
run on Azure Databricks.  For the demo we use local Parquet files as a
drop-in replacement for Delta tables.

Two feature groups are produced and then joined:
  1. ``batch_features``     – derived from CRM / data-lake batch records
  2. ``streaming_features`` – aggregated from CDP / webhook event data (7-day
     rolling window, simulating Structured Streaming output)
  3. ``joined_features``    – point-in-time correct join of (1) and (2), used
                               as input to the ML model
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths  (relative to project root; adapt to ADLS mounts in production)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_BATCH = PROJECT_ROOT / "data" / "raw" / "batch_customers.parquet"
RAW_EVENTS = PROJECT_ROOT / "data" / "raw" / "streaming_events.jsonl"
FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Batch feature group
# ---------------------------------------------------------------------------

def compute_batch_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light-weight feature transformations on top of the raw CRM table.
    In production this runs as a Databricks Job (daily schedule).
    """
    out = df[
        [
            "customer_id",
            "age",
            "tenure_months",
            "monthly_spend",
            "num_products",
            "support_tickets_last_90d",
            "avg_session_duration_minutes",
            "days_since_last_login",
            "snapshot_date",
        ]
    ].copy()

    # Derived features
    out["spend_per_product"] = (out["monthly_spend"] / out["num_products"]).round(2)
    out["is_long_tenure"] = (out["tenure_months"] > 24).astype(int)
    out["high_support_usage"] = (out["support_tickets_last_90d"] > 5).astype(int)

    return out


# ---------------------------------------------------------------------------
# 2. Streaming feature group  (7-day aggregations)
# ---------------------------------------------------------------------------

def compute_streaming_features(events: list[dict]) -> pd.DataFrame:
    """
    Aggregate streaming events into per-customer, 7-day rolling features.
    In production this runs as a Databricks Structured Streaming job that
    writes micro-batch results to a Delta table in the feature store.
    """
    rows = []
    for ev in events:
        rows.append(
            {
                "customer_id": ev["customer_id"],
                "event_type": ev["event_type"],
            }
        )
    ev_df = pd.DataFrame(rows)

    # Pivot event counts
    pivot = (
        ev_df.groupby(["customer_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure expected columns exist even if a type never appeared
    for col in ["click", "purchase", "support_contact", "login", "page_view"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot.rename(
        columns={
            "click": "event_clicks_7d",
            "purchase": "event_purchases_7d",
            "support_contact": "event_support_7d",
            "login": "event_logins_7d",
            "page_view": "event_pageviews_7d",
        }
    )

    pivot["total_events_7d"] = (
        pivot["event_clicks_7d"]
        + pivot["event_purchases_7d"]
        + pivot["event_support_7d"]
        + pivot["event_logins_7d"]
        + pivot["event_pageviews_7d"]
    )
    return pivot


# ---------------------------------------------------------------------------
# 3. Join (point-in-time correct in production via Databricks Feature Store
#    time-travel lookups)
# ---------------------------------------------------------------------------

def join_features(
    batch: pd.DataFrame, streaming: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Left-join batch and streaming features; attach labels for training."""
    joined = batch.merge(streaming, on="customer_id", how="left")
    # Fill customers with no streaming events
    streaming_cols = [
        "event_clicks_7d",
        "event_purchases_7d",
        "event_support_7d",
        "event_logins_7d",
        "event_pageviews_7d",
        "total_events_7d",
    ]
    joined[streaming_cols] = joined[streaming_cols].fillna(0).astype(int)
    # Attach churn label
    joined = joined.merge(
        labels[["customer_id", "churned"]], on="customer_id", how="left"
    )
    return joined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_feature_store() -> pd.DataFrame:
    print("Building feature store …")

    raw_batch = pd.read_parquet(RAW_BATCH)

    with open(RAW_EVENTS) as fh:
        events = [json.loads(line) for line in fh]

    batch_features = compute_batch_features(raw_batch)
    batch_features.to_parquet(FEATURE_STORE_DIR / "batch_features.parquet", index=False)
    print(f"  ✓ batch_features : {batch_features.shape}")

    streaming_features = compute_streaming_features(events)
    streaming_features.to_parquet(
        FEATURE_STORE_DIR / "streaming_features.parquet", index=False
    )
    print(f"  ✓ streaming_features : {streaming_features.shape}")

    joined = join_features(batch_features, streaming_features, raw_batch)
    joined.to_parquet(FEATURE_STORE_DIR / "joined_features.parquet", index=False)
    print(f"  ✓ joined_features    : {joined.shape}")

    return joined


if __name__ == "__main__":
    joined = build_feature_store()
    print("\nSample joined features:")
    print(joined.head(3).to_string(index=False))
