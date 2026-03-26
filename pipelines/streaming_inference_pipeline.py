"""
Streaming Inference Pipeline
==============================
Simulates a Databricks Structured Streaming job that:
  1. Reads new events from Azure Event Hubs (simulated via a local JSONL file)
  2. Enriches them with batch features from the feature store
  3. Runs the ML model in micro-batch mode
  4. Writes scored records to a Delta / streaming output table

In production this job runs continuously on a Databricks cluster.
Azure Event Hubs is the Kafka-compatible source; the Feature Store lookup
uses the Databricks Feature Store client for low-latency point-in-time joins.

For the demo, we simulate the streaming behaviour by processing the events
file in configurable micro-batches.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from models.model import FEATURE_COLUMNS, predict_batch

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = PROJECT_ROOT / "data" / "raw" / "streaming_events.jsonl"
BATCH_FEATURES_PATH = PROJECT_ROOT / "data" / "feature_store" / "batch_features.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "inference_results" / "streaming"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MICRO_BATCH_SIZE = 50          # events per micro-batch (Spark trigger interval analogue)
REGISTERED_MODEL_NAME = "customer_churn_model"
MODEL_VERSION = "1"
# Streaming feature columns contributed by the live event stream
STREAMING_AGGREGATES = ["event_clicks_7d", "event_purchases_7d", "event_support_7d"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_batch_features() -> pd.DataFrame:
    """Load the offline batch features; acts as the offline feature store lookup."""
    return pd.read_parquet(BATCH_FEATURES_PATH).set_index("customer_id")


def _read_events_in_micro_batches(
    path: Path, batch_size: int
) -> Iterator[list[dict]]:
    """Yield successive micro-batches of raw event records."""
    batch: list[dict] = []
    with open(path) as fh:
        for line in fh:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def enrich_events_with_features(
    events: list[dict], batch_features: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a feature-enriched DataFrame for a micro-batch of events.

    Online streaming features (7-day window) are approximated from the
    current micro-batch itself (in production these would come from an
    online feature store / Redis cache populated by the streaming feature
    engineering job).
    """
    ev_df = pd.DataFrame(
        [
            {
                "event_id": e["event_id"],
                "customer_id": e["customer_id"],
                "event_type": e["event_type"],
                "event_timestamp": e["event_timestamp"],
                "source": e["source"],
            }
            for e in events
        ]
    )

    # Micro-batch streaming aggregates (approximate)
    agg = (
        ev_df.groupby(["customer_id", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["click", "purchase", "support_contact"]:
        if col not in agg.columns:
            agg[col] = 0
    agg = agg.rename(
        columns={
            "click": "event_clicks_7d",
            "purchase": "event_purchases_7d",
            "support_contact": "event_support_7d",
        }
    )

    # Latest event per customer in this micro-batch
    latest = ev_df.sort_values("event_timestamp").groupby("customer_id").last().reset_index()
    enriched = latest.merge(agg[["customer_id"] + STREAMING_AGGREGATES], on="customer_id", how="left")

    # Join offline (batch) features – exclude streaming columns that are
    # already present from the micro-batch aggregation step
    batch_cols_to_join = [
        c for c in batch_features.columns if c not in enriched.columns
    ]
    enriched = enriched.join(batch_features[batch_cols_to_join], on="customer_id", how="left")

    # Fill NaN for customers not in the offline store
    for col in FEATURE_COLUMNS:
        if col not in enriched.columns:
            enriched[col] = 0.0
    enriched[FEATURE_COLUMNS] = enriched[FEATURE_COLUMNS].fillna(0.0)

    return enriched


def score_micro_batch(enriched: pd.DataFrame) -> pd.DataFrame:
    """Run inference and attach metadata."""
    scored = predict_batch(enriched[FEATURE_COLUMNS + ["customer_id"]])
    scored["event_id"] = enriched["event_id"].values
    scored["event_type"] = enriched["event_type"].values
    scored["inference_timestamp"] = datetime.now(timezone.utc).isoformat()
    scored["model_name"] = REGISTERED_MODEL_NAME
    scored["model_version"] = MODEL_VERSION
    return scored[
        [
            "customer_id",
            "event_id",
            "event_type",
            "churn_probability",
            "inference_timestamp",
            "model_name",
            "model_version",
        ]
    ]


def write_micro_batch(df: pd.DataFrame, batch_index: int) -> Path:
    """Append micro-batch results (simulates Delta table streaming write)."""
    out_path = OUTPUT_DIR / f"stream_batch_{batch_index:05d}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_streaming_inference(max_batches: int | None = None) -> int:
    """
    Process the event stream in micro-batches.

    Parameters
    ----------
    max_batches : int | None
        Stop after this many micro-batches (``None`` = process everything).

    Returns
    -------
    int
        Total number of events processed.
    """
    print("=== Streaming Inference Pipeline ===")
    print(f"Source : {EVENTS_PATH}")
    print(f"Micro-batch size : {MICRO_BATCH_SIZE} events\n")

    batch_features = load_batch_features()
    total_processed = 0
    batch_index = 0

    for micro_batch in _read_events_in_micro_batches(EVENTS_PATH, MICRO_BATCH_SIZE):
        if max_batches is not None and batch_index >= max_batches:
            break

        enriched = enrich_events_with_features(micro_batch, batch_features)
        scored = score_micro_batch(enriched)
        out_path = write_micro_batch(scored, batch_index)

        total_processed += len(micro_batch)
        high_risk = (scored["churn_probability"] >= 0.7).sum()
        print(
            f"  Batch {batch_index:04d} | events={len(micro_batch):>4d} | "
            f"customers={len(scored):>4d} | high_risk={high_risk} | → {out_path.name}"
        )
        batch_index += 1

    print(f"\n=== Done | total events processed: {total_processed:,} ===\n")
    return total_processed


if __name__ == "__main__":
    run_streaming_inference()
