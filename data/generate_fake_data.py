"""
Synthetic data generator for the ML platform demo.

Produces two datasets:
  - Batch customer data  (CRM / data-lake style, saved as Parquet)
  - Streaming event data (CDP / webhook style, saved as JSON-lines)

Both datasets represent a *customer churn* use-case and share a common
`customer_id` key so they can be joined in the feature store.
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
rng = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_CUSTOMERS = 1_000
N_EVENTS = 5_000
SNAPSHOT_DATE = datetime(2024, 6, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Batch customer data  (simulates a Delta / ADLS table)
# ---------------------------------------------------------------------------

def generate_batch_customer_data(n: int = N_CUSTOMERS) -> pd.DataFrame:
    """Return a DataFrame of synthetic CRM records."""
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n + 1)]

    ages = rng.integers(18, 75, size=n)
    tenure_months = rng.integers(1, 120, size=n)
    monthly_spend = rng.uniform(10.0, 500.0, size=n).round(2)
    num_products = rng.integers(1, 6, size=n)
    support_tickets = rng.integers(0, 20, size=n)
    avg_session_min = rng.uniform(1.0, 60.0, size=n).round(2)
    days_since_login = rng.integers(0, 365, size=n)

    # Synthetic churn label (ground-truth, used only for training)
    churn_score = (
        0.3 * (days_since_login / 365)
        + 0.2 * (support_tickets / 20)
        - 0.1 * (tenure_months / 120)
        - 0.1 * (monthly_spend / 500)
        + rng.uniform(0, 0.3, size=n)
    )
    churned = (churn_score > 0.4).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": ages,
            "tenure_months": tenure_months,
            "monthly_spend": monthly_spend,
            "num_products": num_products,
            "support_tickets_last_90d": support_tickets,
            "avg_session_duration_minutes": avg_session_min,
            "days_since_last_login": days_since_login,
            "country": rng.choice(
                ["US", "UK", "DE", "FR", "AU"], size=n, p=[0.4, 0.2, 0.15, 0.15, 0.1]
            ),
            "plan_type": rng.choice(
                ["free", "basic", "premium", "enterprise"],
                size=n,
                p=[0.3, 0.35, 0.25, 0.1],
            ),
            "snapshot_date": SNAPSHOT_DATE.isoformat(),
            "churned": churned,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Streaming event data (simulates CDP / third-party webhook events)
# ---------------------------------------------------------------------------

EVENT_TYPES = ["click", "purchase", "support_contact", "login", "page_view"]
WEIGHTS = [0.35, 0.10, 0.08, 0.20, 0.27]


def generate_streaming_events(n: int = N_EVENTS) -> list[dict]:
    """Return a list of synthetic event records."""
    customer_ids = [f"CUST_{i:06d}" for i in range(1, N_CUSTOMERS + 1)]
    events = []
    for _ in range(n):
        ts = SNAPSHOT_DATE - timedelta(days=int(rng.integers(0, 7)))
        events.append(
            {
                "event_id": fake.uuid4(),
                "customer_id": random.choice(customer_ids),
                "event_type": random.choices(EVENT_TYPES, WEIGHTS)[0],
                "event_timestamp": ts.isoformat(),
                "properties": {
                    "page": fake.uri_path(),
                    "session_id": fake.uuid4(),
                    "amount": round(random.uniform(5, 300), 2)
                    if random.random() < 0.15
                    else None,
                },
                "source": random.choice(["cdp", "webhook_shopify", "webhook_zendesk"]),
            }
        )
    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating synthetic batch customer data …")
    batch_df = generate_batch_customer_data()
    batch_path = DATA_DIR / "batch_customers.parquet"
    batch_df.to_parquet(batch_path, index=False)
    print(f"  ✓ Saved {len(batch_df):,} records → {batch_path}")

    print("Generating synthetic streaming event data …")
    events = generate_streaming_events()
    events_path = DATA_DIR / "streaming_events.jsonl"
    with open(events_path, "w") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")
    print(f"  ✓ Saved {len(events):,} events → {events_path}")

    # Quick stats
    batch_df_loaded = pd.read_parquet(batch_path)
    print(
        f"\nBatch data shape : {batch_df_loaded.shape}  |  "
        f"churn rate: {batch_df_loaded['churned'].mean():.1%}"
    )
    print(
        f"Event types      : {pd.Series([e['event_type'] for e in events]).value_counts().to_dict()}"
    )


if __name__ == "__main__":
    main()
