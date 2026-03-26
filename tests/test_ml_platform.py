"""
Tests for the core ML platform components.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model import FEATURE_COLUMNS, FakeChurnModel, predict_batch, predict_single


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_feature_df(n: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:06d}" for i in range(n)],
            "age": rng.integers(18, 75, size=n).astype(float),
            "tenure_months": rng.integers(1, 120, size=n).astype(float),
            "monthly_spend": rng.uniform(10, 500, size=n),
            "num_products": rng.integers(1, 6, size=n).astype(float),
            "support_tickets_last_90d": rng.integers(0, 20, size=n).astype(float),
            "avg_session_duration_minutes": rng.uniform(1, 60, size=n),
            "days_since_last_login": rng.integers(0, 365, size=n).astype(float),
            "event_clicks_7d": rng.integers(0, 50, size=n).astype(float),
            "event_purchases_7d": rng.integers(0, 10, size=n).astype(float),
            "event_support_7d": rng.integers(0, 5, size=n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestFakeChurnModel:
    def test_predict_returns_series(self):
        model = FakeChurnModel(seed=42)
        df = make_feature_df()[FEATURE_COLUMNS]
        result = model.predict(None, df)
        assert isinstance(result, pd.Series)

    def test_predict_length_matches_input(self):
        model = FakeChurnModel(seed=42)
        for n in [1, 10, 100]:
            df = make_feature_df(n)[FEATURE_COLUMNS]
            result = model.predict(None, df)
            assert len(result) == n

    def test_predict_probabilities_in_range(self):
        model = FakeChurnModel(seed=42)
        df = make_feature_df(200)[FEATURE_COLUMNS]
        result = model.predict(None, df)
        assert result.between(0.0, 1.0).all(), "All probabilities must be in [0, 1]"

    def test_predict_deterministic_with_same_seed(self):
        df = make_feature_df(10)[FEATURE_COLUMNS]
        r1 = FakeChurnModel(seed=7).predict(None, df)
        r2 = FakeChurnModel(seed=7).predict(None, df)
        pd.testing.assert_series_equal(r1, r2)

    def test_predict_different_seeds_differ(self):
        df = make_feature_df(50)[FEATURE_COLUMNS]
        r1 = FakeChurnModel(seed=1).predict(None, df)
        r2 = FakeChurnModel(seed=2).predict(None, df)
        assert not r1.equals(r2)


class TestPredictBatch:
    def test_returns_dataframe_with_column(self):
        df = make_feature_df(10)
        result = predict_batch(df)
        assert "churn_probability" in result.columns

    def test_output_length(self):
        df = make_feature_df(25)
        result = predict_batch(df)
        assert len(result) == 25

    def test_probabilities_in_range(self):
        df = make_feature_df(100)
        result = predict_batch(df)
        assert result["churn_probability"].between(0.0, 1.0).all()

    def test_missing_column_raises(self):
        df = make_feature_df(5).drop(columns=["age"])
        with pytest.raises(ValueError, match="Missing required feature columns"):
            predict_batch(df)

    def test_original_columns_preserved(self):
        df = make_feature_df(5)
        original_cols = list(df.columns)
        result = predict_batch(df)
        for col in original_cols:
            assert col in result.columns


class TestPredictSingle:
    def test_returns_float(self):
        features = {col: 1.0 for col in FEATURE_COLUMNS}
        result = predict_single(features)
        assert isinstance(result, float)

    def test_probability_in_range(self):
        features = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
        result = predict_single(features)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_compute_batch_features_columns(self):
        from feature_store.feature_engineering import compute_batch_features

        raw = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "age": [30, 45],
                "tenure_months": [12, 36],
                "monthly_spend": [100.0, 250.0],
                "num_products": [2, 4],
                "support_tickets_last_90d": [3, 0],
                "avg_session_duration_minutes": [20.0, 10.0],
                "days_since_last_login": [5, 180],
                "country": ["US", "UK"],
                "plan_type": ["basic", "premium"],
                "snapshot_date": ["2024-06-01", "2024-06-01"],
                "churned": [0, 1],
            }
        )
        result = compute_batch_features(raw)
        expected_cols = {
            "customer_id",
            "age",
            "tenure_months",
            "monthly_spend",
            "num_products",
            "support_tickets_last_90d",
            "avg_session_duration_minutes",
            "days_since_last_login",
            "spend_per_product",
            "is_long_tenure",
            "high_support_usage",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_compute_batch_features_derived_values(self):
        from feature_store.feature_engineering import compute_batch_features

        raw = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "age": [30],
                "tenure_months": [36],
                "monthly_spend": [200.0],
                "num_products": [4],
                "support_tickets_last_90d": [8],
                "avg_session_duration_minutes": [15.0],
                "days_since_last_login": [10],
                "country": ["US"],
                "plan_type": ["basic"],
                "snapshot_date": ["2024-06-01"],
                "churned": [0],
            }
        )
        result = compute_batch_features(raw)
        assert result["spend_per_product"].iloc[0] == pytest.approx(50.0)
        assert result["is_long_tenure"].iloc[0] == 1
        assert result["high_support_usage"].iloc[0] == 1

    def test_compute_streaming_features(self):
        from feature_store.feature_engineering import compute_streaming_features

        events = [
            {"customer_id": "C1", "event_type": "click"},
            {"customer_id": "C1", "event_type": "click"},
            {"customer_id": "C1", "event_type": "purchase"},
            {"customer_id": "C2", "event_type": "support_contact"},
        ]
        result = compute_streaming_features(events)
        c1 = result[result["customer_id"] == "C1"].iloc[0]
        assert c1["event_clicks_7d"] == 2
        assert c1["event_purchases_7d"] == 1

    def test_join_features_no_data_loss(self):
        from feature_store.feature_engineering import join_features

        batch = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "age": [25, 35, 45],
                "tenure_months": [6, 24, 60],
                "monthly_spend": [50.0, 150.0, 300.0],
                "num_products": [1, 2, 3],
                "support_tickets_last_90d": [0, 2, 5],
                "avg_session_duration_minutes": [10.0, 20.0, 30.0],
                "days_since_last_login": [1, 7, 30],
                "snapshot_date": ["2024-06-01"] * 3,
                "spend_per_product": [50.0, 75.0, 100.0],
                "is_long_tenure": [0, 0, 1],
                "high_support_usage": [0, 0, 1],
            }
        )
        streaming = pd.DataFrame(
            {"customer_id": ["C1"], "event_clicks_7d": [5], "event_purchases_7d": [1],
             "event_support_7d": [0], "event_logins_7d": [3], "event_pageviews_7d": [10],
             "total_events_7d": [19]}
        )
        labels = pd.DataFrame({"customer_id": ["C1", "C2", "C3"], "churned": [0, 1, 0]})
        joined = join_features(batch, streaming, labels)
        assert len(joined) == 3
        assert "churned" in joined.columns
        # C2 and C3 have no streaming events → filled with 0
        assert joined.loc[joined["customer_id"] == "C2", "event_clicks_7d"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Serving API tests
# ---------------------------------------------------------------------------

class TestServingAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from serving.inference_service import app

        return TestClient(app)

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_name"] == "customer_churn_model"
        assert "feature_columns" in data

    def test_predict_single(self, client):
        payload = {
            "customer_id": "CUST_000001",
            "age": 35.0,
            "tenure_months": 24.0,
            "monthly_spend": 150.0,
            "num_products": 2.0,
            "support_tickets_last_90d": 3.0,
            "avg_session_duration_minutes": 20.0,
            "days_since_last_login": 10.0,
            "event_clicks_7d": 5.0,
            "event_purchases_7d": 1.0,
            "event_support_7d": 0.0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["customer_id"] == "CUST_000001"
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["model_name"] == "customer_churn_model"

    def test_predict_batch_endpoint(self, client):
        record = {
            "customer_id": "CUST_000002",
            "age": 28.0,
            "tenure_months": 12.0,
            "monthly_spend": 80.0,
            "num_products": 1.0,
            "support_tickets_last_90d": 1.0,
            "avg_session_duration_minutes": 15.0,
            "days_since_last_login": 5.0,
            "event_clicks_7d": 10.0,
            "event_purchases_7d": 2.0,
            "event_support_7d": 0.0,
        }
        payload = {"records": [record, {**record, "customer_id": "CUST_000003"}]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert 0.0 <= pred["churn_probability"] <= 1.0

    def test_predict_batch_too_large(self, client):
        record = {
            "customer_id": "CUST_000001",
            "age": 30.0,
            "tenure_months": 12.0,
            "monthly_spend": 100.0,
            "num_products": 2.0,
            "support_tickets_last_90d": 0.0,
            "avg_session_duration_minutes": 10.0,
            "days_since_last_login": 7.0,
        }
        # 1001 records exceeds the 1000 limit
        payload = {"records": [record] * 1001}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 422  # Validation error

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_predict_requests" in data
        assert "total_predictions_served" in data


# ---------------------------------------------------------------------------
# Batch inference pipeline tests
# ---------------------------------------------------------------------------

class TestBatchInferencePipeline:
    def test_run_inference_adds_column(self):
        from pipelines.batch_inference_pipeline import run_inference

        df = make_feature_df(10)
        result = run_inference(df, model=None)
        assert "churn_probability" in result.columns
        assert result["churn_probability"].between(0.0, 1.0).all()

    def test_run_inference_metadata_columns(self):
        from pipelines.batch_inference_pipeline import run_inference

        df = make_feature_df(5)
        result = run_inference(df, model=None)
        assert "inference_timestamp" in result.columns
        assert "model_name" in result.columns
        assert "model_version" in result.columns


# ---------------------------------------------------------------------------
# Streaming pipeline tests
# ---------------------------------------------------------------------------

class TestStreamingPipeline:
    def test_enrich_events(self):
        from pipelines.streaming_inference_pipeline import enrich_events_with_features

        batch_features = make_feature_df(3).set_index("customer_id")
        cust_ids = list(batch_features.index)
        events = [
            {
                "event_id": "e1",
                "customer_id": cust_ids[0],
                "event_type": "click",
                "event_timestamp": "2024-06-01T10:00:00",
                "source": "cdp",
            },
            {
                "event_id": "e2",
                "customer_id": cust_ids[1],
                "event_type": "purchase",
                "event_timestamp": "2024-06-01T11:00:00",
                "source": "webhook_shopify",
            },
        ]
        enriched = enrich_events_with_features(events, batch_features)
        assert "customer_id" in enriched.columns
        assert len(enriched) == 2

    def test_score_micro_batch(self):
        from pipelines.streaming_inference_pipeline import score_micro_batch

        df = make_feature_df(4)
        df["event_id"] = ["e1", "e2", "e3", "e4"]
        df["event_type"] = ["click", "purchase", "click", "login"]
        scored = score_micro_batch(df)
        assert "churn_probability" in scored.columns
        assert scored["churn_probability"].between(0.0, 1.0).all()
        assert "event_id" in scored.columns


# ---------------------------------------------------------------------------
# Monitoring tests
# ---------------------------------------------------------------------------

class TestMonitoring:
    def test_compute_feature_drift(self):
        from monitoring.model_monitoring import compute_feature_drift

        rng = np.random.default_rng(42)
        n = 200
        reference = pd.DataFrame({col: rng.random(n) for col in FEATURE_COLUMNS})
        current = pd.DataFrame({col: rng.random(n) for col in FEATURE_COLUMNS})
        result = compute_feature_drift(reference, current)
        assert set(result.keys()) == set(FEATURE_COLUMNS)
        for k, v in result.items():
            assert "ks_statistic" in v
            assert "drifted" in v
            assert 0.0 <= v["ks_statistic"] <= 1.0

    def test_compute_prediction_drift(self):
        from monitoring.model_monitoring import compute_prediction_drift

        ref = pd.Series(np.random.default_rng(1).random(300))
        cur = pd.Series(np.random.default_rng(2).random(300) * 0.5)  # shifted distribution
        result = compute_prediction_drift(ref, cur)
        assert "mean_shift" in result
        assert "drifted" in result
        assert isinstance(result["drifted"], bool)

    def test_generate_alerts_high_risk(self):
        from monitoring.model_monitoring import generate_alerts

        # Predictions all above 0.7 → should trigger high-risk alert
        preds = pd.DataFrame({"churn_probability": [0.9] * 100})
        alerts = generate_alerts({}, {}, preds)
        alert_types = [a["type"] for a in alerts]
        assert "high_risk_rate" in alert_types

    def test_generate_alerts_no_alerts(self):
        from monitoring.model_monitoring import generate_alerts

        preds = pd.DataFrame({"churn_probability": [0.1] * 100})
        alerts = generate_alerts({}, {"drifted": False}, preds)
        assert alerts == []
