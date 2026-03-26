# Azure ML Platform – Batch & Real-time Inferencing

A reference implementation of an end-to-end ML platform built on Azure Databricks /
ADLS / MLflow.  It demonstrates:

- **Synthetic data generation** – structured CRM records and CDP / webhook event streams
- **Feature engineering & feature store** – batch and streaming feature groups joined for training
- **Fake ML model** – MLflow-registered `FakeChurnModel` that accepts feature vectors and
  returns a random probability in `[0, 1]` (drop-in replacement for any real model)
- **Batch inference pipeline** – score all customers from the feature store on a schedule
- **Streaming inference pipeline** – score customers in micro-batches as events arrive
- **Real-time serving API** – FastAPI service (`POST /predict`, `POST /predict/batch`)
- **Model & feature monitoring** – KS-test drift detection and alerting

For full architecture, design decisions, and data-flow diagrams see
[`architecture/architecture.md`](architecture/architecture.md).

---

## Repository layout

```
.
├── architecture/
│   └── architecture.md          # Architecture diagram, tool justification, design trade-offs
├── config/
│   └── config.yaml              # Platform configuration (paths, thresholds, model names)
├── data/
│   ├── generate_fake_data.py    # Synthetic CRM + event data generator
│   └── raw/                     # Generated artefacts (git-ignored)
├── feature_store/
│   └── feature_engineering.py  # Batch & streaming feature groups; feature store join
├── models/
│   ├── model.py                 # FakeChurnModel + predict_batch / predict_single helpers
│   └── train.py                 # MLflow training run + model registration
├── pipelines/
│   ├── batch_inference_pipeline.py    # Daily batch scoring job
│   └── streaming_inference_pipeline.py # Micro-batch streaming scoring job
├── serving/
│   └── inference_service.py     # FastAPI real-time serving application
├── monitoring/
│   └── model_monitoring.py      # KS-test drift detection + alert generation
├── tests/
│   └── test_ml_platform.py      # 30 unit/integration tests (pytest)
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python data/generate_fake_data.py
# → data/raw/batch_customers.parquet  (1 000 CRM records)
# → data/raw/streaming_events.jsonl   (5 000 CDP / webhook events)
```

### 3. Build the feature store

```bash
PYTHONPATH=. python feature_store/feature_engineering.py
# → data/feature_store/batch_features.parquet
# → data/feature_store/streaming_features.parquet
# → data/feature_store/joined_features.parquet
```

### 4. Train and register the model

```bash
PYTHONPATH=. python models/train.py
# Registers 'customer_churn_model' v1 in local MLflow registry (mlruns/)
```

### 5. Run batch inference

```bash
PYTHONPATH=. python pipelines/batch_inference_pipeline.py
# → data/inference_results/batch/predictions_<timestamp>.parquet
```

### 6. Run streaming inference (micro-batch simulation)

```bash
PYTHONPATH=. python pipelines/streaming_inference_pipeline.py
# → data/inference_results/streaming/stream_batch_*.parquet  (100 files)
```

### 7. Start the real-time serving API

```bash
PYTHONPATH=. uvicorn serving.inference_service:app --host 0.0.0.0 --port 8000 --reload
```

Then call it:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_000001",
    "age": 35, "tenure_months": 24, "monthly_spend": 150,
    "num_products": 2, "support_tickets_last_90d": 3,
    "avg_session_duration_minutes": 20, "days_since_last_login": 10,
    "event_clicks_7d": 5, "event_purchases_7d": 1, "event_support_7d": 0
  }' | python -m json.tool
```

Expected response:

```json
{
  "customer_id": "CUST_000001",
  "churn_probability": 0.7743,
  "model_name": "customer_churn_model",
  "model_version": "1",
  "inference_timestamp": "2024-06-01T12:00:00+00:00"
}
```

Other endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/model/info` | Model metadata & feature schema |
| `GET` | `/metrics` | Request counters |
| `POST` | `/predict` | Single-customer inference |
| `POST` | `/predict/batch` | Batch inference (up to 1 000 records) |

### 8. Run monitoring

```bash
PYTHONPATH=. python monitoring/model_monitoring.py
# → data/monitoring/monitoring_report_<timestamp>.json
```

### 9. Run tests

```bash
PYTHONPATH=. pytest tests/ -v
# 30 tests in ~3 seconds
```

---

## Architecture overview

See [`architecture/architecture.md`](architecture/architecture.md) for:

- High-level component diagram
- Tool & technology justification
- Batch and streaming data-flow walkthrough
- Model registration & serving flow
- Scalability and performance strategies
- Potential challenges and mitigations
- Key design trade-offs
