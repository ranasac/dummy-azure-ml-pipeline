"""
Real-time Inference Serving API
=================================
A FastAPI application that exposes the ML model as a low-latency REST
endpoint, suitable for real-time / near-real-time inference.

Endpoints
---------
POST /predict         – score a single customer
POST /predict/batch   – score a batch of customers (up to 1 000)
GET  /health          – liveness probe
GET  /metrics         – Prometheus-style counters (for observability)
GET  /model/info      – model metadata

In production this service would be deployed on:
  - Azure Kubernetes Service (AKS) behind Azure API Management, or
  - Azure ML Online Endpoints (managed hosting), or
  - Databricks Model Serving (Mosaic AI)

Usage
-----
    uvicorn serving.inference_service:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from models.model import FEATURE_COLUMNS, predict_batch, predict_single

import pandas as pd

# ---------------------------------------------------------------------------
# App & in-memory counters (replace with Prometheus / Azure Monitor in prod)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Churn Inference API",
    description="Near-real-time ML inference for customer churn probability.",
    version="1.0.0",
)

_counters: dict[str, int] = defaultdict(int)
_start_time = time.time()

MODEL_NAME = "customer_churn_model"
MODEL_VERSION = "1"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., examples=["CUST_000001"])
    age: float = Field(..., ge=0, le=120)
    tenure_months: float = Field(..., ge=0)
    monthly_spend: float = Field(..., ge=0)
    num_products: float = Field(..., ge=1)
    support_tickets_last_90d: float = Field(..., ge=0)
    avg_session_duration_minutes: float = Field(..., ge=0)
    days_since_last_login: float = Field(..., ge=0)
    event_clicks_7d: float = Field(default=0.0, ge=0)
    event_purchases_7d: float = Field(default=0.0, ge=0)
    event_support_7d: float = Field(default=0.0, ge=0)


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    model_name: str
    model_version: str
    inference_timestamp: str


class BatchPredictionRequest(BaseModel):
    records: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
    inference_timestamp: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Infrastructure"])
def health() -> dict[str, Any]:
    """Liveness probe – returns 200 when the service is ready."""
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/info", tags=["Model"])
def model_info() -> dict[str, Any]:
    """Return metadata about the currently loaded model."""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "output": "churn_probability",
        "description": "Predicts probability of customer churn within 30 days.",
    }


@app.get("/metrics", tags=["Observability"])
def metrics() -> dict[str, Any]:
    """Return basic operational counters (Prometheus scrape target in prod)."""
    return {
        "total_predict_requests": _counters["predict"],
        "total_batch_predict_requests": _counters["batch_predict"],
        "total_predictions_served": _counters["predictions_served"],
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: CustomerFeatures) -> PredictionResponse:
    """Score a single customer and return their churn probability."""
    _counters["predict"] += 1
    _counters["predictions_served"] += 1

    feature_dict = {col: getattr(request, col) for col in FEATURE_COLUMNS}
    try:
        prob = predict_single(feature_dict)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        customer_id=request.customer_id,
        churn_probability=prob,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        inference_timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_endpoint(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score a batch of customers (up to 1 000 per call)."""
    _counters["batch_predict"] += 1
    _counters["predictions_served"] += len(request.records)

    rows = [r.model_dump() for r in request.records]
    df = pd.DataFrame(rows)

    try:
        scored = predict_batch(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ts = datetime.now(timezone.utc).isoformat()
    predictions = [
        PredictionResponse(
            customer_id=row["customer_id"],
            churn_probability=row["churn_probability"],
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inference_timestamp=ts,
        )
        for _, row in scored.iterrows()
    ]
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        inference_timestamp=ts,
    )
