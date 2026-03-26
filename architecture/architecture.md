# Architecture: Azure ML Platform with Batch & Real-time Inferencing

## 1. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        AZURE CLOUD BOUNDARY                                     │
│                                                                                 │
│  ┌──────────────────────────────┐   ┌──────────────────────────────────────┐   │
│  │     DATA INGESTION LAYER     │   │         STREAMING INGESTION          │   │
│  │                              │   │                                      │   │
│  │  ┌────────────────────────┐  │   │  ┌─────────────────────────────────┐│   │
│  │  │  Azure Data Factory    │  │   │  │     Azure Event Hubs            ││   │
│  │  │  (Batch ETL / ELT)     │  │   │  │  (Kafka-compatible endpoint)    ││   │
│  │  └──────────┬─────────────┘  │   │  └───────────────┬─────────────────┘│   │
│  │             │                │   │                  │                   │   │
│  │  ┌──────────▼─────────────┐  │   │  Sources:        │                   │   │
│  │  │  CRM / ERP systems     │  │   │  • CDP events    │                   │   │
│  │  │  (structured data)     │  │   │  • Shopify hooks │                   │   │
│  │  └────────────────────────┘  │   │  • Zendesk hooks │                   │   │
│  └──────────────────────────────┘   └──────────────────┼───────────────────┘   │
│                │                                        │                       │
│                ▼                                        ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                  AZURE DATA LAKE STORAGE Gen2 (ADLS)                    │   │
│  │                                                                         │   │
│  │   Landing Zone        Bronze Layer        Silver Layer    Gold Layer    │   │
│  │  ┌───────────┐       ┌──────────────┐   ┌───────────┐  ┌────────────┐  │   │
│  │  │ raw_data/ │──────▶│ delta tables │──▶│ cleansed  │─▶│  feature   │  │   │
│  │  │  .jsonl   │       │  (raw copy)  │   │  tables   │  │   store    │  │   │
│  │  │  .parquet │       └──────────────┘   └───────────┘  └────────────┘  │   │
│  │  └───────────┘                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                │                     │                                          │
│                ▼                     ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │              AZURE DATABRICKS (Unified Analytics Platform)              │   │
│  │                                                                         │   │
│  │  ┌───────────────────┐   ┌──────────────────────────────────────────┐  │   │
│  │  │  Feature Store    │   │         ML Compute Cluster               │  │   │
│  │  │  (Unity Catalog)  │   │                                          │  │   │
│  │  │                   │   │  ┌──────────────┐  ┌─────────────────┐   │  │   │
│  │  │ batch_features    │   │  │ Model        │  │  Batch          │   │  │   │
│  │  │ streaming_feats   │   │  │ Training Job │  │  Inference Job  │   │  │   │
│  │  │ joined_features   │   │  │ (daily/on-   │  │  (scheduled,    │   │  │   │
│  │  │                   │   │  │  demand)     │  │   Databricks    │   │  │   │
│  │  └────────┬──────────┘   │  └──────┬───────┘  │   Jobs API)    │   │  │   │
│  │           │              │         │           └────────────────┘   │  │   │
│  │           │              │         ▼                                │  │   │
│  │  ┌────────▼──────────┐   │  ┌──────────────┐                       │  │   │
│  │  │ Structured        │   │  │ MLflow        │                       │  │   │
│  │  │ Streaming Job     │   │  │ (Tracking &   │                       │  │   │
│  │  │ (micro-batch)     │   │  │  Model Reg.)  │                       │  │   │
│  │  └───────────────────┘   │  └──────┬───────┘                       │  │   │
│  │                          └─────────┼─────────────────────────────┘  │   │
│  └──────────────────────────────────────┼───────────────────────────────┘   │
│                                         │                                    │
│                                         ▼                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    MODEL SERVING LAYER                                 │  │
│  │                                                                        │  │
│  │  ┌────────────────────────────────┐  ┌────────────────────────────┐   │  │
│  │  │  Real-time Serving             │  │  Batch Scoring             │   │  │
│  │  │  (Databricks Model Serving /   │  │  (Databricks Jobs /        │   │  │
│  │  │   Azure ML Online Endpoints /  │  │   Azure ML Pipeline)       │   │  │
│  │  │   AKS + FastAPI)               │  │                            │   │  │
│  │  │                                │  │  Input:  Feature Store     │   │  │
│  │  │  POST /predict                 │  │  Output: Delta table with  │   │  │
│  │  │  POST /predict/batch           │  │          predictions       │   │  │
│  │  │                                │  │                            │   │  │
│  │  │  Latency: < 100 ms             │  │  Throughput: millions/hr   │   │  │
│  │  └────────────────────────────────┘  └────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                         │                                    │
│                                         ▼                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │              MONITORING & OBSERVABILITY                                │  │
│  │                                                                        │  │
│  │  ┌─────────────────────┐  ┌────────────────────┐  ┌─────────────────┐ │  │
│  │  │ Databricks          │  │ Azure Monitor /    │  │  Grafana /      │ │  │
│  │  │ Lakehouse Monitoring│  │ Log Analytics      │  │  Power BI       │ │  │
│  │  │ (feature/data drift)│  │ (metrics, alerts)  │  │  (dashboards)   │ │  │
│  │  └─────────────────────┘  └────────────────────┘  └─────────────────┘ │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │              DOWNSTREAM CONSUMERS                                      │  │
│  │                                                                        │  │
│  │  CRM / Salesforce   BI / Power BI   Email Campaign   API Clients      │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tool and Technology Justification

| Layer | Tool / Service | Justification |
|---|---|---|
| Batch Ingestion | **Azure Data Factory** | Native Azure service; 90+ connectors; serverless, auto-scaling; deep ADLS integration. Avoids 3rd-party licensing cost. |
| Streaming Ingestion | **Azure Event Hubs** | Kafka-compatible API → zero code change for Kafka producers; managed partitioning & retention; scales to millions of events/sec. |
| Storage | **ADLS Gen2 + Delta Lake** | ACID transactions, time travel (point-in-time joins), Z-ordering for fast scans; native Databricks integration. |
| Processing | **Azure Databricks** | Unified batch + Structured Streaming on the same cluster; native MLflow; Unity Catalog for governance; cost-efficient auto-scaling. |
| Feature Store | **Databricks Feature Store (Unity Catalog)** | Point-in-time correct joins; feature lineage; shared access between training and serving with no ETL duplication. |
| Experiment Tracking | **MLflow** (native in Databricks) | Open standard; model registry; artifact storage; integrates with Azure ML if needed. |
| Real-time Serving | **Databricks Model Serving (Mosaic AI)** or **AKS + FastAPI** | Databricks option: zero infra, auto-scaling, A/B routing. AKS option: more control, custom middleware, lower cost at scale. |
| Orchestration | **Databricks Workflows (Jobs API)** | DAG orchestration, cluster lifecycle management; native Databricks; replaces Airflow overhead. |
| Monitoring | **Databricks Lakehouse Monitoring** + **Azure Monitor** | DLM: automated feature/data quality profiles on Delta tables. Azure Monitor: infrastructure and latency metrics, PagerDuty integration. |
| CI/CD | **Azure DevOps / GitHub Actions** | Parameterised notebook/job promotion across environments (dev → staging → prod). |

### Why not use a separate 3rd-party feature store (e.g., Feast, Tecton)?
Databricks Feature Store is deeply integrated with Unity Catalog, MLflow, and Structured Streaming. Introducing Feast or Tecton would add operational overhead (another control plane) without material benefit when the data platform is already 100% Databricks.

---

## 3. Data Flow & Integration

### 3.1 Batch Path

```
CRM / Data Warehouse
        │ (ADF pipeline, daily)
        ▼
ADLS – Bronze Delta table
        │ (Databricks Job – data quality, dedup)
        ▼
ADLS – Silver Delta table
        │ (Databricks Job – feature engineering: spend_per_product, tenure flags …)
        ▼
Feature Store – batch_features (Delta, partitioned by snapshot_date)
        │ (Databricks Job – point-in-time join)
        ▼
Feature Store – joined_features
        │ (Databricks Job – daily batch scoring)
        ▼
ADLS – inference_results/batch  (Delta)
        │
        ▼
Downstream: CRM, Power BI, Email Campaign Engine
```

### 3.2 Streaming Path

```
CDP / Webhooks
        │ (Event Hubs SDK / Kafka producer)
        ▼
Azure Event Hubs (partitioned by customer_id)
        │ (Databricks Structured Streaming – micro-batch, 30 s trigger)
        ▼
ADLS – Bronze streaming Delta table (append-only)
        │ (Streaming Feature Engineering Job – 7-day rolling aggregations)
        ▼
Feature Store – streaming_features (Delta, updated in near real-time)
        │ (Streaming Inference Job – joins batch + streaming features, scores model)
        ▼
ADLS – inference_results/streaming (Delta)
        │
        ▼
Downstream: Real-time alerting, API response cache, CRM trigger
```

### 3.3 Online (Real-time) Serving Path

```
API Client / Application
        │ POST /predict {customer_id, features}
        ▼
Azure API Management (rate limiting, auth, observability)
        │
        ▼
Inference Service (FastAPI on AKS / Databricks Model Serving)
        │  feature lookup from online store (Redis Cache / Delta Sharing)
        ▼
MLflow Pyfunc Model (loaded in-process)
        │
        ▼
Response: {customer_id, churn_probability, model_version}
        │
        ▼
Audit log → Azure Monitor → Databricks Lakehouse Monitoring
```

### 3.4 Model Registration Flow

```
Training Notebook / Job
        │  mlflow.start_run()
        ├──▶ log_params(), log_metrics()
        ├──▶ mlflow.pyfunc.log_model(registered_model_name="customer_churn_model")
        │
        ▼
MLflow Model Registry
        ├── Staging   → automated integration tests
        └── Production → canary deployment (10% traffic) → full rollout
```

---

## 4. Scalability and Performance

| Concern | Strategy |
|---|---|
| **Streaming throughput** | Event Hubs partitioned by `customer_id`; Databricks cluster auto-scales; Structured Streaming checkpoints to ADLS so no event is lost on restart. |
| **Batch volume** | Delta Z-ordering on `customer_id`; photon-enabled Databricks compute; partition pruning by `snapshot_date`; parallel feature computation with Spark. |
| **Serving latency** | Model loaded once at startup (not per-request); Redis/online feature store for sub-millisecond feature lookup; horizontal pod autoscaling on AKS. |
| **Feature store scale** | Delta Lake handles petabytes; Unity Catalog enforces schema evolution; streaming writes use `foreachBatch` to avoid small-file problems. |
| **Model registry** | MLflow Model Registry version-stamps every model; champion/challenger A/B routing at the gateway layer. |
| **Reliability** | Event Hubs: 99.95% SLA, 7-day replay window; ADLS: geo-redundant storage (GRS); Databricks: spot + on-demand hybrid cluster policy. |

---

## 5. Potential Challenges and Mitigation

| Challenge | Risk Level | Mitigation |
|---|---|---|
| **Schema evolution of streaming events** | High | Event Hubs Schema Registry + Delta `mergeSchema`; contract testing in CI pipeline. |
| **Late-arriving events** | Medium | Watermarking in Structured Streaming (`withWatermark`); time-travel in Delta allows retroactive correction. |
| **Feature skew (training vs serving)** | High | Single feature computation logic shared between batch & online path; Databricks Feature Store enforces the same transform. |
| **Point-in-time correctness** | High | Databricks Feature Store `.create_training_set()` performs PITR joins automatically; no manual timestamp handling needed. |
| **Cold-start (new customers)** | Medium | Default population-level priors for unknown `customer_id`; separate model for new-user onboarding. |
| **Model staleness** | Medium | Automated retraining trigger when drift score exceeds threshold (Lakehouse Monitoring → Databricks Workflow). |
| **Data privacy / GDPR** | High | Unity Catalog row/column-level security; PII masking in Bronze→Silver transform; right-to-erasure via Delta `DELETE`. |
| **Cost overrun on streaming cluster** | Medium | Separate cost-optimised streaming cluster (spot, autoscale 2–8 nodes); budget alerts in Azure Cost Management. |

---

## 6. Key Design Trade-offs

### Batch vs. Micro-batch for streaming
We choose **Structured Streaming with micro-batch (30 s trigger)** rather than continuous processing because:
- The downstream use case is near-real-time (not hard real-time), so 30 s latency is acceptable.
- Micro-batch is dramatically simpler to operate and debug than continuous streaming.
- Cost is lower: cluster can scale to zero between triggers.

### Databricks Feature Store vs. Online Feature Store (Redis)
For most CRM/churn use cases a 30-second-stale feature vector is acceptable for the online path. If sub-second freshness is required, add a Redis layer populated by the streaming feature engineering job. We defer this until the SLA requires it to avoid premature complexity.

### Fake model vs. real model
The platform deliberately uses a random-output model to decouple ML engineering from the platform design. The `FakeChurnModel.predict()` interface is identical to what a real scikit-learn / XGBoost / PyTorch model would expose, so **swapping in a real model requires zero platform changes**.
