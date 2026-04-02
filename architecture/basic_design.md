# Basic Design: Kafka Data Ingestion Pipeline for Real-time P2P Payment Risk Evaluation

## 1. Overview

This document describes the high-level architecture of a **Kafka-based data ingestion pipeline** that feeds a **real-time Peer-to-Peer (P2P) payment risk evaluation system** built on the **Azure Databricks** stack. It walks through real-world scaling scenarios and the challenges that emerge at each stage.

### What is P2P Payment Risk Evaluation?

When a user initiates a person-to-person payment (e.g., Zelle, Venmo, Cash App), the platform must decide **in milliseconds** whether the transaction is legitimate or potentially fraudulent. The risk evaluation system ingests transaction events, enriches them with historical and behavioural features, runs an ML model, and returns a risk score before the payment is authorised.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              AZURE CLOUD BOUNDARY                                    │
│                                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────────────┐   │
│  │                         EVENT PRODUCERS                                       │   │
│  │                                                                               │   │
│  │   Mobile App ──┐    Web App ──┐    Partner API ──┐    Batch Settlement ──┐    │   │
│  │                │              │                   │                       │    │   │
│  │                ▼              ▼                   ▼                       ▼    │   │
│  │          ┌─────────────────────────────────────────────────────────────┐       │   │
│  │          │           Azure Event Hubs (Kafka-compatible)              │       │   │
│  │          │                                                             │       │   │
│  │          │  Topic: p2p.transactions        (partitioned by sender_id) │       │   │
│  │          │  Topic: p2p.account-events      (partitioned by user_id)   │       │   │
│  │          │  Topic: p2p.device-signals       (partitioned by device_id)│       │   │
│  │          │  Topic: p2p.kyc-updates          (partitioned by user_id)  │       │   │
│  │          └──────────┬──────────────┬──────────────┬────────────────────┘       │   │
│  └─────────────────────┼──────────────┼──────────────┼───────────────────────────┘   │
│                        │              │              │                                │
│                        ▼              ▼              ▼                                │
│  ┌───────────────────────────────────────────────────────────────────────────────┐   │
│  │                    AZURE DATABRICKS – INGESTION LAYER                         │   │
│  │                                                                               │   │
│  │  ┌────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │  Structured Streaming Jobs (Spark)                                     │   │   │
│  │  │                                                                        │   │   │
│  │  │  • Read from Event Hubs (Kafka protocol)                               │   │   │
│  │  │  • Schema validation & dead-letter routing                             │   │   │
│  │  │  • Write to Bronze Delta tables (append-only, partitioned by date)     │   │   │
│  │  └────────────────────────────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────────────────────────────┘   │
│                        │                                                             │
│                        ▼                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────┐   │
│  │                    ADLS Gen2 – MEDALLION ARCHITECTURE                         │   │
│  │                                                                               │   │
│  │   Bronze (raw)              Silver (cleansed)           Gold (features)       │   │
│  │  ┌───────────────┐        ┌──────────────────┐       ┌──────────────────┐    │   │
│  │  │ txn_raw       │───────▶│ txn_validated    │──────▶│ sender_profile   │    │   │
│  │  │ account_raw   │───────▶│ account_cleansed │──────▶│ velocity_feats   │    │   │
│  │  │ device_raw    │───────▶│ device_cleansed  │──────▶│ graph_feats      │    │   │
│  │  │ kyc_raw       │───────▶│ kyc_normalised   │──────▶│ risk_features    │    │   │
│  │  └───────────────┘        └──────────────────┘       └──────────────────┘    │   │
│  └───────────────────────────────────────────────────────────────────────────────┘   │
│                        │                                                             │
│                        ▼                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────┐   │
│  │               AZURE DATABRICKS – FEATURE ENGINEERING & ML                     │   │
│  │                                                                               │   │
│  │  ┌─────────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐  │   │
│  │  │  Feature Store       │  │  Model Training  │  │  MLflow Model Registry  │  │   │
│  │  │  (Unity Catalog)     │  │  (scheduled /    │  │  (versioning, staging,  │  │   │
│  │  │                      │  │   on-demand)     │  │   production promotion) │  │   │
│  │  │  • sender_profile    │  │                  │  │                         │  │   │
│  │  │  • velocity_features │  │  XGBoost / NN    │  │  champion / challenger  │  │   │
│  │  │  • device_risk_score │  │  fraud models    │  │  A/B deployment         │  │   │
│  │  │  • graph_features    │  │                  │  │                         │  │   │
│  │  └──────────┬──────────┘  └────────┬─────────┘  └────────────┬────────────┘  │   │
│  │             │                      │                          │               │   │
│  └─────────────┼──────────────────────┼──────────────────────────┼───────────────┘   │
│                │                      │                          │                    │
│                ▼                      ▼                          ▼                    │
│  ┌───────────────────────────────────────────────────────────────────────────────┐   │
│  │                 REAL-TIME RISK SCORING SERVICE                                │   │
│  │                                                                               │   │
│  │  ┌──────────────────────────────────────────────────────────────────────┐     │   │
│  │  │  Databricks Model Serving  /  AKS + FastAPI                         │     │   │
│  │  │                                                                      │     │   │
│  │  │  1. Receive transaction event                                        │     │   │
│  │  │  2. Lookup sender + receiver features (Redis / Feature Serving)      │     │   │
│  │  │  3. Run risk model  →  risk_score + risk_label (ALLOW / REVIEW / BLOCK) │  │   │
│  │  │  4. Return decision in < 100 ms                                      │     │   │
│  │  │  5. Publish decision event to p2p.risk-decisions topic               │     │   │
│  │  └──────────────────────────────────────────────────────────────────────┘     │   │
│  └───────────────────────────────────────────────────────────────────────────────┘   │
│                        │                                                             │
│                        ▼                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────┐   │
│  │                 MONITORING & OBSERVABILITY                                    │   │
│  │                                                                               │   │
│  │  Azure Monitor  │  Databricks Lakehouse Monitoring  │  Grafana Dashboards    │   │
│  │  (latency, SLA) │  (feature drift, data quality)    │  (real-time ops view)  │   │
│  └───────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Descriptions

| Component | Technology | Role in Pipeline |
|---|---|---|
| **Event Producers** | Mobile / Web apps, Partner APIs | Generate P2P transaction requests and ancillary signals (device telemetry, KYC changes). |
| **Message Broker** | Azure Event Hubs (Kafka-compatible) | Durable, partitioned event bus. Decouples producers from consumers; provides replay (up to 7 days) and at-least-once delivery. |
| **Ingestion Layer** | Databricks Structured Streaming | Reads Kafka topics via the Spark Kafka connector; validates schemas; writes to Bronze Delta tables. Checkpointed to ADLS for exactly-once processing. |
| **Storage** | ADLS Gen2 + Delta Lake (Medallion architecture) | Bronze → Silver → Gold layers provide progressive data refinement. Delta gives ACID transactions, time travel, and Z-ordering for fast lookups. |
| **Feature Engineering** | Databricks Spark Jobs + Feature Store (Unity Catalog) | Computes real-time and batch features (velocity counts, graph centrality, device risk scores) and registers them for both training and serving. |
| **Model Training** | Databricks ML Runtime + MLflow | Trains fraud detection models (XGBoost, neural networks) on labelled historical data; tracks experiments; registers model versions. |
| **Risk Scoring Service** | Databricks Model Serving or AKS + FastAPI | Synchronous endpoint that scores each transaction in < 100 ms. Looks up pre-computed features and calls the model in-process. |
| **Decision Feedback Loop** | Event Hubs topic `p2p.risk-decisions` | Publishes every risk decision back to Kafka so downstream systems (case management, analytics, model retraining) can consume them. |
| **Monitoring** | Azure Monitor + Lakehouse Monitoring + Grafana | Tracks end-to-end latency, throughput, feature drift, and model performance. Alerts on SLA breaches. |

---

## 4. Real-World Scaling Scenarios & Challenges

The following scenarios illustrate what happens to this architecture as transaction volume grows from a small fintech launch to a large-scale payment network.

### Scenario 1 — Early Launch: ~100 transactions / second

**Context:** A fintech startup launches a P2P payment feature. A few thousand daily active users send money to friends and family.

**Architecture at this scale:**
- Single Event Hubs namespace with 4 partitions
- One small Databricks cluster (2–4 nodes) running Structured Streaming
- Risk scoring via a single Databricks Model Serving endpoint
- Features refreshed in micro-batches every 30 seconds

**Challenges that begin to appear:**

| Challenge | Description | Impact |
|---|---|---|
| **Schema evolution** | Product team adds new fields (e.g., `payment_note`, `linked_card_type`) to the transaction payload. Downstream consumers break on unexpected columns. | Streaming jobs fail; manual intervention required to restart. |
| **Late-arriving events** | Mobile clients on poor networks submit transactions that arrive minutes after they occurred. | Velocity features (e.g., "transactions in last 5 minutes") become inaccurate, causing false positives. |
| **Cold-start users** | New users have no transaction history, so feature vectors are mostly null. | The risk model has low confidence; either too many false blocks (bad UX) or too many false allows (fraud loss). |

**Mitigations:**
- Register schemas in the Event Hubs Schema Registry; enforce `BACKWARD` compatibility.
- Use Structured Streaming watermarks (`withWatermark("event_time", "10 minutes")`) to handle late arrivals.
- Implement population-level default features for new users; route cold-start users through a separate rule-based engine until enough history accumulates.

---

### Scenario 2 — Regional Growth: ~5,000 transactions / second

**Context:** The product gains traction in a major metro. Marketing campaigns drive spikes during weekends and holidays (Diwali, Black Friday). The platform now handles millions of daily transactions.

**Architecture adjustments:**
- Event Hubs scaled to 32 partitions
- Databricks cluster auto-scales from 4 to 16 nodes
- Redis cache added for online feature serving (sub-millisecond lookups)
- Separate streaming and batch clusters to avoid resource contention

**Challenges that emerge:**

| Challenge | Description | Impact |
|---|---|---|
| **Consumer lag during traffic spikes** | Friday evening payment surges cause the streaming consumer to fall behind. The lag between event production and risk scoring grows from seconds to minutes. | Transactions are held in a pending state; users see delays; customer complaints spike. |
| **Hot partitions** | A viral campaign causes one popular user to send/receive thousands of payments. All their events land on the same partition (keyed by `sender_id`), creating a bottleneck. | One Spark task takes 10× longer than others; overall micro-batch latency increases. |
| **Feature-serving skew** | Batch features (refreshed daily) and streaming features (refreshed every 30 s) drift apart. The model was trained on joined features at a point-in-time, but serving uses stale batch data. | Model accuracy degrades silently; fraud catch rate drops 5–10% before anyone notices. |
| **Kafka offset management** | A streaming job crashes mid-checkpoint. On restart, it replays 5 minutes of events, causing duplicate risk evaluations. | Duplicate payment holds frustrate users; reconciliation logic is required. |

**Mitigations:**
- Implement **back-pressure monitoring**: alert when consumer lag exceeds a threshold (e.g., 10,000 events or 60 seconds). Pre-scale clusters before known spikes.
- Mitigate hot partitions by using a **composite partition key** (`sender_id + random_suffix`) and repartitioning within Spark.
- Reduce feature skew by increasing batch feature refresh frequency (every 4 hours instead of daily) or computing near-real-time batch features using Delta Live Tables.
- Leverage Structured Streaming's **exactly-once checkpointing** to ADLS; design the scoring service to be **idempotent** (same transaction ID → same decision).

---

### Scenario 3 — National Scale: ~50,000 transactions / second

**Context:** The platform is now a nationally adopted P2P payment method. Regulatory requirements mandate real-time AML (Anti-Money Laundering) screening alongside fraud detection. Peak traffic during salary days and festivals can hit 100K+ TPS.

**Architecture adjustments:**
- Dedicated Event Hubs cluster (not shared namespace) with 100+ partitions
- Multiple Databricks streaming jobs per topic (fan-out for fraud, AML, analytics)
- Model serving on AKS with horizontal pod autoscaler (50+ pods)
- Redis Cluster (6+ nodes) for feature serving
- Separate Delta tables for fraud decisions and AML decisions (regulatory audit trail)

**Challenges that emerge:**

| Challenge | Description | Impact |
|---|---|---|
| **End-to-end latency budget** | The payment network requires a risk decision in < 50 ms. Feature lookup (Redis) + model inference + network overhead must fit within this budget. At 50K TPS, every added millisecond costs real money. | Cannot meet SLA with a Python-based serving layer; p99 latency exceeds 50 ms. |
| **Multi-model orchestration** | Fraud model, AML model, and velocity-rules engine must all be consulted. Results must be merged into a single ALLOW / REVIEW / BLOCK decision with an explainable reason code. | Sequential model calls blow the latency budget. Parallel calls add complexity (partial failures, timeouts, fallback logic). |
| **Data consistency across topics** | Transaction events, account events, and device signals arrive on different topics at different rates. A risk decision for transaction T1 might use stale device data if the device signal topic has higher lag. | Inconsistent feature vectors lead to non-deterministic risk scores; difficult to reproduce decisions for regulatory audits. |
| **Cluster cost at scale** | Running 100+ Databricks nodes 24/7 across streaming, batch, and training clusters costs $50K–$100K/month. Cost per transaction must stay below business margins. | Finance team demands optimisation; over-optimising risks under-provisioning and SLA breaches. |
| **Regulatory audit requirements** | Regulators require the ability to replay any transaction, see exactly which features were used, and which model version produced the decision. | Must store immutable snapshots of feature vectors alongside decisions; Delta time travel alone is insufficient for cross-table consistency. |

**Mitigations:**
- Move the scoring service from Python to a **low-latency runtime** (e.g., ONNX Runtime, TensorRT, or Java-based serving with Triton Inference Server on AKS).
- Orchestrate models with **parallel async calls** behind a lightweight gateway; use circuit breakers and fallback rules when a model times out.
- Implement **event-time joins** across topics in a single Structured Streaming job using `mapGroupsWithState` to maintain a consistent, time-aligned view.
- Adopt **spot instances** for batch/training workloads; use **reserved capacity** for streaming clusters; implement per-team **cost attribution** via Databricks tags.
- Write a **decision audit record** (transaction + feature snapshot + model version + decision + timestamp) to a dedicated, append-only Delta table with regulatory retention policies.

---

### Scenario 4 — Global Scale: ~500,000+ transactions / second

**Context:** The platform operates across multiple countries and time zones. Always-on, follow-the-sun traffic patterns mean there is no off-peak. Regional data residency laws (GDPR, India's DPDP Act) mandate that data stays within geographic boundaries.

**Architecture adjustments:**
- Multi-region Event Hubs with geo-disaster recovery
- Regional Databricks workspaces (EU, US, India) with Unity Catalog federation
- Regional Redis clusters with read replicas
- Global MLflow model registry with region-specific model variants
- Cross-region event replication for global fraud graph analysis

**Challenges that emerge:**

| Challenge | Description | Impact |
|---|---|---|
| **Cross-region data residency** | An Indian user sends money to a US user. The transaction must be processed in India (sender's data residency) but the receiver's features live in the US workspace. | Feature lookup requires cross-region calls, adding 100+ ms latency; may violate data localisation laws if raw data crosses borders. |
| **Global fraud rings** | Fraud patterns span regions (e.g., compromised accounts in Europe funding mules in Asia). A single-region model cannot detect cross-border collusion. | Region-local models miss 15–20% of sophisticated fraud rings. |
| **Partition exhaustion** | Even with 256 partitions per topic, some partitions carry 10× the average load due to power-law distribution of user activity. | Tail latency (p99.9) becomes unacceptable; rebalancing partitions causes temporary unavailability. |
| **Model versioning across regions** | Rolling out a new model version globally is risky. A model that works well on US transaction patterns may perform poorly on Indian UPI-style transactions. | Simultaneous global deployment causes a spike in false positives in some regions. |
| **Operational complexity** | 6+ Databricks workspaces, 10+ streaming jobs, 50+ AKS pods per region, 3 Redis clusters — the blast radius of a misconfiguration is enormous. | Mean time to recovery (MTTR) for incidents increases; on-call burden grows. |

**Mitigations:**
- Implement a **feature gateway** that serves pre-aggregated, privacy-safe feature vectors across regions without transferring raw data. Use **Delta Sharing** for controlled cross-workspace feature access.
- Train a **global graph-based model** on anonymised, aggregated transaction patterns replicated to a central analytics workspace. Combine global signals with region-local models in a **stacked ensemble**.
- Move to **Azure Event Hubs Premium / Dedicated** with dynamic partition scaling; implement application-level **sharding** (multiple topics per region) to avoid single-topic partition limits.
- Use **canary deployments per region**: roll out new models to 5% of traffic in one region, monitor for 24 hours, then gradually expand. MLflow model aliases (`@champion`, `@challenger`) simplify routing.
- Invest in **Infrastructure as Code** (Terraform / Pulumi), **centralised observability** (Azure Monitor + Grafana Cloud), and **automated runbooks** for common incidents to reduce MTTR.

---

## 5. End-to-End Data Flow: A Single P2P Transaction

To make the architecture concrete, here is the lifecycle of a single payment:

```
1. User A opens the app and sends $50 to User B
   │
   ▼
2. Mobile app publishes event to Event Hubs topic "p2p.transactions"
   {
     "txn_id": "TXN-2026-04-01-00042",
     "sender_id": "USR-A-12345",
     "receiver_id": "USR-B-67890",
     "amount": 50.00,
     "currency": "USD",
     "event_time": "2026-04-01T03:35:12Z",
     "device_id": "DEV-IPHONE-9A3F",
     "ip_address": "203.0.113.42",
     "payment_method": "linked_bank"
   }
   │
   ▼
3. Databricks Structured Streaming reads event from Kafka (< 1 s)
   │
   ├──▶ Writes to Bronze Delta table (raw, immutable)
   │
   ├──▶ Schema validation passes → writes to Silver Delta table
   │
   └──▶ Streaming feature engineering computes:
        • sender velocity: 12 txns in last 24h (above normal)
        • sender-receiver pair: first-time pair
        • device risk: device seen with 3 other accounts (medium risk)
        • amount percentile: 85th percentile for sender
   │
   ▼
4. Features written to Gold layer / Feature Store + Redis cache
   │
   ▼
5. Risk Scoring Service receives the transaction (sync call from payment gateway)
   │
   ├──▶ Feature lookup from Redis (< 5 ms)
   │
   ├──▶ XGBoost model inference (< 10 ms)
   │     risk_score = 0.72 (high)
   │     top_factors: ["first_time_pair", "high_velocity", "multi_account_device"]
   │
   ├──▶ Business rules overlay:
   │     amount < $100 AND score < 0.85 → ALLOW with step-up verification
   │
   └──▶ Decision: ALLOW + request SMS OTP from User A
   │
   ▼
6. Decision published to "p2p.risk-decisions" topic
   │
   ├──▶ Payment gateway receives ALLOW → proceeds with OTP flow
   ├──▶ Case management system logs high-risk decision for analyst review
   └──▶ Decision + feature snapshot written to audit Delta table
```

---

## 6. Summary of Scaling Challenges

| Scale | Key Challenge | Root Cause | Recommended Mitigation |
|---|---|---|---|
| 100 TPS | Schema evolution breaks consumers | No schema governance | Event Hubs Schema Registry + contract tests |
| 100 TPS | Cold-start users cause inaccurate scoring | Sparse feature vectors | Population-level defaults + rule-based fallback |
| 5K TPS | Consumer lag during spikes | Under-provisioned cluster | Back-pressure alerts + pre-scaling + auto-scale |
| 5K TPS | Hot partitions | Skewed partition key | Composite key + in-Spark repartitioning |
| 50K TPS | Latency budget exceeded | Python serving overhead | ONNX Runtime / Triton on AKS |
| 50K TPS | Multi-model orchestration | Sequential scoring calls | Parallel async calls + circuit breakers |
| 50K TPS | Regulatory audit trail | No feature snapshotting | Append-only decision + feature audit table |
| 500K+ TPS | Cross-region data residency | Raw data cannot cross borders | Feature gateway + Delta Sharing |
| 500K+ TPS | Global fraud rings | Region-local models only | Global graph model + stacked ensemble |
| 500K+ TPS | Operational complexity | Too many moving parts | IaC + centralised observability + runbooks |

---

## 7. Technology Stack Summary

| Layer | Azure Databricks Stack Component |
|---|---|
| Message Broker | Azure Event Hubs (Kafka-compatible API) |
| Stream Processing | Databricks Structured Streaming (Spark) |
| Storage | ADLS Gen2 + Delta Lake (Medallion architecture) |
| Feature Store | Databricks Feature Store (Unity Catalog) |
| Online Feature Serving | Azure Cache for Redis |
| Model Training | Databricks ML Runtime + MLflow |
| Model Registry | MLflow Model Registry (Unity Catalog) |
| Real-time Serving | Databricks Model Serving / AKS + FastAPI |
| Orchestration | Databricks Workflows (Jobs API) |
| Monitoring | Azure Monitor + Lakehouse Monitoring + Grafana |
| CI/CD | GitHub Actions + Databricks Asset Bundles |
| Infrastructure | Terraform / Pulumi for Azure resource provisioning |
