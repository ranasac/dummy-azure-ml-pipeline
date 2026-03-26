# Design Trade-offs: Azure ML Platform

This document provides a critical, comparative analysis of every major design decision in the Azure ML Platform. For each stage it evaluates realistic alternatives, their approximate costs, and how the architecture should evolve as traffic grows from **1 request / sec** to **1 million requests / sec**.

---

## Table of Contents

1. [Traffic Scaling Model](#1-traffic-scaling-model)
2. [Data Ingestion – Batch](#2-data-ingestion--batch)
3. [Data Ingestion – Streaming](#3-data-ingestion--streaming)
4. [Storage Layer](#4-storage-layer)
5. [Compute & Processing](#5-compute--processing)
6. [Feature Store](#6-feature-store)
7. [ML Experiment Tracking & Model Registry](#7-ml-experiment-tracking--model-registry)
8. [Real-time Model Serving](#8-real-time-model-serving)
9. [Orchestration](#9-orchestration)
10. [Monitoring & Observability](#10-monitoring--observability)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [End-to-end Architecture Evolution Summary](#12-end-to-end-architecture-evolution-summary)
13. [Total Cost of Ownership Summary](#13-total-cost-of-ownership-summary)

---

## 1. Traffic Scaling Model

The following three phases are used throughout this document to frame trade-offs.

| Phase | Traffic | Requests / Day | Data Volume | Typical Context |
|-------|---------|----------------|-------------|-----------------|
| **Phase 1 – MVP** | ~1 req/sec | ~86 K | GBs / day | Early startup, internal tooling, pilot customers |
| **Phase 2 – Growth** | ~1 K req/sec | ~86 M | TBs / day | Regional launch, B2B SaaS scaling |
| **Phase 3 – Scale** | ~1 M req/sec | ~86 B | PBs / day | Global consumer product, high-frequency fintech/ad-tech |

> **Key insight:** The jump from Phase 2 → Phase 3 is not a linear scale-out. It typically requires architectural re-design: replacing or augmenting components that were "good enough" at 1 K RPS but become bottlenecks at 1 M RPS.

---

## 2. Data Ingestion – Batch

### Options Compared

| Option | Description | Pros | Cons | Approx. Cost |
|--------|-------------|------|------|--------------|
| **Azure Data Factory (current)** | Managed, serverless ETL/ELT with 90+ connectors | Zero infra; native ADLS & Databricks connectors; visual authoring; built-in retry & lineage | Per-activity pricing adds up at high volume; limited Python/custom transform support | $0.25 / 1 K Data Integration Units (DIU)-hour; ~$0–$50/month at Phase 1, ~$200–$800/month at Phase 2 |
| **Databricks Workflows + Notebooks** | Spark-based ELT jobs on Databricks cluster | Same cluster as ML workloads → cost sharing; full Python/Spark flexibility | Requires Databricks cluster running (min cost even idle) | DBU-based; $0.15–$0.55/DBU; ~$100–$400/month at Phase 1 |
| **Apache Airflow (MWAA / Astronomer)** | DAG orchestration with arbitrary operators | Rich ecosystem; battle-tested; multi-cloud | Operational overhead; separate infra to manage | AWS MWAA ~$0.49/env-hour + instance cost; ~$350–$700/month |
| **Azure Synapse Pipelines** | ADF-like service bundled with Synapse | Tight Synapse integration; familiar for ADF users | Vendor lock-in to Synapse; limited benefit if not using Synapse SQL | Comparable to ADF; ~$50–$300/month |
| **dbt + Fivetran** | Best-of-breed ELT stack | dbt SQL transforms are version-controlled; Fivetran handles connectors | Two additional vendors; Fivetran is expensive at scale | Fivetran ~$500–$2 K/month for mid volume; dbt Cloud ~$100–$300/month |

### Decision & Rationale

**Phase 1–2:** Azure Data Factory is the right choice. It requires no cluster management, the cost is proportional to usage, and its connector library covers every common CRM / ERP source.

**Phase 3:** At petabyte volumes ADF DIU costs can spike. The recommendation is to migrate heavy transformation workloads into Databricks Workflows and use ADF solely as a thin orchestration layer for ingestion (copy activity only). This keeps transformation logic in version-controlled Python/Spark notebooks rather than ADF's visual pipeline.

---

## 3. Data Ingestion – Streaming

### Options Compared

| Option | Description | Pros | Cons | Approx. Cost |
|--------|-------------|------|------|--------------|
| **Azure Event Hubs (current)** | Fully managed Kafka-compatible event bus | Kafka API compatibility; managed partitioning; scales to millions of events/sec; 99.95% SLA | Kafka-native features limited (no log compaction until Premium tier); retention max 7 days (Standard) | Standard tier: ~$0.028/million events + $11/throughput unit/month; ~$25–$100/month at Phase 1, ~$200–$2 K/month at Phase 2 |
| **Confluent Cloud (Kafka SaaS)** | Fully managed Apache Kafka | Full Kafka semantics; schema registry included; multi-cloud | Most expensive managed Kafka; separate vendor | ~$0.11/GB ingress + cluster cost; ~$200–$500/month base |
| **Azure Service Bus** | Enterprise messaging queue | Dead-letter queues; sessions; transactions | Not designed for high-throughput streaming; 1 MB message size limit | $0.05–$10 / million operations; ~$10–$50/month |
| **Apache Kafka on AKS** | Self-managed Kafka on Kubernetes | Full control; cheapest at hyperscale | Highest operational overhead; requires Kafka expertise | AKS node cost + operator time; ~$150–$600/month at Phase 2 |
| **Amazon Kinesis Data Streams** | AWS managed streaming | Serverless sharding; good AWS integration | AWS lock-in; poor fit for Azure-native stack | $0.015/shard-hour + $0.014/million PUT records |
| **Azure Event Grid** | Event routing for webhooks/reactive patterns | Great for low-volume event routing; native Azure integration | Not designed for high-throughput data streams | ~$0.60/million operations; free for first 100 K |

### Decision & Rationale

**Phase 1:** Azure Event Hubs Standard tier (2 throughput units) is more than sufficient and costs ~$22/month. The Kafka-compatible endpoint means producers (Shopify, Zendesk, CDP) need zero code changes.

**Phase 2:** Increase throughput units (auto-inflate enabled) or move to Event Hubs Premium for log compaction and longer retention. Budget $500–$1 K/month.

**Phase 3:** At 1 M RPS of events, evaluate Confluent Cloud (Premium) or self-managed Kafka on AKS. The key factor is whether the ops team has Kafka expertise. Confluent Cloud removes that burden at a cost premium (~$5 K–$20 K/month for peak throughput). Alternatively, Azure Event Hubs Dedicated clusters (~$6 K/month per cluster unit) offer fixed-cost, dedicated throughput with no per-message fees.

> **Critical note:** Event Hubs lacks Kafka log compaction on the Standard tier. If the streaming use case requires compacted topics (e.g., latest-state-per-key), upgrade to Premium or switch to Confluent Cloud.

---

## 4. Storage Layer

### Options Compared

| Option | Description | Pros | Cons | Approx. Cost |
|--------|-------------|------|------|--------------|
| **ADLS Gen2 + Delta Lake (current)** | Object storage with ACID Delta format | ACID; time travel; schema evolution; native Databricks; no separate compute to query | Delta format requires Spark or Delta Standalone reader | ~$0.023/GB/month storage + transaction cost; ~$50–$200/month at Phase 1, ~$1–5 K/month at Phase 3 |
| **Azure Blob Storage + Parquet** | Raw object store without Delta | Cheaper; wider reader support | No ACID; no time travel; schema drift is undetected | ~$0.018/GB/month; $10–$100/month at Phase 1 |
| **Snowflake** | Cloud data warehouse | Excellent SQL; separation of compute/storage; rich ecosystem | Expensive at scale; another platform to manage; poor streaming ingest | ~$2/credit; $400–$5 K+/month depending on warehouse size |
| **Google BigQuery** | Serverless analytical DWH | Serverless; great BI/SQL tooling; per-query pricing | GCP-only; vendor lock-in; not ideal for ML feature serving | $5/TB queried; $0.02/GB/month storage |
| **Apache Iceberg on ADLS** | Open table format alternative to Delta | Multi-engine support (Spark, Flink, Trino); vendor neutral | Less mature Databricks integration; operational complexity | Same underlying storage cost as Delta |
| **Azure Synapse Analytics** | Integrated DWH + Spark | One-stop shop; Spark + SQL in one service | Performance lags Databricks for ML workloads; not best-of-breed for streaming | Dedicated SQL Pool from ~$1.20/DWU-hour; Serverless: $5/TB |

### Decision & Rationale

**All phases:** ADLS Gen2 + Delta Lake is the strongest choice for an ML platform:
- ACID guarantees prevent partial-write corruption during concurrent batch + streaming writes.
- Time travel enables reproducible training datasets (retroactive feature joins).
- Z-ordering reduces data scan cost for feature lookups by customer ID.

**Phase 3 addition:** At petabyte scale, introduce **Delta Lake liquid clustering** (replaces static Z-ordering) and **Delta Sharing** for cross-team feature access without data duplication. Consider tiering cold data to **Azure Blob Cool/Archive** (~$0.01/GB/month) while keeping hot data on ADLS Premium ($0.023/GB/month).

---

## 5. Compute & Processing

### Options Compared

| Option | Description | Pros | Cons | Approx. Cost |
|--------|-------------|------|------|--------------|
| **Azure Databricks (current)** | Managed Spark + Photon on Azure VMs | Best Spark performance (Photon); Unity Catalog; native MLflow; unified batch + streaming | Premium pricing; DBU surcharge on top of VM cost | Standard DBU: ~$0.15/DBU; Premium: ~$0.55/DBU; ~$500–$2 K/month at Phase 1, ~$5 K–$30 K/month at Phase 2 |
| **Azure HDInsight** | Managed Hadoop/Spark on Azure | Cheaper than Databricks; open-source Spark | No Photon; poor ML tooling integration; slower; less active development | Standard VMs + HDI surcharge; ~30% cheaper than Databricks |
| **Azure Synapse Spark Pools** | Spark within Synapse workspace | Native Synapse integration; no separate service | 20–40% slower than Databricks Photon; poor streaming support | ~$0.12/vCore-hour; potentially cheaper than Databricks at Phase 1 |
| **AWS EMR** | Managed Spark on AWS | Mature; spot instance integration; Graviton support | AWS-only; no Unity Catalog equivalent; poor Azure integration | ~$0.27/DBU equivalent; spot can save 60–70% |
| **Google Dataproc** | Managed Spark on GCP | Cheap with preemptible VMs; BigQuery connector | GCP-only; limited ML tooling vs Databricks | ~$0.01/vCPU-hour + VM; generally cheapest managed Spark |
| **Azure Container Apps / Kubernetes Jobs** | Container-based batch jobs | Full control; any framework | No Spark; requires custom orchestration | Pay per vCPU/memory-second; ~$0.000024/vCPU-sec |

### Decision & Rationale

**Phase 1:** Azure Databricks on spot/preemptible instances with auto-scaling (min 0 workers during idle) keeps costs to $300–$800/month. Databricks Community or Azure Free tier for development.

**Phase 2:** Enable Photon-accelerated compute for feature engineering and batch scoring. Separate cluster policies for streaming (always-on, small) vs batch (scale to zero). Budget $10 K–$30 K/month.

**Phase 3:** At this scale, Databricks cost can be optimised by:
1. Moving archival/cold-path workloads to Azure Synapse Serverless SQL (pay-per-query).
2. Using Databricks reserved instance pricing (up to 40% discount for 1-year commit).
3. Separating latency-sensitive serving compute from batch ETL workloads to avoid noisy-neighbour effects.

---

## 6. Feature Store

### Options Compared

| Option | Description | Pros | Cons | Approx. Cost |
|--------|-------------|------|------|--------------|
| **Databricks Feature Store / Unity Catalog (current)** | Integrated feature store within Databricks | Zero additional infra; point-in-time joins; lineage; no data duplication; Delta-backed | Only works well if Databricks is the primary platform; no dedicated online store | Included in Databricks premium pricing |
| **Feast (open-source)** | Lightweight, framework-agnostic feature store | Vendor-neutral; pluggable backends; active community | Operational overhead; no GUI; limited governance; manual infra setup | Free (OSS); infra cost of backends (Redis ~$50–$500/month) |
| **Tecton** | Enterprise, fully managed feature platform | Real-time + batch in one; automatic backfill; great DX | Expensive; US-centric; separate vendor to manage | ~$5 K–$20 K+/month for enterprise |
| **Hopsworks** | Open-source feature store + ML platform | Full-featured; HSFS Python SDK; online + offline | Requires dedicated cluster; complex setup | OSS: self-host; cloud managed: ~$2 K–$10 K/month |
| **Redis (online store only)** | In-memory feature cache for low-latency serving | Sub-millisecond reads; simple ops | Offline store not included; data freshness coupling; cost at large key space | Azure Cache for Redis: ~$54/month (C1 Standard) to ~$3 K/month (P4) |
| **Custom ETL → PostgreSQL/DynamoDB** | Hand-rolled feature materialisation | Full control | High maintenance; no lineage; training-serving skew risk | Postgres Flexible Server: ~$50–$500/month |

### Decision & Rationale

**Phase 1:** Databricks Feature Store (Unity Catalog) with no separate online store. Feature vectors are materialised to Delta and served via the FastAPI service (Delta read on each request is acceptable at 1 RPS).

**Phase 2:** The online serving path becomes the bottleneck. At 1 K RPS with a 20 ms feature lookup budget, a Delta table read is too slow (~100–500 ms). **Add Redis** as an online feature store, populated by the streaming feature engineering job. The offline path (training, batch scoring) remains on Delta. Cost addition: ~$200–$500/month for Redis.

**Phase 3:** Redis at 1 M RPS requires a cluster (Azure Cache for Redis Enterprise ~$5 K–$10 K/month). Evaluate Tecton or Hopsworks as a managed platform if operational overhead of maintaining the Redis materialisation pipeline becomes untenable.

> **Feature skew risk:** The biggest hidden cost of a split offline/online store is training-serving skew. Whatever architecture is chosen, the *same* feature transform code must run in both training (offline) and serving (online) contexts. Databricks Feature Store enforces this by design. Custom solutions require disciplined engineering to avoid silent accuracy degradation.

---

## 7. ML Experiment Tracking & Model Registry

### Options Compared

| Option | Description | Pros | Cons | Approx. Cost |
|--------|-------------|------|------|--------------|
| **MLflow on Databricks (current)** | Open-source MLflow hosted in Databricks workspace | No extra cost; native integration; model registry; artifact versioning | UI is basic; limited collaboration features outside Databricks | Included in Databricks |
| **Azure Machine Learning** | Azure's managed ML platform | Native Azure integration; dataset versioning; responsible AI; CI/CD pipelines | Heavier service; overlaps with Databricks capabilities; additional cost | ~$0 for basic registry; compute charged separately; ~$100–$500/month for managed endpoints |
| **Weights & Biases (W&B)** | Purpose-built ML experiment tracker | Best-in-class experiment visualisation; sweep hyperparam optimisation; team collaboration | SaaS subscription cost; data leaves Azure environment | ~$0 (personal) to ~$1 K–$5 K/month (team/enterprise) |
| **Neptune.ai** | Cloud experiment tracker | Good Python API; metadata versioning; team features | Another SaaS vendor; smaller community than W&B | ~$99–$999/month depending on seats |
| **DVC (Data Version Control)** | Git-based data & model versioning | Free; Git-native; no separate server | Not a tracking server; requires separate metrics logging | Free (OSS) |
| **SageMaker Experiments** | AWS-native tracking | AWS integration | AWS-only; limited if not on AWS | Included in SageMaker; compute billed separately |

### Decision & Rationale

**Phase 1:** MLflow on Databricks is zero-additional-cost and sufficient for solo/small teams. The Unity Catalog model registry provides access control and lineage.

**Phase 2:** As the ML team grows beyond 5–10 people, W&B becomes attractive for its experiment comparison UI, sweep scheduling, and report sharing. However, the additional $2 K–$5 K/month cost requires justification via productivity gains.

**Phase 3:** At scale, the model registry becomes critical infrastructure. The recommendation is to invest in Azure Machine Learning's managed registry (integrated with Azure AD, RBAC, audit logging) while keeping MLflow for experiment logging. MLflow can log to Azure ML as the backend tracking server, preserving code portability.

---

## 8. Real-time Model Serving

This is the most traffic-sensitive layer and the area where design decisions have the largest cost and latency implications.

### Options Compared

| Option | Description | Latency | Throughput | Approx. Cost | Operational Complexity |
|--------|-------------|---------|------------|--------------|----------------------|
| **Databricks Model Serving – Mosaic AI (current option A)** | Serverless GPU/CPU model hosting in Databricks | 10–80 ms | Auto-scales to millions/day | Pay-per-token/request; ~$0.10–$0.40 per 1 K requests | Low – fully managed |
| **AKS + FastAPI (current option B)** | Custom container on Azure Kubernetes | 5–20 ms | Limited by cluster size; HPA scales out | AKS nodes ($0.10–$0.40/vCPU-hour); ~$200–$600/month (Phase 1), $3 K–$15 K/month (Phase 3) | High – own Kubernetes ops |
| **Azure ML Online Endpoints** | Managed real-time serving in Azure ML | 10–50 ms | Auto-scales with Azure ML compute | ~$0.05/hour per instance + compute; ~$200–$1 K/month | Medium – Azure ML-managed |
| **Azure Container Apps** | Serverless containers; KEDA event-driven scaling | 10–50 ms (cold start ~1–3 s) | Scale-to-zero; bursts well | ~$0.000024/vCPU-sec; near-zero idle cost | Low-Medium |
| **Azure Functions (HTTP trigger)** | Serverless functions | 5–20 ms (warm), 200 ms–2 s (cold) | Scales automatically but cold starts hurt | Consumption plan: ~$0.20/million executions | Low – but not ideal for ML |
| **AWS SageMaker Real-time Endpoints** | AWS managed inference | 10–60 ms | Auto-scales with SageMaker | ~$0.23–$0.77/hour per instance; similar to AKS | Medium |
| **Triton Inference Server on GPU** | NVIDIA's high-performance inference server | 1–5 ms | Very high (GPU batching) | GPU instance cost (~$1–$3/hour) | High |

### Decision & Rationale

**Phase 1 (1 RPS):**
Azure Container Apps or a single AKS pod (1 replica) running FastAPI. Cost is nearly zero at idle (scale-to-zero with Container Apps). **Recommended: Azure Container Apps** – zero idle cost, no cluster management, scales automatically.

Approx. cost: **$5–$30/month** (minimal traffic, mostly idle).

**Phase 2 (1 K RPS):**
1 K RPS sustained requires ~5–20 FastAPI pods (depending on model complexity). Options:
- **AKS + FastAPI with HPA**: Predictable cost, good performance, requires Kubernetes expertise. Budget ~$2 K–$5 K/month.
- **Databricks Model Serving**: Fully managed, auto-scales instantly. At 1 K RPS × $0.15/1 K requests = ~$3.8 K/month for inference only (excluding feature lookup).
- **Azure ML Online Endpoints**: Good middle ground; managed autoscaling with Azure ML compute.

**Phase 3 (1 M RPS):**
This requires a fundamental re-think. A single-tier FastAPI service cannot handle 1 M RPS without extreme horizontal scaling.

Architecture changes required:
1. **API Gateway (Azure API Management Premium)** – rate limiting, routing, DDoS protection. Cost: ~$3 K/month.
2. **Caching layer (Redis / CDN)** – if the same `customer_id` is requested repeatedly, return cached prediction (with TTL). Can reduce backend load by 80–95% for some use cases.
3. **Async/queue-based inference** – for non-latency-critical requests, push to a queue (Event Hubs) and return results asynchronously. Smooths traffic spikes.
4. **Model compilation** – use ONNX export + ONNX Runtime or TorchScript to reduce per-inference CPU/GPU time by 2–10×.
5. **GPU batching with Triton** – if the model is neural-network-based, Triton's dynamic batching can serve thousands of RPS per GPU.
6. **Multi-region deployment** – Azure Traffic Manager / Front Door to route traffic to the nearest region, reducing latency and improving resilience.

Approx. cost at 1 M RPS: **$50 K–$200 K+/month** (highly dependent on model size, caching effectiveness, and batch vs real-time split).

---

## 9. Orchestration

### Options Compared

| Option | Description | Pros | Cons | Approx. Cost |
|--------|-------------|------|------|--------------|
| **Databricks Workflows (current)** | Native DAG orchestration in Databricks | Zero additional infra; native cluster lifecycle; easy cross-task data passing | Databricks-only; limited non-Spark task types | Included in Databricks platform |
| **Apache Airflow (Apache 2.0)** | Industry-standard DAG orchestrator | Massive ecosystem; any operator type; provider packages for every cloud | Requires its own infra (K8s or managed); operational overhead | Self-hosted: ~$150–$500/month; Astronomer Cloud: ~$1 K–$3 K/month |
| **Azure Data Factory** | ADF as orchestrator for Databricks/SQL tasks | Native Azure; good for simple pipelines with existing ADF investment | Visual-only; limited Python/custom operators; per-activity cost | ~$0.25/1 K DIU-hours + activity runs |
| **Prefect** | Modern Python-native orchestrator | Strong Python DX; automatic retries; cloud dashboard | Relatively newer; less battle-tested than Airflow | Prefect Cloud: $0–$1.5 K+/month; self-hosted free |
| **Dagster** | Asset-centric orchestration | Best-in-class data asset lineage; strong typing | Steeper learning curve; smaller community | OSS free; Dagster Cloud: $0–$2 K+/month |
| **GitHub Actions** | CI/CD-driven job scheduling | Zero infra for simple scheduled jobs; integrated with code | Not designed for data pipelines; limited retry/observability | Included in GitHub; $0.008/minute for extra runners |

### Decision & Rationale

**Phase 1:** Databricks Workflows covers 100% of the pipeline requirements with no additional cost or infra. Recommended.

**Phase 2:** As the team grows and pipelines expand beyond Databricks (e.g., calling external APIs, sending emails, triggering Salesforce), Airflow or Prefect becomes valuable. Prefect is the modern default for Python-native teams; Airflow for teams with existing investment.

**Phase 3:** At this scale, orchestration reliability is critical. The recommendation is either:
- **Managed Airflow (Astronomer)** for maximum reliability with a managed ops model.
- **Dagster** if asset-level lineage and data contracts are a priority.

In all phases, Databricks Workflows handles the ML-specific DAG sub-tasks; the external orchestrator triggers the Databricks jobs via the Jobs API.

---

## 10. Monitoring & Observability

### Options Compared

| Option | Description | Pros | Cons | Approx. Cost |
|--------|-------------|------|------|--------------|
| **Databricks Lakehouse Monitoring (current)** | Automated Delta table quality & drift profiles | Native; no code required; integrates with Unity Catalog | Databricks-only; no infrastructure metrics | Included in Databricks Unity Catalog |
| **Azure Monitor + Log Analytics** | Azure-native metrics & logs | Deep Azure integration; alerting; dashboards; PagerDuty | Limited ML-specific metrics (no drift out of box) | ~$2.30/GB ingested; ~$50–$500/month at Phase 1–2 |
| **Grafana + Prometheus** | Open-source metrics stack | Best-in-class dashboards; flexible; wide adoption | Requires infra; persistent storage for Prometheus | Self-hosted: ~$100–$300/month; Grafana Cloud: free–$200+/month |
| **Datadog** | Full-stack SaaS observability | APM, logs, infra, custom metrics in one pane | Expensive at scale; can be $50–$100/host/month | ~$15–$30/host/month + APM + log ingest; $1 K–$20 K/month at Phase 2–3 |
| **Evidently AI** | Open-source ML monitoring (data & model drift) | Purpose-built for ML; report generation; free | Requires custom integration and hosting | Free (OSS); cloud managed available |
| **Arize AI / WhyLabs** | ML observability platforms | Drift, data quality, explainability in one platform | SaaS cost; US-centric data residency concerns | ~$500–$5 K+/month |
| **Monte Carlo** | Data observability (pipeline reliability) | Best-in-class data freshness, volume, schema checks | Expensive; focused on data quality, not ML model metrics | ~$2 K–$10 K+/month |

### Decision & Rationale

**Phase 1:** Databricks Lakehouse Monitoring for feature/data drift + Azure Monitor for infrastructure metrics. Total cost: near zero (included in Databricks + Azure Monitor base).

**Phase 2:** Add **Grafana + Prometheus** (or Grafana Cloud) to get real-time latency histograms, error rates, and throughput dashboards. Use **Evidently AI** (OSS) for model/data drift reports. Cost: ~$200–$500/month additional.

**Phase 3:** At scale, the investment in Datadog (or equivalent APM) for full-stack observability becomes justified by the operational savings from faster incident resolution. The cost (~$10 K–$50 K/month) is small relative to the revenue impact of a 1-minute outage at 1 M RPS. For ML-specific monitoring, **Arize AI** or **WhyLabs** provides the depth of drift analysis that generic APM tools cannot.

**Critical metric to monitor at every phase:**
- **Feature freshness** – if streaming features are stale, predictions degrade silently.
- **Prediction distribution drift** – the output distribution of churn probability shifting is an early warning of data or model issues.
- **p99 latency** – 99th-percentile latency, not average, drives user experience at scale.

---

## 11. CI/CD Pipeline

### Options Compared

| Option | Description | Pros | Cons | Approx. Cost |
|--------|-------------|------|------|--------------|
| **GitHub Actions (current)** | YAML-based workflows triggered by Git events | Native GitHub integration; free for open-source; rich marketplace | Limited compute for large test suites; no built-in Databricks notebook promotion | Free (public) / $0.008/minute (private) |
| **Azure DevOps Pipelines** | Microsoft's CI/CD service | Deep Azure integration; YAML or GUI; Databricks extension | Microsoft ecosystem dependency; GUI-heavy | Free 1,800 min/month; $40/extra parallel job/month |
| **Jenkins** | Self-hosted CI/CD server | Maximum flexibility; any plugin | High operational overhead; security patching burden | Self-hosted: ~$100–$300/month in compute |
| **Databricks Asset Bundles (DABs)** | Infrastructure-as-code for Databricks resources | Version-control Databricks jobs/clusters; dev→staging→prod promotion | Databricks-specific; does not handle non-Databricks resources | Included in Databricks |
| **Terraform + GitHub Actions** | IaC for all Azure resources + GitHub Actions CI | Full infrastructure reproducibility; GitOps | Terraform learning curve; state management overhead | GitHub Actions cost + Azure resource cost |

### Decision & Rationale

**Phase 1:** GitHub Actions + Databricks Asset Bundles (DABs). DABs provide job/cluster configuration-as-code and environment promotion (dev → staging → prod) without additional tooling. Cost: free for public repos, ~$20–$50/month for private.

**Phase 2:** Add Terraform for Azure resource provisioning (ADLS, Event Hubs, AKS). Use GitHub Actions for the full CI/CD loop: lint → test → build Docker image → deploy to staging → integration test → promote to prod.

**Phase 3:** Consider Azure DevOps if the organisation is heavily invested in the Microsoft ecosystem. Otherwise, GitHub Actions with self-hosted runners on AKS provides the flexibility and scale needed. Invest in a **dedicated MLOps platform** (e.g., Azure ML Pipelines, Kubeflow) to manage the complexity of A/B testing, canary deployments, and rollback at scale.

---

## 12. End-to-end Architecture Evolution Summary

### Phase 1: MVP (~1 req/sec, GBs/day)

```
CRM / ERP
    │  Azure Data Factory (batch ETL)
    ▼
ADLS Gen2 + Delta Lake (Bronze → Silver → Gold)
    │  Databricks Workflows (feature engineering)
    ▼
Databricks Feature Store (offline only, Delta)
    │  Databricks Model Serving OR
    │  Single AKS pod / Azure Container Apps (FastAPI)
    ▼
API Client

Monitoring: Databricks Lakehouse Monitoring + Azure Monitor (basic)
CI/CD:      GitHub Actions + Databricks Asset Bundles
Cost:       ~$1,000–$3,000/month total
```

**Key characteristics:** Simplicity over scale. Minimise operational surface area. Prefer managed services (ADF, Container Apps, Databricks-managed feature store). Accept higher per-unit cost in exchange for zero ops overhead.

---

### Phase 2: Growth (~1,000 req/sec, TBs/day)

```
CRM / ERP + CDP / Webhooks
    │  Azure Data Factory (batch) + Azure Event Hubs (streaming)
    ▼
ADLS Gen2 + Delta Lake (Medallion architecture)
    │  Azure Databricks + Photon (batch + structured streaming)
    ▼
Databricks Feature Store (offline: Delta) + Redis (online cache, <5 ms)
    │  Azure API Management → AKS + FastAPI (HPA: 5–20 replicas)
    │  (Redis feature cache warms online requests)
    ▼
API Client

Monitoring: Databricks Lakehouse Monitoring + Azure Monitor + Grafana + Evidently AI
CI/CD:      GitHub Actions + Terraform + Databricks Asset Bundles
Cost:       ~$15,000–$50,000/month total
```

**Key architectural additions vs Phase 1:**
- Redis online feature store (eliminates Delta read latency from serving path).
- Azure API Management (rate limiting, auth, observability at API layer).
- Horizontal Pod Autoscaler on AKS (reactive scale-out for burst traffic).
- Separate Databricks cluster policies for streaming (always-on) vs batch (scale-to-zero).
- Champion/challenger A/B routing at the API Gateway layer.

---

### Phase 3: Scale (~1,000,000 req/sec, PBs/day)

```
CRM / ERP + CDP / Webhooks + High-volume event streams
    │  Azure Data Factory (batch) + Azure Event Hubs Dedicated OR Confluent Cloud
    ▼
ADLS Gen2 + Delta Lake (Liquid Clustering, tiered storage, Delta Sharing)
    │  Azure Databricks (reserved capacity, auto-optimise)
    ▼
Tecton / Databricks Feature Store + Redis Enterprise Cluster (online store)
    │
    │  Azure Front Door (global routing, CDN, DDoS)
    ▼
Azure API Management Premium (multi-region)
    │
    ├── Cache layer (Redis) – serve cached predictions for repeated requests
    │
    ├── Real-time path (< 50 ms SLA):
    │       AKS (multi-region, 50–500 replicas, GPU nodes for neural models)
    │       Triton Inference Server (dynamic GPU batching)
    │
    └── Async path (> 50 ms SLA):
            Azure Event Hubs → Databricks Structured Streaming → Delta predictions
            Results fetched via polling or webhook callback

Monitoring: Datadog (full-stack APM) + Arize AI (ML drift) + Monte Carlo (data quality)
CI/CD:      GitHub Actions + Terraform + Databricks Asset Bundles + GitOps (Argo CD)
Cost:       ~$100,000–$500,000+/month total
```

**Key architectural additions vs Phase 2:**
- **Multi-region active-active deployment** (Azure Front Door routes to nearest healthy region).
- **Prediction caching** (Redis TTL cache on `customer_id` → reduces backend load 80–95% for high-repeat traffic).
- **Async inference path** for non-latency-critical bulk requests (smooths traffic spikes, cheaper compute).
- **GPU inference with Triton** for neural-network models (10–100× throughput vs CPU).
- **Delta Lake liquid clustering** + tiered storage to manage petabyte-scale feature tables economically.
- **Event Hubs Dedicated** or Confluent Cloud for guaranteed throughput SLAs at extreme event volumes.
- **GitOps with Argo CD** for Kubernetes deployment management at scale.

---

## 13. Total Cost of Ownership Summary

The table below shows estimated monthly costs for each phase. Costs are Azure list prices (April 2025); actual costs depend on reserved instance discounts, spot/preemptible usage, and data volumes.

| Component | Phase 1 (~1 RPS) | Phase 2 (~1 K RPS) | Phase 3 (~1 M RPS) |
|-----------|-----------------|---------------------|---------------------|
| **Compute (Databricks)** | $500–$1,500 | $8,000–$20,000 | $50,000–$150,000 |
| **Storage (ADLS Gen2 + Delta)** | $50–$200 | $500–$2,000 | $5,000–$20,000 |
| **Streaming (Event Hubs)** | $25–$100 | $500–$2,000 | $6,000–$20,000 (Dedicated) |
| **Serving (AKS / Container Apps)** | $10–$100 | $2,000–$8,000 | $30,000–$100,000 |
| **Online Feature Store (Redis)** | $0 | $200–$1,000 | $5,000–$15,000 |
| **API Management** | $50–$200 | $500–$2,000 | $3,000–$10,000 |
| **Monitoring (Datadog/Arize)** | $50–$200 | $1,000–$5,000 | $15,000–$50,000 |
| **Networking (Front Door, bandwidth)** | $10–$50 | $200–$1,000 | $5,000–$20,000 |
| **CI/CD & Misc.** | $20–$100 | $500–$1,000 | $2,000–$5,000 |
| **TOTAL (estimated)** | **~$1,000–$3,000** | **~$15,000–$50,000** | **~$120,000–$400,000** |

> **Cost optimisation strategies:**
> - **Reserved instances**: 1- or 3-year Azure reservations cut VM costs by 30–60%.
> - **Spot/preemptible for batch**: Databricks spot clusters for non-time-critical batch jobs save 60–80%.
> - **Caching**: Aggressive prediction caching at Phase 3 can reduce serving compute by 80–95%.
> - **Auto-scale to zero**: Use Container Apps / Databricks scale-to-zero for workloads with variable traffic.
> - **Data lifecycle policies**: Automatically tier data older than 90 days to ADLS Cool/Archive.
> - **Committed use discounts**: Negotiate Databricks commit-based pricing for Phase 2+ workloads.

---

## Decision Matrix: When to Upgrade

Use this table to trigger architectural upgrades before hitting bottlenecks.

| Metric | Threshold | Recommended Action |
|--------|-----------|-------------------|
| Real-time serving p99 latency | > 200 ms | Add Redis online feature cache; scale out serving pods |
| Feature freshness lag | > 5 minutes | Move from micro-batch (30 s) to continuous Structured Streaming |
| Batch scoring job duration | > 4 hours | Optimise with Delta Z-ordering / liquid clustering; add Photon compute |
| Event Hubs throughput units | > 80% utilisation | Enable auto-inflate or upgrade to Premium/Dedicated tier |
| Monthly Databricks spend | > $30 K | Negotiate reserved DBU pricing; evaluate spot cluster policies |
| Serving error rate | > 0.1% | Investigate model/feature store health; add circuit breaker at API Gateway |
| Prediction drift score | > 0.1 (KS statistic) | Trigger automated model retraining pipeline |
| Team size for ML | > 10 engineers | Evaluate dedicated ML platform tooling (Tecton, Arize, W&B) |
