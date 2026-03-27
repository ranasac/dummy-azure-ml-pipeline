# Deployment Guide – Azure & Databricks End-to-End

This guide walks you through every step required to deploy this ML platform on
Azure / Databricks and run the full end-to-end pipeline: from raw data through
feature engineering, model training, batch / streaming inference, real-time
serving, and monitoring.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Provision Azure Resources](#2-provision-azure-resources)
3. [Local Development Setup](#3-local-development-setup)
4. [Configure the Databricks Workspace](#4-configure-the-databricks-workspace)
5. [Upload the Code to Databricks](#5-upload-the-code-to-databricks)
6. [Update `config/config.yaml`](#6-update-configconfigyaml)
7. [Generate Synthetic Data (or Ingest Real Data)](#7-generate-synthetic-data-or-ingest-real-data)
8. [Run Feature Engineering](#8-run-feature-engineering)
9. [Train and Register the Model](#9-train-and-register-the-model)
10. [Deploy the Batch Inference Job](#10-deploy-the-batch-inference-job)
11. [Deploy the Streaming Inference Job](#11-deploy-the-streaming-inference-job)
12. [Deploy the Real-time Serving API](#12-deploy-the-real-time-serving-api)
13. [Run the Monitoring Job](#13-run-the-monitoring-job)
14. [End-to-End Verification](#14-end-to-end-verification)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Prerequisites

### 1.1 Required accounts and subscriptions

| Requirement | Notes |
|---|---|
| Azure subscription | Contributor or Owner role on the target subscription |
| Azure Databricks workspace | Premium tier required for Unity Catalog and MLflow Model Registry |
| GitHub / Azure DevOps account | For source control and CI/CD (optional for initial deploy) |

### 1.2 Local tooling

Install the following on your workstation before proceeding:

```bash
# Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az version   # ≥ 2.60

# Databricks CLI v2
pip install databricks-cli
databricks --version   # ≥ 0.200.0

# Python 3.10+
python --version   # ≥ 3.10

# Git
git --version
```

### 1.3 Environment variables used throughout this guide

Set these in your shell (or in CI/CD secrets) before running any commands:

```bash
export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
export AZURE_RESOURCE_GROUP="rg-ml-platform"
export AZURE_LOCATION="eastus"                        # change to your preferred region

# ADLS Gen2
export ADLS_ACCOUNT_NAME="mlplatformstorage"          # globally unique, no hyphens
export ADLS_CONTAINER="ml-platform"

# Event Hubs
export EVENTHUB_NAMESPACE="mlplatformeh"
export EVENTHUB_NAME="customer-events"

# Databricks
export DATABRICKS_WORKSPACE_NAME="mlplatform-databricks"
export DATABRICKS_HOST="https://<workspace-id>.azuredatabricks.net"
export DATABRICKS_TOKEN="<your-pat-token>"            # see step 4.1

# Azure Container Apps (serving)
export ACR_NAME="mlplatformacr"                       # globally unique, no hyphens
export CONTAINER_APP_ENV="mlplatform-env"
export CONTAINER_APP_NAME="churn-serving"
```

---

## 2. Provision Azure Resources

All commands below use the Azure CLI. Run them in order.

### 2.1 Login and set subscription

```bash
az login
az account set --subscription "$AZURE_SUBSCRIPTION_ID"
```

### 2.2 Create a resource group

```bash
az group create \
  --name "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_LOCATION"
```

### 2.3 Create ADLS Gen2 storage account

```bash
az storage account create \
  --name "$ADLS_ACCOUNT_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --hns true                # hierarchical namespace = ADLS Gen2

# Create the container used as the data lake root
az storage fs create \
  --name "$ADLS_CONTAINER" \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --auth-mode login
```

Create the expected directory structure inside the container:

```bash
for DIR in raw_data feature_store model_artifacts inference_results/batch \
           inference_results/streaming checkpoints/streaming_inference monitoring; do
  az storage fs directory create \
    --name "$DIR" \
    --file-system "$ADLS_CONTAINER" \
    --account-name "$ADLS_ACCOUNT_NAME" \
    --auth-mode login
done
```

### 2.4 Create Azure Event Hubs (streaming path)

```bash
# Namespace (shared infrastructure)
az eventhubs namespace create \
  --name "$EVENTHUB_NAMESPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_LOCATION" \
  --sku Standard

# Event Hub (the actual topic)
az eventhubs eventhub create \
  --name "$EVENTHUB_NAME" \
  --namespace-name "$EVENTHUB_NAMESPACE" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --partition-count 4 \
  --message-retention 3

# Capture the connection string for later
EVENTHUB_CONN_STR=$(az eventhubs namespace authorization-rule keys list \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --namespace-name "$EVENTHUB_NAMESPACE" \
  --name RootManageSharedAccessKey \
  --query primaryConnectionString -o tsv)
echo "Event Hubs connection string: $EVENTHUB_CONN_STR"
```

### 2.5 Create the Azure Databricks workspace

```bash
az databricks workspace create \
  --name "$DATABRICKS_WORKSPACE_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_LOCATION" \
  --sku premium

# Retrieve the workspace URL
DATABRICKS_HOST=$(az databricks workspace show \
  --name "$DATABRICKS_WORKSPACE_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query workspaceUrl -o tsv)
export DATABRICKS_HOST="https://$DATABRICKS_HOST"
echo "Databricks workspace URL: $DATABRICKS_HOST"
```

### 2.6 Create Azure Container Registry (for the serving image)

```bash
az acr create \
  --name "$ACR_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --sku Basic \
  --admin-enabled true
```

---

## 3. Local Development Setup

### 3.1 Clone the repository

```bash
git clone https://github.com/ranasac/dummy-azure-ml-pipeline.git
cd dummy-azure-ml-pipeline
```

### 3.2 Create and activate a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 3.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 Verify the local setup by running the tests

```bash
PYTHONPATH=. pytest tests/ -v
# Expected: 30 tests pass in ~3 seconds
```

---

## 4. Configure the Databricks Workspace

### 4.1 Generate a Personal Access Token (PAT)

1. Open `$DATABRICKS_HOST` in a browser and log in.
2. Click your username (top-right) → **Settings** → **Developer** → **Access tokens**.
3. Click **Generate new token**, set a 90-day lifetime, copy the token.
4. Store it:

```bash
export DATABRICKS_TOKEN="<paste-token-here>"
```

### 4.2 Configure the Databricks CLI

```bash
databricks configure --token
# Host:  paste $DATABRICKS_HOST
# Token: paste $DATABRICKS_TOKEN
```

### 4.3 Store secrets in Databricks Secret Scope

Secret scopes keep credentials out of notebooks and job configs.

```bash
# Create a scope
databricks secrets create-scope ml-platform-scope

# ADLS storage account key
ADLS_KEY=$(az storage account keys list \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --query "[0].value" -o tsv)

databricks secrets put-secret ml-platform-scope adls-account-key \
  --string-value "$ADLS_KEY"

# Event Hubs connection string
databricks secrets put-secret ml-platform-scope eventhub-conn-str \
  --string-value "$EVENTHUB_CONN_STR"
```

### 4.4 Create a general-purpose cluster

```bash
databricks clusters create --json '{
  "cluster_name": "ml-platform-cluster",
  "spark_version": "15.4.x-scala2.12",
  "node_type_id": "Standard_DS3_v2",
  "autoscale": { "min_workers": 1, "max_workers": 4 },
  "autotermination_minutes": 60,
  "spark_env_vars": {
    "PYSPARK_PYTHON": "/databricks/python3/bin/python3"
  },
  "spark_conf": {
    "spark.databricks.delta.preview.enabled": "true"
  }
}'
```

Note the `cluster_id` printed by the command – you will need it in step 10 and 11.

### 4.5 Mount ADLS Gen2 in Databricks (optional, for notebook access)

Create and run the following cell in any Databricks notebook to mount ADLS:

```python
configs = {
    "fs.azure.account.key.<ADLS_ACCOUNT_NAME>.dfs.core.windows.net":
        dbutils.secrets.get(scope="ml-platform-scope", key="adls-account-key")
}
dbutils.fs.mount(
    source=f"abfss://ml-platform@<ADLS_ACCOUNT_NAME>.dfs.core.windows.net/",
    mount_point="/mnt/ml-platform",
    extra_configs=configs
)
dbutils.fs.ls("/mnt/ml-platform")   # verify the mount
```

Replace `<ADLS_ACCOUNT_NAME>` with the value of `$ADLS_ACCOUNT_NAME`.

### 4.6 Enable Unity Catalog (recommended for production)

Follow the [official Unity Catalog setup guide](https://docs.databricks.com/en/data-governance/unity-catalog/get-started.html)
to attach a metastore to the workspace. This enables feature lineage, RBAC on Delta tables,
and automated point-in-time joins from the Databricks Feature Store.

---

## 5. Upload the Code to Databricks

### Option A – Databricks Repos (recommended)

Databricks Repos syncs directly with your Git provider.

1. In the Databricks UI, go to **Workspace** → **Repos** → **Add Repo**.
2. Paste the GitHub repository URL: `https://github.com/ranasac/dummy-azure-ml-pipeline.git`
3. Click **Create Repo**. Databricks will clone the repo into `/Repos/<username>/dummy-azure-ml-pipeline`.
4. To update later, click the repo → **Pull**.

From the CLI:

```bash
databricks repos create \
  --url https://github.com/ranasac/dummy-azure-ml-pipeline.git \
  --provider gitHub
```

### Option B – Workspace file upload

```bash
# Upload entire source tree to Databricks workspace
databricks workspace import-dir . /ml-platform --overwrite
```

### 5.1 Install Python dependencies on the cluster

Install the project's dependencies as a cluster-level library:

```bash
CLUSTER_ID="<your-cluster-id>"   # from step 4.4

databricks libraries install \
  --cluster-id "$CLUSTER_ID" \
  --requirements /path/to/requirements.txt
```

Alternatively, add a cluster **init script** that runs `pip install -r requirements.txt`
at cluster start-up.

---

## 6. Update `config/config.yaml`

Edit `config/config.yaml` to point at the real Azure resources provisioned in step 2:

```yaml
platform:
  name: azure-ml-platform
  environment: prod                        # or dev / staging

storage:
  adls_account: "<ADLS_ACCOUNT_NAME>"      # e.g. mlplatformstorage
  container: "ml-platform"
  feature_store_path: "feature_store"
  model_artifacts_path: "model_artifacts"
  raw_data_path: "raw_data"
  streaming_path: "streaming_data"

mlflow:
  # Use the Databricks-managed MLflow tracking server:
  tracking_uri: "databricks"
  experiment_name: "/Shared/customer_churn_experiment"
  registered_model_name: "customer_churn_model"

feature_store:
  batch_feature_table: "batch_features"
  streaming_feature_table: "streaming_features"
  joined_feature_table: "joined_features"

model:
  name: "customer_churn_model"
  version: "1"
  input_features:
    - age
    - tenure_months
    - monthly_spend
    - num_products
    - support_tickets_last_90d
    - avg_session_duration_minutes
    - days_since_last_login
    - event_clicks_7d
    - event_purchases_7d
    - event_support_7d
  output: "churn_probability"

batch_inference:
  input_path: "feature_store/joined_features"
  output_path: "inference_results/batch"
  schedule: "0 2 * * *"                   # daily at 02:00 UTC

streaming_inference:
  consumer_group: "$Default"
  checkpoint_path: "checkpoints/streaming_inference"
  output_path: "inference_results/streaming"
  max_events_per_trigger: 1000

serving:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout_seconds: 30

monitoring:
  drift_threshold: 0.1
  alert_email: "ml-ops@example.com"
  metrics_path: "monitoring/metrics"
  alert_path: "monitoring/alerts"
```

> **Tip:** When `mlflow.tracking_uri` is set to `"databricks"`, the MLflow SDK automatically
> uses the workspace-level MLflow server that comes bundled with every Databricks workspace.
> No separate MLflow infrastructure is required.

---

## 7. Generate Synthetic Data (or Ingest Real Data)

### Option A – Local synthetic data (demo / development)

Run the data generator locally and upload the output files to ADLS:

```bash
# Generate fake data
python data/generate_fake_data.py
# → data/raw/batch_customers.parquet  (1 000 CRM records)
# → data/raw/streaming_events.jsonl   (5 000 CDP / webhook events)

# Upload to ADLS Gen2
az storage fs file upload \
  --source data/raw/batch_customers.parquet \
  --path raw_data/batch_customers.parquet \
  --file-system "$ADLS_CONTAINER" \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --auth-mode login

az storage fs file upload \
  --source data/raw/streaming_events.jsonl \
  --path raw_data/streaming_events.jsonl \
  --file-system "$ADLS_CONTAINER" \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --auth-mode login
```

### Option B – Real CRM data via Azure Data Factory

1. In the Azure portal, create an **Azure Data Factory** resource.
2. Create a **Linked Service** pointing to your CRM or data warehouse.
3. Create a pipeline with a **Copy Data** activity that writes to:
   `abfss://ml-platform@<ADLS_ACCOUNT_NAME>.dfs.core.windows.net/raw_data/`
4. Schedule the pipeline to run daily.

### Option C – Stream events via Azure Event Hubs

Produce events using the `azure-eventhub` SDK (already in `requirements.txt`):

```python
import json
from azure.eventhub import EventHubProducerClient, EventData

conn_str = "<EVENTHUB_CONN_STR>"
producer = EventHubProducerClient.from_connection_string(
    conn_str, eventhub_name="customer-events"
)

events = [
    {"event_id": "EVT_001", "customer_id": "CUST_000001",
     "event_type": "click", "event_timestamp": "2024-06-01T10:00:00Z"},
    # ... more events
]

with producer:
    batch = producer.create_batch()
    for e in events:
        batch.add(EventData(json.dumps(e)))
    producer.send_batch(batch)
```

---

## 8. Run Feature Engineering

### 8.1 Run locally (development)

```bash
PYTHONPATH=. python feature_store/feature_engineering.py
# → data/feature_store/batch_features.parquet
# → data/feature_store/streaming_features.parquet
# → data/feature_store/joined_features.parquet
```

### 8.2 Run as a Databricks Job (production)

Create a Databricks Job that runs `feature_store/feature_engineering.py`:

```bash
databricks jobs create --json '{
  "name": "feature-engineering",
  "tasks": [
    {
      "task_key": "feature_engineering",
      "python_wheel_task": {
        "entry_point": "feature_store/feature_engineering.py"
      },
      "existing_cluster_id": "<your-cluster-id>",
      "libraries": [
        {"pypi": {"package": "mlflow==3.10.1"}},
        {"pypi": {"package": "pyarrow==16.1.0"}},
        {"pypi": {"package": "pandas==2.2.2"}}
      ]
    }
  ],
  "schedule": {
    "quartz_cron_expression": "0 0 1 * * ?",
    "timezone_id": "UTC"
  }
}'
```

Trigger the job manually for the first run:

```bash
JOB_ID="<job-id-from-above>"
databricks jobs run-now --job-id "$JOB_ID"
```

---

## 9. Train and Register the Model

### 9.1 Run locally (development)

```bash
PYTHONPATH=. python models/train.py
# Logs experiment to mlruns/ (local MLflow)
# Registers 'customer_churn_model' v1 in the local MLflow Model Registry
```

### 9.2 Run as a Databricks Job (production)

The training job points `mlflow.tracking_uri` to `"databricks"` so all runs are
logged to the workspace-level MLflow server and the model is registered in the
Databricks Unity Catalog Model Registry.

```bash
databricks jobs create --json '{
  "name": "model-training",
  "tasks": [
    {
      "task_key": "train_model",
      "spark_python_task": {
        "python_file": "/Repos/<username>/dummy-azure-ml-pipeline/models/train.py"
      },
      "existing_cluster_id": "<your-cluster-id>",
      "libraries": [
        {"pypi": {"package": "mlflow==3.10.1"}},
        {"pypi": {"package": "scikit-learn==1.5.0"}},
        {"pypi": {"package": "pandas==2.2.2"}},
        {"pypi": {"package": "pyarrow==16.1.0"}}
      ]
    }
  ]
}'
```

Run the job:

```bash
databricks jobs run-now --job-id "<training-job-id>"
```

After the job completes, verify the model in the Databricks UI:
**Machine Learning** → **Models** → `customer_churn_model`

### 9.3 Promote the model to Production

```bash
# Using MLflow CLI
export MLFLOW_TRACKING_URI="databricks"
mlflow models transition-model-version-stage \
  --model-name customer_churn_model \
  --version 1 \
  --stage Production
```

---

## 10. Deploy the Batch Inference Job

### 10.1 Run locally (development)

```bash
PYTHONPATH=. python pipelines/batch_inference_pipeline.py
# → data/inference_results/batch/predictions_<timestamp>.parquet
```

### 10.2 Deploy as a scheduled Databricks Job (production)

```bash
databricks jobs create --json '{
  "name": "batch-inference",
  "tasks": [
    {
      "task_key": "batch_inference",
      "spark_python_task": {
        "python_file": "/Repos/<username>/dummy-azure-ml-pipeline/pipelines/batch_inference_pipeline.py"
      },
      "existing_cluster_id": "<your-cluster-id>",
      "libraries": [
        {"pypi": {"package": "mlflow==3.10.1"}},
        {"pypi": {"package": "pandas==2.2.2"}},
        {"pypi": {"package": "pyarrow==16.1.0"}}
      ]
    }
  ],
  "schedule": {
    "quartz_cron_expression": "0 0 2 * * ?",
    "timezone_id": "UTC",
    "pause_status": "UNPAUSED"
  },
  "email_notifications": {
    "on_failure": ["ml-ops@example.com"]
  }
}'
```

This creates a job that runs automatically at **02:00 UTC every day** and emails
`ml-ops@example.com` on failure.

Trigger an immediate run to verify:

```bash
databricks jobs run-now --job-id "<batch-inference-job-id>"
```

Verify output in ADLS:

```bash
az storage fs file list \
  --path inference_results/batch \
  --file-system "$ADLS_CONTAINER" \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --auth-mode login
```

---

## 11. Deploy the Streaming Inference Job

The streaming job reads events continuously from Azure Event Hubs, aggregates
features in a 30-second micro-batch window, joins with the batch feature store,
and scores the model.

### 11.1 Run locally (development / simulation)

The local version reads from `data/raw/streaming_events.jsonl` and simulates
micro-batches without Event Hubs:

```bash
PYTHONPATH=. python pipelines/streaming_inference_pipeline.py
# → data/inference_results/streaming/stream_batch_*.parquet  (100 files)
```

### 11.2 Deploy as a Databricks Streaming Job (production)

The production streaming job runs indefinitely using Databricks Structured Streaming.

```bash
databricks jobs create --json '{
  "name": "streaming-inference",
  "tasks": [
    {
      "task_key": "streaming_inference",
      "spark_python_task": {
        "python_file": "/Repos/<username>/dummy-azure-ml-pipeline/pipelines/streaming_inference_pipeline.py",
        "parameters": ["--streaming", "true"]
      },
      "existing_cluster_id": "<your-cluster-id>",
      "libraries": [
        {"pypi": {"package": "mlflow==3.10.1"}},
        {"pypi": {"package": "azure-eventhub==5.11.7"}},
        {"pypi": {"package": "delta-spark==3.2.0"}},
        {"pypi": {"package": "pandas==2.2.2"}}
      ]
    }
  ],
  "max_concurrent_runs": 1
}'
```

In the Databricks notebook / job, inject the Event Hubs connection string from
Databricks Secrets:

```python
import pyspark.sql.functions as F

conn_str = dbutils.secrets.get(scope="ml-platform-scope", key="eventhub-conn-str")

df_stream = (
    spark.readStream
    .format("eventhubs")
    .options(**{"eventhubs.connectionString": sc._jvm.com.microsoft.azure.eventhubs.EventHubsUtils.encrypt(conn_str)})
    .load()
)
```

> **Note:** The Databricks Structured Streaming connector for Event Hubs is available
> as a Maven library: `com.microsoft.azure:azure-eventhubs-spark_2.12:2.3.22`.
> Add it to your cluster's **Libraries** tab under **Maven**.

---

## 12. Deploy the Real-time Serving API

The FastAPI application (`serving/inference_service.py`) can be deployed on:

- **Azure Container Apps** (easiest, serverless) – covered below
- **Azure Kubernetes Service (AKS)** – for full control and advanced scaling
- **Databricks Model Serving (Mosaic AI)** – zero infrastructure option

### Option A – Azure Container Apps (recommended for most teams)

#### Step 1 – Create a Dockerfile

Create `Dockerfile` in the repository root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "serving.inference_service:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Step 2 – Build and push the Docker image

```bash
# Login to Azure Container Registry
az acr login --name "$ACR_NAME"

# Build and tag
docker build -t "${ACR_NAME}.azurecr.io/churn-serving:latest" .

# Push
docker push "${ACR_NAME}.azurecr.io/churn-serving:latest"
```

#### Step 3 – Create the Container Apps environment and deploy

```bash
# Install the Container Apps CLI extension (once)
az extension add --name containerapp --upgrade

# Create the environment
az containerapp env create \
  --name "$CONTAINER_APP_ENV" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --location "$AZURE_LOCATION"

# Get the ACR password
ACR_PASSWORD=$(az acr credential show \
  --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

# Deploy the container app
az containerapp create \
  --name "$CONTAINER_APP_NAME" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --environment "$CONTAINER_APP_ENV" \
  --image "${ACR_NAME}.azurecr.io/churn-serving:latest" \
  --registry-server "${ACR_NAME}.azurecr.io" \
  --registry-username "$ACR_NAME" \
  --registry-password "$ACR_PASSWORD" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 10 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --query "properties.configuration.ingress.fqdn" -o tsv
```

The command prints the public HTTPS URL, e.g.
`churn-serving.xxxxxxxxx.eastus.azurecontainerapps.io`.

#### Step 4 – Verify the deployment

```bash
SERVING_URL="https://<fqdn-from-above>"

# Health check
curl "${SERVING_URL}/health"

# Single prediction
curl -X POST "${SERVING_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_000001",
    "age": 35, "tenure_months": 24, "monthly_spend": 150,
    "num_products": 2, "support_tickets_last_90d": 3,
    "avg_session_duration_minutes": 20, "days_since_last_login": 10,
    "event_clicks_7d": 5, "event_purchases_7d": 1, "event_support_7d": 0
  }'
```

### Option B – Databricks Model Serving (Mosaic AI)

Databricks Model Serving hosts the MLflow model directly with no Docker or
Kubernetes required.

1. In the Databricks UI, go to **Machine Learning** → **Models** →
   `customer_churn_model` → **Serving**.
2. Click **Enable serving** and choose an instance type (e.g., `CPU Small`).
3. Click **Create**.

After provisioning (~2 minutes), note the endpoint URL shown in the UI.

Call the endpoint:

```bash
curl -X POST "${DATABRICKS_HOST}/serving-endpoints/customer_churn_model/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [{
      "customer_id": "CUST_000001",
      "age": 35, "tenure_months": 24, "monthly_spend": 150,
      "num_products": 2, "support_tickets_last_90d": 3,
      "avg_session_duration_minutes": 20, "days_since_last_login": 10,
      "event_clicks_7d": 5, "event_purchases_7d": 1, "event_support_7d": 0
    }]
  }'
```

### Option C – Azure Kubernetes Service (AKS)

Use AKS when you need advanced networking, custom middleware, or very
high throughput (> 10,000 req/s). The steps are:

```bash
# Create AKS cluster
az aks create \
  --name mlplatform-aks \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --node-count 3 \
  --node-vm-size Standard_DS2_v2 \
  --attach-acr "$ACR_NAME" \
  --generate-ssh-keys

az aks get-credentials \
  --name mlplatform-aks \
  --resource-group "$AZURE_RESOURCE_GROUP"

# Apply Kubernetes deployment and service manifests
# (Create deployment.yaml with the container spec from Option A above)
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc churn-serving     # note the EXTERNAL-IP
```

---

## 13. Run the Monitoring Job

### 13.1 Run locally

```bash
PYTHONPATH=. python monitoring/model_monitoring.py
# → data/monitoring/monitoring_report_<timestamp>.json
```

The report contains:
- Feature drift (KS-test) for each input feature
- Prediction distribution drift
- High-risk customer rate alerts
- Data quality (missing value rates)

### 13.2 Deploy as a daily Databricks Job

```bash
databricks jobs create --json '{
  "name": "model-monitoring",
  "tasks": [
    {
      "task_key": "monitoring",
      "spark_python_task": {
        "python_file": "/Repos/<username>/dummy-azure-ml-pipeline/monitoring/model_monitoring.py"
      },
      "existing_cluster_id": "<your-cluster-id>",
      "libraries": [
        {"pypi": {"package": "pandas==2.2.2"}},
        {"pypi": {"package": "scipy==1.13.0"}},
        {"pypi": {"package": "pyarrow==16.1.0"}},
        {"pypi": {"package": "structlog==24.2.0"}}
      ]
    }
  ],
  "schedule": {
    "quartz_cron_expression": "0 0 3 * * ?",
    "timezone_id": "UTC",
    "pause_status": "UNPAUSED"
  }
}'
```

---

## 14. End-to-End Verification

Follow these steps in order to confirm the full platform is running correctly.

### Step 1 – Generate or confirm data is in ADLS

```bash
az storage fs file list \
  --path raw_data \
  --file-system "$ADLS_CONTAINER" \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --auth-mode login
# Expected: batch_customers.parquet and streaming_events.jsonl
```

### Step 2 – Confirm feature store tables

```bash
az storage fs file list \
  --path feature_store \
  --file-system "$ADLS_CONTAINER" \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --auth-mode login
# Expected: batch_features.parquet, streaming_features.parquet, joined_features.parquet
```

If missing, trigger the feature engineering job:

```bash
databricks jobs run-now --job-id "<feature-engineering-job-id>"
```

### Step 3 – Confirm model is registered

```bash
# Using MLflow CLI (set MLFLOW_TRACKING_URI=databricks first for production)
export MLFLOW_TRACKING_URI="databricks"
mlflow models list --name customer_churn_model
# Expected: at least version 1 in Staging or Production
```

### Step 4 – Trigger a manual batch inference run

```bash
databricks jobs run-now --job-id "<batch-inference-job-id>"
# Wait for completion (~1-2 minutes)
```

Verify output:

```bash
az storage fs file list \
  --path inference_results/batch \
  --file-system "$ADLS_CONTAINER" \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --auth-mode login
# Expected: predictions_<timestamp>.parquet
```

### Step 5 – Verify the real-time serving API

```bash
# Health check
curl "${SERVING_URL}/health"
# Expected: {"status":"healthy","uptime_seconds":...}

# Model info
curl "${SERVING_URL}/model/info"
# Expected: {"model_name":"customer_churn_model","model_version":"1",...}

# Single prediction
curl -s -X POST "${SERVING_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_000001",
    "age": 35, "tenure_months": 24, "monthly_spend": 150,
    "num_products": 2, "support_tickets_last_90d": 3,
    "avg_session_duration_minutes": 20, "days_since_last_login": 10,
    "event_clicks_7d": 5, "event_purchases_7d": 1, "event_support_7d": 0
  }' | python -m json.tool
# Expected:
# {
#   "customer_id": "CUST_000001",
#   "churn_probability": 0.xxxx,
#   "model_name": "customer_churn_model",
#   "model_version": "1",
#   "inference_timestamp": "2024-..."
# }

# Batch prediction (2 records)
curl -s -X POST "${SERVING_URL}/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"customer_id":"CUST_000001","age":35,"tenure_months":24,"monthly_spend":150,
       "num_products":2,"support_tickets_last_90d":3,"avg_session_duration_minutes":20,
       "days_since_last_login":10,"event_clicks_7d":5,"event_purchases_7d":1,"event_support_7d":0},
      {"customer_id":"CUST_000002","age":50,"tenure_months":6,"monthly_spend":80,
       "num_products":1,"support_tickets_last_90d":0,"avg_session_duration_minutes":5,
       "days_since_last_login":30,"event_clicks_7d":1,"event_purchases_7d":0,"event_support_7d":0}
    ]
  }' | python -m json.tool
# Expected: {"predictions":[...],"total":2,"inference_timestamp":"..."}

# Swagger UI (interactive docs)
open "${SERVING_URL}/docs"
```

### Step 6 – Verify streaming inference output

If the streaming job is running, check for output files:

```bash
az storage fs file list \
  --path inference_results/streaming \
  --file-system "$ADLS_CONTAINER" \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --auth-mode login
# Expected: stream_batch_*.parquet files
```

### Step 7 – Run the monitoring report

```bash
databricks jobs run-now --job-id "<monitoring-job-id>"
# Wait for completion, then check:
az storage fs file list \
  --path monitoring \
  --file-system "$ADLS_CONTAINER" \
  --account-name "$ADLS_ACCOUNT_NAME" \
  --auth-mode login
# Expected: monitoring_report_<timestamp>.json
```

### Step 8 – Run the full test suite

```bash
PYTHONPATH=. pytest tests/ -v
# Expected: 30 tests pass
```

---

## 15. Troubleshooting

### MLflow cannot connect to Databricks registry

```
mlflow.exceptions.MlflowException: RESOURCE_DOES_NOT_EXIST
```

**Fix:** Ensure `MLFLOW_TRACKING_URI=databricks` and `DATABRICKS_HOST`/`DATABRICKS_TOKEN`
environment variables are set, or that you have run `databricks configure --token`.

---

### Batch inference job fails with "Feature store file not found"

```
FileNotFoundError: data/feature_store/joined_features.parquet
```

**Fix:** Run the feature engineering step first (step 8). The batch inference pipeline
depends on the joined feature table being available.

---

### Container App deployment fails with "image not found"

```
ERROR: (ImagePullError) Failed to pull image
```

**Fix:**
1. Confirm the image was pushed: `az acr repository list --name "$ACR_NAME"`
2. Confirm the registry credentials were passed to `az containerapp create` (`--registry-server`, `--registry-username`, `--registry-password`).

---

### Streaming job cannot connect to Event Hubs

```
com.microsoft.azure.eventhubs.AuthorizationFailedException
```

**Fix:**
1. Verify the connection string stored in Databricks Secrets:
   `databricks secrets get-secret ml-platform-scope eventhub-conn-str`
2. Confirm the Event Hubs namespace firewall allows access from the Databricks cluster's
   IP range, or enable the **Allow trusted Microsoft services** option in the Event Hubs
   network settings.

---

### `pytest` fails with import errors

```
ModuleNotFoundError: No module named 'models'
```

**Fix:** Always set `PYTHONPATH=.` before running pytest so that the project root
is on the Python path:

```bash
PYTHONPATH=. pytest tests/ -v
```

---

### Databricks Job stuck in "Pending" state

**Fix:** The cluster may be in a terminated state. Either:
- Set `autotermination_minutes` to a higher value in the cluster config, or
- Change the job to use **Job Compute** (a cluster that starts and stops with
  each run) instead of referencing a long-running cluster.

---

## Summary of Job Schedule

| Job | Schedule (UTC) | Dependencies |
|---|---|---|
| Feature Engineering | 01:00 daily | Raw data in ADLS |
| Model Training | On-demand / weekly | Feature Engineering complete |
| Batch Inference | 02:00 daily | Feature Engineering complete, model registered |
| Streaming Inference | Continuous | Event Hubs, Feature Engineering |
| Model Monitoring | 03:00 daily | Batch Inference complete |

---

## Further Reading

- [Azure Databricks documentation](https://docs.databricks.com/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Azure Container Apps](https://learn.microsoft.com/en-us/azure/container-apps/)
- [Azure Event Hubs for Kafka](https://learn.microsoft.com/en-us/azure/event-hubs/event-hubs-for-kafka-ecosystem-overview)
- [ADLS Gen2 with Databricks](https://learn.microsoft.com/en-us/azure/databricks/connect/storage/azure-storage)
- [`architecture/architecture.md`](architecture/architecture.md) – full architecture diagram and tool justification
- [`architecture/trade_offs.md`](architecture/trade_offs.md) – cost/performance comparison at 1 req/s → 1 M req/s
