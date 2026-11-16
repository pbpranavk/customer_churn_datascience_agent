# Customer Churn Prediction Agent with TabPFN, Google ADK and Vertex AI

A portfolio-grade **educational prototype** that lets you ask an ADK agent questions about a synthetic telecom-style customer dataset and score individual customers with **TabPFN-3.5-Fast**.

## What it does

- `dataset_summary`: data size, churn prevalence, and contract distribution
- `evaluate_churn_model`: TabPFN fit + held-out 25% test-set metrics (ROC AUC, F1, precision, recall, accuracy)
- `predict_customer_churn`: churn probability and illustrative risk band for customer attributes
- Gemini (via Google ADK) selects tools and presents results conversationally

**Architecture:** User -> Gemini ADK agent -> Python tools -> TabPFN classifier -> structured JSON -> response.

## Requirements

- Python 3.11 or 3.12, Google Cloud project, Vertex AI enabled, application-default credentials
- [Prior Labs account/license acceptance](https://ux.priorlabs.ai) for TabPFN-3.5-Fast weights
- CPU with sufficient RAM; GPU improves performance

> **Important:** TabPFN-3.5-Fast **model weights are non-commercial**. This repository contains no model weights. Deploying a demo to a cloud environment does not automatically grant commercial usage rights. Review the [Prior Labs licensing terms](https://github.com/PriorLabs/TabPFN#license) before using real business data or in production.

## Run locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
# Set TABPFN_TOKEN if needed after accepting the model license.
gcloud auth application-default login
adk web .
```

Open the local URL printed by ADK and select `churn_agent`.

**Try these prompts:**

1. `Summarize the customer churn dataset.`
2. `Evaluate the churn model. What's the ROC AUC?`
3. `Predict churn for a customer with 4 months tenure, $105 monthly charges, 4 support tickets, month_to_month contract, electronic_check payment, and no auto-pay.`

The first model call downloads TabPFN weights and fits on ~450 rows, which may take time. The classifier is cached *within each Python worker*, not across process restarts. The dataset is a synthetic example, so metrics are not real-world results.

## Deploy to Vertex AI Agent Engine / Agent Runtime

The documented ADK CLI supports `adk deploy agent_engine` (Agent Runtime branding may vary). This is **a deployment recipe, not a verified live deployment**.

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
gcloud config set project "$PROJECT_ID"
gcloud services enable aiplatform.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

gcloud auth application-default login
# Make sure the deployed runtime can access TabPFN model weights legally.
# For headless environments, configure TABPFN_TOKEN securely in the runtime.
# Do NOT commit your token to Git, publish it in .env, or pass it as plain text on a shared CLI.

adk deploy agent_engine \
  --project="$PROJECT_ID" \
  --region="$REGION" \
  --display_name="Customer Churn TabPFN Agent" \
  churn_agent
```

**Deployment caveats:** The deployed agent requires network access to retrieve licensed model weights on first use, sufficient RAM, runtime authentication/configuration for Gemini, and secure handling of the TabPFN token. In some environments, model caching and startup delays make this direct on-demand inference design unsuitable. For a robust production design, move TabPFN inference into a separately managed model service with controlled weights and lifecycle; point the ADK tool at that endpoint. Validate Agent Runtime compatibility, IAM and memory settings before relying on the remote instance. Cloud charges may apply.

The packaged `churn_agent/data/customers.csv` is intentionally inside the agent directory so it can be shipped with the source-based deployment.

## Tests

```bash
pip install pytest
python -m pytest -q
```

Unit tests mock the inference call and do not download TabPFN weights. They **do not** constitute an end-to-end model or cloud deployment test.

## Regenerate synthetic data

```bash
python scripts/generate_data.py
```

The target is generated from a known simulated probability function. It is **not** a benchmark for any real retention program. Do not interpret model predictions or hypothetical retention suggestions as causal explanations.

## References

- [Google ADK documentation](https://adk.dev/)
- [ADK Agent Runtime deployment](https://adk.dev/deploy/agent-runtime/deploy/)
- [Prior Labs TabPFN](https://github.com/PriorLabs/TabPFN)
