# Deployment Guide

## Local Demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make train-baseline
make build-rag
make app
```

## Docker

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

## FastAPI

```bash
make api
```

Open:

```text
http://localhost:8000/docs
```

## Azure Deployment Options

1. Azure App Service for Streamlit
2. Azure Container Apps for Dockerized app
3. Azure Machine Learning endpoint for model scoring
4. Azure Storage / Blob Storage for raw datasets and model artifacts
5. Azure Key Vault for OpenAI keys and secrets
6. Azure Application Insights for monitoring

## Minimum Production Checklist

- [ ] Secrets stored outside GitHub
- [ ] Input text redaction added before LLM calls
- [ ] Model artifacts versioned
- [ ] RAG policy documents versioned
- [ ] Human approval required for customer-facing responses
- [ ] Logs include prediction, confidence, retrieved policy, and final action
- [ ] Monitoring dashboard tracks drift and low-confidence cases
