# VLLM Proxy

A proxy for vLLM.

## Development Setup

### 1. Create and activate virtual environment

```bash
# Create virtual environment using specific Python version
poetry env use 3.11.12

# Activate virtual environment
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Install development dependencies
poetry install
```

### 3. Run for local development

```bash
# Run local mock vllm
cd docker/local
docker compose -f docker-compose.local.yml up -d

# Run vllm-proxy server locally
cd ../..
uvicorn src.app.main:app --host 0.0.0.0 --reload
```

## Tests

```bash
# Run all tests
pytest tests

# Run specific test file
pytest tests/app/test_openai.py
```