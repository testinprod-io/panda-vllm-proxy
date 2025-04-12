# VLLM Proxy

A proxy for vLLM.

## Development Setup

### 1. Create and activate virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install dependencies

```bash
# Install development dependencies
pip install
```

## Run for development

```bash
# Run production server
uvicorn main:app --host 0.0.0.0 --reload

# Run development server
fastapi dev main.py --host 0.0.0.0
```

## Production 

### Build for production

```bash
bash docker/build.sh
```

### Run for production

```bash
cd docker
docker compose up -d
```

### Run for local development

```bash
cd docker/local
docker compose -f docker-compose.local.yml up -d
```

## Tests

```bash
# Run all tests
pytest tests

# Run specific test file
pytest tests/app/test_openai.py
```