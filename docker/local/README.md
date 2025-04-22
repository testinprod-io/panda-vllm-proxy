# Local Development

This is for local development of vllm-proxy, where the host does not have capabilities on NVIDIA GPUs.

## Run local mock vllm server
```bash
docker compose -f docker-compose.local.yml up -d
```