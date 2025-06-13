FROM vllm/vllm-openai:v0.9.1

RUN pip install "ray[default]" pyarrow pandas

RUN apt-get update -y && apt-get install -y numactl
