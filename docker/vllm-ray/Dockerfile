FROM vllm/vllm-openai:v0.9.1

RUN apt-get update -y && apt-get install -y jq

RUN pip install "ray[default]" pyarrow pandas

COPY modify-tokenizer.sh  /modify-tokenizer.sh
RUN chmod +x /modify-tokenizer.sh
