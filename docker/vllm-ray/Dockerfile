FROM vllm/vllm-openai:v0.9.1

RUN pip install "ray[default]" pyarrow pandas

COPY modify-tokenizer.sh  /modify-tokenizer.sh
RUN chmod +x /modify-tokenizer.sh
