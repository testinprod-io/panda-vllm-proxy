FROM python:3.11

WORKDIR /app
COPY mock_vllm.py .

RUN pip install fastapi uvicorn

EXPOSE 8001

CMD ["uvicorn", "mock_vllm:app", "--host", "0.0.0.0", "--port", "8001"] 