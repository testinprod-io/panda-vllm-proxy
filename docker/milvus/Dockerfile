FROM python:3.11.12

RUN pip install apscheduler

WORKDIR /app
COPY docker/milvus/cron/cron.py ./
COPY docker/milvus/cron/compaction.py ./
COPY docker/milvus/cron/log.py ./

CMD ["python", "cron.py"]