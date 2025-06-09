#!/bin/sh
set -e

sleep 60

if [ -n "$HF_HOME" ]; then
  if [ ! -d "$HF_HOME" ]; then
    mkdir -p "$HF_HOME"
    echo "Created directory: $HF_HOME"
  fi
fi

: "${WORKERS:=1}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

echo "Starting Gunicorn with $WORKERS workers on $HOST:$PORT"
exec gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    --workers $WORKERS \
    --timeout 180 \
    --bind $HOST:$PORT
