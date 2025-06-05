#!/bin/sh
set -e

: "${WORKERS:=1}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

echo "Starting Gunicorn with $WORKERS workers on $HOST:$PORT"
exec gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    --workers $WORKERS \
    --timeout 180 \
    --bind $HOST:$PORT
