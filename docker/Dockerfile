# ─── Builder ────────────────────────────────────────────────────────────────
# python:3.11.12-slim
FROM python@sha256:9c85d1d49df54abca1c5db3b4016400e198e9e9bb699f32f1ef8e5c0c2149ccf AS builder

# install build deps, then clean up apt cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential=12.9 \
       curl=7.88.1-10+deb12u12 \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip & install Poetry
RUN pip install --upgrade pip==23.3.1 \
    && pip install poetry==2.1.3

WORKDIR /app

# copy only dependency manifests
COPY pyproject.toml poetry.lock* ./

# install dependencies into system Python (no virtualenv)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# copy app code
COPY src .


# ─── Final ──────────────────────────────────────────────────────────────────
# python:3.11.12-slim
FROM python@sha256:9c85d1d49df54abca1c5db3b4016400e198e9e9bb699f32f1ef8e5c0c2149ccf

WORKDIR /app

# install runtime needs, then clean up
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# bring in installed Python packages and app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn
COPY --from=builder /app /app
RUN chmod +x /app/entrypoint.sh

# ensure logs are unbuffered
ENV PYTHONUNBUFFERED=1

# add non-root user and give app ownership
RUN useradd --create-home appuser \
    && chown -R appuser /app
USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
