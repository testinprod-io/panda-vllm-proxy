FROM python:3.11

# Install Poppler utilities (system dependency for pdf2image)
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
WORKDIR /tmp
RUN pip install poetry==1.8.5
COPY ./pyproject.toml ./poetry.lock* ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes \
    && pip install --no-cache-dir --upgrade -r requirements.txt \
    && rm -rf requirements.txt

# Copy source code
WORKDIR /app
COPY src ./
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
