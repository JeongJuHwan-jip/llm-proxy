FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cache)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Install package (editable not needed in production)
RUN pip install --no-cache-dir --no-deps .

# Create data directory for SQLite
RUN mkdir -p /app/data

EXPOSE 8000

ENTRYPOINT ["llm-proxy", "start", "--config", "/app/config.yaml"]
