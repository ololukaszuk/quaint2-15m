FROM python:3.11-slim

LABEL maintainer="BTC ML System"
LABEL service="chainlink-poller"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy shared modules
COPY ../shared /app/shared

# Copy service files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory
RUN mkdir -p /app/logs

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); from shared.db_client import create_db_client_from_env; db = create_db_client_from_env(); exit(0 if db.health_check() else 1)"

# Run service
CMD ["python", "-u", "app.py"]
