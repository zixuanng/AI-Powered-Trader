FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY .env.example .env

# Create data directory
RUN mkdir -p data models

# Expose Prometheus metrics port
EXPOSE 8000

# Default: run the trading engine
ENTRYPOINT ["python", "scripts/live_paper.py"]
