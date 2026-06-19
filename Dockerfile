FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
# Install CPU-only torch first to keep image size manageable
# (GPU training runs on Modal A100, not in this container)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy project
COPY . .

# Set Python path so src/ imports work
ENV PYTHONPATH=/app

# Expose both ports
EXPOSE 8000
EXPOSE 8501

#uvicorn src.api.main:app --reload --port 8000 streamlit run src/ui/app.py     