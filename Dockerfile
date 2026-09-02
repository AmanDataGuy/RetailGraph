# RetailGraph — HuggingFace Spaces (Docker SDK) image.
# Builds a single container running the FastAPI backend + Streamlit UI.
# The fine-tuned Qwen2-VL weights are NOT here — inference ran offline; the app
# only queries Neo4j (AuraDB) + Qdrant Cloud + Groq, all via env-var secrets.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as UID 1000; it needs a writable HOME for caches.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements-docker.txt .
RUN pip install --no-cache-dir --user -r requirements-docker.txt

# Bake the embedding model into the image so the first query isn't a cold download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .
RUN chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]
