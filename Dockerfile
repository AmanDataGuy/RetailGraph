# # Dockerfile - one image for the FastAPI backend + Streamlit UI
#
# - Used by docker-compose (api + streamlit services) and by HF Spaces (start.sh runs both).
# - The fine-tuned Qwen2-VL weights are NOT here: extraction ran offline; the app only
#   queries Neo4j + Qdrant + Groq, configured through env vars.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as UID 1000; it needs a writable HOME for caches.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# Dependencies first, in their own layer, so code edits don't reinstall them.
COPY --chown=user pyproject.toml README.md ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/req.txt \
    && pip install --no-cache-dir --user -r /tmp/req.txt

# Bake the embedding model into the image so the first query isn't a cold download.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .
RUN pip install --no-cache-dir --user --no-deps . && chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]
