#!/bin/bash
# HF Spaces gives one container + one public port. Run the API internally and
# Streamlit on 7860 (the port HF exposes). app.py calls http://127.0.0.1:8000,
# which resolves in-container, so no code change is needed.
set -e
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
exec streamlit run app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
