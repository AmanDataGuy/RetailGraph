#!/bin/bash
# One container + one public port (true on HF Spaces, Render, and most
# single-service PaaS hosts). Run the API internally and Streamlit on the
# platform-assigned $PORT — HF Spaces sets PORT=7860 itself; Render assigns
# its own dynamic port at runtime, so this falls back to 7860 only when
# nothing sets PORT (e.g. running the image locally). app.py calls
# http://127.0.0.1:8000, which resolves in-container, so no code change
# elsewhere is needed for either platform.
set -e
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
exec streamlit run app.py \
  --server.port "${PORT:-7860}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
