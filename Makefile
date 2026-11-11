# # Makefile - one-word shortcuts for running, testing and loading RetailGraph
#
# - Every target calls a script that exists in this repo.
# - Run `make` (or `make help`) to list them.

.PHONY: help run test lint ingest train benchmark docker-up docker-down

PYTHON  := python
COMPOSE := docker compose
# Lets `retailgraph.*` import without `pip install -e .` (also covers training/ and eval/).
export PYTHONPATH := src:.

help:
	@echo ""
	@echo "  make run          Start FastAPI (:8000) + Streamlit (:8501)"
	@echo "  make test         Run the unit tests"
	@echo "  make lint         Run Ruff"
	@echo "  make ingest       Create Neo4j indexes, load the graph, load Qdrant vectors"
	@echo "  make train        Fine-tune Qwen2-VL on Modal (needs a Modal account)"
	@echo "  make benchmark    Run the 30-query vector/graph/GraphRAG benchmark"
	@echo "  make docker-up    Start Neo4j + API + UI with docker compose"
	@echo "  make docker-down  Stop them"
	@echo ""

run:
	uvicorn retailgraph.api.main:app --reload --port 8000 &
	streamlit run src/retailgraph/ui/app.py

test:
	pytest tests -q

lint:
	ruff check src tests

ingest:
	$(PYTHON) scripts/create_indexes.py
	$(PYTHON) -m retailgraph.graph.builder
	$(PYTHON) -m retailgraph.graph.vector_store

train:
	modal run training/finetune_qwen.py

benchmark:
	$(PYTHON) scripts/run_benchmark.py

docker-up:
	$(COMPOSE) up --build

docker-down:
	$(COMPOSE) down
