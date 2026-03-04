.PHONY: run test lint ingest train benchmark docker-up reset-graph review-queue help

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON     := python
STREAMLIT  := streamlit
PYTEST     := pytest
COMPOSE    := docker compose

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  RetailGraph — available commands"
	@echo ""
	@echo "  make run            Start FastAPI + Streamlit"
	@echo "  make test           Run full pytest suite"
	@echo "  make lint           Run Ruff + Black checks"
	@echo "  make ingest         Extract → validate → load into Neo4j"
	@echo "  make train          Fine-tune Qwen2-VL (requires H100)"
	@echo "  make benchmark      Run 50 test queries against ground truth"
	@echo "  make docker-up      Spin up full stack (Neo4j, Kafka, Redis, MLflow)"
	@echo "  make reset-graph    ⚠️  Wipe Neo4j and rebuild indexes (dev only)"
	@echo "  make review-queue   Open Streamlit human review UI"
	@echo ""

# ── Development ───────────────────────────────────────────────────────────────
run:
	@echo "▶  Starting FastAPI..."
	$(PYTHON) main.py &
	@echo "▶  Starting Streamlit..."
	$(STREAMLIT) run app.py

test:
	@echo "▶  Running tests..."
	$(PYTEST) tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	@echo "▶  Running Ruff..."
	ruff check .
	@echo "▶  Running Black..."
	black --check .

# ── Pipeline ──────────────────────────────────────────────────────────────────
ingest:
	@echo "▶  Running extraction..."
	$(PYTHON) scripts/run_extraction.py --input data/raw/train.csv
	@echo "▶  Building graph..."
	$(PYTHON) scripts/build_graph.py --input data/extracted/entities.jsonl

train:
	@echo "▶  Starting fine-tuning (ensure H100 is available)..."
	$(PYTHON) training/finetune_qwen.py

benchmark:
	@echo "▶  Running benchmark suite..."
	$(PYTHON) evaluation/benchmark.py

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	@echo "▶  Spinning up full stack..."
	$(COMPOSE) up --build

docker-down:
	$(COMPOSE) down

# ── Graph ─────────────────────────────────────────────────────────────────────
reset-graph:
	@echo "⚠️  This will WIPE the entire Neo4j graph. Are you sure? [y/N]" && read ans && [ $${ans:-N} = y ]
	$(PYTHON) scripts/reset_graph.py
	$(PYTHON) scripts/create_indexes.py
	@echo "✅  Graph wiped and indexes rebuilt."

# ── UI ────────────────────────────────────────────────────────────────────────
review-queue:
	$(STREAMLIT) run app_pages/review_queue.py
