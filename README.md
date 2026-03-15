# RetailGraph

**Multimodal entity extraction and knowledge graph platform for large-scale grocery product catalogs.**

Reads a raw product listing + image → extracts a validated structured entity → loads it into a typed knowledge graph queryable in plain English via a LangGraph agent.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qwen2-VL](https://img.shields.io/badge/Qwen2--VL-7B-purple?style=flat-square)](https://github.com/QwenLM/Qwen2-VL)
[![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-blueviolet?style=flat-square)](https://github.com/unslothai/unsloth)
[![Modal](https://img.shields.io/badge/Modal-A100_80GB-black?style=flat-square)](https://modal.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-018bff?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-dc244c?style=flat-square)](https://qdrant.tech/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-green?style=flat-square)](https://langchain.com/langgraph)

---

## Results

Evaluated on 678 held-out examples. Zero parse errors across all runs.

| Field | Accuracy |
|---|---|
| brand | 100.0% |
| pack_size | 100.0% |
| price | 100.0% |
| allergen_list | 91.5% |
| quantity_unit | 92.6% |
| quantity_value | 88.2% |
| dietary_tags | 85.7% |
| category | 78.3% |
| **Overall** | **94.2%** |

Training: 5,369 verified pairs across 3 rounds. Round 2 self-training failed (category collapsed 82% → 66% due to pseudo-label noise). Fixed with GPT-4o-mini verification — 487 category corrections, 298 tag corrections, $0.15 total cost.

---

## Pipeline

```
  ┌──────────────────────────────────────────────────────────────────┐
  │         75,000 Amazon US grocery products + product images       │
  └─────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  WEAK SUPERVISION                                                │
  │  31 Snorkel labeling functions · keyword · regex · price signal  │
  │  LabelModel learns LF accuracy without ground truth             │
  │  prob ≥ 0.85 → accept  ·  0.50–0.85 → review  ·  < 0.50 → skip │
  └─────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  TRAINING DATA                                                   │
  │  3,208 GPT-4o-mini pairs · 500 visual pairs (image + text)      │
  │  + 2,161 pseudo-labels verified by GPT-4o-mini  =  5,369 total  │
  │  487 category fixes · 298 dietary tag fixes · $0.15 total       │
  └─────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  FINE-TUNING                                                     │
  │  Qwen2-VL 7B · QLoRA rank 16 · Unsloth · Modal A100 80GB        │
  │  40.4M / 8.33B trainable params (0.48%) · ~2.9 hrs · ~$29       │
  │  94.2% overall · 0 parse errors · 8.7s avg latency              │
  └─────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  BATCH EXTRACTION + CONFIDENCE SCORING                          │
  │  JSON validity · Pydantic schema · field completeness           │
  │  weak supervision agreement  →  3 confidence buckets           │
  │  2,161 high-conf extractions from 3,920 processed products      │
  └──────────────────┬───────────────────────┬──────────────────────┘
                     │                       │
                     ▼                       ▼
  ┌────────────────────────┐    ┌────────────────────────────────────┐
  │  NEO4J AURADB          │    │  QDRANT CLOUD                      │
  │                        │    │                                    │
  │  2,160 Product nodes   │    │  2,161 vectors · dim 384           │
  │  1,041 Brand nodes     │    │  all-MiniLM-L6-v2 embeddings       │
  │  17 Category nodes     │    │  Filtered vector search            │
  │  6,248 relationships   │    │  Payload: price · tags · category  │
  │                        │    │                                    │
  │  BELONGS_TO · MADE_BY  │    │  Semantic similarity + metadata    │
  │  HAS_TAG · CONTAINS_ALLERGEN│  filtering in single query         │
  └──────────┬─────────────┘    └───────────────┬────────────────────┘
             │                                  │
             └────────────────┬─────────────────┘
                              │
                              ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  GRAPHRAG HYBRID SEARCH                                          │
  │  Qdrant semantic search → top-50 candidates                     │
  │  Neo4j constraint verification → price · allergens · tags       │
  │  hybrid score = 0.7 × semantic + 0.3 × quality                  │
  └─────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │  LANGGRAPH AGENT  (in progress)                                  │
  │  Pydantic AI structured outputs · Redis semantic cache          │
  │  NL query → intent → Cypher / GraphRAG → formatted answer       │
  │  LangSmith tracing · every query logged and auditable           │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Why This Stack

**GraphRAG over pure vector search** — Qdrant finds semantically similar candidates. Neo4j verifies hard constraints: price limits, allergen exclusions, dietary certifications. A query like *"vegan, nut-free, under $5"* needs all three simultaneously true — vectors approximate, graphs enforce.

**Weak supervision over manual labeling** — 75,000 products at 30 seconds each is 625 hours of labeling. Snorkel's LabelModel combines 31 noisy labeling functions into calibrated probabilistic labels with no ground truth required.

**QLoRA over full fine-tuning** — Full fine-tuning Qwen2-VL 7B requires ~56GB VRAM. QLoRA at rank 16 fits on a single A100 80GB, trains 0.48% of parameters, and matches full fine-tuning accuracy for domain adaptation tasks.

**Verification over raw self-training** — Self-training with an 82% accurate teacher amplifies errors. GPT-4o-mini verification of 2,161 pseudo-labels corrected 22.5% of categories and 13.8% of dietary tags before retraining — at $0.15 total cost. This is the weak supervision + strong verification pattern used at scale.

---

## What Didn't Work (and What I Learned)

Round 2 self-training failed. Using the model's own pseudo-labels as training data caused category accuracy to collapse from 82% to 66.2% — the model learned to repeat its own mistakes more confidently. The fix: verify pseudo-labels with a stronger model before retraining, not after. Data quality beats data quantity every time.

---

## Status

| Phase | Description | Status |
|---|---|---|
| 1–2 | Weak supervision · schema · normalization pipeline | ✅ Complete |
| 3 | Training data generation — 5,369 verified pairs | ✅ Complete |
| 4–5 | Qwen2-VL fine-tuning + evaluation — 94.2% | ✅ Complete |
| 6 | Batch extraction + GPT-4o-mini verification loop | ✅ Complete |
| 7 | Neo4j + Qdrant + GraphRAG hybrid search | ✅ Complete |
| 8 | LLM-as-Judge evaluation + LangSmith observability | 🔄 Next |
| 9 | Kafka real-time streaming pipeline | Planned |
| 10 | LangGraph agent + Pydantic AI + Redis semantic cache | Planned |
| 11 | FastAPI + MCP server endpoints | Planned |
| 12 | Streamlit frontend — 4 pages | Planned |
| 13 | GraphRAG vs VectorRAG benchmark — 50 queries | Planned |

---

[huggingface.co/datasets/amanDS5153/retailgraph-products](https://huggingface.co/datasets/amanDS5153/retailgraph-products) · [github.com/AmanDataGuy/RetailGraph](https://github.com/AmanDataGuy/RetailGraph)