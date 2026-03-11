# RetailGraph

**Multimodal entity extraction and knowledge graph platform for large-scale grocery product catalogs.**

Reads a raw product listing + image → extracts a validated structured entity → loads it into a typed knowledge graph queryable in plain English.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qwen2-VL](https://img.shields.io/badge/Qwen2--VL-7B-purple?style=flat-square)](https://github.com/QwenLM/Qwen2-VL)
[![Modal](https://img.shields.io/badge/Modal-A100_80GB-black?style=flat-square)](https://modal.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-018bff?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)

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
| dietary_tags | 85.7% |
| quantity_value | 88.2% |
| category | 78.3% |
| **Overall** | **94.2%** |

Training data: 5,369 verified pairs (3,208 GPT-4o-mini seed + 2,161 pseudo-labels verified by GPT-4o-mini). Three rounds of training, each measurably improving quality.

---

## Pipeline

```
75,000 Amazon US grocery products + product images
              │
              ▼
   ┌─────────────────────┐
   │  Weak Supervision   │   31 labeling functions (keyword, regex, price signal)
   │  Snorkel LabelModel │   learns LF accuracy without ground truth
   └──────────┬──────────┘
              │  prob ≥ 0.85 → auto-accept │ 0.50–0.85 → review │ < 0.50 → reject
              ▼
   ┌─────────────────────┐
   │  Training Data      │   3,208 GPT-4o-mini pairs + 500 visual pairs
   │  Generation         │   proportional sampling across 11 categories
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Fine-tuning        │   Qwen2-VL 7B · QLoRA rank 16 · Modal A100 80GB
   │  Unsloth + QLoRA    │   40.4M / 8.33B trainable params (0.48%)
   └──────────┬──────────┘
              │  94.2% overall · 0 parse errors · 8.7s avg latency
              ▼
   ┌─────────────────────┐
   │  Batch Extraction   │   Confidence scorer: JSON validity + Pydantic
   │  + Verification     │   + field completeness + weak supervision agreement
   └──────────┬──────────┘
              │  2,161 high-conf extractions → GPT-4o-mini verification
              │  487 category fixes · 298 dietary tag fixes · $0.15 total
              ▼
   ┌─────────────────────┐
   │  Knowledge Graph    │   Neo4j AuraDB · Cypher queries
   │  (Phase 7)          │   Product → Brand, Category, Ingredient, VisualAttribute
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  NL Query Agent     │   LangGraph state machine · Text-to-Cypher
   │  (Phase 8)          │   intent → filters → Cypher → formatted answer
   └─────────────────────┘
```

---

## Why This Stack

**Graph over vector store** — A query like *"vegan, nut-free, under $5"* needs all three conditions simultaneously true, not approximately similar. Cypher enforces hard constraints exactly. Vectors approximate.

**Weak supervision over manual labeling** — 75,000 products at 30 seconds each is 625 hours of labeling. Snorkel's LabelModel combines 31 noisy heuristics into calibrated probabilistic labels with no ground truth required.

**QLoRA over full fine-tuning** — Full fine-tuning Qwen2-VL 7B needs ~56GB VRAM. QLoRA at rank 16 fits on one A100 80GB, trains 0.48% of parameters, and matches full fine-tuning accuracy for domain adaptation tasks.

**Verification over raw self-training** — Using the model's own pseudo-labels as training data amplifies its errors. Running GPT-4o-mini verification on the 2,161 high-confidence extractions corrected 22.5% of categories and 13.8% of dietary tags before retraining — at $0.15 total cost.

---

## Status

| Phase | Description | Status |
|---|---|---|
| 1–2 | Weak supervision, schema, normalization pipeline | ✅ Complete |
| 3 | Training data generation (5,369 verified pairs) | ✅ Complete |
| 4–5 | Qwen2-VL fine-tuning + evaluation (94.2%) | ✅ Complete |
| 6 | Batch extraction + GPT-4o-mini verification loop | ✅ Complete |
| 7 | Neo4j knowledge graph construction | 🔄 In progress |
| 8 | LangGraph natural language query agent | Planned |
| 9–10 | FastAPI + Kafka streaming + Streamlit frontend | Planned |

---

## Dataset

75,000 Amazon US grocery product listings with product images.

[huggingface.co/datasets/amanDS5153/retailgraph-products](https://huggingface.co/datasets/amanDS5153/retailgraph-products)

---

## Author

[GitHub](https://github.com/AmanDataGuy/RetailGraph)