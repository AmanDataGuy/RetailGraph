<div align="center">

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

</div>

---

## Problem Statement

Product catalogs are unstructured by default. A listing like *"Maggi 2-Minute Noodles Masala 70g"* is a flat string to a machine. Three problems compound at scale:

- **No structure** — brand, category, allergens, dietary certifications are buried in raw text
- **No semantics** — keyword search can't answer *"vegan, nut-free snacks under $5"* reliably
- **No relationships** — which brands share allergens? Which categories overlap in dietary tags?

RetailGraph solves all three: fine-tuned VLM extracts structured entities from text and images, a knowledge graph stores the relationships, and a GraphRAG agent answers natural language queries with hard constraint enforcement — not approximations.

---

## System Architecture

![RetailGraph Pipeline](assets/pipeline.svg)

---

## Performance

### Extraction Accuracy — 678 held-out validation examples, zero parse errors

| Field | Accuracy | Notes |
|---|---|---|
| brand | 100.0% | ✅ |
| pack_size | 100.0% | ✅ |
| price | 100.0% | ✅ |
| allergen_list | 91.5% | ✅ |
| quantity_unit | 92.6% | ✅ |
| quantity_value | 88.2% | — |
| dietary_tags | 85.7% | — |
| category | 78.3% | — |
| **Overall** | **94.2%** | |

### Training Rounds Comparison

| Round | Training Data | Overall | Category | Dietary Tags | Notes |
|---|---|---|---|---|---|
| Round 1 | 3,208 clean pairs | 94.3% | 82.0% | 83.3% | Baseline |
| Round 2 | + 2,161 raw pseudo-labels | 91.8% | 66.2% | 81.0% | ❌ Self-training failed |
| Round 3 | + 2,161 verified pseudo-labels | **94.2%** | 78.3% | **85.7%** | ✅ GPT-4o-mini fix |

### Knowledge Graph Stats

| Metric | Value |
|---|---|
| Product nodes | 2,160 |
| Brand nodes | 1,041 |
| Category nodes | 17 |
| DietaryTag nodes | 17 |
| Allergen nodes | 47 |
| Total relationships | 6,248 |
| Vectors in Qdrant | 2,161 · dim 384 |
| Neo4j query latency | < 50ms |
| Qdrant search latency | < 100ms |

---

## Technical Deep Dive

### Phase 1–2 · Weak Supervision + Schema

**Challenge:** 75,000 products with no labels. Manual labeling at 30 seconds each = 625 hours.

**Solution:** 31 Snorkel labeling functions covering dietary tags (organic, kosher, gluten-free, vegan, keto) and categories (11 types including Hinglish signals: *masala, atta, dal, chawal*). Snorkel's LabelModel learns each LF's accuracy and correlation structure without ground truth, producing calibrated probabilistic labels.

**Results:** Products above 0.85 confidence auto-accepted. 50 most uncertain products routed to active learning queue ranked by entropy × disagreement × category rarity.

---

### Phase 3 · Training Data Generation

**Challenge:** Get high-quality labeled examples without paying for 75k manual annotations.

**Solution:** Proportional sampling across all 11 categories → GPT-4o-mini generates structured JSON extractions for 3,208 products. 500 visual pairs generated with GPT-4o (image + text). 500 synthetic Python template pairs for underrepresented categories.

**Cost:** ~$3.50 total for seed dataset generation.

---

### Phase 4–5 · Fine-tuning + Evaluation

**Challenge:** Full fine-tuning Qwen2-VL 7B requires ~56GB VRAM and days of training.

**Solution:** QLoRA rank 16 via Unsloth — freezes 99.52% of parameters, trains 40.4M out of 8.33B. Fits on a single A100 80GB with gradient accumulation of 8. Three training rounds with measurable per-round improvement.

| Config | Value |
|---|---|
| Base model | Qwen2-VL 7B (Alibaba) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Batch size | 4 per device |
| Gradient accumulation | 8 (effective batch 32) |
| Epochs | 3 |
| Final training loss | 0.02 |
| Training cost | ~$29/run |

---

### Phase 6 · Batch Extraction + Confidence Scoring

**Challenge:** Model inference on 75k products without human review of every output.

**Solution:** 4-component confidence scorer (JSON validity +0.30, Pydantic validation +0.30, field completeness +0.20, weak supervision agreement +0.20). Three output buckets: high_conf (> 0.85), review (0.60–0.85), failed (< 0.60).

**Self-training failure and fix:** Round 2 used raw pseudo-labels from an 82% accurate teacher — category accuracy collapsed to 66.2%. Root cause: 18% wrong labels amplified through retraining. Fix: GPT-4o-mini verification pass on all 2,161 high-conf extractions corrected 487 categories (22.5%) and 298 dietary tags (13.8%) at $0.15 total. Round 3 restored 94.2%.

---

### Phase 7 · Neo4j + Qdrant + GraphRAG

**Challenge:** Enable both exact constraint queries and semantic similarity search over the same product catalog.

**Solution:** Dual-database architecture — Neo4j for structured facts, Qdrant for semantic vectors.

**Neo4j** stores typed nodes (Product, Brand, Category, DietaryTag, Allergen) with explicit relationships. Cypher queries enforce hard constraints: `price < $5 AND HAS_TAG vegan AND NOT CONTAINS_ALLERGEN nuts` — exact, not approximate.

**Qdrant** stores 384-dim embeddings of catalog text via `all-MiniLM-L6-v2`. Supports filtered vector search — semantic similarity AND metadata constraints in a single query.

**GraphRAG hybrid search** combines both:
- Semantic query → Qdrant retrieves top-50 candidates by cosine similarity
- Neo4j verifies hard constraints on those 50 candidates
- Hybrid score = 0.7 × semantic + 0.3 × quality
- Result: semantically relevant AND constraint-satisfying products

```
"vegan protein bar under $10 with no nuts"
         │
         ▼
   Qdrant: finds semantically similar products (protein bars, snacks)
         │
         ▼
   Neo4j: filters → HAS_TAG vegan, price ≤ 10, NOT CONTAINS_ALLERGEN nuts
         │
         ▼
   Ranked results: correct AND relevant
```

---

## Why This Stack

**GraphRAG over pure vector search** — Vectors approximate. Graphs enforce. *"Vegan, nut-free, under $5"* requires all three simultaneously true — a cosine similarity score can't guarantee that. Neo4j Cypher can.

**Weak supervision over manual labeling** — Snorkel's LabelModel combines 31 noisy heuristics into calibrated labels. Zero human annotation budget, production-quality training signal.

**QLoRA over full fine-tuning** — 0.48% of parameters trained. Same accuracy for domain adaptation tasks. $29/run instead of $200+/run.

**Verification over raw self-training** — Self-training amplifies a model's existing errors when the teacher accuracy is below ~95%. GPT-4o-mini verification at $0.15 is cheaper and more reliable than another full training run.

**LLM-as-Judge over field accuracy only** — Field accuracy measures exact string match. LLM-as-Judge measures semantic correctness — the pattern used by OpenAI and Anthropic for their own model evaluations.

---

## What Didn't Work

Round 2 self-training failed. Using the model's own outputs as training data caused category accuracy to collapse from 82% to 66.2% — the model reinforced its own errors. This is a known self-training failure mode: when teacher accuracy is below ~95% on the target field, pseudo-label noise exceeds the signal from more data.

The fix was verification before retraining, not after: route all pseudo-labels through a stronger model (GPT-4o-mini) to correct only the uncertain fields (category, dietary_tags). Keep the fields the weak model got right (brand 100%, price 100%, pack_size 100%). Data quality beats data quantity.

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

<div align="center">

*"Vectors guess. Graphs know. GraphRAG does both."*

[Dataset](https://huggingface.co/datasets/amanDS5153/retailgraph-products) · [GitHub](https://github.com/AmanDataGuy/RetailGraph)

</div>