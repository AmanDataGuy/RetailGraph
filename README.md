<div align="center">

# RetailGraph

**Multimodal Knowledge Graph Platform for Enterprise Product Catalogs**

Retail · Grocery · Healthcare · B2B Procurement · Regulated Industries

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qwen2-VL](https://img.shields.io/badge/Qwen2--VL-7B-purple?style=flat-square)](https://github.com/QwenLM/Qwen2-VL)
[![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-blueviolet?style=flat-square)](https://github.com/unslothai/unsloth)
[![Snorkel](https://img.shields.io/badge/Snorkel-weak_supervision-orange?style=flat-square)](https://snorkel.ai/)
[![Modal](https://img.shields.io/badge/Modal-A100_80GB-black?style=flat-square)](https://modal.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-018bff?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-green?style=flat-square)](https://langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

</div>

---

## Overview

Product catalogs are unstructured by default. A listing like *"Maggi 2-Minute Noodles Masala 70g"* is a flat string to a machine. RetailGraph reads that string alongside the product image, extracts a validated structured entity, and loads it into a typed knowledge graph queryable in plain English.

The pipeline is fully local after fine-tuning, has zero ongoing labeling cost, and is self-improving — each inference round feeds high-confidence extractions back into the training set, measurably raising accuracy with each cycle.

**Why a graph over a vector store?**

A vector search on *"vegan snacks under $5 with no nuts"* returns semantically similar products. A Cypher query on a knowledge graph returns the *correct* products — because `price`, `dietary_tags`, and `allergen_list` are typed properties on real nodes, not dimensions in an embedding space. Hard constraints (price ranges, allergen exclusions, certification requirements) are enforced exactly, not approximated.

---

## Current Status

| Phase | Description | Status |
|---|---|---|
| 1 | EDA, weak supervision, active learning | Complete |
| 2 | Schema design, validation, normalization pipeline | Complete |
| 3 | Training data generation (3,208 seed pairs) | Complete |
| 4 | Qwen2-VL 7B fine-tuning on Modal A100 | Complete |
| 5 | Model evaluation on 678-example validation set | Complete |
| 6 | Self-training loop — batched inference on 20k products | In progress |
| 7 | Neo4j knowledge graph construction | Planned |
| 8 | LangGraph natural language query agent | Planned |
| 9 | FastAPI + Apache Kafka streaming ingestion | Planned |
| 10 | Streamlit frontend + MLflow experiment tracking | Planned |

---

## Evaluation Results (Phase 5 — 678 validation examples)

Fine-tuned Qwen2-VL 7B, QLoRA rank 16, trained on Modal A100 80GB. Final training loss: 0.02.

| Field | Accuracy | Status |
|---|---|---|
| brand | 100.0% | Pass |
| pack_size | 100.0% | Pass |
| price | 100.0% | Pass |
| allergen_list | 91.6% | Pass |
| quantity_unit | 92.0% | Pass |
| quantity_value | 88.3% | Below target |
| dietary_tags | 83.3% | Below target |
| category | 82.0% | Below target |
| packaging_type | 100.0% | Pass |
| packaging_color | 100.0% | Pass |
| has_brand_logo | 100.0% | Pass |
| **Overall** | **94.3%** | |

`category`, `dietary_tags`, and `quantity_value` are the targets for Round 2 retraining after self-training on 20k products. Self-training is expected to push overall accuracy past 96%.

---

## Architecture

```
Raw Catalog CSV + Product Images (75,000 Amazon US grocery products)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1 — Zero-Label Data Preparation                  │
│                                                         │
│  31 labeling functions (keyword, regex, Hinglish,       │
│  price signal, image URL signal)                        │
│       │                                                 │
│       ▼                                                 │
│  Snorkel LabelModel — learns LF accuracy +              │
│  correlation without ground truth labels                │
│       │                                                 │
│  prob >= 0.85 → auto_accept                             │
│  prob  0.50–0.85 → active learning review queue         │
│  prob <  0.50 → auto_reject                             │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2 — Schema + Validation Pipeline                 │
│                                                         │
│  ProductEntity (Pydantic v2)                            │
│    item_name, brand, category, quantity_value,          │
│    quantity_unit, pack_size, price, dietary_tags,       │
│    allergen_list, packaging_type, packaging_color,      │
│    has_brand_logo, extraction_confidence                │
│                                                         │
│  Normalizer — RapidFuzz fuzzy matching against          │
│  YAML domain adapters (brand aliases, unit aliases,     │
│  dietary tag aliases)                                   │
│                                                         │
│  Validator — Pydantic validation + 3-step retry loop   │
│  (error message fed back to model on each retry)        │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3 — Training Data Generation                     │
│                                                         │
│  2,228 pairs  GPT-4o-mini text extraction    ~$0.35     │
│    500 pairs  GPT-4o vision (image + text)   ~$2.00     │
│    600 pairs  Python templates (synthetic)   $0.00      │
│  ─────────────────────────────────────────────────      │
│  3,208 total  80/20 train/val split                     │
│                                                         │
│  Proportional sampling across 11 categories            │
│  (weak_labels.csv from Phase 1)                         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 4 — Fine-tuning                                  │
│                                                         │
│  Base model:   Qwen/Qwen2-VL-7B-Instruct               │
│  Method:       QLoRA via Unsloth (4-bit quantization)  │
│  Hardware:     Modal A100 80GB                          │
│  Duration:     ~1hr 45min, 303 steps                    │
│  LoRA rank:    16 | alpha: 32 | dropout: 0.05          │
│  Trainable:    40.4M / 8.33B params (0.48%)            │
│  Final loss:   0.02                                     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 5 — Evaluation                                   │
│                                                         │
│  678 validation examples, 8.27s avg latency            │
│  94.3% overall accuracy                                 │
│  0 parse errors                                         │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 6 — Self-Training Loop (in progress)             │
│                                                         │
│  Batched inference (batch_size=8) on 20k products      │
│                                                         │
│  Confidence scorer (pure Python, 4 components):        │
│    +0.30  JSON validity                                 │
│    +0.30  Required fields + controlled vocab check     │
│    +0.20  Field completeness                            │
│    +0.20  Weak supervision category agreement          │
│                                                         │
│  conf > 0.85 → high_conf.jsonl → Round 2 retraining   │
│  conf 0.60–0.85 → review.jsonl → spot check            │
│  conf < 0.60 → failed.jsonl → discard                  │
│                                                         │
│  Round 2: 3,208 seed + ~14-16k pseudo-labels           │
│  Expected: category 82% → 90%+, overall 94% → 96%+    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 7 — Knowledge Graph (planned)                    │
│                                                         │
│  ~20k clean ProductEntity records → Neo4j AuraDB       │
│                                                         │
│  Nodes:   Product, Brand, Category,                     │
│           Ingredient, VisualAttribute                   │
│                                                         │
│  Edges:   MANUFACTURED_BY, BELONGS_TO,                 │
│           CONTAINS, HAS_VISUAL, DUPLICATE_OF           │
│                                                         │
│  Deduplication: exact SKU → RapidFuzz fuzzy →          │
│  embedding cosine similarity                            │
│  Duplicates linked via DUPLICATE_OF, not deleted       │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 8 — Natural Language Query Agent (planned)       │
│                                                         │
│  LangGraph typed state machine                          │
│                                                         │
│  User query                                             │
│    → Intent extractor                                   │
│    → Filter builder (price, tags, allergens, brand)    │
│    → Schema-aware Text-to-Cypher generator             │
│    → Neo4j execution                                    │
│    → Answer formatter + raw Cypher audit log           │
│                                                         │
│  Every query returns the generated Cypher alongside    │
│  the answer — fully transparent and auditable          │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 9–10 — API + Frontend (planned)                  │
│                                                         │
│  FastAPI — REST API over the graph                      │
│  Apache Kafka — streaming ingestion pipeline           │
│  Streamlit — search UI, review queue, graph analytics  │
│  MLflow — training curves, Round 1 vs Round 2 charts   │
└─────────────────────────────────────────────────────────┘
```

---

## Knowledge Graph Schema (Phase 7)

```
(Product)
  ├── id: UUID
  ├── item_name: String
  ├── price: Float
  ├── quantity_value: Float
  ├── quantity_unit: String
  ├── pack_size: Integer
  ├── dietary_tags: List[String]
  ├── allergen_list: List[String]
  ├── extraction_confidence: Float
  └── quality_score: Integer (0–100)

(Brand)           ← Product -[:MANUFACTURED_BY]→ Brand
  ├── name: String
  └── canonical_name: String

(Category)        ← Product -[:BELONGS_TO]→ Category
  ├── name: String
  └── domain: String

(Ingredient)      ← Product -[:CONTAINS]→ Ingredient
  ├── name: String
  ├── canonical_name: String
  └── is_allergen: Boolean

(VisualAttribute) ← Product -[:HAS_VISUAL]→ VisualAttribute
  ├── packaging_type: String
  ├── packaging_color: String
  └── has_brand_logo: Boolean
```

---

## Technical Decisions

**Why Qwen2-VL over a text-only model?**
Visual attributes (packaging type, color, logo presence) exist only in the product image — they cannot be inferred from text. Qwen2-VL reads both modalities in a single forward pass. The 500 visual training pairs teach it to populate `packaging_type`, `packaging_color`, and `has_brand_logo` from images; the 2,708 text pairs teach the core extraction task.

**Why weak supervision over manual labeling?**
75,000 products at 30 seconds each is 625 hours of human work. Snorkel's LabelModel combines 31 noisy labeling functions — keyword rules, regex patterns, Hinglish brand signals, price heuristics — into probabilistic labels without any ground truth. Products above 0.85 confidence are auto-accepted; the rest go to an active learning queue ranked by entropy, LF disagreement, and category rarity.

**Why QLoRA over full fine-tuning?**
Qwen2-VL 7B in float16 requires ~14GB VRAM. Full fine-tuning at batch size 4 requires ~56GB. QLoRA (4-bit quantization, LoRA rank 16) reduces trainable parameters to 0.48% of the model and fits on an A100 80GB with headroom for gradient accumulation steps of 8.

**Why a knowledge graph over a vector store?**
Vector similarity cannot enforce hard constraints. A query like *"vegan, under $5, nut-free"* requires all three conditions to be simultaneously true — not approximately similar. Cypher executes this as a typed property filter with exact semantics. The graph also captures relationships (brand ownership, ingredient sharing) that embeddings flatten away.

**Why self-training?**
The 3,208 seed pairs cover the distribution well but not exhaustively. Running the fine-tuned model on 20,000 unseen products and collecting high-confidence extractions as pseudo-labels is cheaper than generating more GPT-4o-mini pairs and produces examples from the actual target distribution. Round 2 trains on ~17-19k pairs — a 5-6x increase in training data at near-zero marginal cost.

---

## Project Structure

```
RetailGraph/
├── data/
│   ├── raw/                        # train.csv — 75k Amazon US grocery products
│   ├── images/train/               # product images (gitignored)
│   ├── extracted/
│   │   ├── weak_labels.csv         # Snorkel output — category + dietary probs
│   │   └── review_queue.csv        # active learning — top 50 uncertain products
│   └── training/
│       ├── train.jsonl             # 3,208 training pairs
│       ├── val.jsonl               # 678 validation pairs
│       └── visual_pairs.jsonl      # 500 image+text pairs
│
├── src/
│   ├── extraction/
│   │   ├── schemas.py              # ProductEntity Pydantic v2 model
│   │   ├── normalizer.py           # RapidFuzz fuzzy matching + YAML adapters
│   │   ├── validator.py            # Pydantic validation + 3-step retry loop
│   │   ├── prompt_templates.py     # system prompt, few-shot examples, retry builder
│   │   └── weak_supervision/
│   │       ├── labeling_functions.py   # 31 LFs across dietary tags + categories
│   │       └── label_model.py          # Snorkel pipeline + threshold routing
│   │
│   ├── graph/                      # Neo4j — Phase 7 (planned)
│   ├── agent/                      # LangGraph query agent — Phase 8 (planned)
│   ├── api/                        # FastAPI — Phase 9 (planned)
│   └── streaming/                  # Kafka producer/consumer — Phase 9 (planned)
│
├── training/
│   ├── configs/qwen2vl_lora.yaml   # LoRA rank, alpha, target modules, scheduler
│   ├── generate_pairs.py           # GPT-4o-mini seed pair generation
│   ├── generate_visual_pairs.py    # GPT-4o vision pair generation
│   ├── generate_synthetic.py       # Python template synthetic pairs
│   ├── finetune_qwen.py            # Unsloth SFTTrainer — runs on Modal A100
│   └── evaluate.py                 # per-field accuracy report — runs on Modal A100
│
├── src/extraction/extractor.py     # Phase 6 — batched inference + confidence scorer
├── notebooks/                      # EDA, schema design, prompt ablations
├── mlruns/                         # MLflow training logs (local)
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Vision-Language Model | Qwen2-VL 7B (Alibaba) |
| Fine-tuning | Unsloth + QLoRA, 4-bit quantization |
| Training infrastructure | Modal A100 80GB |
| Weak supervision | Snorkel LabelModel |
| Schema validation | Pydantic v2 |
| Fuzzy normalization | RapidFuzz |
| Knowledge graph | Neo4j AuraDB |
| Query agent | LangGraph typed state machine |
| Streaming | Apache Kafka |
| API | FastAPI |
| Frontend | Streamlit |
| Experiment tracking | MLflow |

---

## Dataset

75,000 Amazon US grocery product listings with product images. Text fields include raw catalog descriptions, item names, and prices. Images are product packaging photographs.

Dataset: [huggingface.co/datasets/amanDS5153/retailgraph-products](https://huggingface.co/datasets/amanDS5153/retailgraph-products)

---

## Author

[GitHub](https://github.com/AmanDataGuy/RetailGraph)