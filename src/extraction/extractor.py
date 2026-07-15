# =============================================================================
# RetailGraph — Full Catalog Extraction Pipeline
# =============================================================================
#
# HOW TO RUN:
#
#   Quick test (100 products):
#     modal run src/extraction/extractor.py --target 100
#
#   Full run (20,000 products):
#     modal run src/extraction/extractor.py --target 20000
#
#   Resume interrupted run:
#     modal run src/extraction/extractor.py --target 20000 --resume
#
#   Download results:
#     modal volume get retailgraph-models extraction_results ./extraction/
#
# OUTPUT FILES (inside Modal Volume /models/extraction_results/):
#   high_conf.jsonl   — confidence >85%, used for Round 2 retraining
#   review.jsonl      — confidence 60-85%, spot check manually
#   failed.jsonl      — confidence <60% or invalid JSON, discard
#   summary.json      — counts, stats, timing
#   checkpoint.json   — resume state
#
# =============================================================================

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import modal

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("retailgraph.extractor")


# =============================================================================
# SECTION 1 — MODAL SETUP
# =============================================================================

app = modal.App("retailgraph-extract")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers>=4.45.0,<5.0.0",
        "accelerate",
        "Pillow>=10.0.0",
        "pandas",
        "qwen-vl-utils",
    )
)

data_volume  = modal.Volume.from_name("retailgraph-data",   create_if_missing=False)
model_volume = modal.Volume.from_name("retailgraph-models", create_if_missing=False)

REMOTE_DATA_DIR    = Path("/data")
REMOTE_MODELS_DIR  = Path("/models")
REMOTE_MODEL_DIR   = REMOTE_MODELS_DIR / "qwen2vl_retailgraph_v1"
REMOTE_TRAIN_CSV   = REMOTE_DATA_DIR   / "raw"       / "train.csv"
REMOTE_WEAK_LABELS = REMOTE_DATA_DIR   / "extracted" / "weak_labels.csv"
REMOTE_RESULTS_DIR = REMOTE_MODELS_DIR / "extraction_results"
REMOTE_CHECKPOINT  = REMOTE_RESULTS_DIR / "checkpoint.json"


# =============================================================================
# SECTION 2 — CONFIGURATION
# =============================================================================

HIGH_CONF_THRESHOLD = 0.85
REVIEW_THRESHOLD    = 0.60

BATCH_SIZE       = 4     # safe for A100 80GB with transformers float16
CHECKPOINT_EVERY = 500
MAX_NEW_TOKENS   = 300


# =============================================================================
# SECTION 3 — ROPE SCALING PATCH
#
# Unsloth strips rope_scaling from config.json on save.
# We patch it at runtime via model.named_modules() after loading.
# =============================================================================

QWEN2VL_ROPE_SCALING = {
    "type": "mrope",
    "mrope_section": [16, 24, 24],
}


def patch_rope_scaling(model) -> None:
    """
    Patches rope_scaling on all attention layers after model load.
    Required because Unsloth strips this from the saved config.
    """
    import torch
    patched = 0
    for name, module in model.named_modules():
        if hasattr(module, "rope_scaling"):
            module.rope_scaling = QWEN2VL_ROPE_SCALING
            patched += 1
    if patched:
        log.warning(f"Patched rope_scaling on {patched} attention layer(s).")
    else:
        log.info("No rope_scaling to patch.")


# =============================================================================
# SECTION 4 — CONFIDENCE SCORER
# =============================================================================
#
# Pure Python — NO imports from src/. The src/ directory is not mounted
# in the Modal container. All logic is self-contained here.

VALID_CATEGORIES = {
    "Coffee & Tea", "Breakfast & Cereal", "Meat & Seafood",
    "Soups & Canned Goods", "Pasta & Noodles", "Bread & Bakery",
    "Protein Bars & Snacks", "Supplements & Health",
    "Grains, Beans & Legumes", "Oils & Vinegars", "Nuts & Seeds",
    "Personal Care & Beauty", "Spices & Seasonings",
    "Condiments & Sauces", "Baking & Cooking", "Snacks & Candy",
    "Beverages", "Non-Food", "Unknown",
}

VALID_UNITS = {"oz", "fl oz", "lb", "ct", "g", "kg", "ml", "L"}

REQUIRED_FIELDS = ["item_name", "price", "quantity_value", "quantity_unit", "category"]

KEY_FIELDS = [
    "item_name", "price", "quantity_value", "quantity_unit",
    "category", "brand", "dietary_tags", "allergen_list",
]


def _is_present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True


def score_extraction(
    prediction: dict,
    sample_id: str,
    weak_label_category: Optional[str] = None,
) -> float:
    score = 0.30  # JSON validity — always awarded if we reach this function

    n_required = sum(1 for f in REQUIRED_FIELDS if _is_present(prediction.get(f)))
    req_ratio  = n_required / len(REQUIRED_FIELDS)

    cat_valid  = prediction.get("category")      in VALID_CATEGORIES
    unit_valid = prediction.get("quantity_unit") in VALID_UNITS
    if cat_valid and unit_valid:
        req_ratio = min(req_ratio * 1.1, 1.0)

    score += 0.30 * req_ratio

    filled       = sum(1 for f in KEY_FIELDS if _is_present(prediction.get(f)))
    completeness = filled / len(KEY_FIELDS)
    score       += 0.20 * completeness

    if weak_label_category is not None:
        pred_cat = str(prediction.get("category", "")).strip().lower()
        weak_cat = str(weak_label_category).strip().lower()
        if pred_cat and pred_cat == weak_cat:
            score += 0.20
        elif pred_cat and (pred_cat in weak_cat or weak_cat in pred_cat):
            score += 0.10
    else:
        score += 0.20 * completeness

    return round(min(score, 1.0), 4)


def bucket(confidence: float) -> str:
    if confidence >= HIGH_CONF_THRESHOLD:
        return "high_conf"
    elif confidence >= REVIEW_THRESHOLD:
        return "review"
    return "failed"


# =============================================================================
# SECTION 5 — SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a product data extraction model for a grocery e-commerce platform.

Your job is to read a raw product catalog listing and extract structured information into a valid JSON object.

## OUTPUT RULES — FOLLOW EXACTLY

1. Return ONLY a valid JSON object. No markdown, no backticks, no explanation, no preamble.
2. Never add fields not in the schema below.
3. Use null for any field you cannot determine.
4. Use ONLY the exact values from the controlled vocabulary lists below.
5. extraction_confidence: your confidence as a float 0.0-1.0. Never output 1.0.

## OUTPUT SCHEMA

{
  "product_id": "string — use the Product ID provided",
  "item_name": "string — clean product name without size/quantity info. Max 120 chars.",
  "brand": "string or null",
  "category": "string — MUST be from allowed categories below, or null",
  "quantity_value": "number or null",
  "quantity_unit": "string — MUST be from canonical units below, or null",
  "pack_size": "integer or null — from Pack of N pattern",
  "price": "number or null",
  "dietary_tags": ["array — only from allowed tags below, empty [] if none"],
  "allergen_list": ["array — lowercase allergen names, empty [] if none"],
  "extraction_confidence": "float 0.0-1.0"
}

## CANONICAL UNITS — use EXACTLY these strings
oz, fl oz, lb, g, kg, ml, L, ct

Unit rules: Ounce/ounces/OZ → oz | Fluid Ounce/fl. oz. → fl oz | Count/count/Ct → ct
Pound/pounds/LB → lb | Gram/grams → g | Milliliter → ml | Liter/liters → L
If unit missing or unclear → ct

## ALLOWED CATEGORIES — use EXACTLY these strings
Beverages, Coffee & Tea, Snacks & Candy, Condiments & Sauces,
Grains, Beans & Legumes, Baking & Cooking, Spices & Seasonings,
Supplements & Health, Nuts & Seeds, Personal Care & Beauty,
Protein Bars & Snacks, Breakfast & Cereal, Meat & Seafood,
Soups & Canned Goods, Pasta & Noodles, Bread & Bakery,
Oils & Vinegars, Non-Food, Unknown

## ALLOWED DIETARY TAGS — only use these exact strings
organic, kosher, gluten-free, non-GMO, vegan, keto, paleo,
dairy-free, sugar-free, nut-free, soy-free, high-protein,
low-calorie, caffeine-free, allergen-free, cruelty-free

## ITEM NAME RULES
- Remove size/quantity: "Smucker's Peanut Butter 16oz" → "Smucker's Natural Peanut Butter"
- Remove pack info: "McCormick Garlic Powder (Pack of 6)" → "McCormick Garlic Powder"
- Always keep the brand name
- Max 120 characters"""


def build_prompt(sample_id: str, catalog_content: str, price: Optional[float]) -> str:
    # Hard cap at 800 chars — prevents model continuing catalog text
    # instead of generating JSON when input is truncated mid-sentence.
    content = catalog_content.strip()[:800]

    if price is not None and "Price:" not in content:
        content = f"{content}\nPrice: {price}"

    return (
        "Extract product information from the following grocery product listing.\n"
        "Return ONLY a valid JSON object matching the schema. "
        "No explanation, no markdown, no backticks.\n\n"
        f"Product ID: {sample_id}\n\n"
        f"Product listing:\n{content}"
    )


# =============================================================================
# SECTION 6 — BATCHED INFERENCE (transformers)
# =============================================================================

def run_batch(
    model,
    processor,
    batch: list[dict],
) -> list[dict]:
    """
    Runs one transformers generation pass over a batch of products.

    Args:
        model:     Qwen2VLForConditionalGeneration on CUDA
        processor: Qwen2VLProcessor with padding_side='left'
        batch:     list of dicts — keys: sample_id, catalog_content, price

    Returns:
        list of dicts — keys: sample_id, prediction (dict), raw_output (str)
    """
    import torch

    # Build chat-formatted messages for each item in the batch
    all_messages = []
    for item in batch:
        all_messages.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(
                item["sample_id"],
                item["catalog_content"],
                item["price"],
            )},
        ])

    # Apply chat template and tokenize the full batch at once
    texts = [
        processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        for msgs in all_messages
    ]

    inputs = processor(
        text=texts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,          # required when do_sample=False to silence warning
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    results = []
    for i, item in enumerate(batch):
        # Slice off the input tokens — only decode the generated part
        new_tokens = output_ids[i][input_len:]
        raw_output = processor.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        # Log non-JSON outputs for debugging
        if not raw_output.startswith("{"):
            log.warning(
                f"RAW OUTPUT sample_id={item['sample_id']}: "
                f"{repr(raw_output[:300])}"
            )

        # Parse JSON — strip markdown fences if present
        prediction = {}
        try:
            clean = raw_output
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            prediction = json.loads(clean.strip())
        except json.JSONDecodeError:
            pass  # stays {} — bucketed as failed

        results.append({
            "sample_id":  item["sample_id"],
            "prediction": prediction,
            "raw_output": raw_output[:500],
        })

    return results


# =============================================================================
# SECTION 7 — CHECKPOINT HELPERS
# =============================================================================

def load_checkpoint() -> set[str]:
    if not REMOTE_CHECKPOINT.exists():
        return set()
    try:
        data      = json.loads(REMOTE_CHECKPOINT.read_text(encoding="utf-8"))
        processed = set(data.get("processed_ids", []))
        log.info(f"Resuming from checkpoint: {len(processed)} already done.")
        return processed
    except Exception as e:
        log.warning(f"Could not read checkpoint ({e}) — starting fresh.")
        return set()


def save_checkpoint(processed_ids: set[str]) -> None:
    REMOTE_CHECKPOINT.write_text(
        json.dumps({
            "processed_ids": list(processed_ids),
            "timestamp":     datetime.now().isoformat(),
        }),
        encoding="utf-8",
    )


def append_result(record: dict, output_file: Path) -> None:
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# SECTION 8 — MAIN EXTRACTION FUNCTION (runs on Modal A100)
# =============================================================================

@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={
        str(REMOTE_DATA_DIR):   data_volume,
        str(REMOTE_MODELS_DIR): model_volume,
    },
    timeout=43200,  # 12 hours
)
def run_extraction(target: int = 20_000, resume: bool = False):
    import torch
    import pandas as pd
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

    # ── Load model ─────────────────────────────────────────────────────────────
    log.info(f"Loading model from {REMOTE_MODEL_DIR}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(REMOTE_MODEL_DIR),
        dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda").eval()

    # Patch rope_scaling — Unsloth strips this on save
    patch_rope_scaling(model)
    log.info("Model loaded on GPU in float16.")

    # ── Load processor ─────────────────────────────────────────────────────────
    processor = Qwen2VLProcessor.from_pretrained(
        str(REMOTE_MODEL_DIR),
        trust_remote_code=True,
    )
    # Left-padding required for decoder-only batch inference
    processor.tokenizer.padding_side = "left"

    # ── Load train.csv ─────────────────────────────────────────────────────────
    if not REMOTE_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"train.csv not found at {REMOTE_TRAIN_CSV}.\n"
            "Upload it first:\n"
            "  modal volume put retailgraph-data data/raw/train.csv /raw/train.csv"
        )

    log.info(f"Loading products from {REMOTE_TRAIN_CSV}...")
    df = pd.read_csv(REMOTE_TRAIN_CSV)
    log.info(f"Total products available: {len(df)}")

    # ── Load weak labels (optional) ────────────────────────────────────────────
    weak_labels: dict = {}
    if REMOTE_WEAK_LABELS.exists():
        wl_df = pd.read_csv(REMOTE_WEAK_LABELS, usecols=["sample_id", "category"])
        weak_labels = dict(
            zip(wl_df["sample_id"].astype(str), wl_df["category"])
        )
        log.info(f"Loaded {len(weak_labels)} weak labels for confidence scoring.")
    else:
        log.warning(
            "weak_labels.csv not found — confidence scorer running without "
            "weak supervision component."
        )

    # ── Setup output files ─────────────────────────────────────────────────────
    REMOTE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    high_conf_file = REMOTE_RESULTS_DIR / "high_conf.jsonl"
    review_file    = REMOTE_RESULTS_DIR / "review.jsonl"
    failed_file    = REMOTE_RESULTS_DIR / "failed.jsonl"

    # ── Handle resume ──────────────────────────────────────────────────────────
    already_done = load_checkpoint() if resume else set()
    if already_done:
        df = df[~df["sample_id"].astype(str).isin(already_done)]
        log.info(
            f"Skipping {len(already_done)} already processed. "
            f"Remaining: {len(df)}"
        )

    remaining = target - len(already_done)
    df = df.head(remaining)
    log.info(
        f"Processing {len(df)} products "
        f"(target={target}, batch_size={BATCH_SIZE})..."
    )

    # ── Tracking ───────────────────────────────────────────────────────────────
    processed_ids = set(already_done)
    counts        = {"high_conf": 0, "review": 0, "failed": 0, "parse_error": 0}
    latencies     = []
    total         = len(df)
    rows          = df.to_dict("records")
    n_done        = 0

    # ── Main loop ──────────────────────────────────────────────────────────────
    for batch_start in range(0, total, BATCH_SIZE):
        batch_rows = rows[batch_start: batch_start + BATCH_SIZE]

        batch_input = []
        for row in batch_rows:
            raw_price = row.get("price")
            price     = float(raw_price) if pd.notna(raw_price) else None
            batch_input.append({
                "sample_id":       str(row["sample_id"]),
                "catalog_content": str(row.get("catalog_content", "")),
                "price":           price,
            })

        t0      = time.time()
        results = run_batch(model, processor, batch_input)
        elapsed = time.time() - t0
        latencies.append(elapsed)

        for item_input, result in zip(batch_input, results):
            sample_id  = result["sample_id"]
            prediction = result["prediction"]

            if not prediction:
                counts["parse_error"] += 1
                counts["failed"]      += 1
                conf = 0.0
                bkt  = "failed"
            else:
                weak_cat = weak_labels.get(sample_id)
                conf     = score_extraction(prediction, sample_id, weak_cat)
                bkt      = bucket(conf)
                counts[bkt] += 1

            record = {
                "sample_id":       sample_id,
                "confidence":      conf,
                "bucket":          bkt,
                "prediction":      prediction,
                "catalog_content": item_input["catalog_content"][:300],
                "price":           item_input["price"],
            }

            if bkt == "high_conf":
                append_result(record, high_conf_file)
            elif bkt == "review":
                append_result(record, review_file)
            else:
                append_result(record, failed_file)

            processed_ids.add(sample_id)

        n_done += len(batch_rows)

        batch_num = (batch_start // BATCH_SIZE) + 1
        if batch_num % 10 == 0 or n_done >= total:
            recent  = latencies[-min(10, len(latencies)):]
            avg_lat = sum(recent) / len(recent)
            log.info(
                f"[{n_done:>6}/{total}] ({100*n_done/total:.1f}%) | "
                f"high={counts['high_conf']} "
                f"review={counts['review']} "
                f"failed={counts['failed']} | "
                f"batch={elapsed:.1f}s avg10={avg_lat:.1f}s"
            )

        if n_done % CHECKPOINT_EVERY < BATCH_SIZE or n_done >= total:
            save_checkpoint(processed_ids)
            _save_summary(counts, latencies, n_done, total)
            model_volume.commit()
            log.info(f"Checkpoint saved at {n_done} products.")

    # ── Final save ─────────────────────────────────────────────────────────────
    _save_summary(counts, latencies, n_done, total, final=True)
    model_volume.commit()

    report = _build_report(counts, latencies, n_done)
    print(report)
    return report


# =============================================================================
# SECTION 9 — SUMMARY AND REPORT
# =============================================================================

def _save_summary(
    counts: dict,
    latencies: list,
    n_done: int,
    total: int,
    final: bool = False,
) -> None:
    total_bucketed = sum(counts[k] for k in ("high_conf", "review", "failed"))
    summary = {
        "timestamp":    datetime.now().isoformat(),
        "final":        final,
        "total_target": total,
        "processed":    n_done,
        "counts":       counts,
        "rates": {
            "high_conf_pct": round(100 * counts["high_conf"] / max(total_bucketed, 1), 1),
            "review_pct":    round(100 * counts["review"]    / max(total_bucketed, 1), 1),
            "failed_pct":    round(100 * counts["failed"]    / max(total_bucketed, 1), 1),
        },
        "avg_batch_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "est_s_per_product":   round(
            (sum(latencies) / len(latencies)) / BATCH_SIZE, 2
        ) if latencies else 0,
    }
    out = REMOTE_RESULTS_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_report(counts: dict, latencies: list, n_done: int) -> str:
    total      = sum(counts[k] for k in ("high_conf", "review", "failed"))
    avg_lat    = sum(latencies) / len(latencies) if latencies else 0
    s_per_prod = avg_lat / BATCH_SIZE

    sep   = "=" * 65
    lines = [
        sep,
        "RETAILGRAPH — EXTRACTION REPORT",
        sep,
        f"Products processed  : {n_done}",
        f"Avg s/product       : {s_per_prod:.2f}s  (batch_size={BATCH_SIZE})",
        f"Avg batch latency   : {avg_lat:.2f}s",
        "",
        "── CONFIDENCE BUCKETS ───────────────────────────────────────",
        f"  High conf  (>85%) : {counts['high_conf']:>6}  "
            f"({100*counts['high_conf']/max(total,1):.1f}%)  → Round 2 retraining",
        f"  Review (60-85%)   : {counts['review']:>6}  "
            f"({100*counts['review']/max(total,1):.1f}%)  → spot check",
        f"  Failed    (<60%)  : {counts['failed']:>6}  "
            f"({100*counts['failed']/max(total,1):.1f}%)  → discard",
        f"    parse errors    : {counts['parse_error']:>6}",
        "",
        "── NEXT STEP ────────────────────────────────────────────────",
        f"  {counts['high_conf']} high-confidence pairs + 3,208 original "
            f"= ~{3208 + counts['high_conf']:,} total training pairs.",
        "  Run Round 2 retraining: modal run training/finetune_qwen.py",
        sep,
        "Download results:",
        "  modal volume get retailgraph-models extraction_results ./extraction/",
        sep,
    ]
    return "\n".join(lines)


# =============================================================================
# SECTION 10 — LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main(target: int = 20_000, resume: bool = False):
    """
    Triggers extraction on Modal A100 from your laptop.

    Usage:
        modal run src/extraction/extractor.py --target 100
        modal run src/extraction/extractor.py --target 20000
        modal run src/extraction/extractor.py --target 20000 --resume
        modal volume get retailgraph-models extraction_results ./extraction/
    """
    if resume:
        log.info("Resume mode: will skip already-processed products.")
    log.info(f"Target: {target:,} products | Batch size: {BATCH_SIZE}")
    log.info("Triggering extraction on Modal A100...")
    log.info("Check live progress at: https://modal.com/apps")

    report = run_extraction.remote(target=target, resume=resume)
    print("\n" + report)

    log.info("Extraction complete.")
    log.info("Download results:")
    log.info("  modal volume get retailgraph-models extraction_results ./extraction/")