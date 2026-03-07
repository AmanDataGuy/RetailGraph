# training/generate_pairs.py
"""
Phase 4.2 — Training Pair Generation.

PURPOSE:
    Generates 3,000 high-quality (input → output) training pairs for
    fine-tuning Qwen2-VL on the ProductEntity extraction task.

    Each pair is:
        INPUT  = catalog_content (raw text from train.csv) + price
        OUTPUT = ProductEntity JSON (validated + normalized)

WHY 3,000:
    These are the SEED pairs. After fine-tuning Qwen2-VL on them,
    the self-training loop runs the fine-tuned model on the remaining
    ~72,000 products and progressively expands the training set.

PIPELINE:
    train.csv (75k rows)
        → sample_products()         — 3,000 proportional across 11 categories
        → generate_with_groq()      — Groq Llama-3.1-8b-instant (free tier)
        → validate_extraction()     — Phase 2 validator (JSON → Pydantic → confidence)
        → normalize_product()       — Phase 2 normalizer
        → save_pairs()              — 80/20 train/val split
        → data/training/train.jsonl — 2,400 training pairs
        → data/training/val.jsonl   — 600 validation pairs
        → data/training/failed_generation.csv — failed products

API COST:
    Groq Llama-3.1-8b-instant is FREE (rate limited).
    GPT-4o-mini fallback: ~$1.50 for 3,000 pairs.
    Switch via GENERATION_MODEL env var.

RATE LIMITING:
    Groq free tier: 30 requests/minute.
    We sleep 2s between requests → ~1,800 per hour → 3,000 takes ~1.7 hours.
    Run overnight or use --fast flag to skip sleep (risks 429 errors).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from src.extraction.prompt_templates import build_full_messages, build_retry_messages
from src.extraction.validator import validate_extraction, validate_with_retry

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Total seed pairs to generate
TARGET_PAIRS = 3_000

# 80/20 split
TRAIN_RATIO = 0.80

# Groq model — free tier, fast, good enough for seed data generation
GROQ_MODEL = "llama-3.1-8b-instant"

# Sleep between requests to respect Groq free tier rate limit (30 req/min)
SLEEP_SECONDS = 2.0

# Max retries per product if first extraction fails validation
MAX_RETRIES = 2

# Minimum confidence to accept a pair into the training set.
# Lower than validator.py's MIN_CONFIDENCE (0.60) because we normalize
# after extraction — some products genuinely have sparse listings.
MIN_CONFIDENCE = 0.55

# Output paths
OUTPUT_DIR        = Path("data/training")
TRAIN_JSONL       = OUTPUT_DIR / "train.jsonl"
VAL_JSONL         = OUTPUT_DIR / "val.jsonl"
FAILED_CSV        = OUTPUT_DIR / "failed_generation.csv"
PROGRESS_JSONL    = OUTPUT_DIR / "progress.jsonl"  # checkpoint file

# Input
TRAIN_CSV         = Path("data/raw/train.csv")

# The 11 categories — must match CATEGORY_NAMES in label_model.py
CATEGORY_NAMES = [
    "Beverages",
    "Coffee & Tea",
    "Snacks & Candy",
    "Condiments & Sauces",
    "Grains, Beans & Legumes",
    "Baking & Cooking",
    "Spices & Seasonings",
    "Supplements & Health",
    "Nuts & Seeds",
    "Personal Care & Beauty",
    "Protein Bars & Snacks",
]


# ── Step 1 — Proportional sampling ───────────────────────────────────────────

def sample_products(
    df: pd.DataFrame,
    n: int = TARGET_PAIRS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample n products proportionally across all 11 categories.

    WHY PROPORTIONAL:
        If we sampled randomly, rare categories (Protein Bars, Personal Care)
        would get maybe 10 products each. The fine-tuned model would barely
        see them during training and perform poorly on them.

        Proportional sampling ensures every category gets AT LEAST
        floor(n / n_categories) = 272 examples, with larger categories
        getting more in proportion to their size.

    HOW:
        We use weak_labels.csv (Phase 3 output) to know which category
        each product belongs to. Products without a category label get
        sampled from the "unknown" pool to fill any remaining quota.

    Args:
        df:   full train.csv DataFrame (75k rows)
        n:    total products to sample (default 3,000)
        seed: random seed for reproducibility

    Returns:
        DataFrame of n rows with columns: sample_id, catalog_content, price
    """
    weak_labels_path = Path("data/extracted/weak_labels.csv")

    if weak_labels_path.exists():
        logger.info("Using weak_labels.csv for proportional sampling...")
        labels = pd.read_csv(weak_labels_path, usecols=["sample_id", "category", "category_status"])

        # Only use auto_accepted categories — they're the most reliable
        accepted = labels[labels["category_status"] == "auto_accept"].copy()
        logger.info("Auto-accepted products for sampling: %d", len(accepted))

        # Merge with full df to get catalog_content and price
        merged = df.merge(accepted[["sample_id", "category"]], on="sample_id", how="inner")

        # Proportional allocation per category
        category_counts = merged["category"].value_counts()
        total_accepted  = len(merged)
        allocations     = {}

        for cat, count in category_counts.items():
            # Allocate proportionally, minimum 10 per category
            allocated = max(10, int(n * count / total_accepted))
            allocations[cat] = allocated

        # Scale down if total allocation exceeds n
        total_allocated = sum(allocations.values())
        if total_allocated > n:
            scale = n / total_allocated
            allocations = {cat: max(10, int(v * scale)) for cat, v in allocations.items()}

        # Sample per category
        sampled_parts = []
        for cat, quota in allocations.items():
            cat_df = merged[merged["category"] == cat]
            actual = min(quota, len(cat_df))
            sampled_parts.append(cat_df.sample(n=actual, random_state=seed))

        sampled = pd.concat(sampled_parts).drop_duplicates(subset="sample_id")

        # Fill remaining quota with random products if needed
        remaining = n - len(sampled)
        if remaining > 0:
            already_sampled = set(sampled["sample_id"])
            leftover = df[~df["sample_id"].isin(already_sampled)]
            extra = leftover.sample(n=min(remaining, len(leftover)), random_state=seed)
            sampled = pd.concat([sampled, extra])

    else:
        # Fallback — pure random sample if weak labels don't exist yet
        logger.warning(
            "weak_labels.csv not found — falling back to random sampling. "
            "Run Phase 3 first for proportional sampling."
        )
        sampled = df.sample(n=min(n, len(df)), random_state=seed)

    sampled = sampled.reset_index(drop=True)
    logger.info("Sampled %d products for pair generation.", len(sampled))
    return sampled


# ── Step 2 — LLM extraction ───────────────────────────────────────────────────

def build_groq_client() -> Groq:
    """
    Build and return a Groq client.

    Reads GROQ_API_KEY from .env via load_dotenv().
    Raises clear error if key is missing — common setup mistake.

    Returns:
        Groq client instance.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found in environment. "
            "Add it to your .env file: GROQ_API_KEY=gsk_..."
        )
    return Groq(api_key=api_key)


def call_groq(
    client: Groq,
    messages: list[dict],
    model: str = GROQ_MODEL,
) -> str:
    """
    Call Groq API with a list of messages and return the response text.

    WHY MAX_TOKENS=500:
        A valid ProductEntity JSON is ~250–350 chars. 500 tokens is
        generous enough to handle edge cases without wasting API quota.

    WHY TEMPERATURE=0:
        We want deterministic, structured JSON output — not creative
        variation. temperature=0 means the model always picks the
        highest probability token → most consistent JSON structure.

    Args:
        client:   Groq client from build_groq_client()
        messages: list of message dicts from build_full_messages()
        model:    Groq model string (default: llama-3.1-8b-instant)

    Returns:
        Raw response string from the model.

    Raises:
        Exception: passes through Groq API errors for caller to handle.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def extract_product(
    client: Groq,
    sample_id: int,
    catalog_content: str,
    price: Optional[float],
    include_few_shot: bool = True,
) -> tuple[bool, Optional[dict], str]:
    """
    Extract ProductEntity JSON for one product with up to MAX_RETRIES retries.

    Flow:
        1. Build messages with prompt_templates.build_full_messages()
        2. Call Groq → get raw JSON string
        3. Validate with validator.validate_with_retry()
           - JSON parse check
           - Pydantic validation
           - Confidence threshold check
        4. If failed → retry with error feedback in prompt
        5. Return (success, entity_dict, raw_output)

    Args:
        client:          Groq client
        sample_id:       product ID (for logging and failed.csv)
        catalog_content: raw catalog text
        price:           product price
        include_few_shot: whether to include few-shot examples in prompt

    Returns:
        (success, entity_dict, raw_output)
        success     — True if validation passed
        entity_dict — ProductEntity as dict (None if failed)
        raw_output  — last raw string from model (for debugging failures)
    """
    messages    = build_full_messages(catalog_content, price, include_few_shot)
    raw_output  = ""
    last_error  = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = call_groq(client, messages)
        except Exception as e:
            logger.warning("Groq API error on product %s attempt %d: %s", sample_id, attempt, e)
            time.sleep(5)  # back off on API errors
            continue

        # Validate through Phase 2 pipeline
        result = validate_extraction(raw_output, str(sample_id), attempt=attempt)

        if result.success:
            entity_dict = result.entity.model_dump()
            return True, entity_dict, raw_output

        # Build retry message with error feedback
        last_error = result.error or "unknown error"
        logger.debug(
            "Product %s attempt %d failed at step '%s': %s",
            sample_id, attempt, result.step, last_error
        )

        if attempt < MAX_RETRIES:
            error_description = _format_retry_error(result.step, last_error)
            messages = build_retry_messages(messages, raw_output, error_description)
            time.sleep(SLEEP_SECONDS)

    return False, None, raw_output


def _format_retry_error(step: str, error: str) -> str:
    """
    Format a human-readable retry error message for the model.

    Mirrors the logic in validator.py's _format_error_for_retry() but
    tailored for the generation context — we want the model to understand
    what went wrong and fix it.

    Args:
        step:  validation step that failed (json_parse, pydantic, confidence)
        error: error message from the validator

    Returns:
        Formatted string to inject into the retry prompt.
    """
    if step == "json_parse":
        return f"JSON parse error: {error}. Return ONLY raw JSON — no backticks, no explanation."
    elif step == "pydantic":
        return f"Schema validation error: {error}. Fix the field and return corrected JSON only."
    elif step == "confidence":
        return (
            f"Confidence too low: {error}. "
            f"If the product information is clear, increase extraction_confidence. "
            f"Return corrected JSON only."
        )
    else:
        return f"Validation error at step '{step}': {error}. Return corrected JSON only."


# ── Step 3 — Save pairs ───────────────────────────────────────────────────────

def load_progress() -> set[int]:
    """
    Load already-processed sample_ids from the progress checkpoint file.

    WHY CHECKPOINTING:
        Generating 3,000 pairs takes ~1.7 hours on Groq free tier.
        If the script crashes or you close your laptop halfway through,
        you'd lose all progress. The checkpoint file lets you resume
        from where you left off.

    Returns:
        Set of sample_ids that have already been processed.
    """
    if not PROGRESS_JSONL.exists():
        return set()

    processed = set()
    with open(PROGRESS_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                processed.add(record["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info("Resuming: %d products already processed.", len(processed))
    return processed


def save_progress(sample_id: int, success: bool) -> None:
    """
    Append one progress record to the checkpoint file.

    Args:
        sample_id: product ID
        success:   whether extraction succeeded
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": sample_id, "success": success}) + "\n")


def save_pair(pair: dict, output_file: Path) -> None:
    """
    Append one training pair to a JSONL file.

    WHY JSONL (JSON Lines):
        Each line is one complete JSON object. JSONL is the standard
        format for fine-tuning datasets because:
        1. You can stream it line by line without loading the whole file
        2. Appending is safe — no need to rewrite the entire file
        3. Qwen2-VL's fine-tuning script expects JSONL format

    Pair format:
        {
          "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "{...json...}"}
          ]
        }

    This is the standard chat fine-tuning format used by Qwen2-VL,
    LLaMA, and most modern instruction-tuned models.

    Args:
        pair:        training pair dict with 'messages' key
        output_file: path to the JSONL file to append to
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def log_failure(sample_id: int, catalog_content: str, error: str) -> None:
    """
    Log a failed product to failed_generation.csv.

    Failed products can be retried manually or skipped.
    Having a log of failures helps diagnose patterns — e.g. if all
    failures are from one category, that category's products might
    need a different prompt strategy.

    Args:
        sample_id:       product ID
        catalog_content: raw catalog text (truncated to 200 chars)
        error:           last error message
    """
    import csv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = FAILED_CSV.exists()

    with open(FAILED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "catalog_preview", "error"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "sample_id":       sample_id,
            "catalog_preview": str(catalog_content)[:200],
            "error":           str(error)[:300],
        })


# ── Step 4 — Train/val split ──────────────────────────────────────────────────

def split_pairs(
    pairs: list[dict],
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split pairs into train and validation sets.

    WHY 80/20:
        Standard ML split. 2,400 train pairs is enough to fine-tune
        Qwen2-VL meaningfully. 600 val pairs gives a reliable estimate
        of generalization performance during fine-tuning.

    WHY SHUFFLE BEFORE SPLIT:
        Products are ordered by category in our sampled DataFrame.
        Without shuffling, all Beverages would go to train and all
        Protein Bars to val — badly unbalanced split.

    Args:
        pairs:       list of training pair dicts
        train_ratio: fraction for training (default 0.80)
        seed:        random seed for reproducibility

    Returns:
        (train_pairs, val_pairs)
    """
    import random
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    split_idx   = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:split_idx]
    val_pairs   = shuffled[split_idx:]

    logger.info(
        "Split: %d train / %d val (%.0f%% / %.0f%%)",
        len(train_pairs), len(val_pairs),
        100 * train_ratio, 100 * (1 - train_ratio),
    )
    return train_pairs, val_pairs


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_generation(
    target: int = TARGET_PAIRS,
    fast: bool = False,
    include_few_shot: bool = True,
    train_csv: Path = TRAIN_CSV,
) -> dict:
    """
    Run the full training pair generation pipeline.

    Steps:
        1. Load train.csv
        2. Sample target products proportionally
        3. Skip already-processed products (checkpoint resume)
        4. For each product: extract → validate → save pair or log failure
        5. Split into train/val JSONL files
        6. Print summary stats

    Args:
        target:           how many pairs to generate (default 3,000)
        fast:             skip sleep between requests (risks rate limit 429s)
        include_few_shot: include few-shot examples in prompts
        train_csv:        path to train.csv (parameterized for testing)

    Returns:
        Summary dict with counts: total, success, failed, train, val
    """
    logger.info("═" * 60)
    logger.info("Starting training pair generation — target: %d pairs", target)
    logger.info("═" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading %s...", train_csv)
    df = pd.read_csv(train_csv)
    logger.info("Loaded %d products.", len(df))

    # ── Sample ─────────────────────────────────────────────────────────────
    sampled = sample_products(df, n=target)

    # ── Resume from checkpoint ─────────────────────────────────────────────
    already_done = load_progress()
    todo = sampled[~sampled["sample_id"].isin(already_done)]
    logger.info(
        "To process: %d (skipping %d already done)",
        len(todo), len(already_done)
    )

    # ── Build Groq client ──────────────────────────────────────────────────
    client = build_groq_client()

    # ── Generate pairs ─────────────────────────────────────────────────────
    success_count = 0
    failed_count  = 0
    pairs         = []

    for i, row in enumerate(todo.itertuples(index=False)):
        sample_id       = int(row.sample_id)
        catalog_content = str(row.catalog_content)
        price           = float(row.price) if pd.notna(row.price) else None

        success, entity_dict, raw_output = extract_product(
            client=client,
            sample_id=sample_id,
            catalog_content=catalog_content,
            price=price,
            include_few_shot=include_few_shot,
        )

        if success and entity_dict:
            # Build training pair in chat fine-tuning format
            messages = build_full_messages(catalog_content, price, include_few_shot)
            pair = {
                "messages": messages + [{
                    "role":    "assistant",
                    "content": json.dumps(entity_dict, ensure_ascii=False),
                }]
            }
            pairs.append(pair)
            success_count += 1
        else:
            log_failure(sample_id, catalog_content, raw_output[:300])
            failed_count += 1

        save_progress(sample_id, success)

        # Progress log every 50 products
        if (i + 1) % 50 == 0:
            logger.info(
                "Progress: %d/%d — success=%d failed=%d",
                i + 1, len(todo), success_count, failed_count
            )

        if not fast:
            time.sleep(SLEEP_SECONDS)

    # ── Split and save ─────────────────────────────────────────────────────
    logger.info("Generation complete. Splitting into train/val...")
    train_pairs, val_pairs = split_pairs(pairs)

    # Clear existing JSONL files before writing (fresh split)
    TRAIN_JSONL.unlink(missing_ok=True)
    VAL_JSONL.unlink(missing_ok=True)

    for pair in train_pairs:
        save_pair(pair, TRAIN_JSONL)
    for pair in val_pairs:
        save_pair(pair, VAL_JSONL)

    # ── Summary ────────────────────────────────────────────────────────────
    summary = {
        "total_attempted": len(todo),
        "success":         success_count,
        "failed":          failed_count,
        "success_rate":    round(success_count / max(len(todo), 1), 3),
        "train_pairs":     len(train_pairs),
        "val_pairs":       len(val_pairs),
    }

    logger.info("═" * 60)
    logger.info("GENERATION SUMMARY")
    logger.info("  Total attempted : %d", summary["total_attempted"])
    logger.info("  Success         : %d (%.1f%%)", summary["success"], 100 * summary["success_rate"])
    logger.info("  Failed          : %d", summary["failed"])
    logger.info("  Train pairs     : %d → %s", summary["train_pairs"], TRAIN_JSONL)
    logger.info("  Val pairs       : %d → %s", summary["val_pairs"],   VAL_JSONL)
    if failed_count > 0:
        logger.info("  Failures logged : %s", FAILED_CSV)
    logger.info("═" * 60)

    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training pairs for Qwen2-VL fine-tuning"
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_PAIRS,
        help=f"Number of pairs to generate (default: {TARGET_PAIRS})"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip sleep between requests (risks Groq rate limit 429s)"
    )
    parser.add_argument(
        "--no-few-shot", action="store_true",
        help="Disable few-shot examples in prompts (faster, slightly less accurate)"
    )
    args = parser.parse_args()

    run_generation(
        target=args.target,
        fast=args.fast,
        include_few_shot=not args.no_few_shot,
    )