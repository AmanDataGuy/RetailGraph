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
        → generate_with_openai()    — GPT-4o-mini
        → validate_extraction()     — Phase 2 validator (JSON → Pydantic → confidence)
        → normalize_product()       — Phase 2 normalizer
        → save_pairs()              — 80/20 train/val split
        → data/training/train.jsonl — 2,400 training pairs
        → data/training/val.jsonl   — 600 validation pairs
        → data/training/failed_generation.csv — failed products

API COST:
    GPT-4o-mini: ~$0.15-0.30 for 3,000 pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.extraction.prompt_templates import build_full_messages, build_retry_messages
from src.extraction.validator import validate_extraction

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_PAIRS   = 3_000
TRAIN_RATIO    = 0.80
OPENAI_MODEL   = "gpt-4o-mini"
SLEEP_SECONDS  = 0.5
MAX_RETRIES    = 2
MIN_CONFIDENCE = 0.55

OUTPUT_DIR     = Path("data/training")
TRAIN_JSONL    = OUTPUT_DIR / "train.jsonl"
VAL_JSONL      = OUTPUT_DIR / "val.jsonl"
FAILED_CSV     = OUTPUT_DIR / "failed_generation.csv"
PROGRESS_JSONL = OUTPUT_DIR / "progress.jsonl"
TRAIN_CSV      = Path("data/raw/train.csv")

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
    weak_labels_path = Path("data/extracted/weak_labels.csv")

    if weak_labels_path.exists():
        logger.info("Using weak_labels.csv for proportional sampling...")
        labels = pd.read_csv(
            weak_labels_path,
            usecols=["sample_id", "category", "category_status"]
        )
        accepted = labels[labels["category_status"] == "auto_accept"].copy()
        logger.info("Auto-accepted products for sampling: %d", len(accepted))

        merged = df.merge(
            accepted[["sample_id", "category"]], on="sample_id", how="inner"
        )

        category_counts = merged["category"].value_counts()
        total_accepted  = len(merged)
        allocations     = {}

        for cat, count in category_counts.items():
            allocated = max(10, int(n * count / total_accepted))
            allocations[cat] = allocated

        total_allocated = sum(allocations.values())
        if total_allocated > n:
            scale = n / total_allocated
            allocations = {
                cat: max(10, int(v * scale)) for cat, v in allocations.items()
            }

        sampled_parts = []
        for cat, quota in allocations.items():
            cat_df = merged[merged["category"] == cat]
            actual = min(quota, len(cat_df))
            sampled_parts.append(cat_df.sample(n=actual, random_state=seed))

        sampled = pd.concat(sampled_parts).drop_duplicates(subset="sample_id")

        remaining = n - len(sampled)
        if remaining > 0:
            already_sampled = set(sampled["sample_id"])
            leftover = df[~df["sample_id"].isin(already_sampled)]
            extra = leftover.sample(
                n=min(remaining, len(leftover)), random_state=seed
            )
            sampled = pd.concat([sampled, extra])

    else:
        logger.warning(
            "weak_labels.csv not found — falling back to random sampling. "
            "Run Phase 3 first for proportional sampling."
        )
        sampled = df.sample(n=min(n, len(df)), random_state=seed)

    sampled = sampled.reset_index(drop=True)
    logger.info("Sampled %d products for pair generation.", len(sampled))
    return sampled


# ── Step 2 — OpenAI client ────────────────────────────────────────────────────

def build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found in environment. "
            "Add it to your .env file: OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)


def call_openai(
    client: OpenAI,
    messages: list[dict],
    model: str = OPENAI_MODEL,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def extract_product(
    client: OpenAI,
    sample_id: int,
    catalog_content: str,
    price: Optional[float],
    include_few_shot: bool = True,
) -> tuple[bool, Optional[dict], str]:
    messages   = build_full_messages(catalog_content, price, include_few_shot)
    raw_output = ""
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = call_openai(client, messages)
        except Exception as e:
            logger.warning(
                "OpenAI API error on product %s attempt %d: %s",
                sample_id, attempt, e
            )
            time.sleep(5)
            continue

        result = validate_extraction(raw_output, str(sample_id), attempt=attempt)

        if result.success:
            entity_dict = result.entity.model_dump()
            return True, entity_dict, raw_output

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
    if step == "json_parse":
        return (
            f"JSON parse error: {error}. "
            f"Return ONLY raw JSON — no backticks, no explanation."
        )
    elif step == "pydantic":
        return (
            f"Schema validation error: {error}. "
            f"Fix the field and return corrected JSON only."
        )
    elif step == "confidence":
        return (
            f"Confidence too low: {error}. "
            f"If the product information is clear, increase extraction_confidence. "
            f"Return corrected JSON only."
        )
    else:
        return (
            f"Validation error at step '{step}': {error}. "
            f"Return corrected JSON only."
        )


# ── Step 3 — Checkpointing & saving ──────────────────────────────────────────

def load_progress() -> set[int]:
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": sample_id, "success": success}) + "\n")


def save_pair(pair: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def log_failure(sample_id: int, catalog_content: str, error: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = FAILED_CSV.exists()

    with open(FAILED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "catalog_preview", "error"]
        )
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
    logger.info("═" * 60)
    logger.info("Starting training pair generation — target: %d pairs", target)
    logger.info("═" * 60)

    logger.info("Loading %s...", train_csv)
    df = pd.read_csv(train_csv)
    logger.info("Loaded %d products.", len(df))

    sampled      = sample_products(df, n=target)
    already_done = load_progress()
    todo         = sampled[~sampled["sample_id"].isin(already_done)]

    logger.info(
        "To process: %d (skipping %d already done)",
        len(todo), len(already_done)
    )

    client        = build_openai_client()
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

        if (i + 1) % 50 == 0:
            logger.info(
                "Progress: %d/%d — success=%d failed=%d",
                i + 1, len(todo), success_count, failed_count
            )

        if not fast:
            time.sleep(SLEEP_SECONDS)

    logger.info("Generation complete. Splitting into train/val...")
    train_pairs, val_pairs = split_pairs(pairs)

    TRAIN_JSONL.unlink(missing_ok=True)
    VAL_JSONL.unlink(missing_ok=True)

    for pair in train_pairs:
        save_pair(pair, TRAIN_JSONL)
    for pair in val_pairs:
        save_pair(pair, VAL_JSONL)

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


# ── CLI ───────────────────────────────────────────────────────────────────────

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
        help="Skip sleep between requests"
    )
    parser.add_argument(
        "--no-few-shot", action="store_true",
        help="Disable few-shot examples in prompts"
    )
    args = parser.parse_args()

    run_generation(
        target=args.target,
        fast=args.fast,
        include_few_shot=not args.no_few_shot,
    )