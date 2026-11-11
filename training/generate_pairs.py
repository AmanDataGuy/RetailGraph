"""# generate_pairs.py - builds the text seed pairs (catalog text -> ProductEntity JSON) for Qwen2-VL fine-tuning

- Input: data/raw/train.csv (+ data/extracted/weak_labels.csv for category-proportional sampling).
- Output: data/training/train.jsonl + val.jsonl (80/20), failed_generation.csv, progress.jsonl (resume checkpoint).
- Teacher: Groq openai/gpt-oss-120b. The existing 3,000 pairs were made with GPT-4o-mini (2026-03).
- Run: python training/generate_pairs.py [--target N] [--fast] [--no-few-shot]
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

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from retailgraph.extraction.prompt_templates import build_full_messages, build_retry_messages
from retailgraph.extraction.validator import validate_extraction

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TARGET_PAIRS   = 3_000
TRAIN_RATIO    = 0.80
GROQ_MODEL     = "openai/gpt-oss-120b"   # same model as src/retailgraph/agent/llm.py
SLEEP_SECONDS  = 0.5
MAX_RETRIES    = 2

OUTPUT_DIR     = Path("data/training")
TRAIN_JSONL    = OUTPUT_DIR / "train.jsonl"
VAL_JSONL      = OUTPUT_DIR / "val.jsonl"
FAILED_CSV     = OUTPUT_DIR / "failed_generation.csv"
PROGRESS_JSONL = OUTPUT_DIR / "progress.jsonl"
TRAIN_CSV      = Path("data/raw/train.csv")


# Pick n products, proportional to weak-label categories (min 10 each), else a plain random sample.
def sample_products(df: pd.DataFrame, n: int = TARGET_PAIRS, seed: int = 42) -> pd.DataFrame:
    weak_labels_path = Path("data/extracted/weak_labels.csv")

    if weak_labels_path.exists():
        logger.info("Using weak_labels.csv for proportional sampling...")
        labels = pd.read_csv(weak_labels_path, usecols=["sample_id", "category", "category_status"])
        accepted = labels[labels["category_status"] == "auto_accept"]
        logger.info("Auto-accepted products for sampling: %d", len(accepted))

        merged = df.merge(accepted[["sample_id", "category"]], on="sample_id", how="inner")
        category_counts = merged["category"].value_counts()
        allocations = {cat: max(10, int(n * count / len(merged))) for cat, count in category_counts.items()}

        total_allocated = sum(allocations.values())
        if total_allocated > n:
            scale = n / total_allocated
            allocations = {cat: max(10, int(v * scale)) for cat, v in allocations.items()}

        sampled_parts = []
        for cat, quota in allocations.items():
            cat_df = merged[merged["category"] == cat]
            sampled_parts.append(cat_df.sample(n=min(quota, len(cat_df)), random_state=seed))
        sampled = pd.concat(sampled_parts).drop_duplicates(subset="sample_id")

        remaining = n - len(sampled)
        if remaining > 0:
            leftover = df[~df["sample_id"].isin(set(sampled["sample_id"]))]
            sampled = pd.concat([sampled, leftover.sample(n=min(remaining, len(leftover)), random_state=seed)])
    else:
        logger.warning(
            "weak_labels.csv not found — falling back to random sampling. "
            "Run weak supervision first for proportional sampling."
        )
        sampled = df.sample(n=min(n, len(df)), random_state=seed)

    sampled = sampled.reset_index(drop=True)
    logger.info("Sampled %d products for pair generation.", len(sampled))
    return sampled


# Make a Groq client from GROQ_API_KEY, or fail loudly.
def build_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise OSError(
            "GROQ_API_KEY not found in environment. "
            "Add it to your .env file: GROQ_API_KEY=gsk_..."
        )
    return Groq(api_key=api_key)


# Send one chat request to the teacher model and return its text.
def call_groq(client: Groq, messages: list[dict], model: str = GROQ_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500,
        temperature=0,
        reasoning_effort="low",   # gpt-oss reasons first; keep it from eating max_tokens
    )
    return response.choices[0].message.content.strip()


# Ask the teacher to extract one product, retrying with the validation error; returns (ok, entity, raw).
def extract_product(
    client: Groq,
    sample_id: int,
    catalog_content: str,
    price: float | None,
    include_few_shot: bool = True,
) -> tuple[bool, dict | None, str]:
    messages   = build_full_messages(catalog_content, price, include_few_shot)
    raw_output = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = call_groq(client, messages)
        except Exception as e:
            logger.warning("Groq API error on product %s attempt %d: %s", sample_id, attempt, e)
            time.sleep(5)
            continue

        result = validate_extraction(raw_output, str(sample_id), attempt=attempt)
        if result.success:
            return True, result.entity.model_dump(), raw_output

        last_error = result.error or "unknown error"
        logger.debug("Product %s attempt %d failed at step '%s': %s", sample_id, attempt, result.step, last_error)

        if attempt < MAX_RETRIES:
            messages = build_retry_messages(messages, raw_output, _format_retry_error(result.step, last_error))
            time.sleep(SLEEP_SECONDS)

    return False, None, raw_output


# Turn a validation failure into a correction hint for the retry prompt.
def _format_retry_error(step: str, error: str) -> str:
    if step == "json_parse":
        return f"JSON parse error: {error}. Return ONLY raw JSON — no backticks, no explanation."
    if step == "pydantic":
        return f"Schema validation error: {error}. Fix the field and return corrected JSON only."
    if step == "confidence":
        return (
            f"Confidence too low: {error}. "
            f"If the product information is clear, increase extraction_confidence. "
            f"Return corrected JSON only."
        )
    return f"Validation error at step '{step}': {error}. Return corrected JSON only."


# Read the sample_ids already processed in an earlier run.
def load_progress() -> set[int]:
    if not PROGRESS_JSONL.exists():
        return set()

    processed = set()
    with open(PROGRESS_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                processed.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    logger.info("Resuming: %d products already processed.", len(processed))
    return processed


# Append one processed sample_id to the resume checkpoint.
def save_progress(sample_id: int, success: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": sample_id, "success": success}) + "\n")


# Append one training pair as a JSONL line.
def save_pair(pair: dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


# Append a failed product to failed_generation.csv (header on first write).
def log_failure(sample_id: int, catalog_content: str, error: str) -> None:
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


# Shuffle with a fixed seed and split into train/val lists.
def split_pairs(pairs: list[dict], train_ratio: float = TRAIN_RATIO, seed: int = 42) -> tuple[list[dict], list[dict]]:
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_pairs, val_pairs = shuffled[:split_idx], shuffled[split_idx:]
    logger.info(
        "Split: %d train / %d val (%.0f%% / %.0f%%)",
        len(train_pairs), len(val_pairs), 100 * train_ratio, 100 * (1 - train_ratio),
    )
    return train_pairs, val_pairs


# Full pipeline: sample, extract with the teacher, checkpoint, split and write train/val JSONL.
def run_generation(
    target: int = TARGET_PAIRS,
    fast: bool = False,
    include_few_shot: bool = True,
    train_csv: Path = TRAIN_CSV,
) -> dict:
    logger.info("Starting training pair generation — target: %d pairs", target)
    df = pd.read_csv(train_csv)
    logger.info("Loaded %d products from %s.", len(df), train_csv)

    sampled      = sample_products(df, n=target)
    already_done = load_progress()
    todo         = sampled[~sampled["sample_id"].isin(already_done)]
    logger.info("To process: %d (skipping %d already done)", len(todo), len(already_done))

    client        = build_groq_client()
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
            pairs.append({
                "messages": messages + [{
                    "role":    "assistant",
                    "content": json.dumps(entity_dict, ensure_ascii=False),
                }]
            })
            success_count += 1
        else:
            log_failure(sample_id, catalog_content, raw_output[:300])
            failed_count += 1

        save_progress(sample_id, success)

        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d — success=%d failed=%d", i + 1, len(todo), success_count, failed_count)
        if not fast:
            time.sleep(SLEEP_SECONDS)

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
    logger.info(
        "Done: %d attempted, %d ok (%.1f%%), %d failed -> %d train (%s) / %d val (%s)",
        summary["total_attempted"], summary["success"], 100 * summary["success_rate"], summary["failed"],
        summary["train_pairs"], TRAIN_JSONL, summary["val_pairs"], VAL_JSONL,
    )
    if failed_count > 0:
        logger.info("Failures logged: %s", FAILED_CSV)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training pairs for Qwen2-VL fine-tuning")
    parser.add_argument("--target", type=int, default=TARGET_PAIRS,
                        help=f"Number of pairs to generate (default: {TARGET_PAIRS})")
    parser.add_argument("--fast", action="store_true", help="Skip sleep between requests")
    parser.add_argument("--no-few-shot", action="store_true", help="Disable few-shot examples in prompts")
    args = parser.parse_args()
    run_generation(target=args.target, fast=args.fast, include_few_shot=not args.no_few_shot)
