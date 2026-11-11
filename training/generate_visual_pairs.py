"""# generate_visual_pairs.py - builds image + text -> JSON pairs so Qwen2-VL learns the visual fields

- Input: data/raw/train.csv, data/images/train/<id>.jpg, and the text-pair checkpoint data/training/progress.jsonl.
- Output: data/training/visual_pairs.jsonl ({"messages", "image_path"}; no base64), visual_progress.jsonl, visual_failed.csv.
- Teacher: Groq qwen/qwen3.8-27b. The existing 500 pairs were made with GPT-4o (2026-03).
- Run after generate_pairs.py: python training/generate_visual_pairs.py [--fast]
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from retailgraph.extraction.validator import validate_extraction

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TARGET_VISUAL  = 500
N_BARE         = 300      # image is the main signal
N_MEDIUM       = 150      # image fills in sparse text
N_RICH         = 50       # text + image consistency

GROQ_VISION_MODEL = "qwen/qwen3.8-27b"   # Groq vision model (checked 2026-11-11); images must be ≥32px a side
SLEEP_SECONDS  = 1.0      # stay under Groq free-tier RPM
MAX_RETRIES    = 2

BARE_MAX_CHARS   = 150    # tier thresholds from the EDA
MEDIUM_MAX_CHARS = 600

OUTPUT_DIR           = Path("data/training")
IMAGES_DIR           = Path("data/images/train")
TRAIN_CSV            = Path("data/raw/train.csv")
TEXT_PROGRESS_JSONL  = OUTPUT_DIR / "progress.jsonl"
VIS_PAIRS_JSONL      = OUTPUT_DIR / "visual_pairs.jsonl"
VIS_PROGRESS_JSONL   = OUTPUT_DIR / "visual_progress.jsonl"
VIS_FAILED_CSV       = OUTPUT_DIR / "visual_failed.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VISUAL_SYSTEM_PROMPT = """You are a product data extraction model for a grocery e-commerce platform.

You will receive a product image and catalog text. Extract structured information into a valid JSON object.

## OUTPUT RULES
1. Return ONLY a valid JSON object. No markdown, no backticks, no explanation.
2. Never add fields not in the schema. Never omit required fields.
3. Use ONLY the exact values from the controlled vocabulary lists below.
4. For empty lists always use [] never null.

## OUTPUT SCHEMA
{
  "item_name": string,
  "price": float,
  "quantity_value": float,
  "quantity_unit": string,
  "category": string,
  "dietary_tags": [string],
  "allergen_list": [string],
  "packaging_type": string,
  "packaging_color": string,
  "has_brand_logo": boolean,
  "extraction_confidence": float
}

## VISUAL FIELDS — LOOK AT THE IMAGE CAREFULLY
packaging_type: bottle | jar | can | bag | box | pouch | carton | sachet | tube | unknown
  bottle  = liquid in a bottle (ketchup, juice, oil)
  jar     = wide-mouth container (peanut butter, jam, salsa)
  can     = metal cylinder (beans, soup, tuna)
  bag     = flexible plastic bag (chips, flour, rice)
  box     = rigid cardboard (cereal, pasta, crackers)
  pouch   = stand-up flexible pouch (coffee, protein powder)
  carton  = milk/juice carton
  sachet  = small single-serve packet
  tube    = squeezable tube (tomato paste, condiment)
  unknown = image unclear

packaging_color: dominant color of the outer packaging.
  Use common color names: red, blue, white, green, yellow, orange, brown, black, purple.
  Use null if image is too small or unclear.

has_brand_logo: true if brand logo is clearly visible on packaging. false if not. null if unclear.

## CANONICAL UNITS
oz, fl oz, lb, g, kg, ml, l, ct, pack

## ALLOWED CATEGORIES
Beverages, Coffee & Tea, Snacks & Candy, Condiments & Sauces,
Grains, Beans & Legumes, Baking & Cooking, Spices & Seasonings,
Supplements & Health, Nuts & Seeds, Personal Care & Beauty, Protein Bars & Snacks

## ALLOWED DIETARY TAGS
organic, kosher, gluten-free, non-GMO, vegan, keto, paleo,
dairy-free, sugar-free, nut-free, soy-free, high-protein,
low-calorie, caffeine-free, allergen-free

## CONFIDENCE
0.85-0.95: clear image + rich text
0.70-0.84: clear image + sparse text
0.55-0.69: unclear image or very sparse text
Never output 1.0. Never below 0.40."""


# Read a JSONL checkpoint and return the sample_ids it holds (only successful ones if only_success).
def _read_ids(path: Path, only_success: bool) -> set[int]:
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if not only_success or record.get("success", False):
                    ids.add(int(record["sample_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return ids


# IDs that succeeded in text pair generation; visual pairs reuse these products.
def load_processed_ids() -> set[int]:
    if not TEXT_PROGRESS_JSONL.exists():
        raise FileNotFoundError(f"{TEXT_PROGRESS_JSONL} not found. Run generate_pairs.py first.")
    processed = _read_ids(TEXT_PROGRESS_JSONL, only_success=True)
    logger.info("Loaded %d successfully processed text pair IDs.", len(processed))
    return processed


# IDs already handled by an earlier visual run (resume support).
def load_visual_checkpoint() -> set[int]:
    if not VIS_PROGRESS_JSONL.exists():
        return set()
    done = _read_ids(VIS_PROGRESS_JSONL, only_success=False)
    logger.info("Resuming: %d visual pairs already done.", len(done))
    return done


# Append one product ID to the visual checkpoint.
def save_visual_checkpoint(sample_id: int, success: bool) -> None:
    with open(VIS_PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": sample_id, "success": success}) + "\n")


# Bucket a listing by text length: bare (<=150), medium (<=600) or rich.
def assign_tier(text_len: int) -> str:
    if text_len <= BARE_MAX_CHARS:
        return "bare"
    if text_len <= MEDIUM_MAX_CHARS:
        return "medium"
    return "rich"


# Find the product image on disk (.jpg, .jpeg or .png), or None.
def find_image_path(sample_id: int) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png"]:
        path = IMAGES_DIR / f"{sample_id}{ext}"
        if path.exists():
            return path
    return None


# Pick text-pair products that have an image and are not done yet: 300 bare / 150 medium / 50 rich.
def sample_products(
    df: pd.DataFrame,
    processed_ids: set[int],
    already_done: set[int],
    seed: int = 42,
) -> pd.DataFrame:
    pool = df[df["sample_id"].isin(processed_ids) & ~df["sample_id"].isin(already_done)].copy()
    logger.info("Pool before image filter: %d products", len(pool))

    pool["text_len"]   = pool["catalog_content"].str.len()
    pool["tier"]       = pool["text_len"].apply(assign_tier)
    pool["image_path"] = pool["sample_id"].apply(lambda sid: find_image_path(int(sid)))
    pool               = pool[pool["image_path"].notna()].copy()

    tier_counts = pool["tier"].value_counts()
    logger.info(
        "Pool after image filter: %d | bare=%d medium=%d rich=%d", len(pool),
        tier_counts.get("bare", 0), tier_counts.get("medium", 0), tier_counts.get("rich", 0),
    )

    parts = []
    for tier, quota in {"bare": N_BARE, "medium": N_MEDIUM, "rich": N_RICH}.items():
        tier_df = pool[pool["tier"] == tier]
        actual  = min(quota, len(tier_df))
        parts.append(tier_df.sample(n=actual, random_state=seed))
        logger.info("  Sampled %d / %d %s products", actual, len(tier_df), tier)

    result = pd.concat(parts).drop_duplicates(subset="sample_id").reset_index(drop=True)
    logger.info("Total sampled: %d products", len(result))
    return result


# Read an image as base64 for the API call only (never stored in the JSONL).
def encode_image_b64(image_path: Path) -> str | None:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning("Failed to encode image %s: %s", image_path, e)
        return None


# Build the system + (image, text) user messages for the vision model.
def build_visual_messages(
    sample_id: int,
    catalog_content: str,
    price: float | None,
    image_b64: str,
) -> list[dict]:
    price_str = f"\nPrice: ${price}" if price is not None else ""
    return [
        {"role": "system", "content": VISUAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {
                    "type": "text",
                    "text": (
                        f"Product ID: {sample_id}{price_str}\n\n"
                        f"Catalog listing:\n{catalog_content.strip()}\n\n"
                        "Extract all product information. "
                        "Use the image to fill packaging_type, packaging_color, and has_brand_logo."
                    ),
                },
            ],
        },
    ]


# Send one vision request; raises on API error (the caller retries).
def call_groq_vision(client: Groq, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0,
        reasoning_effort="none",   # Qwen3 thinks first by default; that would eat max_tokens
    )
    return response.choices[0].message.content.strip()


# The vision model often returns null for empty lists; turn those into [] before validation.
def fix_null_lists(raw_json: str) -> str:
    try:
        parsed = json.loads(raw_json)
        if parsed.get("dietary_tags") is None:
            parsed["dietary_tags"] = []
        if parsed.get("allergen_list") is None:
            parsed["allergen_list"] = []
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw_json  # not valid JSON — let the validator report it


# Extract one product from image + text with retries; returns (ok, entity with flat visual fields, messages).
def extract_visual_product(
    client: Groq,
    sample_id: int,
    catalog_content: str,
    price: float | None,
    image_b64: str,
) -> tuple[bool, dict | None, list[dict]]:
    messages = build_visual_messages(sample_id, catalog_content, price, image_b64)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = call_groq_vision(client, messages)
        except Exception as e:
            logger.warning("[%s] Groq vision API error attempt %d: %s", sample_id, attempt, e)
            time.sleep(5)
            continue

        raw_output = fix_null_lists(raw_output)
        result = validate_extraction(raw_output, str(sample_id), attempt=attempt)

        if result.success:
            entity_dict = result.entity.model_dump()
            # The model returns visual fields at top level; copy them flat into the training target.
            try:
                raw_parsed = json.loads(raw_output)
                for vfield in ["packaging_type", "packaging_color", "has_brand_logo"]:
                    val = raw_parsed.get(vfield)
                    if val is not None:
                        entity_dict[vfield] = val
            except Exception:
                pass
            return True, entity_dict, messages

        # Retry with plain-text turns only: the vision model rejects a second image_url block,
        # which is why build_retry_messages() is not used here.
        if attempt < MAX_RETRIES:
            logger.debug("[%s] attempt %d failed at '%s': %s", sample_id, attempt, result.step, result.error)
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": (
                        f"That response failed validation: {result.error}\n"
                        "Fix the error and return corrected JSON only. "
                        "No markdown, no explanation."
                    ),
                },
            ]
            time.sleep(SLEEP_SECONDS)

    return False, None, messages


# Append one pair to visual_pairs.jsonl: text-only messages + the image path (Qwen2-VL loads the image from disk).
def save_visual_pair(messages: list[dict], entity_dict: dict, image_path: Path) -> None:
    clean_messages = []
    for m in messages:
        if isinstance(m["content"], str):
            clean_messages.append({"role": m["role"], "content": m["content"]})
        elif isinstance(m["content"], list):
            text_parts = [
                block["text"]
                for block in m["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if text_parts:
                clean_messages.append({"role": m["role"], "content": " ".join(text_parts)})

    pair = {
        "messages": clean_messages + [{
            "role":    "assistant",
            "content": json.dumps(entity_dict, ensure_ascii=False),
        }],
        "image_path": str(image_path),
    }
    with open(VIS_PAIRS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


# Append a failed product to visual_failed.csv (header on first write).
def log_visual_failure(sample_id: int, catalog_content: str, error: str) -> None:
    write_header = not VIS_FAILED_CSV.exists()
    with open(VIS_FAILED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "catalog_preview", "error"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "sample_id":       sample_id,
            "catalog_preview": str(catalog_content)[:200],
            "error":           str(error)[:300],
        })


# Full pipeline: sample products, call the vision model per image, save pairs and checkpoint as we go.
def run_visual_generation(target: int = TARGET_VISUAL, fast: bool = False) -> dict:
    logger.info("Starting visual pair generation — target: %d pairs", target)

    df = pd.read_csv(TRAIN_CSV)
    logger.info("Loaded %d products from train.csv", len(df))

    processed_ids = load_processed_ids()
    already_done  = load_visual_checkpoint()
    sampled       = sample_products(df, processed_ids, already_done)
    todo          = sampled[~sampled["sample_id"].isin(already_done)]
    logger.info("To process: %d (skipping %d already done)", len(todo), len(already_done))

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise OSError("GROQ_API_KEY not found in .env file.")
    client = Groq(api_key=api_key)

    success_count = 0
    failed_count  = 0

    for i, row in enumerate(todo.itertuples(index=False)):
        sample_id       = int(row.sample_id)
        catalog_content = str(row.catalog_content)
        price           = float(row.price) if pd.notna(row.price) else None
        image_path      = Path(str(row.image_path))

        image_b64 = encode_image_b64(image_path)
        if image_b64 is None:
            logger.warning("[%s] Could not encode image — skipping.", sample_id)
            log_visual_failure(sample_id, catalog_content, "image encode failed")
            save_visual_checkpoint(sample_id, success=False)
            failed_count += 1
            continue

        success, entity_dict, messages = extract_visual_product(
            client=client,
            sample_id=sample_id,
            catalog_content=catalog_content,
            price=price,
            image_b64=image_b64,
        )

        if success and entity_dict:
            save_visual_pair(messages, entity_dict, image_path)
            success_count += 1
        else:
            log_visual_failure(sample_id, catalog_content, "all retries failed")
            failed_count += 1

        save_visual_checkpoint(sample_id, success)

        if (i + 1) % 50 == 0:
            logger.info("Progress: %d/%d — success=%d failed=%d", i + 1, len(todo), success_count, failed_count)
        if not fast:
            time.sleep(SLEEP_SECONDS)

    summary = {
        "total_attempted": len(todo),
        "success":         success_count,
        "failed":          failed_count,
        "success_rate":    round(success_count / max(len(todo), 1), 3),
    }
    logger.info(
        "Done: %d attempted, %d ok (%.1f%%), %d failed -> %s",
        summary["total_attempted"], summary["success"], 100 * summary["success_rate"], summary["failed"],
        VIS_PAIRS_JSONL,
    )
    if failed_count > 0:
        logger.info("Failures logged: %s", VIS_FAILED_CSV)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual training pairs for Qwen2-VL fine-tuning")
    parser.add_argument("--target", type=int, default=TARGET_VISUAL,
                        help=f"Number of visual pairs to generate (default: {TARGET_VISUAL})")
    parser.add_argument("--fast", action="store_true", help="Skip sleep between requests")
    args = parser.parse_args()
    run_visual_generation(target=args.target, fast=args.fast)
