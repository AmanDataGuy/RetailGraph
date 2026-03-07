# training/generate_visual_pairs.py
"""
Phase 4.3 — Visual Training Pair Generation.

PURPOSE:
    Generates 500 multimodal (image + text → JSON) training pairs for
    fine-tuning Qwen2-VL on the ProductEntity extraction task.

    Unlike generate_pairs.py (text only via GPT-4o-mini), this script
    sends BOTH the product image AND text to GPT-4o vision so that
    visual fields (packaging_type, packaging_color, has_brand_logo)
    are properly filled in the output JSON.

    During Qwen2-VL fine-tuning, both image + text will be passed as
    input — so the model learns to use visual signal for these fields.

WHY 500 (not 3,000):
    GPT-4o vision is ~10x more expensive than GPT-4o-mini.
    500 pairs costs ~$1.50 and is sufficient to teach visual field
    extraction across all packaging types and categories.
    The 3,000 text pairs handle everything else.

SAMPLING STRATEGY (from already-processed text pair products only):
    300 bare products    (60%) — image is the PRIMARY signal here
    150 medium products  (30%) — image supplements sparse text
     50 rich products    (10%) — teaches text + image consistency
    ─────────────────────────────────────────────────────────────
    500 total, proportional across all 18 categories

    WHY same products as text pairs?
    Model sees Smucker's jar twice:
        Text pair  → {"packaging_type": null, ...}      (GPT-4o-mini, no image)
        Visual pair → {"packaging_type": "jar", ...}    (GPT-4o, sees image)
    This is data augmentation — same product, different modality.
    Model learns to fill visual fields from image signal.

PIPELINE:
    data/training/progress.jsonl   (already-processed IDs)
        → load_processed_ids()     — get IDs from text pair generation
        → load_tiers()             — assign bare/medium/rich by text length
        → sample_products()        — 300/150/50 proportional split
        → encode_image()           — load image from disk → base64
        → call_gpt4o_vision()      — GPT-4o with image + text
        → validate_extraction()    — Phase 2 validator
        → save_pair()              — append to visual_pairs.jsonl
        → save_checkpoint()        — update visual_progress.jsonl

PAIR FORMAT (stored in JSONL):
    {
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},   ← text only (for Qwen input)
            {"role": "assistant", "content": "{...}"}  ← correct JSON
        ],
        "image_path": "data/images/train/158784.jpg"   ← Qwen loads this at training time
    }

    NOTE: We do NOT store base64 in the JSONL — that would make the file
    ~17GB. Instead we store the image path. The fine-tuning script
    (training/finetune_qwen.py) loads images from disk during training.

API COST:
    GPT-4o vision: ~$1.50 for 500 pairs (image detail="low" saves tokens).
    Total with text pairs: ~$2.00 on OpenAI.

OUTPUTS:
    data/training/visual_pairs.jsonl      — 500 visual pairs
    data/training/visual_progress.jsonl   — checkpoint (resume support)
    data/training/visual_failed.csv       — products that failed all retries
"""

from __future__ import annotations

import argparse
import base64
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
TARGET_VISUAL  = 500
N_BARE         = 300     # 60% — image is primary signal
N_MEDIUM       = 150     # 30% — image supplements text
N_RICH         = 50      # 10% — text + image consistency

OPENAI_MODEL   = "gpt-4o"
SLEEP_SECONDS  = 1.0     # GPT-4o has stricter rate limits than mini
MAX_RETRIES    = 2
MIN_CONFIDENCE = 0.55

# Text length thresholds for tier assignment (matches EDA findings)
BARE_MAX_CHARS   = 150   # ≤150 chars → bare
MEDIUM_MAX_CHARS = 600   # 151–600 chars → medium
                         # >600 chars → rich

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR          = Path("data/training")
IMAGES_DIR          = Path("data/images/train")    # images live in train/ subdir
TRAIN_CSV           = Path("data/raw/train.csv")

TEXT_PROGRESS_JSONL = OUTPUT_DIR / "progress.jsonl"         # from generate_pairs.py
VIS_PAIRS_JSONL     = OUTPUT_DIR / "visual_pairs.jsonl"
VIS_PROGRESS_JSONL  = OUTPUT_DIR / "visual_progress.jsonl"  # our checkpoint
VIS_FAILED_CSV      = OUTPUT_DIR / "visual_failed.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Visual system prompt ──────────────────────────────────────────────────────
# Extends the base system prompt with explicit visual field instructions.
# Kept separate from prompt_templates.py because it's GPT-4o vision only —
# the base SYSTEM_PROMPT in prompt_templates.py is used for text-only calls.

VISUAL_SYSTEM_PROMPT = """You are a product data extraction model for a grocery e-commerce platform.

You will receive a product image and catalog text. Extract structured information into a valid JSON object.

## OUTPUT RULES — FOLLOW EXACTLY
1. Return ONLY a valid JSON object. No markdown, no backticks, no explanation, no preamble.
2. Never add fields that are not in the schema below.
3. Never omit required fields (item_name, price, quantity_value, quantity_unit, category, extraction_confidence).
4. Use ONLY the exact values from the controlled vocabulary lists below.
5. If a field cannot be determined, use the default value specified.

## OUTPUT SCHEMA
{
  "item_name": string,           // Clean product name WITHOUT size/quantity. Max 120 chars.
  "price": float,                // Product price as a positive number. Required.
  "quantity_value": float,       // Numeric amount (e.g. 16.0 for 16oz). Default: 1.0
  "quantity_unit": string,       // MUST be one of the canonical units below.
  "category": string,            // MUST be one of the allowed categories below.
  "dietary_tags": [string],      // Only from allowed list. Empty [] if none.
  "allergen_list": [string],     // Allergens mentioned. Lowercase. Empty [] if none.
  "packaging_type": string,      // LOOK AT THE IMAGE. One of: bottle, bag, box, jar, can, sachet, pouch, tube, carton, unknown
  "packaging_color": string,     // LOOK AT THE IMAGE. Dominant packaging color. null if unclear.
  "has_brand_logo": boolean,     // LOOK AT THE IMAGE. Is a brand logo clearly visible? null if unclear.
  "extraction_confidence": float // Confidence in this extraction. Float between 0.0 and 1.0.
}

## VISUAL FIELD INSTRUCTIONS — READ CAREFULLY
packaging_type: Study the image carefully.
  bottle  → liquid product in a bottle (ketchup, juice, sauce, oil)
  jar     → wide-mouth glass/plastic container (peanut butter, jam, salsa)
  can     → metal cylinder (canned beans, soup, tuna)
  bag     → flexible plastic/mylar bag (chips, flour, rice, frozen items)
  box     → rigid cardboard box (cereal, pasta, crackers, tea bags)
  pouch   → stand-up flexible pouch (baby food, coffee, protein powder)
  carton  → milk/juice carton or egg carton
  sachet  → small single-serve packet or envelope
  tube    → squeezable tube (toothpaste, tomato paste, condiment)
  unknown → image unclear or packaging not visible

packaging_color: The DOMINANT color of the outer packaging.
  Examples: "red", "blue", "white", "green", "yellow", "orange", "brown", "black", "purple"
  Use null if image is unclear or very small.

has_brand_logo: true if you can clearly see a brand logo or brand name on the packaging.
  false if no logo visible. null if image is too small/blurry to tell.

## CANONICAL UNITS — use EXACTLY these strings
oz, fl oz, lb, g, kg, ml, l, ct, pack

## ALLOWED CATEGORIES — use EXACTLY these strings
Beverages, Coffee & Tea, Snacks & Candy, Condiments & Sauces,
Grains, Beans & Legumes, Baking & Cooking, Spices & Seasonings,
Supplements & Health, Nuts & Seeds, Personal Care & Beauty,
Protein Bars & Snacks

## ALLOWED DIETARY TAGS — only use these exact strings
organic, kosher, gluten-free, non-GMO, vegan, keto, paleo,
dairy-free, sugar-free, nut-free, soy-free, high-protein,
low-calorie, caffeine-free, allergen-free

## CONFIDENCE CALIBRATION
  0.90–0.97 → Rich text + clear image: most fields determinable
  0.75–0.89 → Medium text OR clear image: most fields determinable
  0.60–0.74 → Sparse text, image fills gaps
  0.45–0.59 → Very sparse text, image partially unclear
  Never output 1.0. Never output below 0.40 for an existing product."""


# ── Step 1 — Load processed IDs from text pair generation ────────────────────

def load_processed_ids() -> set[int]:
    """
    Load product IDs already processed by generate_pairs.py.

    We only generate visual pairs for products that already have
    text pairs — this enables data augmentation (same product,
    both modalities) and avoids wasting GPT-4o calls on products
    that failed text extraction.

    Returns:
        Set of sample_id integers that succeeded in text pair generation.
    """
    if not TEXT_PROGRESS_JSONL.exists():
        raise FileNotFoundError(
            f"{TEXT_PROGRESS_JSONL} not found. "
            "Run generate_pairs.py first (Phase 4.2)."
        )

    processed = set()
    with open(TEXT_PROGRESS_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                # Only include successfully processed products
                if record.get("success", False):
                    processed.add(int(record["sample_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    logger.info("Loaded %d successfully processed text pair IDs.", len(processed))
    return processed


# ── Step 2 — Checkpoint for visual pairs ─────────────────────────────────────

def load_visual_checkpoint() -> set[int]:
    """
    Load already-processed visual pair IDs (for resume support).

    Same pattern as load_progress() in generate_pairs.py.
    Allows resuming a crashed run without re-processing products.

    Returns:
        Set of sample_id integers already processed in this run.
    """
    if not VIS_PROGRESS_JSONL.exists():
        return set()

    done = set()
    with open(VIS_PROGRESS_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(int(json.loads(line.strip())["sample_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    logger.info("Resuming visual generation: %d already done.", len(done))
    return done


def save_visual_checkpoint(sample_id: int, success: bool) -> None:
    """Append one product ID to the visual checkpoint file."""
    with open(VIS_PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": sample_id, "success": success}) + "\n")


# ── Step 3 — Proportional sampling ───────────────────────────────────────────

def assign_tier(text_len: int) -> str:
    """
    Assign content tier from text length.

    Proxy for ContentTier enum — avoids importing ProductEntity
    just for sampling. Thresholds match EDA findings (Phase 1).

    bare   → ≤150 chars  (item name only, warehouse-style)
    medium → 151–600 chars (some bullets, partial info)
    rich   → >600 chars  (full description with bullets)
    """
    if text_len <= BARE_MAX_CHARS:
        return "bare"
    elif text_len <= MEDIUM_MAX_CHARS:
        return "medium"
    return "rich"


def find_image_path(sample_id: int) -> Optional[Path]:
    """
    Find the image file for a product.

    Tries .jpg and .jpeg extensions since download script
    may have saved either. Returns None if not found.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        path = IMAGES_DIR / f"{sample_id}{ext}"
        if path.exists():
            return path
    return None


def sample_products(
    df: pd.DataFrame,
    processed_ids: set[int],
    already_done: set[int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample 500 products for visual pair generation.

    Strategy:
        - Only from products that succeeded in text pair generation
        - Only products that have a downloadable image on disk
        - Skip products already processed in a previous visual run
        - Weighted: 300 bare / 150 medium / 50 rich
        - Within each tier, proportional across all 18 categories

    Args:
        df:            full train.csv DataFrame
        processed_ids: IDs that succeeded in text pair generation
        already_done:  IDs already processed in this visual run
        seed:          random seed for reproducibility

    Returns:
        DataFrame of sampled products with columns:
        sample_id, catalog_content, price, tier
    """
    # Filter to text-pair-processed products not yet done visually
    pool = df[
        df["sample_id"].isin(processed_ids) &
        ~df["sample_id"].isin(already_done)
    ].copy()

    logger.info("Pool before image filter: %d products", len(pool))

    # Assign tiers
    pool["text_len"] = pool["catalog_content"].str.len()
    pool["tier"]     = pool["text_len"].apply(assign_tier)

    # Filter to products with images on disk
    pool["image_path"] = pool["sample_id"].apply(
        lambda sid: find_image_path(int(sid))
    )
    pool = pool[pool["image_path"].notna()].copy()

    logger.info(
        "Pool after image filter: %d products | bare=%d medium=%d rich=%d",
        len(pool),
        len(pool[pool["tier"] == "bare"]),
        len(pool[pool["tier"] == "medium"]),
        len(pool[pool["tier"] == "rich"]),
    )

    # Sample per tier
    parts = []
    quotas = {"bare": N_BARE, "medium": N_MEDIUM, "rich": N_RICH}

    for tier, quota in quotas.items():
        tier_df  = pool[pool["tier"] == tier]
        actual   = min(quota, len(tier_df))
        sampled  = tier_df.sample(n=actual, random_state=seed)
        parts.append(sampled)
        logger.info("  Sampled %d / %d %s products", actual, len(tier_df), tier)

    result = pd.concat(parts).drop_duplicates(subset="sample_id").reset_index(drop=True)
    logger.info("Total sampled: %d products", len(result))
    return result


# ── Step 4 — Image encoding ───────────────────────────────────────────────────

def encode_image_b64(image_path: Path) -> Optional[str]:
    """
    Read image from disk and encode to base64 string.

    Base64 is required by the GPT-4o vision API.
    NOT stored in JSONL — only used for the API call.
    The image_path is stored instead (fine-tuning script
    loads images from disk).

    Returns:
        Base64-encoded string, or None if file cannot be read.
    """
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.warning("Failed to encode image %s: %s", image_path, e)
        return None


# ── Step 5 — GPT-4o vision call ───────────────────────────────────────────────

def build_visual_messages(
    sample_id: int,
    catalog_content: str,
    price: Optional[float],
    image_b64: str,
) -> list[dict]:
    """
    Build the full message list for a GPT-4o vision API call.

    Structure:
        [system]   visual system prompt (with packaging field instructions)
        [user]     image (base64) + text (catalog_content + price)

    Why detail="low"?
        GPT-4o "low" detail uses ~85 tokens per image vs ~1,000+ for "high".
        For packaging type/color/logo detection, low detail is sufficient.
        This saves ~$1 over 500 images.

    Args:
        sample_id:       product ID (for product_id field in output)
        catalog_content: raw catalog text from train.csv
        price:           product price (may be None)
        image_b64:       base64-encoded image string

    Returns:
        List of message dicts ready for client.chat.completions.create()
    """
    content = catalog_content.strip()
    if price is not None and "Price:" not in content:
        content = f"{content}\nPrice: {price}"

    return [
        {
            "role": "system",
            "content": VISUAL_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url":    f"data:image/jpeg;base64,{image_b64}",
                        "detail": "low",   # saves ~$1 over 500 images
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Product ID: {sample_id}\n\n"
                        f"Catalog listing:\n{content}\n\n"
                        "Extract all product information. Use the image to fill "
                        "packaging_type, packaging_color, and has_brand_logo."
                    ),
                },
            ],
        },
    ]


def call_gpt4o_vision(client: OpenAI, messages: list[dict]) -> str:
    """
    Call GPT-4o with vision. Returns raw response text.

    Raises on API error — caller handles retries.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ── Step 6 — Extract with retry ───────────────────────────────────────────────

def extract_visual_product(
    client: OpenAI,
    sample_id: int,
    catalog_content: str,
    price: Optional[float],
    image_b64: str,
) -> tuple[bool, Optional[dict], list[dict]]:
    """
    Run GPT-4o vision extraction with up to MAX_RETRIES attempts.

    On failure, builds a retry message with the specific error description
    so GPT-4o can correct itself — same retry pattern as generate_pairs.py.

    Args:
        client:          OpenAI client
        sample_id:       product ID
        catalog_content: raw catalog text
        price:           product price
        image_b64:       base64 image

    Returns:
        (success, entity_dict, messages_used)
        success=True  → entity_dict has validated ProductEntity fields
        success=False → entity_dict is None
        messages_used → the messages list for building the final pair
    """
    messages   = build_visual_messages(sample_id, catalog_content, price, image_b64)
    raw_output = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = call_gpt4o_vision(client, messages)
        except Exception as e:
            logger.warning(
                "[%s] GPT-4o API error attempt %d: %s",
                sample_id, attempt, e
            )
            time.sleep(5)
            continue

        result = validate_extraction(raw_output, str(sample_id), attempt=attempt)

        if result.success:
            return True, result.entity.model_dump(), messages

        logger.debug(
            "[%s] attempt %d failed at step '%s': %s",
            sample_id, attempt, result.step, result.error
        )

        # Build retry message with error feedback (same as generate_pairs.py)
        if attempt < MAX_RETRIES:
            messages = build_retry_messages(messages, raw_output, result.error or "")
            time.sleep(SLEEP_SECONDS)

    return False, None, messages


# ── Step 7 — Save pairs and failures ─────────────────────────────────────────

def save_visual_pair(
    messages: list[dict],
    entity_dict: dict,
    image_path: Path,
) -> None:
    """
    Append one visual pair to visual_pairs.jsonl.

    Pair format:
        {
            "messages": [..., {"role": "assistant", "content": "{...}"}],
            "image_path": "data/images/train/158784.jpg"
        }

    WHY store image_path and not base64?
        500 images × ~200KB average = ~100MB base64.
        The fine-tuning script (finetune_qwen.py) loads images
        from disk at training time instead.

    The "messages" field uses the TEXT-ONLY format (no image_url block)
    because Qwen2-VL expects images passed separately, not in messages.
    Only the assistant output (correct JSON) matters for the loss function.
    """
    # For Qwen fine-tuning, rebuild text-only messages (no base64 block)
    # GPT-4o visual messages contain base64 which we don't want in JSONL
    text_messages = [m for m in messages if isinstance(m["content"], str)]

    pair = {
        "messages": text_messages + [{
            "role":    "assistant",
            "content": json.dumps(entity_dict, ensure_ascii=False),
        }],
        "image_path": str(image_path),
    }

    with open(VIS_PAIRS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(pair, ensure_ascii=False) + "\n")


def log_visual_failure(sample_id: int, catalog_content: str, error: str) -> None:
    """Append a failed product to visual_failed.csv."""
    write_header = not VIS_FAILED_CSV.exists()
    with open(VIS_FAILED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "catalog_preview", "error"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "sample_id":       sample_id,
            "catalog_preview": str(catalog_content)[:200],
            "error":           str(error)[:300],
        })


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_visual_generation(
    target: int = TARGET_VISUAL,
    fast: bool = False,
) -> dict:
    """
    Run the full visual pair generation pipeline.

    1. Load processed IDs from generate_pairs.py
    2. Load visual checkpoint (for resume support)
    3. Sample 500 products (300 bare / 150 medium / 50 rich)
    4. For each product: encode image → call GPT-4o → validate → save
    5. Print summary

    Args:
        target: number of visual pairs to generate (default: 500)
        fast:   skip sleep between requests (not recommended, rate limits)

    Returns:
        Summary dict with success/failure counts.
    """
    logger.info("═" * 60)
    logger.info("Starting visual pair generation — target: %d pairs", target)
    logger.info("═" * 60)

    # Load data
    df = pd.read_csv(TRAIN_CSV)
    logger.info("Loaded %d products from train.csv", len(df))

    processed_ids = load_processed_ids()
    already_done  = load_visual_checkpoint()

    sampled = sample_products(df, processed_ids, already_done)
    todo    = sampled[~sampled["sample_id"].isin(already_done)]

    logger.info(
        "To process: %d (skipping %d already done)",
        len(todo), len(already_done)
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found in environment. "
            "Add it to your .env file."
        )
    client = OpenAI(api_key=api_key)

    success_count = 0
    failed_count  = 0

    for i, row in enumerate(todo.itertuples(index=False)):
        sample_id       = int(row.sample_id)
        catalog_content = str(row.catalog_content)
        price           = float(row.price) if pd.notna(row.price) else None
        image_path      = Path(str(row.image_path))

        # Encode image
        image_b64 = encode_image_b64(image_path)
        if image_b64 is None:
            logger.warning("[%s] Could not encode image, skipping.", sample_id)
            log_visual_failure(sample_id, catalog_content, "image encode failed")
            save_visual_checkpoint(sample_id, success=False)
            failed_count += 1
            continue

        # Extract
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
            logger.info(
                "Progress: %d/%d — success=%d failed=%d",
                i + 1, len(todo), success_count, failed_count
            )

        if not fast:
            time.sleep(SLEEP_SECONDS)

    summary = {
        "total_attempted": len(todo),
        "success":         success_count,
        "failed":          failed_count,
        "success_rate":    round(success_count / max(len(todo), 1), 3),
    }

    logger.info("═" * 60)
    logger.info("VISUAL GENERATION SUMMARY")
    logger.info("  Total attempted : %d", summary["total_attempted"])
    logger.info("  Success         : %d (%.1f%%)", summary["success"], 100 * summary["success_rate"])
    logger.info("  Failed          : %d", summary["failed"])
    logger.info("  Output          : %s", VIS_PAIRS_JSONL)
    if failed_count > 0:
        logger.info("  Failures logged : %s", VIS_FAILED_CSV)
    logger.info("═" * 60)

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate visual training pairs for Qwen2-VL fine-tuning"
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_VISUAL,
        help=f"Number of visual pairs to generate (default: {TARGET_VISUAL})"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip sleep between requests (use carefully — rate limits)"
    )
    args = parser.parse_args()

    run_visual_generation(
        target=args.target,
        fast=args.fast,
    )