# training/generate_visual_pairs.py
"""
Phase 4.3 — Visual Training Pair Generation.

PURPOSE:
    Generates 500 multimodal (image + text → JSON) training pairs using
    GPT-4o vision. Unlike generate_pairs.py (text only via GPT-4o-mini),
    this script sends BOTH the product image AND text to GPT-4o so that
    visual fields (packaging_type, packaging_color, has_brand_logo) are
    properly filled in the output JSON.

    During Qwen2-VL fine-tuning, both image + text are passed as input —
    so the model learns to use visual signal for those fields.

WHY 500 (not 3,000):
    GPT-4o vision is ~10x more expensive than GPT-4o-mini.
    500 pairs costs ~$2 and is sufficient to teach visual field extraction
    across all packaging types. The 3,000 text pairs handle everything else.

SAMPLING STRATEGY (from already-processed text pair products only):
    300 bare products    (60%) — image is the PRIMARY signal
    150 medium products  (30%) — image supplements sparse text
     50 rich products    (10%) — teaches text + image consistency

    WHY same products as text pairs?
    Data augmentation — same product, different modality:
        Text pair   → {"packaging_type": null, ...}   (no image seen)
        Visual pair → {"packaging_type": "jar",  ...}  (image seen)
    Model learns to fill visual fields from image during fine-tuning.

PAIR FORMAT stored in JSONL:
    {
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "Product ID: ...\nCatalog: ..."},
            {"role": "assistant", "content": "{...json with visual fields...}"}
        ],
        "image_path": "data/images/train/158784.jpg"
    }

    NOTE: base64 NOT stored in JSONL — that would make the file ~17GB.
    finetune_qwen.py loads images from disk at training time using image_path.

API COST:
    GPT-4o with detail="low": ~$1.85-2.20 for 500 pairs total.

OUTPUTS:
    data/training/visual_pairs.jsonl     — 500 visual pairs (append-safe)
    data/training/visual_progress.jsonl  — checkpoint (resume support)
    data/training/visual_failed.csv      — products that failed all retries
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
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

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
N_BARE         = 300      # 60% — image is primary signal
N_MEDIUM       = 150      # 30% — supplements sparse text
N_RICH         = 50       # 10% — consistency training

OPENAI_MODEL   = "gpt-4o"
SLEEP_SECONDS  = 1.0      # GPT-4o rate limits are stricter than mini
MAX_RETRIES    = 2

# Text length thresholds matching EDA (Phase 1)
BARE_MAX_CHARS   = 150
MEDIUM_MAX_CHARS = 600

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR           = Path("data/training")
IMAGES_DIR           = Path("data/images/train")
TRAIN_CSV            = Path("data/raw/train.csv")
TEXT_PROGRESS_JSONL  = OUTPUT_DIR / "progress.jsonl"
VIS_PAIRS_JSONL      = OUTPUT_DIR / "visual_pairs.jsonl"
VIS_PROGRESS_JSONL   = OUTPUT_DIR / "visual_progress.jsonl"
VIS_FAILED_CSV       = OUTPUT_DIR / "visual_failed.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── System prompt ─────────────────────────────────────────────────────────────
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


# ── Step 1 — Load processed IDs from text pair generation ────────────────────

def load_processed_ids() -> set[int]:
    """
    Load product IDs that succeeded in text pair generation (generate_pairs.py).

    WHY only from text pairs:
        - Ensures data augmentation (same product in both text and visual sets)
        - Avoids wasting GPT-4o calls on products that failed extraction before
    """
    if not TEXT_PROGRESS_JSONL.exists():
        raise FileNotFoundError(
            f"{TEXT_PROGRESS_JSONL} not found. "
            "Run generate_pairs.py first (Phase 4.2)."
        )

    processed = set()
    with open(TEXT_PROGRESS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("success", False):
                    processed.add(int(record["sample_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    logger.info("Loaded %d successfully processed text pair IDs.", len(processed))
    return processed


# ── Step 2 — Checkpoint ───────────────────────────────────────────────────────

def load_visual_checkpoint() -> set[int]:
    """Load already-processed visual pair IDs for resume support."""
    if not VIS_PROGRESS_JSONL.exists():
        return set()

    done = set()
    with open(VIS_PROGRESS_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["sample_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    logger.info("Resuming: %d visual pairs already done.", len(done))
    return done


def save_visual_checkpoint(sample_id: int, success: bool) -> None:
    """Append one product ID to the visual checkpoint file."""
    with open(VIS_PROGRESS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": sample_id, "success": success}) + "\n")


# ── Step 3 — Sampling ─────────────────────────────────────────────────────────

def assign_tier(text_len: int) -> str:
    """
    Assign content tier by text length.
    Thresholds match Phase 1 EDA findings:
        bare   ≤ 150 chars  (item name only)
        medium 151–600 chars (partial info)
        rich   > 600 chars  (full description)
    """
    if text_len <= BARE_MAX_CHARS:
        return "bare"
    elif text_len <= MEDIUM_MAX_CHARS:
        return "medium"
    return "rich"


def find_image_path(sample_id: int) -> Optional[Path]:
    """Try .jpg, .jpeg, .png extensions. Returns None if not found."""
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

    Rules:
        - Only from products that succeeded in text pair generation
        - Only products with a downloadable image on disk
        - Skip already done (resume support)
        - Weighted: 300 bare / 150 medium / 50 rich
    """
    pool = df[
        df["sample_id"].isin(processed_ids) &
        ~df["sample_id"].isin(already_done)
    ].copy()

    logger.info("Pool before image filter: %d products", len(pool))

    pool["text_len"]   = pool["catalog_content"].str.len()
    pool["tier"]       = pool["text_len"].apply(assign_tier)
    pool["image_path"] = pool["sample_id"].apply(lambda sid: find_image_path(int(sid)))
    pool               = pool[pool["image_path"].notna()].copy()

    logger.info(
        "Pool after image filter: %d | bare=%d medium=%d rich=%d",
        len(pool),
        len(pool[pool["tier"] == "bare"]),
        len(pool[pool["tier"] == "medium"]),
        len(pool[pool["tier"] == "rich"]),
    )

    parts   = []
    quotas  = {"bare": N_BARE, "medium": N_MEDIUM, "rich": N_RICH}

    for tier, quota in quotas.items():
        tier_df = pool[pool["tier"] == tier]
        actual  = min(quota, len(tier_df))
        sampled = tier_df.sample(n=actual, random_state=seed)
        parts.append(sampled)
        logger.info("  Sampled %d / %d %s products", actual, len(tier_df), tier)

    result = pd.concat(parts).drop_duplicates(subset="sample_id").reset_index(drop=True)
    logger.info("Total sampled: %d products", len(result))
    return result


# ── Step 4 — Image encoding ───────────────────────────────────────────────────

def encode_image_b64(image_path: Path) -> Optional[str]:
    """
    Encode image to base64 string for GPT-4o vision API.

    NOT stored in JSONL — only used for the API call.
    image_path is stored instead; finetune_qwen.py loads images from disk.
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
    Build GPT-4o messages with image + text input.

    detail="low" saves ~$1 over 500 images.
    Low detail uses ~85 tokens per image vs 1,000+ for high.
    Sufficient for packaging type/color/logo detection.
    """
    price_str = f"\nPrice: ${price}" if price is not None else ""

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
                        "detail": "low",
                    },
                },
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


def call_gpt4o_vision(client: OpenAI, messages: list[dict]) -> str:
    """Call GPT-4o vision. Raises on API error — caller handles retries."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def fix_null_lists(raw_json: str) -> str:
    """
    Convert null → [] for dietary_tags and allergen_list.

    GPT-4o consistently returns null for empty lists even when instructed
    not to. This post-processes the raw output before Pydantic validation
    so we never get 'Input should be a valid list' errors.
    """
    try:
        parsed = json.loads(raw_json)
        if parsed.get("dietary_tags") is None:
            parsed["dietary_tags"] = []
        if parsed.get("allergen_list") is None:
            parsed["allergen_list"] = []
        return json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw_json  # not valid JSON — let validator handle it


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

    Returns:
        (success, entity_dict, original_messages)
        success=True  → entity_dict contains validated ProductEntity fields
                        including visual fields extracted from the raw output
        success=False → entity_dict is None
        original_messages → the first-turn messages (for building pairs)
    """
    messages = build_visual_messages(sample_id, catalog_content, price, image_b64)

    for attempt in range(1, MAX_RETRIES + 1):
        raw_output = ""
        try:
            raw_output = call_gpt4o_vision(client, messages)
        except Exception as e:
            logger.warning(
                "[%s] GPT-4o API error attempt %d: %s", sample_id, attempt, e
            )
            time.sleep(5)
            continue

        # Fix null lists before Pydantic sees them
        raw_output = fix_null_lists(raw_output)

        result = validate_extraction(raw_output, str(sample_id), attempt=attempt)

        if result.success:
            entity_dict = result.entity.model_dump()

            # Inject visual fields from raw GPT-4o output into entity_dict.
            # ProductEntity stores these under entity.visual but GPT-4o returns
            # them at top level — we flatten them for the training pair.
            try:
                raw_parsed = json.loads(raw_output)
                for vfield in ["packaging_type", "packaging_color", "has_brand_logo"]:
                    val = raw_parsed.get(vfield)
                    if val is not None:
                        entity_dict[vfield] = val
            except Exception:
                pass

            return True, entity_dict, messages

        # Build retry: append assistant failure + new user correction request.
        # We do NOT use build_retry_messages() here because that function
        # appends to messages which already contain image_url blocks —
        # GPT-4o rejects subsequent image_url blocks in the same conversation.
        # Instead we add plain text turns only after the first turn.
        if attempt < MAX_RETRIES:
            logger.debug(
                "[%s] attempt %d failed at '%s': %s",
                sample_id, attempt, result.step, result.error
            )
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


# ── Step 7 — Save pairs ───────────────────────────────────────────────────────

def save_visual_pair(
    messages: list[dict],
    entity_dict: dict,
    image_path: Path,
) -> None:
    """
    Append one visual pair to visual_pairs.jsonl.

    Pair format:
        {
            "messages": [system, user_text, assistant_json],
            "image_path": "data/images/train/158784.jpg"
        }

    WHY user message is text-only (not image_url):
        Qwen2-VL receives images as a separate input during fine-tuning,
        not embedded in messages. We store image_path so finetune_qwen.py
        can load it from disk. The user message stores the catalog text only.

    WHY NOT store base64 in JSONL:
        500 images × ~200KB = ~100MB base64 in one file.
        Storing paths keeps the file small (~5MB).
    """
    # Extract text-only user message from the multimodal messages list.
    # The first user message has content=[image_url_block, text_block].
    # We extract just the text block for storage.
    clean_messages = []
    for m in messages:
        if isinstance(m["content"], str):
            # System message and retry turns — keep as-is
            clean_messages.append({"role": m["role"], "content": m["content"]})
        elif isinstance(m["content"], list):
            # First user message — extract text block only
            text_parts = [
                block["text"]
                for block in m["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if text_parts:
                clean_messages.append({
                    "role":    m["role"],
                    "content": " ".join(text_parts),
                })

    pair = {
        "messages": clean_messages + [{
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
    2. Load visual checkpoint (resume support)
    3. Sample 500 products (300 bare / 150 medium / 50 rich)
    4. For each: encode image → call GPT-4o → validate → save pair
    5. Print summary

    Returns:
        Summary dict with success/failure counts.
    """
    logger.info("═" * 60)
    logger.info("Starting visual pair generation — target: %d pairs", target)
    logger.info("═" * 60)

    df = pd.read_csv(TRAIN_CSV)
    logger.info("Loaded %d products from train.csv", len(df))

    processed_ids = load_processed_ids()
    already_done  = load_visual_checkpoint()
    sampled       = sample_products(df, processed_ids, already_done)
    todo          = sampled[~sampled["sample_id"].isin(already_done)]

    logger.info(
        "To process: %d (skipping %d already done)",
        len(todo), len(already_done)
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not found in .env file."
        )
    client = OpenAI(api_key=api_key)

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
        help="Skip sleep between requests"
    )
    args = parser.parse_args()

    run_visual_generation(
        target=args.target,
        fast=args.fast,
    )