"""
RetailGraph - Round 2 Training Data Preparation
Merges high_conf extractions from Phase 6 with original train.jsonl
Run from project root: python prepare_round2.py
"""
import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
EXTRACTION_FILE = Path("data/training/verified_extractions.jsonl")               # raw extraction file in project root
ORIGINAL_TRAIN  = Path("data/training/train.jsonl")
OUTPUT_TRAIN    = Path("data/training/train_r3.jsonl")

# ── System prompt (matches original training data) ────────────────────────────
SYSTEM_PROMPT = """You are a product data extraction model for a grocery e-commerce platform.

Your job is to read a raw product catalog listing and extract structured information into a valid JSON object.

## OUTPUT RULES — FOLLOW EXACTLY

1. Return ONLY a valid JSON object. No markdown, no backticks, no explanation, no preamble.
2. Never add fields that are not in the schema below.
3. Never omit required fields (item_name, price, quantity_value, quantity_unit, category, extraction_confidence).
4. Use ONLY the exact values from the controlled vocabulary lists below.
5. If a field cannot be determined from the text, use the default value specified.

## OUTPUT SCHEMA

{
  "item_name": string,           // Clean product name WITHOUT size/quantity. Max 120 chars.
  "price": float,                // Product price as a positive number. Required.
  "quantity_value": float,       // Numeric amount (e.g. 16.0 for 16oz). Default: 1.0
  "quantity_unit": string,       // MUST be one of the canonical units below.
  "category": string,            // MUST be one of the allowed categories below.
  "dietary_tags": [string],      // List of applicable tags. Only from allowed list. Empty [] if none.
  "allergen_list": [string],     // Allergens mentioned. Lowercase. Empty [] if none.
  "extraction_confidence": float // Your confidence in this extraction. Float between 0.0 and 1.0.
}

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
low-calorie, caffeine-free, allergen-free"""


def clean_prediction(pred: dict) -> dict:
    """Strip Phase 6 extra fields, keep only schema fields."""
    SCHEMA_FIELDS = {
        "item_name", "price", "quantity_value", "quantity_unit",
        "category", "dietary_tags", "allergen_list", "extraction_confidence"
    }
    cleaned = {}
    for field in SCHEMA_FIELDS:
        val = pred.get(field)
        # Defaults for missing required fields
        if field == "quantity_value" and val is None:
            val = 1.0
        if field == "quantity_unit" and val is None:
            val = "ct"
        if field == "dietary_tags" and val is None:
            val = []
        if field == "allergen_list" and val is None:
            val = []
        if field == "extraction_confidence" and val is None:
            val = 0.85
        cleaned[field] = val
    return cleaned


def make_training_example(catalog_content: str, prediction: dict, price: float) -> dict:
    """Convert extraction result into train.jsonl format."""
    user_content = (
        "Extract product information from the following grocery product listing.\n"
        "Return ONLY a valid JSON object matching the schema. "
        "No explanation, no markdown, no backticks.\n\n"
        f"Product listing:\n{catalog_content}\n"
        f"Price: {price}"
    )
    assistant_content = json.dumps(clean_prediction(prediction), indent=2)

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main():
    # ── Load original training data ───────────────────────────────────────────
    print(f"Loading original training data from {ORIGINAL_TRAIN}...")
    original = []
    with open(ORIGINAL_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                original.append(json.loads(line))
    print(f"  Original pairs: {len(original)}")

    # ── Load high_conf extractions ────────────────────────────────────────────
    print(f"Loading high_conf extractions from {EXTRACTION_FILE}...")
    new_pairs = []
    skipped = 0
    with open(EXTRACTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if record.get("bucket") != "high_conf":
                continue

            pred = record.get("prediction", {})
            catalog = record.get("catalog_content", "")
            price = record.get("price", 0.0)

            # Skip if prediction is empty or missing item_name
            if not pred or not pred.get("item_name"):
                skipped += 1
                continue

            example = make_training_example(catalog, pred, price)
            new_pairs.append(example)

    print(f"  High-conf pairs: {len(new_pairs)}")
    print(f"  Skipped: {skipped}")

    # ── Merge and write ───────────────────────────────────────────────────────
    merged = original + new_pairs
    print(f"  Total merged: {len(merged)}")

    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for example in merged:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\nSaved to {OUTPUT_TRAIN}")
    print(f"Round 2 training set: {len(merged)} pairs")
    print(f"  Original:  {len(original)}")
    print(f"  New:       {len(new_pairs)}")
    print(f"\nNext step:")
    print(f"  Upload to Modal and run: modal run training/finetune_qwen.py")


if __name__ == "__main__":
    main()

