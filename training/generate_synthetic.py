"""
Phase 4.3 — Synthetic Data Generation for Underrepresented Categories.

PURPOSE:
    After generate_pairs.py creates 3,000 real seed pairs, some categories
    are still underrepresented — the model won't see enough examples of
    Protein Bars, Personal Care, or Supplements to learn them well.

    This script generates SYNTHETIC product listings for those categories
    using templates + randomization. No API calls needed — pure Python.

WHY SYNTHETIC DATA:
    Fine-tuning on imbalanced data causes the model to:
    - Overpredict common categories (Beverages, Snacks)
    - Miss rare categories entirely (Protein Bars: ~400 products in 75k)

    Adding synthetic examples for rare categories balances the training
    distribution without spending more API quota.

CAP AT 20%:
    Synthetic data is less realistic than real data. Too much of it
    hurts generalization. We cap at 20% of total training pairs
    (~600 of 3,000) so real data always dominates.

OUTPUT:
    Appends synthetic pairs to:
        data/training/train.jsonl  (80% of synthetic)
        data/training/val.jsonl    (20% of synthetic)

    Also saves a manifest:
        data/training/synthetic_manifest.csv
    so you know exactly which pairs are synthetic vs real.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from src.extraction.prompt_templates import (
    ALLOWED_CATEGORIES,
    ALLOWED_DIETARY_TAGS,
    CANONICAL_UNITS,
    build_full_messages,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Maximum synthetic pairs as fraction of total training set
MAX_SYNTHETIC_RATIO = 0.20

# Default total training set size (real pairs) — used to compute cap
DEFAULT_REAL_PAIRS = 3_000

# 80/20 split for synthetic pairs (same as real pairs)
TRAIN_RATIO = 0.80

# Output paths
OUTPUT_DIR         = Path("data/training")
TRAIN_JSONL        = OUTPUT_DIR / "train.jsonl"
VAL_JSONL          = OUTPUT_DIR / "val.jsonl"
SYNTHETIC_MANIFEST = OUTPUT_DIR / "synthetic_manifest.csv"

# Categories that typically get fewer real pairs due to low dataset frequency.
# These are the targets for synthetic augmentation.
UNDERREPRESENTED_CATEGORIES = [
    "Protein Bars & Snacks",
    "Personal Care & Beauty",
    "Supplements & Health",
]

# ── Template data ─────────────────────────────────────────────────────────────
# Each category has:
#   brands      — realistic brand names for that category
#   products    — product name templates (use {flavor}, {size} placeholders)
#   flavors     — flavor/variant options
#   sizes       — (quantity_value, quantity_unit) pairs
#   tags        — common dietary tags for this category
#   allergens   — common allergens for this category
#   price_range — (min, max) price in dollars

CATEGORY_TEMPLATES: dict[str, dict] = {
    "Protein Bars & Snacks": {
        "brands": [
            "RXBAR", "Clif Bar", "Quest Nutrition", "KIND", "LaraBar",
            "ONE Protein", "Built Bar", "Fulfil", "Think!", "No Cow",
        ],
        "products": [
            "{brand} Protein Bar {flavor}",
            "{brand} {flavor} Protein Bar",
            "{brand} Whole Food Protein Bar {flavor}",
            "{brand} High Protein Bar {flavor}",
        ],
        "flavors": [
            "Chocolate Chip Cookie Dough", "Peanut Butter Chocolate",
            "Blueberry", "Chocolate Sea Salt", "Coconut Chocolate",
            "Mixed Berry", "Cinnamon Roll", "Birthday Cake",
            "Mint Chocolate Chip", "Salted Caramel",
        ],
        "sizes": [
            (1.76, "oz"), (2.12, "oz"), (2.5, "oz"),
            (12, "ct"), (6, "ct"), (4, "ct"),
        ],
        "tags": ["gluten-free", "high-protein", "non-GMO", "kosher"],
        "allergens": ["milk", "eggs", "peanuts", "tree nuts", "soy"],
        "price_range": (1.99, 34.99),
        "confidence_range": (0.88, 0.96),
    },

    "Personal Care & Beauty": {
        "brands": [
            "Tom's of Maine", "Dr. Bronner's", "Burt's Bees", "Native",
            "Arm & Hammer", "Colgate", "Dove", "Cetaphil", "Neutrogena",
            "Alba Botanica",
        ],
        "products": [
            "{brand} {variant} {product_type}",
            "{brand} Natural {product_type}",
            "{brand} {product_type} {variant}",
        ],
        "flavors": [
            "Lavender", "Peppermint", "Coconut Oil", "Charcoal",
            "Sensitive Skin", "Fragrance Free", "Aloe Vera",
            "Tea Tree", "Rose", "Citrus",
        ],
        "variants": [
            "Moisturizing", "Whitening", "Natural", "Organic",
            "Sensitive", "Clarifying", "Soothing",
        ],
        "product_types": [
            "Toothpaste", "Deodorant", "Body Wash", "Shampoo",
            "Conditioner", "Lotion", "Face Wash", "Lip Balm",
            "Hand Cream", "Sunscreen",
        ],
        "sizes": [
            (4, "oz"), (8, "oz"), (12, "oz"), (16, "oz"),
            (3.5, "oz"), (6, "fl oz"), (10, "fl oz"),
        ],
        "tags": ["organic", "vegan", "cruelty-free", "non-GMO", "gluten-free"],
        "allergens": [],
        "price_range": (3.99, 24.99),
        "confidence_range": (0.82, 0.94),
    },

    "Supplements & Health": {
        "brands": [
            "Garden of Life", "Nature's Way", "Solgar", "NOW Foods",
            "Thorne", "Pure Encapsulations", "Jarrow Formulas",
            "Nordic Naturals", "Vitacost", "Rainbow Light",
        ],
        "products": [
            "{brand} {product_type} {variant}",
            "{brand} {variant} {product_type}",
            "{brand} Organic {product_type}",
        ],
        "flavors": [
            "Unflavored", "Vanilla", "Chocolate", "Berry",
            "Citrus", "Lemon", "Natural Flavor",
        ],
        "variants": [
            "1000mg", "500mg", "2000 IU", "5000 IU",
            "Complex", "Extra Strength", "Advanced",
            "Daily", "Ultra",
        ],
        "product_types": [
            "Vitamin D3", "Vitamin C", "Fish Oil", "Probiotics",
            "Magnesium", "Zinc", "B12", "Omega-3",
            "Multivitamin", "Collagen Peptides", "Whey Protein",
        ],
        "sizes": [
            (60, "ct"), (90, "ct"), (120, "ct"), (180, "ct"),
            (30, "ct"), (1, "lb"), (2, "lb"),
        ],
        "tags": [
            "gluten-free", "non-GMO", "vegan", "kosher",
            "dairy-free", "soy-free",
        ],
        "allergens": ["milk", "soy", "eggs", "fish"],
        "price_range": (8.99, 49.99),
        "confidence_range": (0.85, 0.95),
    },
}


# ── Catalog content generation ────────────────────────────────────────────────

def _generate_catalog_content(category: str, rng: random.Random) -> tuple[str, dict]:
    """
    Generate a realistic catalog_content string and its ground truth JSON.

    Mimics the structure of real Amazon grocery catalog listings:
        Item Name: <name>
        Bullet Point 1: <feature>
        Bullet Point 2: <feature>
        Value: <quantity_value>
        Unit: <raw_unit>

    Also generates the expected ProductEntity JSON output so we have
    a training pair without needing to call any LLM.

    Args:
        category: one of UNDERREPRESENTED_CATEGORIES
        rng:      seeded random.Random instance for reproducibility

    Returns:
        (catalog_content, entity_dict) tuple
    """
    tmpl = CATEGORY_TEMPLATES[category]

    # ── Build product name ─────────────────────────────────────────────────
    brand        = rng.choice(tmpl["brands"])
    name_template = rng.choice(tmpl["products"])
    flavor       = rng.choice(tmpl["flavors"])

    if category == "Personal Care & Beauty":
        variant      = rng.choice(tmpl["variants"])
        product_type = rng.choice(tmpl["product_types"])
        item_name    = (
            name_template
            .replace("{brand}", brand)
            .replace("{variant}", variant)
            .replace("{product_type}", product_type)
            .replace("{flavor}", flavor)
        )
    elif category == "Supplements & Health":
        variant      = rng.choice(tmpl["variants"])
        product_type = rng.choice(tmpl["product_types"])
        item_name    = (
            name_template
            .replace("{brand}", brand)
            .replace("{variant}", variant)
            .replace("{product_type}", product_type)
            .replace("{flavor}", flavor)
        )
    else:
        item_name = (
            name_template
            .replace("{brand}", brand)
            .replace("{flavor}", flavor)
        )

    # ── Size ───────────────────────────────────────────────────────────────
    qty_value, qty_unit = rng.choice(tmpl["sizes"])

    # Raw unit as it appears in catalog (may differ from canonical)
    raw_unit_map = {
        "oz": rng.choice(["oz", "Ounce", "ounces", "OZ"]),
        "fl oz": rng.choice(["fl oz", "Fluid Ounce", "fl. oz."]),
        "ct": rng.choice(["ct", "Count", "count", "Counts"]),
        "lb": rng.choice(["lb", "Pound", "pounds", "LB"]),
    }
    raw_unit = raw_unit_map.get(qty_unit, qty_unit)

    # ── Price ──────────────────────────────────────────────────────────────
    price = round(rng.uniform(*tmpl["price_range"]), 2)

    # ── Dietary tags ───────────────────────────────────────────────────────
    # Sample 0–3 tags from the category's common tags
    n_tags = rng.randint(0, min(3, len(tmpl["tags"])))
    tags   = rng.sample(tmpl["tags"], n_tags)

    # ── Allergens ─────────────────────────────────────────────────────────
    # 40% chance of having allergens
    if tmpl["allergens"] and rng.random() < 0.4:
        n_allergens = rng.randint(1, min(2, len(tmpl["allergens"])))
        allergens   = rng.sample(tmpl["allergens"], n_allergens)
    else:
        allergens = []

    # ── Bullet points ─────────────────────────────────────────────────────
    bullets = _generate_bullets(category, tags, allergens, rng)

    # ── Catalog content string ─────────────────────────────────────────────
    catalog_lines = [f"Item Name: {item_name}"]
    for i, bullet in enumerate(bullets, 1):
        catalog_lines.append(f"Bullet Point {i}: {bullet}")
    catalog_lines.append(f"Value: {qty_value}")
    catalog_lines.append(f"Unit: {raw_unit}")
    catalog_content = "\n".join(catalog_lines)

    # ── Ground truth entity ────────────────────────────────────────────────
    confidence = round(rng.uniform(*tmpl["confidence_range"]), 2)
    entity_dict = {
        "item_name":             item_name,
        "price":                 price,
        "quantity_value":        float(qty_value),
        "quantity_unit":         qty_unit,
        "category":              category,
        "dietary_tags":          sorted(tags),
        "allergen_list":         sorted(allergens),
        "extraction_confidence": confidence,
    }

    return catalog_content, entity_dict


def _generate_bullets(
    category: str,
    tags: list[str],
    allergens: list[str],
    rng: random.Random,
) -> list[str]:
    """
    Generate 2–4 realistic bullet point descriptions for a product.

    Bullets mention dietary tags and allergens naturally — the way
    real Amazon listings do. This teaches the model to extract these
    from inline text rather than structured fields.

    Args:
        category:  product category
        tags:      dietary tags for this product
        allergens: allergens for this product
        rng:       seeded random.Random instance

    Returns:
        List of 2–4 bullet strings.
    """
    bullets = []

    # Tag bullets
    tag_bullets = {
        "gluten-free":   "Certified gluten-free — safe for celiac and gluten-sensitive diets",
        "organic":       "USDA certified organic — no synthetic pesticides or fertilizers",
        "vegan":         "100% plant-based and vegan — no animal products",
        "non-GMO":       "Non-GMO Project Verified — made with non-genetically modified ingredients",
        "kosher":        "Kosher certified — meets strict kosher dietary standards",
        "high-protein":  "High protein content — excellent for muscle recovery and satiety",
        "dairy-free":    "Completely dairy-free — suitable for lactose intolerant individuals",
        "soy-free":      "Soy-free formula — ideal for those with soy sensitivities",
        "cruelty-free":  "Cruelty-free and never tested on animals",
    }

    for tag in tags:
        if tag in tag_bullets:
            bullets.append(tag_bullets[tag])

    # Allergen bullet
    if allergens:
        allergen_str = ", ".join(allergens)
        bullets.append(f"Contains: {allergen_str}")

    # Generic quality bullet
    generic = [
        "Made with premium quality ingredients",
        "No artificial colors or preservatives",
        "Sustainably sourced ingredients",
        "Manufactured in a GMP certified facility",
        "Third-party tested for purity and potency",
    ]
    bullets.append(rng.choice(generic))

    # Shuffle and cap at 4
    rng.shuffle(bullets)
    return bullets[:4]


# ── Quota calculation ─────────────────────────────────────────────────────────

def compute_synthetic_quota(
    real_pairs: int = DEFAULT_REAL_PAIRS,
    max_ratio: float = MAX_SYNTHETIC_RATIO,
    target_categories: Optional[list[str]] = None,
) -> dict[str, int]:
    """
    Compute how many synthetic pairs to generate per category.

    Logic:
        total_synthetic_cap = real_pairs * max_ratio  (e.g. 600)
        Divide evenly across target categories.
        Minimum 50 per category.

    Args:
        real_pairs:         number of real pairs already generated
        max_ratio:          cap as fraction of real pairs (default 0.20)
        target_categories:  which categories to augment
                            (default: UNDERREPRESENTED_CATEGORIES)

    Returns:
        Dict mapping category name → number of synthetic pairs to generate.
    """
    if target_categories is None:
        target_categories = UNDERREPRESENTED_CATEGORIES

    total_cap   = int(real_pairs * max_ratio)
    per_category = max(50, total_cap // len(target_categories))

    # Don't exceed total cap
    quotas = {}
    remaining = total_cap
    for cat in target_categories:
        allocated = min(per_category, remaining)
        quotas[cat] = allocated
        remaining  -= allocated
        if remaining <= 0:
            break

    logger.info(
        "Synthetic quota: %d total cap, %d categories → %s",
        total_cap, len(target_categories), quotas,
    )
    return quotas


# ── Pair generation ───────────────────────────────────────────────────────────

def generate_synthetic_pairs(
    quotas: dict[str, int],
    seed: int = 42,
) -> list[dict]:
    """
    Generate synthetic training pairs for all target categories.

    For each category, generates `quota` pairs using templates.
    Each pair is in the same chat fine-tuning format as real pairs:
        {
          "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "catalog_content + price"},
            {"role": "assistant", "content": "{...entity json...}"}
          ],
          "synthetic": true,
          "category":  "Protein Bars & Snacks"
        }

    The "synthetic": true flag lets downstream code filter synthetic
    pairs out if needed.

    Args:
        quotas: dict from compute_synthetic_quota()
        seed:   random seed for reproducibility

    Returns:
        List of training pair dicts.
    """
    rng   = random.Random(seed)
    pairs = []

    for category, n in quotas.items():
        if category not in CATEGORY_TEMPLATES:
            logger.warning("No template for category '%s' — skipping.", category)
            continue

        logger.info("Generating %d synthetic pairs for '%s'...", n, category)
        category_pairs = 0

        for _ in range(n):
            try:
                catalog_content, entity_dict = _generate_catalog_content(category, rng)
                price = entity_dict["price"]

                # Build messages in the same format as real pairs
                messages = build_full_messages(
                    catalog_content,
                    price=price,
                    include_few_shot=False,  # no few-shot for synthetic — keeps tokens low
                )
                pair = {
                    "messages": messages + [{
                        "role":    "assistant",
                        "content": json.dumps(entity_dict, ensure_ascii=False),
                    }],
                    "synthetic": True,
                    "category":  category,
                }
                pairs.append(pair)
                category_pairs += 1

            except Exception as e:
                logger.warning("Failed to generate synthetic pair for '%s': %s", category, e)
                continue

        logger.info("  Generated %d / %d pairs for '%s'.", category_pairs, n, category)

    logger.info("Total synthetic pairs generated: %d", len(pairs))
    return pairs


# ── Save pairs ────────────────────────────────────────────────────────────────

def save_synthetic_pairs(
    pairs: list[dict],
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Split synthetic pairs 80/20 and append to existing train/val JSONL files.

    Appends rather than overwrites — the real pairs from generate_pairs.py
    are already in these files. We just add synthetic pairs on top.

    Also writes a manifest CSV recording every synthetic pair so you know
    exactly what was added and from which category.

    Args:
        pairs:       synthetic pairs from generate_synthetic_pairs()
        train_ratio: fraction for training (default 0.80)
        seed:        random seed

    Returns:
        (n_train, n_val) counts of synthetic pairs saved.
    """
    import random as _random
    rng = _random.Random(seed)

    shuffled = list(pairs)
    rng.shuffle(shuffled)

    split_idx   = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:split_idx]
    val_pairs   = shuffled[split_idx:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Append to JSONL files ──────────────────────────────────────────────
    with open(TRAIN_JSONL, "a", encoding="utf-8") as f:
        for pair in train_pairs:
            # Strip internal fields before saving — model doesn't need them
            saveable = {"messages": pair["messages"]}
            f.write(json.dumps(saveable, ensure_ascii=False) + "\n")

    with open(VAL_JSONL, "a", encoding="utf-8") as f:
        for pair in val_pairs:
            saveable = {"messages": pair["messages"]}
            f.write(json.dumps(saveable, ensure_ascii=False) + "\n")

    # ── Write manifest ─────────────────────────────────────────────────────
    manifest_exists = SYNTHETIC_MANIFEST.exists()
    with open(SYNTHETIC_MANIFEST, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "split", "item_name"])
        if not manifest_exists:
            writer.writeheader()
        for pair in train_pairs:
            entity = json.loads(pair["messages"][-1]["content"])
            writer.writerow({
                "category":  pair["category"],
                "split":     "train",
                "item_name": entity.get("item_name", ""),
            })
        for pair in val_pairs:
            entity = json.loads(pair["messages"][-1]["content"])
            writer.writerow({
                "category":  pair["category"],
                "split":     "val",
                "item_name": entity.get("item_name", ""),
            })

    logger.info(
        "Saved %d synthetic train pairs → %s", len(train_pairs), TRAIN_JSONL
    )
    logger.info(
        "Saved %d synthetic val pairs → %s",   len(val_pairs),   VAL_JSONL
    )
    logger.info("Manifest → %s", SYNTHETIC_MANIFEST)

    return len(train_pairs), len(val_pairs)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_synthetic_generation(
    real_pairs: int = DEFAULT_REAL_PAIRS,
    target_categories: Optional[list[str]] = None,
    seed: int = 42,
) -> dict:
    """
    Run the full synthetic generation pipeline.

    Steps:
        1. Compute per-category quotas (capped at 20% of real pairs)
        2. Generate synthetic catalog content + ground truth JSON
        3. Wrap in chat fine-tuning format
        4. Split 80/20 and append to train/val JSONL
        5. Write manifest CSV

    Args:
        real_pairs:         number of real pairs (used to compute cap)
        target_categories:  categories to augment (default: underrepresented ones)
        seed:               random seed

    Returns:
        Summary dict with counts per category and totals.
    """
    logger.info("═" * 60)
    logger.info("Starting synthetic data generation")
    logger.info("Real pairs: %d — cap: %.0f%%", real_pairs, MAX_SYNTHETIC_RATIO * 100)
    logger.info("═" * 60)

    quotas = compute_synthetic_quota(
        real_pairs=real_pairs,
        target_categories=target_categories,
    )

    pairs = generate_synthetic_pairs(quotas=quotas, seed=seed)

    n_train, n_val = save_synthetic_pairs(pairs, seed=seed)

    summary = {
        "quotas":        quotas,
        "total_generated": len(pairs),
        "train_pairs":   n_train,
        "val_pairs":     n_val,
        "synthetic_cap": int(real_pairs * MAX_SYNTHETIC_RATIO),
    }

    logger.info("═" * 60)
    logger.info("SYNTHETIC GENERATION SUMMARY")
    for cat, quota in quotas.items():
        logger.info("  %-35s %d pairs", cat, quota)
    logger.info("  Total generated : %d", len(pairs))
    logger.info("  Train           : %d", n_train)
    logger.info("  Val             : %d", n_val)
    logger.info("═" * 60)

    return summary


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_synthetic_generation()