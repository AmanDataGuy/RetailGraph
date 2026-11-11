"""# generate_synthetic.py - template-made pairs for the 3 rarest categories (no API calls)

- Appends to data/training/train.jsonl / val.jsonl (80/20) and writes data/training/synthetic_manifest.csv.
- Capped at 20% of the real pair count so real data dominates; seeded, so reruns give the same pairs.
- Run after generate_pairs.py: python training/generate_synthetic.py
"""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path

from retailgraph.extraction.prompt_templates import build_full_messages

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

MAX_SYNTHETIC_RATIO = 0.20   # synthetic cap, as a fraction of real pairs
DEFAULT_REAL_PAIRS  = 3_000
TRAIN_RATIO         = 0.80

OUTPUT_DIR         = Path("data/training")
TRAIN_JSONL        = OUTPUT_DIR / "train.jsonl"
VAL_JSONL          = OUTPUT_DIR / "val.jsonl"
SYNTHETIC_MANIFEST = OUTPUT_DIR / "synthetic_manifest.csv"

UNDERREPRESENTED_CATEGORIES = [
    "Protein Bars & Snacks",
    "Personal Care & Beauty",
    "Supplements & Health",
]

# Per-category building blocks: brands, name templates, flavors/variants, (qty, unit) sizes, tags, allergens, ranges.
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


# Build one fake catalog listing and its ground-truth entity JSON (rng call order matters for reproducibility).
def _generate_catalog_content(category: str, rng: random.Random) -> tuple[str, dict]:
    tmpl = CATEGORY_TEMPLATES[category]

    brand         = rng.choice(tmpl["brands"])
    name_template = rng.choice(tmpl["products"])
    flavor        = rng.choice(tmpl["flavors"])

    if category in ("Personal Care & Beauty", "Supplements & Health"):
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
        item_name = name_template.replace("{brand}", brand).replace("{flavor}", flavor)

    qty_value, qty_unit = rng.choice(tmpl["sizes"])

    # All four choices are drawn every time so the rng sequence stays fixed.
    raw_unit_map = {
        "oz": rng.choice(["oz", "Ounce", "ounces", "OZ"]),
        "fl oz": rng.choice(["fl oz", "Fluid Ounce", "fl. oz."]),
        "ct": rng.choice(["ct", "Count", "count", "Counts"]),
        "lb": rng.choice(["lb", "Pound", "pounds", "LB"]),
    }
    raw_unit = raw_unit_map.get(qty_unit, qty_unit)

    price = round(rng.uniform(*tmpl["price_range"]), 2)

    n_tags = rng.randint(0, min(3, len(tmpl["tags"])))
    tags   = rng.sample(tmpl["tags"], n_tags)

    if tmpl["allergens"] and rng.random() < 0.4:
        allergens = rng.sample(tmpl["allergens"], rng.randint(1, min(2, len(tmpl["allergens"]))))
    else:
        allergens = []

    bullets = _generate_bullets(category, tags, allergens, rng)

    catalog_lines = [f"Item Name: {item_name}"]
    catalog_lines += [f"Bullet Point {i}: {bullet}" for i, bullet in enumerate(bullets, 1)]
    catalog_lines += [f"Value: {qty_value}", f"Unit: {raw_unit}"]
    catalog_content = "\n".join(catalog_lines)

    entity_dict = {
        "item_name":             item_name,
        "price":                 price,
        "quantity_value":        float(qty_value),
        "quantity_unit":         qty_unit,
        "category":              category,
        "dietary_tags":          sorted(tags),
        "allergen_list":         sorted(allergens),
        "extraction_confidence": round(rng.uniform(*tmpl["confidence_range"]), 2),
    }
    return catalog_content, entity_dict


# Write 2-4 shuffled bullet lines that mention the tags and allergens inline, like real listings.
def _generate_bullets(
    category: str,
    tags: list[str],
    allergens: list[str],
    rng: random.Random,
) -> list[str]:
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
    bullets = [tag_bullets[tag] for tag in tags if tag in tag_bullets]
    if allergens:
        bullets.append(f"Contains: {', '.join(allergens)}")

    generic = [
        "Made with premium quality ingredients",
        "No artificial colors or preservatives",
        "Sustainably sourced ingredients",
        "Manufactured in a GMP certified facility",
        "Third-party tested for purity and potency",
    ]
    bullets.append(rng.choice(generic))

    rng.shuffle(bullets)
    return bullets[:4]


# Split the synthetic cap evenly across categories (min 50 each, never above the total cap).
def compute_synthetic_quota(
    real_pairs: int = DEFAULT_REAL_PAIRS,
    max_ratio: float = MAX_SYNTHETIC_RATIO,
    target_categories: list[str] | None = None,
) -> dict[str, int]:
    if target_categories is None:
        target_categories = UNDERREPRESENTED_CATEGORIES

    total_cap    = int(real_pairs * max_ratio)
    per_category = max(50, total_cap // len(target_categories))

    quotas = {}
    remaining = total_cap
    for cat in target_categories:
        allocated = min(per_category, remaining)
        quotas[cat] = allocated
        remaining  -= allocated
        if remaining <= 0:
            break

    logger.info("Synthetic quota: %d total cap, %d categories → %s", total_cap, len(target_categories), quotas)
    return quotas


# Make the synthetic pairs in the same chat format as real pairs, tagged with "synthetic" and "category".
def generate_synthetic_pairs(quotas: dict[str, int], seed: int = 42) -> list[dict]:
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
                messages = build_full_messages(catalog_content, price=entity_dict["price"], include_few_shot=False)
                pairs.append({
                    "messages": messages + [{
                        "role":    "assistant",
                        "content": json.dumps(entity_dict, ensure_ascii=False),
                    }],
                    "synthetic": True,
                    "category":  category,
                })
                category_pairs += 1
            except Exception as e:
                logger.warning("Failed to generate synthetic pair for '%s': %s", category, e)

        logger.info("  Generated %d / %d pairs for '%s'.", category_pairs, n, category)

    logger.info("Total synthetic pairs generated: %d", len(pairs))
    return pairs


# Split 80/20, append only "messages" to train/val JSONL, and append each pair to the manifest CSV.
def save_synthetic_pairs(pairs: list[dict], train_ratio: float = TRAIN_RATIO, seed: int = 42) -> tuple[int, int]:
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    split_idx   = int(len(shuffled) * train_ratio)
    train_pairs = shuffled[:split_idx]
    val_pairs   = shuffled[split_idx:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, split in ((TRAIN_JSONL, train_pairs), (VAL_JSONL, val_pairs)):
        with open(path, "a", encoding="utf-8") as f:
            for pair in split:
                f.write(json.dumps({"messages": pair["messages"]}, ensure_ascii=False) + "\n")

    manifest_exists = SYNTHETIC_MANIFEST.exists()
    with open(SYNTHETIC_MANIFEST, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "split", "item_name"])
        if not manifest_exists:
            writer.writeheader()
        for split_name, split in (("train", train_pairs), ("val", val_pairs)):
            for pair in split:
                entity = json.loads(pair["messages"][-1]["content"])
                writer.writerow({
                    "category":  pair["category"],
                    "split":     split_name,
                    "item_name": entity.get("item_name", ""),
                })

    logger.info("Saved %d synthetic train pairs → %s", len(train_pairs), TRAIN_JSONL)
    logger.info("Saved %d synthetic val pairs → %s", len(val_pairs), VAL_JSONL)
    logger.info("Manifest → %s", SYNTHETIC_MANIFEST)
    return len(train_pairs), len(val_pairs)


# Full pipeline: quotas -> synthetic pairs -> append to train/val + manifest; returns a summary dict.
def run_synthetic_generation(
    real_pairs: int = DEFAULT_REAL_PAIRS,
    target_categories: list[str] | None = None,
    seed: int = 42,
) -> dict:
    logger.info("Starting synthetic data generation — real pairs: %d, cap: %.0f%%",
                real_pairs, MAX_SYNTHETIC_RATIO * 100)

    quotas = compute_synthetic_quota(real_pairs=real_pairs, target_categories=target_categories)
    pairs = generate_synthetic_pairs(quotas=quotas, seed=seed)
    n_train, n_val = save_synthetic_pairs(pairs, seed=seed)

    for cat, quota in quotas.items():
        logger.info("  %-35s %d pairs", cat, quota)
    logger.info("Total generated: %d (train %d / val %d)", len(pairs), n_train, n_val)

    return {
        "quotas":          quotas,
        "total_generated": len(pairs),
        "train_pairs":     n_train,
        "val_pairs":       n_val,
        "synthetic_cap":   int(real_pairs * MAX_SYNTHETIC_RATIO),
    }


if __name__ == "__main__":
    run_synthetic_generation()
