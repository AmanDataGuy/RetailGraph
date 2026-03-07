from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml
from rapidfuzz import fuzz, process


# ── Load adapter once at module level ────────────────────────────────────────
def _load_adapter(domain: str = "retail") -> dict:
    path = Path(__file__).parent.parent / "domain_adapter" / f"{domain}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_ADAPTER = _load_adapter("retail")

# Pull lookup tables out of adapter
_UNIT_ALIASES: dict[str, str] = _ADAPTER.get("unit_aliases", {})
_DIETARY_ALIASES: dict[str, str] = _ADAPTER.get("dietary_tag_aliases", {})
_ALLERGEN_ALIASES: dict[str, str] = _ADAPTER.get("allergen_aliases", {})
_BRAND_ALIASES: dict[str, str] = _ADAPTER.get("brand_aliases", {})

_VALID_UNITS: list[str] = ["oz", "fl oz", "lb", "ct", "g", "kg", "ml", "L"]
_VALID_CATEGORIES: list[str] = [
    "Coffee & Tea", "Breakfast & Cereal", "Meat & Seafood",
    "Soups & Canned Goods", "Pasta & Noodles", "Bread & Bakery",
    "Protein Bars & Snacks", "Supplements & Health",
    "Grains, Beans & Legumes", "Oils & Vinegars", "Nuts & Seeds",
    "Personal Care & Beauty", "Spices & Seasonings",
    "Condiments & Sauces", "Baking & Cooking", "Snacks & Candy",
    "Beverages", "Non-Food", "Unknown",
]
_VALID_DIETARY_TAGS: list[str] = _ADAPTER.get("dietary_tags_controlled", [])
_VALID_ALLERGENS: list[str] = _ADAPTER.get("allergens_controlled", [])

# Fuzzy match threshold — below this score we return None
_FUZZY_THRESHOLD = 80


# ── Unit normalization ────────────────────────────────────────────────────────
def normalize_unit(raw: Optional[str]) -> Optional[str]:
    """
    Normalize a raw unit string to one of 8 canonical units.
    Layer 1: exact match against unit_aliases
    Layer 2: fuzzy match against canonical units
    Layer 3: return None
    """
    if raw is None:
        return None

    raw = raw.strip()

    # handle string "None" from CSV
    if raw.lower() == "none" or not raw:
        return None

    # Layer 1 — exact alias match
    if raw in _UNIT_ALIASES:
        return _UNIT_ALIASES[raw]

    # Layer 1b — case-insensitive alias match
    raw_lower = raw.lower()
    for alias, canonical in _UNIT_ALIASES.items():
        if alias.lower() == raw_lower:
            return canonical

    # already canonical — passthrough
    if raw in _VALID_UNITS:
        return raw

    # Layer 2 — fuzzy match against canonical units
    result = process.extractOne(
        raw, _VALID_UNITS, scorer=fuzz.ratio
    )
    if result and result[1] >= _FUZZY_THRESHOLD:
        return result[0]

    # Layer 3 — give up
    return None


# ── Dietary tag normalization ─────────────────────────────────────────────────
def normalize_dietary_tags(raw_tags: Optional[list[str]]) -> Optional[list[str]]:
    """
    Normalize a list of raw dietary tag strings to canonical forms.
    Returns deduplicated list of valid canonical tags or None.
    """
    if not raw_tags:
        return None

    seen: set[str] = set()
    result: list[str] = []

    for raw in raw_tags:
        if not raw:
            continue

        raw_clean = raw.strip().lower()

        # Layer 1 — exact alias match
        canonical = _DIETARY_ALIASES.get(raw_clean)

        # Layer 1b — try original case
        if canonical is None:
            canonical = _DIETARY_ALIASES.get(raw.strip())

        # already canonical
        if canonical is None and raw_clean in _VALID_DIETARY_TAGS:
            canonical = raw_clean

        # Layer 2 — fuzzy match against valid tags
        if canonical is None:
            match = process.extractOne(
                raw_clean, _VALID_DIETARY_TAGS, scorer=fuzz.ratio
            )
            if match and match[1] >= _FUZZY_THRESHOLD:
                canonical = match[0]

        # add if found and not duplicate
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    return result if result else None


# ── Allergen normalization ─────────────────────────────────────────────────────
def normalize_allergens(raw_allergens: Optional[list[str]]) -> Optional[list[str]]:
    """
    Normalize a list of raw allergen strings to canonical forms.
    """
    if not raw_allergens:
        return None

    seen: set[str] = set()
    result: list[str] = []

    for raw in raw_allergens:
        if not raw:
            continue

        raw_clean = raw.strip().lower()

        # Layer 1 — exact alias match
        canonical = _ALLERGEN_ALIASES.get(raw_clean)

        # already canonical
        if canonical is None and raw_clean in _VALID_ALLERGENS:
            canonical = raw_clean

        # Layer 2 — fuzzy match
        if canonical is None:
            match = process.extractOne(
                raw_clean, _VALID_ALLERGENS, scorer=fuzz.ratio
            )
            if match and match[1] >= _FUZZY_THRESHOLD:
                canonical = match[0]

        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)

    return result if result else None


# ── Brand normalization ───────────────────────────────────────────────────────
def normalize_brand(raw: Optional[str]) -> Optional[str]:
    """
    Normalize a raw brand string using alias table then fuzzy match.
    """
    if raw is None:
        return None

    raw = raw.strip()
    if not raw:
        return None

    raw_lower = raw.lower()

    # Layer 1 — exact alias match (lowercase)
    if raw_lower in _BRAND_ALIASES:
        return _BRAND_ALIASES[raw_lower]

    # Layer 2 — fuzzy match against known brand values
    known_brands = list(set(_BRAND_ALIASES.values()))
    match = process.extractOne(
        raw, known_brands, scorer=fuzz.token_sort_ratio
    )
    if match and match[1] >= 90:  # higher threshold for brands
        return match[0]

    # Layer 3 — return as-is, cleaned up
    return raw.title() if raw.islower() else raw


# ── Category normalization ─────────────────────────────────────────────────────
def normalize_category(raw: Optional[str]) -> Optional[str]:
    """
    Normalize a raw category string to one of 19 valid categories.
    """
    if raw is None:
        return None

    raw = raw.strip()
    if not raw:
        return None

    # Layer 1 — exact match
    if raw in _VALID_CATEGORIES:
        return raw

    # Layer 2 — fuzzy match
    match = process.extractOne(
        raw, _VALID_CATEGORIES, scorer=fuzz.token_sort_ratio
    )
    if match and match[1] >= _FUZZY_THRESHOLD:
        return match[0]

    return "Unknown"


# ── Pack size extraction ──────────────────────────────────────────────────────
_PACK_PATTERNS = [
    r"[Pp]ack\s+of\s+(\d+)",       # Pack of 12
    r"(\d+)\s*-?\s*[Pp]ack",       # 12-Pack, 12 pack
    r"(\d+)\s*[Cc]ount",           # 12 Count, 12count
    r"[Ss]et\s+of\s+(\d+)",        # Set of 6
    r"[Bb]ox\s+of\s+(\d+)",        # Box of 24
    r"[Cc]ase\s+of\s+(\d+)",       # Case of 6
    r"\((\d+)\)\s*(?:pk|pack|ct)", # (12) pk
]

def extract_pack_size(text: Optional[str]) -> Optional[int]:
    """
    Extract pack size integer from product text.
    Returns None if no pack size pattern found.
    """
    if not text:
        return None

    for pattern in _PACK_PATTERNS:
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            # sanity check — pack sizes over 1000 are almost certainly wrong
            if 1 < value <= 1000:
                return value

    return None


# ── Item name cleaning ────────────────────────────────────────────────────────
def clean_item_name(raw: Optional[str]) -> Optional[str]:
    """
    Clean raw item name — strip whitespace, collapse internal spaces,
    remove non-printable characters.
    """
    if not raw:
        return None

    # strip leading/trailing whitespace
    clean = raw.strip()

    # remove non-printable chars
    clean = re.sub(r"[^\x20-\x7E]", "", clean)

    # collapse multiple spaces
    clean = re.sub(r"\s{2,}", " ", clean)

    # remove trailing punctuation artifacts
    clean = clean.rstrip(".,;:")

    if len(clean) < 2:
        return None

    return clean


# ── Full product normalization ────────────────────────────────────────────────
def normalize_product(raw: dict) -> dict:
    """
    Run all normalization on a raw extracted product dict.
    Returns a cleaned dict ready for ProductEntity validation.
    """
    return {
        "product_id":   str(raw.get("product_id", "")).strip(),
        "item_name":    clean_item_name(raw.get("item_name")),
        "brand":        normalize_brand(raw.get("brand")),
        "category":     normalize_category(raw.get("category")),
        "quantity_value": raw.get("quantity_value"),
        "quantity_unit": normalize_unit(raw.get("quantity_unit")),
        "pack_size":    raw.get("pack_size") or extract_pack_size(raw.get("item_name")),
        "price":        raw.get("price"),
        "dietary_tags": normalize_dietary_tags(raw.get("dietary_tags")),
        "allergen_list": normalize_allergens(raw.get("allergen_list")),
        "description":  raw.get("description"),
        "bullet_points": raw.get("bullet_points"),
        "ingredients":  raw.get("ingredients"),
        "visual":       raw.get("visual"),
        "image_url":    raw.get("image_url"),
        "extraction_confidence": raw.get("extraction_confidence", 0.0),
    }