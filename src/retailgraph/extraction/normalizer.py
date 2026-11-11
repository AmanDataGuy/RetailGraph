"""
# normalizer.py - maps raw extracted values onto the canonical vocabularies

- Each normalizer tries an alias table from retail.yaml, then a fuzzy match, then gives up.
- normalize_product() runs them all; validator.py calls it before building a ProductEntity.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from retailgraph.extraction.schemas import (
    _ADAPTER,
    VALID_ALLERGENS,
    VALID_CATEGORIES,
    VALID_DIETARY_TAGS,
    VALID_UNITS,
)

_UNIT_ALIASES: dict[str, str] = _ADAPTER.get("unit_aliases", {})
_DIETARY_ALIASES: dict[str, str] = _ADAPTER.get("dietary_tag_aliases", {})
_ALLERGEN_ALIASES: dict[str, str] = _ADAPTER.get("allergen_aliases", {})
_BRAND_ALIASES: dict[str, str] = _ADAPTER.get("brand_aliases", {})

_FUZZY_THRESHOLD = 80  # below this fuzzy score we treat it as no match

_PACK_PATTERNS = [
    r"[Pp]ack\s+of\s+(\d+)",        # Pack of 12
    r"(\d+)\s*-?\s*[Pp]ack",        # 12-Pack, 12 pack
    r"(\d+)\s*[Cc]ount",            # 12 Count, 12count
    r"[Ss]et\s+of\s+(\d+)",         # Set of 6
    r"[Bb]ox\s+of\s+(\d+)",         # Box of 24
    r"[Cc]ase\s+of\s+(\d+)",        # Case of 6
    r"\((\d+)\)\s*(?:pk|pack|ct)",  # (12) pk
]


# Return the best fuzzy match from choices if it scores at least threshold, else None.
def _fuzzy(raw: str, choices: list[str], scorer=fuzz.ratio, threshold: int = _FUZZY_THRESHOLD) -> str | None:
    match = process.extractOne(raw, choices, scorer=scorer)
    return match[0] if match and match[1] >= threshold else None


# Map a raw unit string to one of the 8 canonical units, or None.
def normalize_unit(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = raw.strip()
    if raw.lower() == "none" or not raw:  # the CSV has the literal string "None"
        return None
    if raw in _UNIT_ALIASES:
        return _UNIT_ALIASES[raw]
    for alias, canonical in _UNIT_ALIASES.items():
        if alias.lower() == raw.lower():
            return canonical
    # Case-insensitive, so the prompt's lowercase "l" maps to "L" instead of being dropped.
    exact = next((u for u in VALID_UNITS if u.lower() == raw.lower()), None)
    if exact:
        return exact
    return _fuzzy(raw, VALID_UNITS)


# Map each raw string to a canonical value (alias, exact, then fuzzy) and dedupe; empty becomes None.
def _normalize_list(raw_items: list[str] | None, aliases: dict[str, str], valid: list[str],
                    try_original_case: bool = False) -> list[str] | None:
    if not raw_items:
        return None
    result: list[str] = []
    for raw in raw_items:
        if not raw:
            continue
        raw_clean = raw.strip().lower()
        canonical = aliases.get(raw_clean)
        if canonical is None and try_original_case:
            canonical = aliases.get(raw.strip())
        if canonical is None and raw_clean in valid:
            canonical = raw_clean
        if canonical is None:
            canonical = _fuzzy(raw_clean, valid)
        if canonical and canonical not in result:
            result.append(canonical)
    return result or None


# Normalize raw dietary tags to the controlled list.
def normalize_dietary_tags(raw_tags: list[str] | None) -> list[str] | None:
    return _normalize_list(raw_tags, _DIETARY_ALIASES, VALID_DIETARY_TAGS, try_original_case=True)


# Normalize raw allergens to the controlled list.
def normalize_allergens(raw_allergens: list[str] | None) -> list[str] | None:
    return _normalize_list(raw_allergens, _ALLERGEN_ALIASES, VALID_ALLERGENS)


# Map a brand via the alias table, then a strict fuzzy match; otherwise return it cleaned up.
def normalize_brand(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw.lower() in _BRAND_ALIASES:
        return _BRAND_ALIASES[raw.lower()]
    match = _fuzzy(raw, list(set(_BRAND_ALIASES.values())), scorer=fuzz.token_sort_ratio, threshold=90)
    if match is not None:
        return match
    return raw.title() if raw.islower() else raw


# Map a raw category to one of the 19 valid categories; unmatched text becomes "Unknown".
def normalize_category(raw: str | None) -> str | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    if raw in VALID_CATEGORIES:
        return raw
    return _fuzzy(raw, VALID_CATEGORIES, scorer=fuzz.token_sort_ratio) or "Unknown"


# Find a pack size such as "Pack of 12" or "24-Pack" in text; only 2..1000 counts.
def extract_pack_size(text: str | None) -> int | None:
    if not text:
        return None
    for pattern in _PACK_PATTERNS:
        match = re.search(pattern, text)
        if match and 1 < int(match.group(1)) <= 1000:
            return int(match.group(1))
    return None


# Trim, drop non-printable characters, collapse spaces and trailing punctuation; too short becomes None.
def clean_item_name(raw: str | None) -> str | None:
    if not raw:
        return None
    clean = re.sub(r"[^\x20-\x7E]", "", raw.strip())
    clean = re.sub(r"\s{2,}", " ", clean).rstrip(".,;:")
    return clean if len(clean) >= 2 else None


# Run every normalizer over a raw extracted dict; the result is ready for ProductEntity.
def normalize_product(raw: dict) -> dict:
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
