"""
src/graph/deduplicator.py

Deduplicates products before Neo4j ingestion.

Stage 1 — Exact match:  item_name + brand + quantity_value + quantity_unit
Stage 2 — Fuzzy match:  RapidFuzz token_sort_ratio >= 92 on item_name

Usage:
    from src.graph.deduplicator import deduplicate
    clean, duplicates = deduplicate(products)
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 92          # % similarity to flag as probable duplicate
DUPLICATES_LOG  = Path("data/extracted/duplicates.csv")


@dataclass
class DuplicatePair:
    original_id:   str
    duplicate_id:  str
    original_name: str
    duplicate_name: str
    match_type:    str   # "exact" or "fuzzy"
    similarity:    float


def _exact_key(product: dict) -> str:
    """Build a normalized exact-match key from a product prediction."""
    pred = product.get("prediction", {})
    name  = (pred.get("item_name") or "").lower().strip()
    brand = (pred.get("brand") or "").lower().strip()
    qty   = str(pred.get("quantity_value") or "")
    unit  = (pred.get("quantity_unit") or "").lower().strip()
    return f"{name}|{brand}|{qty}|{unit}"


def _normalize_name(name: Optional[str]) -> str:
    """Lowercase and strip for fuzzy comparison."""
    return (name or "").lower().strip()


def deduplicate(products: list[dict]) -> tuple[list[dict], list[DuplicatePair]]:
    """
    Deduplicate a list of extraction dicts.

    Returns:
        clean      — list of products with exact duplicates removed
        dup_pairs  — all duplicate pairs found (exact + fuzzy)
    """
    seen_keys:  dict[str, str] = {}   # exact_key → sample_id
    seen_names: list[tuple[str, str]] = []  # (normalized_name, sample_id)

    clean:     list[dict]          = []
    dup_pairs: list[DuplicatePair] = []
    skipped:   set[str]            = set()

    for product in products:
        sid  = product.get("sample_id", "unknown")
        pred = product.get("prediction", {})
        name = _normalize_name(pred.get("item_name"))
        key  = _exact_key(product)

        # ── Stage 1: Exact match ──────────────────────────────────────────────
        if key in seen_keys:
            original_sid = seen_keys[key]
            dup_pairs.append(DuplicatePair(
                original_id   = original_sid,
                duplicate_id  = sid,
                original_name = pred.get("item_name", ""),
                duplicate_name = pred.get("item_name", ""),
                match_type    = "exact",
                similarity    = 100.0,
            ))
            skipped.add(sid)
            continue

        # ── Stage 2: Fuzzy match ──────────────────────────────────────────────
        fuzzy_match_found = False
        if name:
            for existing_name, existing_sid in seen_names:
                score = fuzz.token_sort_ratio(name, existing_name)
                if score >= FUZZY_THRESHOLD:
                    dup_pairs.append(DuplicatePair(
                        original_id   = existing_sid,
                        duplicate_id  = sid,
                        original_name = existing_name,
                        duplicate_name = name,
                        match_type    = "fuzzy",
                        similarity    = score,
                    ))
                    fuzzy_match_found = True
                    break  # flag once, keep the product (don't skip fuzzy)

        # Add to clean list (fuzzy duplicates kept — just flagged)
        seen_keys[key] = sid
        if name:
            seen_names.append((name, sid))
        clean.append(product)

    return clean, dup_pairs


def log_duplicates(dup_pairs: list[DuplicatePair]) -> None:
    """Save duplicate pairs to CSV for review."""
    DUPLICATES_LOG.parent.mkdir(parents=True, exist_ok=True)

    with open(DUPLICATES_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "match_type", "similarity",
            "original_id", "original_name",
            "duplicate_id", "duplicate_name",
        ])
        writer.writeheader()
        for pair in sorted(dup_pairs, key=lambda x: -x.similarity):
            writer.writerow({
                "match_type":    pair.match_type,
                "similarity":    f"{pair.similarity:.1f}",
                "original_id":   pair.original_id,
                "original_name": pair.original_name,
                "duplicate_id":  pair.duplicate_id,
                "duplicate_name": pair.duplicate_name,
            })

    print(f"  Duplicate log saved → {DUPLICATES_LOG}")


def run(input_file: str = "data/training/verified_extractions.jsonl") -> list[dict]:
    """
    Load extractions, deduplicate, log results, return clean list.
    Called by builder.py before ingestion.
    """
    print(f"Loading extractions from {input_file}...")
    products = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    products.append(json.loads(line))
                except Exception:
                    continue

    print(f"  Loaded: {len(products)} products")

    clean, dup_pairs = deduplicate(products)

    exact_dups = [d for d in dup_pairs if d.match_type == "exact"]
    fuzzy_dups = [d for d in dup_pairs if d.match_type == "fuzzy"]

    print(f"\nDeduplication results:")
    print(f"  Original count:      {len(products)}")
    print(f"  Exact duplicates:    {len(exact_dups)} (removed)")
    print(f"  Fuzzy duplicates:    {len(fuzzy_dups)} (flagged, kept)")
    print(f"  Clean count:         {len(clean)}")

    if dup_pairs:
        log_duplicates(dup_pairs)
        print(f"\nTop fuzzy matches:")
        for pair in sorted(fuzzy_dups, key=lambda x: -x.similarity)[:5]:
            print(f"  [{pair.similarity:.0f}%] '{pair.original_name}' ↔ '{pair.duplicate_name}'")

    return clean


if __name__ == "__main__":
    clean = run()
    print(f"\n✅ Ready to ingest {len(clean)} clean products into Neo4j.")