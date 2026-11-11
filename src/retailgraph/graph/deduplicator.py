"""
# deduplicator.py - removes duplicate products before Neo4j ingestion

- Exact match (name + brand + quantity + unit) is dropped; fuzzy name match (>= 92) is only flagged.
- `run()` is called by builder.py; also runnable as `python -m retailgraph.graph.deduplicator`.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 92          # % similarity to flag as probable duplicate
DUPLICATES_LOG  = Path("data/extracted/duplicates.csv")


# One duplicate found: which product it copies and how close the match was.
@dataclass
class DuplicatePair:
    original_id:   str
    duplicate_id:  str
    original_name: str
    duplicate_name: str
    match_type:    str   # "exact" or "fuzzy"
    similarity:    float


# Build the normalized key two products must share to count as exact duplicates.
def _exact_key(product: dict) -> str:
    pred = product.get("prediction", {})
    name  = (pred.get("item_name") or "").lower().strip()
    brand = (pred.get("brand") or "").lower().strip()
    qty   = str(pred.get("quantity_value") or "")
    unit  = (pred.get("quantity_unit") or "").lower().strip()
    return f"{name}|{brand}|{qty}|{unit}"


# Drop exact duplicates and flag fuzzy ones; return (clean products, all duplicate pairs).
def deduplicate(products: list[dict]) -> tuple[list[dict], list[DuplicatePair]]:
    seen_keys:  dict[str, str] = {}         # exact_key -> sample_id
    seen_names: list[tuple[str, str]] = []  # (normalized_name, sample_id)
    clean:     list[dict]          = []
    dup_pairs: list[DuplicatePair] = []

    for product in products:
        sid  = product.get("sample_id", "unknown")
        pred = product.get("prediction", {})
        name = (pred.get("item_name") or "").lower().strip()
        key  = _exact_key(product)

        if key in seen_keys:
            dup_pairs.append(DuplicatePair(
                original_id   = seen_keys[key],
                duplicate_id  = sid,
                original_name = pred.get("item_name", ""),
                duplicate_name = pred.get("item_name", ""),
                match_type    = "exact",
                similarity    = 100.0,
            ))
            continue

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
                    break  # flag once; fuzzy duplicates are kept

        seen_keys[key] = sid
        if name:
            seen_names.append((name, sid))
        clean.append(product)

    return clean, dup_pairs


# Save duplicate pairs to CSV for manual review.
def log_duplicates(dup_pairs: list[DuplicatePair]) -> None:
    DUPLICATES_LOG.parent.mkdir(parents=True, exist_ok=True)
    fields = ["match_type", "similarity", "original_id", "original_name",
              "duplicate_id", "duplicate_name"]
    with open(DUPLICATES_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pair in sorted(dup_pairs, key=lambda x: -x.similarity):
            row = {k: getattr(pair, k) for k in fields}
            row["similarity"] = f"{pair.similarity:.1f}"
            writer.writerow(row)

    print(f"  Duplicate log saved → {DUPLICATES_LOG}")


# Load the JSONL extractions, deduplicate, log the duplicates, return the clean list.
def run(input_file: str = "data/training/verified_extractions.jsonl") -> list[dict]:
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

    print("\nDeduplication results:")
    print(f"  Original count:      {len(products)}")
    print(f"  Exact duplicates:    {len(exact_dups)} (removed)")
    print(f"  Fuzzy duplicates:    {len(fuzzy_dups)} (flagged, kept)")
    print(f"  Clean count:         {len(clean)}")

    if dup_pairs:
        log_duplicates(dup_pairs)
        print("\nTop fuzzy matches:")
        for pair in sorted(fuzzy_dups, key=lambda x: -x.similarity)[:5]:
            print(f"  [{pair.similarity:.0f}%] '{pair.original_name}' ↔ '{pair.duplicate_name}'")

    return clean


if __name__ == "__main__":
    clean = run()
    print(f"\n✅ Ready to ingest {len(clean)} clean products into Neo4j.")
