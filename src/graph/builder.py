"""
src/graph/builder.py

Ingests clean ProductEntity extractions into Neo4j AuraDB.

Nodes:    Product, Brand, Category, DietaryTag, Allergen
Edges:    BELONGS_TO, MADE_BY, HAS_TAG, CONTAINS_ALLERGEN

Usage:
    python src/graph/builder.py
    python src/graph/builder.py --input data/training/verified_extractions.jsonl
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.graph.deduplicator import run as deduplicate

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

BATCH_SIZE = 500


# ── Quality Score ─────────────────────────────────────────────────────────────

def compute_quality_score(pred: dict) -> int:
    """
    0-100 quality score per product.
      40 pts — field completeness
      40 pts — extraction confidence
      20 pts — key fields present (item_name, price, category)
    """
    SCORED_FIELDS = [
        "item_name", "brand", "category", "quantity_value",
        "quantity_unit", "price", "dietary_tags", "allergen_list",
        "extraction_confidence"
    ]
    filled = sum(1 for f in SCORED_FIELDS if pred.get(f) not in (None, [], ""))
    completeness = int((filled / len(SCORED_FIELDS)) * 40)

    confidence = pred.get("extraction_confidence") or 0.0
    conf_score = int(min(confidence, 1.0) * 40)

    key_fields = all([
        pred.get("item_name"),
        pred.get("price") is not None,
        pred.get("category"),
    ])
    key_score = 20 if key_fields else 0

    return completeness + conf_score + key_score


# ── Cypher Statements ─────────────────────────────────────────────────────────

CREATE_PRODUCT = """
MERGE (p:Product {product_id: $product_id})
SET p.item_name            = $item_name,
    p.price                = $price,
    p.quantity_value       = $quantity_value,
    p.quantity_unit        = $quantity_unit,
    p.pack_size            = $pack_size,
    p.dietary_tags         = $dietary_tags,
    p.allergen_list        = $allergen_list,
    p.extraction_confidence = $extraction_confidence,
    p.quality_score        = $quality_score,
    p.catalog_content      = $catalog_content
"""

CREATE_BRAND_REL = """
MERGE (b:Brand {name: $brand_name})
WITH b
MATCH (p:Product {product_id: $product_id})
MERGE (p)-[:MADE_BY]->(b)
"""

CREATE_CATEGORY_REL = """
MERGE (c:Category {name: $category_name})
WITH c
MATCH (p:Product {product_id: $product_id})
MERGE (p)-[:BELONGS_TO]->(c)
"""

CREATE_TAG_REL = """
MERGE (t:DietaryTag {name: $tag_name})
WITH t
MATCH (p:Product {product_id: $product_id})
MERGE (p)-[:HAS_TAG]->(t)
"""

CREATE_ALLERGEN_REL = """
MERGE (a:Allergen {name: $allergen_name})
WITH a
MATCH (p:Product {product_id: $product_id})
MERGE (p)-[:CONTAINS_ALLERGEN]->(a)
"""


# ── Batch Ingestion ───────────────────────────────────────────────────────────

def ingest_batch(session, batch: list[dict]) -> dict:
    """Ingest a single batch of products. Returns counts."""
    counts = {"products": 0, "brands": 0, "tags": 0, "allergens": 0, "categories": 0}

    for product in batch:
        pred = product.get("prediction", {})
        sid  = product.get("sample_id", "unknown")

        # Normalize fields
        product_id   = pred.get("product_id") or sid
        item_name    = pred.get("item_name") or ""
        price        = pred.get("price")
        qty_value    = pred.get("quantity_value")
        qty_unit     = pred.get("quantity_unit") or ""
        pack_size    = pred.get("pack_size")
        dietary_tags = pred.get("dietary_tags") or []
        allergens    = pred.get("allergen_list") or []
        confidence   = pred.get("extraction_confidence") or 0.0
        brand        = pred.get("brand") or ""
        category     = pred.get("category") or "Unknown"
        catalog      = product.get("catalog_content") or ""
        quality      = compute_quality_score(pred)

        # Create Product node
        session.run(CREATE_PRODUCT, {
            "product_id":           product_id,
            "item_name":            item_name,
            "price":                float(price) if price is not None else None,
            "quantity_value":       float(qty_value) if qty_value is not None else None,
            "quantity_unit":        qty_unit,
            "pack_size":            int(pack_size) if pack_size is not None else None,
            "dietary_tags":         dietary_tags,
            "allergen_list":        allergens,
            "extraction_confidence": float(confidence),
            "quality_score":        quality,
            "catalog_content":      catalog[:500],  # truncate to save storage
        })
        counts["products"] += 1

        # Brand relationship
        if brand and brand.strip():
            session.run(CREATE_BRAND_REL, {
                "product_id": product_id,
                "brand_name": brand.strip(),
            })
            counts["brands"] += 1

        # Category relationship
        if category:
            session.run(CREATE_CATEGORY_REL, {
                "product_id":    product_id,
                "category_name": category,
            })
            counts["categories"] += 1

        # DietaryTag relationships
        for tag in dietary_tags:
            if tag and tag.strip():
                session.run(CREATE_TAG_REL, {
                    "product_id": product_id,
                    "tag_name":   tag.strip(),
                })
                counts["tags"] += 1

        # Allergen relationships
        for allergen in allergens:
            if allergen and allergen.strip():
                session.run(CREATE_ALLERGEN_REL, {
                    "product_id":   product_id,
                    "allergen_name": allergen.strip().lower(),
                })
                counts["allergens"] += 1

    return counts


def ingest(products: list[dict], driver) -> None:
    """Ingest all products in batches of BATCH_SIZE."""
    total = len(products)
    totals = {"products": 0, "brands": 0, "tags": 0, "allergens": 0, "categories": 0}

    print(f"\nIngesting {total} products into Neo4j (batch size={BATCH_SIZE})...")

    with driver.session(database=DATABASE) as session:
        for i in range(0, total, BATCH_SIZE):
            batch = products[i:i + BATCH_SIZE]
            counts = ingest_batch(session, batch)
            for k in totals:
                totals[k] += counts[k]
            end = min(i + BATCH_SIZE, total)
            print(f"  [{end}/{total}] batch complete")

    print(f"\nIngestion complete!")
    print(f"  Products ingested:    {totals['products']}")
    print(f"  Brand relationships:  {totals['brands']}")
    print(f"  Category rels:        {totals['categories']}")
    print(f"  DietaryTag rels:      {totals['tags']}")
    print(f"  Allergen rels:        {totals['allergens']}")
    print(f"  Total relationships:  {sum(totals.values()) - totals['products']}")


# ── Graph Summary ─────────────────────────────────────────────────────────────

def print_summary(driver) -> None:
    """Print node and relationship counts after ingestion."""
    queries = {
        "Product nodes":      "MATCH (p:Product) RETURN count(p) AS n",
        "Brand nodes":        "MATCH (b:Brand) RETURN count(b) AS n",
        "Category nodes":     "MATCH (c:Category) RETURN count(c) AS n",
        "DietaryTag nodes":   "MATCH (t:DietaryTag) RETURN count(t) AS n",
        "Allergen nodes":     "MATCH (a:Allergen) RETURN count(a) AS n",
        "MADE_BY rels":       "MATCH ()-[:MADE_BY]->() RETURN count(*) AS n",
        "BELONGS_TO rels":    "MATCH ()-[:BELONGS_TO]->() RETURN count(*) AS n",
        "HAS_TAG rels":       "MATCH ()-[:HAS_TAG]->() RETURN count(*) AS n",
        "CONTAINS_ALLERGEN":  "MATCH ()-[:CONTAINS_ALLERGEN]->() RETURN count(*) AS n",
    }

    print("\n── Graph Summary ─────────────────────────────────────────")
    with driver.session(database=DATABASE) as session:
        for label, query in queries.items():
            result = session.run(query)
            count = result.single()["n"]
            print(f"  {label:<25} {count:>6}")
    print("──────────────────────────────────────────────────────────")
    print("✅ Graph is ready.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/training/verified_extractions.jsonl",
        help="Path to verified extractions JSONL file"
    )
    args = parser.parse_args()

    # Step 1 — Deduplicate
    clean_products = deduplicate(args.input)

    # Step 2 — Connect
    print(f"\nConnecting to Neo4j...")
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    driver.verify_connectivity()
    print("  ✅ Connected")

    # Step 3 — Ingest
    ingest(clean_products, driver)

    # Step 4 — Summary
    print_summary(driver)

    driver.close()


if __name__ == "__main__":
    main()