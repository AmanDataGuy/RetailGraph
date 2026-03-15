"""
scripts/create_indexes.py

Creates all Neo4j constraints and indexes for RetailGraph.
Run ONCE before any data ingestion.

Usage:
    python scripts/create_indexes.py
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ── Uniqueness Constraints ─────────────────────────────────────────────────────
# Prevents duplicate nodes — safe to re-run (IF NOT EXISTS)
CONSTRAINTS = [
    "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE",
    "CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:DietaryTag) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT allergen_name IF NOT EXISTS FOR (a:Allergen) REQUIRE a.name IS UNIQUE",
]

# ── Property Indexes ───────────────────────────────────────────────────────────
# Speeds up common query filters
INDEXES = [
    "CREATE INDEX product_price IF NOT EXISTS FOR (p:Product) ON (p.price)",
    "CREATE INDEX product_confidence IF NOT EXISTS FOR (p:Product) ON (p.extraction_confidence)",
    "CREATE INDEX product_quantity_unit IF NOT EXISTS FOR (p:Product) ON (p.quantity_unit)",
]

# ── Fulltext Indexes ───────────────────────────────────────────────────────────
# Enables fuzzy name search — used by the agent for brand/product lookups
FULLTEXT = [
    "CREATE FULLTEXT INDEX product_name_ft IF NOT EXISTS FOR (p:Product) ON EACH [p.item_name]",
    "CREATE FULLTEXT INDEX brand_name_ft IF NOT EXISTS FOR (b:Brand) ON EACH [b.name]",
]


def run(driver):
    with driver.session(database=DATABASE) as session:

        print("Creating uniqueness constraints...")
        for stmt in CONSTRAINTS:
            session.run(stmt)
            label = stmt.split("CONSTRAINT ")[1].split(" ")[0]
            print(f"  ✅ {label}")

        print("\nCreating property indexes...")
        for stmt in INDEXES:
            session.run(stmt)
            label = stmt.split("INDEX ")[1].split(" ")[0]
            print(f"  ✅ {label}")

        print("\nCreating fulltext indexes...")
        for stmt in FULLTEXT:
            session.run(stmt)
            label = stmt.split("INDEX ")[1].split(" ")[0]
            print(f"  ✅ {label}")

        # Verify
        print("\nVerifying...")
        result = session.run("SHOW CONSTRAINTS")
        constraints = [r["name"] for r in result]
        print(f"  Total constraints: {len(constraints)}")

        result = session.run("SHOW INDEXES")
        indexes = [r["name"] for r in result]
        print(f"  Total indexes:     {len(indexes)}")

        print("\n✅ All constraints and indexes created. Ready for ingestion.")


if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    driver.verify_connectivity()
    run(driver)
    driver.close()