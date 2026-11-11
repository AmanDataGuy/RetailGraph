"""
# create_indexes.py - creates the Neo4j uniqueness constraints and indexes RetailGraph needs

- Safe to re-run (IF NOT EXISTS). Run once before ingestion: `python scripts/create_indexes.py`.
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# One node per id/name.
CONSTRAINTS = [
    "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE",
    "CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:DietaryTag) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT allergen_name IF NOT EXISTS FOR (a:Allergen) REQUIRE a.name IS UNIQUE",
]

# Speed up common filters.
INDEXES = [
    "CREATE INDEX product_price IF NOT EXISTS FOR (p:Product) ON (p.price)",
    "CREATE INDEX product_confidence IF NOT EXISTS FOR (p:Product) ON (p.extraction_confidence)",
    "CREATE INDEX product_quantity_unit IF NOT EXISTS FOR (p:Product) ON (p.quantity_unit)",
]

# Fuzzy name search for brand/product lookups.
FULLTEXT = [
    "CREATE FULLTEXT INDEX product_name_ft IF NOT EXISTS FOR (p:Product) ON EACH [p.item_name]",
    "CREATE FULLTEXT INDEX brand_name_ft IF NOT EXISTS FOR (b:Brand) ON EACH [b.name]",
]


# Create every constraint and index, then print how many exist.
def run(driver):
    steps = [
        ("Creating uniqueness constraints...", CONSTRAINTS, "CONSTRAINT "),
        ("\nCreating property indexes...", INDEXES, "INDEX "),
        ("\nCreating fulltext indexes...", FULLTEXT, "INDEX "),
    ]
    with driver.session(database=DATABASE) as session:
        for title, statements, keyword in steps:
            print(title)
            for stmt in statements:
                session.run(stmt)
                print(f"  ✅ {stmt.split(keyword)[1].split(' ')[0]}")

        print("\nVerifying...")
        print(f"  Total constraints: {len(list(session.run('SHOW CONSTRAINTS')))}")
        print(f"  Total indexes:     {len(list(session.run('SHOW INDEXES')))}")

        print("\n✅ All constraints and indexes created. Ready for ingestion.")


if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    driver.verify_connectivity()
    run(driver)
    driver.close()
