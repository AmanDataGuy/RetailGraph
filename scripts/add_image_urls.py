from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

load_dotenv()

HF_BASE = "https://huggingface.co/datasets/amanDS5153/retailgraph-products/resolve/main"

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

with driver.session(database=os.getenv("NEO4J_DATABASE")) as s:
    ids = [rec[0] for rec in s.run("MATCH (p:Product) RETURN p.product_id")]
    print(f"Found {len(ids)} products in Neo4j")

    for i, pid in enumerate(ids):
        url = f"{HF_BASE}/{pid}.jpg"
        s.run(
            "MATCH (p:Product {product_id: $pid}) SET p.image_url = $url",
            pid=pid,
            url=url
        )
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{len(ids)}")

    print(f"Done. Updated {len(ids)} products with image_url.")

driver.close()