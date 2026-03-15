"""
src/graph/queries.py

Pre-built Cypher query templates for RetailGraph.
Used by the LangGraph agent for structured queries.

Usage:
    from src.graph.queries import GraphQueries
    gq = GraphQueries()
    results = gq.get_products(tags=["vegan"], max_price=10.0)
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


class GraphQueries:

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def _run(self, cypher: str, params: dict = {}) -> list[dict]:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, params)
            return [dict(r) for r in result]

    # ── 1. Filter by tags + price + category ─────────────────────────────────

    def get_products(
        self,
        tags: list[str] = None,
        category: str = None,
        max_price: float = None,
        min_price: float = None,
        exclude_allergens: list[str] = None,
        brand: str = None,
        limit: int = 20,
    ) -> list[dict]:
        """
        Multi-constraint product filter.
        Most commonly used template — handles 80% of agent queries.
        """
        match_clauses = ["MATCH (p:Product)"]
        where_clauses = []
        params        = {"limit": limit}

        if category:
            match_clauses.append("MATCH (p)-[:BELONGS_TO]->(c:Category {name: $category})")
            params["category"] = category

        if brand:
            match_clauses.append("MATCH (p)-[:MADE_BY]->(br:Brand {name: $brand})")
            params["brand"] = brand

        if tags:
            for i, tag in enumerate(tags):
                match_clauses.append(f"MATCH (p)-[:HAS_TAG]->(:DietaryTag {{name: $tag_{i}}})")
                params[f"tag_{i}"] = tag

        if exclude_allergens:
            for i, allergen in enumerate(exclude_allergens):
                where_clauses.append(
                    f"NOT EXISTS {{ MATCH (p)-[:CONTAINS_ALLERGEN]->(:Allergen {{name: $excl_{i}}}) }}"
                )
                params[f"excl_{i}"] = allergen.lower()

        if max_price is not None:
            where_clauses.append("p.price <= $max_price")
            params["max_price"] = max_price

        if min_price is not None:
            where_clauses.append("p.price >= $min_price")
            params["min_price"] = min_price

        cypher = "\n".join(match_clauses)
        if where_clauses:
            cypher += "\nWHERE " + " AND ".join(where_clauses)

        cypher += """
OPTIONAL MATCH (p)-[:MADE_BY]->(brand_node)
OPTIONAL MATCH (p)-[:BELONGS_TO]->(cat_node)
RETURN p.product_id   AS product_id,
       p.item_name    AS item_name,
       brand_node.name AS brand,
       cat_node.name  AS category,
       p.price        AS price,
       p.dietary_tags     AS dietary_tags,
       p.allergen_list    AS allergen_list,
       p.quantity_value   AS quantity_value,
       p.quantity_unit    AS quantity_unit,
       p.quality_score    AS quality_score
ORDER BY p.quality_score DESC
LIMIT $limit
"""
        return self._run(cypher, params)

    # ── 2. Get products by brand ──────────────────────────────────────────────

    def get_products_by_brand(self, brand: str, limit: int = 20) -> list[dict]:
        """All products made by a specific brand."""
        cypher = """
MATCH (p:Product)-[:MADE_BY]->(b:Brand {name: $brand})
OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
RETURN p.product_id AS product_id,
       p.item_name  AS item_name,
       b.name       AS brand,
       c.name       AS category,
       p.price      AS price,
       p.dietary_tags  AS dietary_tags,
       p.quality_score AS quality_score
ORDER BY p.price ASC
LIMIT $limit
"""
        return self._run(cypher, {"brand": brand, "limit": limit})

    # ── 3. Get products by category ───────────────────────────────────────────

    def get_products_by_category(self, category: str, limit: int = 20) -> list[dict]:
        """All products in a specific category."""
        cypher = """
MATCH (p:Product)-[:BELONGS_TO]->(c:Category {name: $category})
OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
RETURN p.product_id AS product_id,
       p.item_name  AS item_name,
       b.name       AS brand,
       c.name       AS category,
       p.price      AS price,
       p.dietary_tags  AS dietary_tags,
       p.quality_score AS quality_score
ORDER BY p.quality_score DESC
LIMIT $limit
"""
        return self._run(cypher, {"category": category, "limit": limit})

    # ── 4. Get similar products ───────────────────────────────────────────────

    def get_similar_products(self, product_id: str, limit: int = 10) -> list[dict]:
        """
        Find products similar to a given product.
        Similarity = shared category + at least 1 shared dietary tag.
        """
        cypher = """
MATCH (p:Product {product_id: $product_id})-[:BELONGS_TO]->(c:Category)
MATCH (p)-[:HAS_TAG]->(t:DietaryTag)
MATCH (similar:Product)-[:BELONGS_TO]->(c)
MATCH (similar)-[:HAS_TAG]->(t)
WHERE similar.product_id <> $product_id
OPTIONAL MATCH (similar)-[:MADE_BY]->(b:Brand)
WITH similar, b, c, count(t) AS shared_tags
ORDER BY shared_tags DESC
RETURN similar.product_id AS product_id,
       similar.item_name  AS item_name,
       b.name             AS brand,
       c.name             AS category,
       similar.price      AS price,
       similar.dietary_tags AS dietary_tags,
       shared_tags
LIMIT $limit
"""
        return self._run(cypher, {"product_id": product_id, "limit": limit})

    # ── 5. Allergen exclusion ─────────────────────────────────────────────────

    def get_allergen_safe_products(
        self,
        allergens: list[str],
        category: str = None,
        limit: int = 20
    ) -> list[dict]:
        """Products that do NOT contain specified allergens."""
        params = {"limit": limit}
        match  = "MATCH (p:Product)"
        where  = []

        if category:
            match += "\nMATCH (p)-[:BELONGS_TO]->(c:Category {name: $category})"
            params["category"] = category

        for i, allergen in enumerate(allergens):
            where.append(
                f"NOT EXISTS {{ MATCH (p)-[:CONTAINS_ALLERGEN]->(:Allergen {{name: $a_{i}}}) }}"
            )
            params[f"a_{i}"] = allergen.lower()

        cypher = match
        if where:
            cypher += "\nWHERE " + " AND ".join(where)
        cypher += """
OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
OPTIONAL MATCH (p)-[:BELONGS_TO]->(cat)
RETURN p.product_id AS product_id,
       p.item_name  AS item_name,
       b.name       AS brand,
       cat.name     AS category,
       p.price      AS price,
       p.dietary_tags  AS dietary_tags,
       p.allergen_list AS allergen_list,
       p.quality_score AS quality_score
ORDER BY p.quality_score DESC
LIMIT $limit
"""
        return self._run(cypher, params)

    # ── 6. Brand stats ────────────────────────────────────────────────────────

    def get_brand_stats(self, brand: str) -> dict:
        """Analytics for a specific brand."""
        cypher = """
MATCH (p:Product)-[:MADE_BY]->(b:Brand {name: $brand})
OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
OPTIONAL MATCH (p)-[:HAS_TAG]->(t:DietaryTag)
RETURN count(DISTINCT p)    AS product_count,
       avg(p.price)         AS avg_price,
       min(p.price)         AS min_price,
       max(p.price)         AS max_price,
       collect(DISTINCT c.name) AS categories,
       collect(DISTINCT t.name) AS dietary_tags
"""
        results = self._run(cypher, {"brand": brand})
        return results[0] if results else {}

    # ── 7. Category stats ─────────────────────────────────────────────────────

    def get_category_stats(self) -> list[dict]:
        """Overview of all categories — product counts and avg price."""
        cypher = """
MATCH (p:Product)-[:BELONGS_TO]->(c:Category)
RETURN c.name           AS category,
       count(p)         AS product_count,
       round(avg(p.price), 2) AS avg_price,
       round(min(p.price), 2) AS min_price,
       round(max(p.price), 2) AS max_price
ORDER BY product_count DESC
"""
        return self._run(cypher)

    # ── 8. Dietary tag stats ──────────────────────────────────────────────────

    def get_dietary_tag_stats(self) -> list[dict]:
        """How many products have each dietary tag."""
        cypher = """
MATCH (p:Product)-[:HAS_TAG]->(t:DietaryTag)
RETURN t.name    AS tag,
       count(p)  AS product_count
ORDER BY product_count DESC
"""
        return self._run(cypher)

    # ── 9. Top brands ─────────────────────────────────────────────────────────

    def get_top_brands(self, limit: int = 20) -> list[dict]:
        """Brands with most products."""
        cypher = """
MATCH (p:Product)-[:MADE_BY]->(b:Brand)
RETURN b.name     AS brand,
       count(p)   AS product_count,
       round(avg(p.price), 2) AS avg_price
ORDER BY product_count DESC
LIMIT $limit
"""
        return self._run(cypher, {"limit": limit})

    # ── 10. Price range search ────────────────────────────────────────────────

    def get_products_by_price_range(
        self,
        min_price: float,
        max_price: float,
        category: str = None,
        limit: int = 20
    ) -> list[dict]:
        """Products within a price range, optionally filtered by category."""
        params = {"min_price": min_price, "max_price": max_price, "limit": limit}
        match  = "MATCH (p:Product)\nWHERE p.price >= $min_price AND p.price <= $max_price"

        if category:
            match += "\nMATCH (p)-[:BELONGS_TO]->(c:Category {name: $category})"
            params["category"] = category

        cypher = match + """
OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
OPTIONAL MATCH (p)-[:BELONGS_TO]->(cat)
RETURN p.product_id AS product_id,
       p.item_name  AS item_name,
       b.name       AS brand,
       cat.name     AS category,
       p.price      AS price,
       p.dietary_tags  AS dietary_tags,
       p.quality_score AS quality_score
ORDER BY p.price ASC
LIMIT $limit
"""
        return self._run(cypher, params)

    def close(self):
        self.driver.close()


# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gq = GraphQueries()

    print("=" * 60)
    print("1. Vegan snacks under $10")
    results = gq.get_products(tags=["vegan"], category="Snacks & Candy", max_price=10.0, limit=3)
    for r in results:
        print(f"  {r['item_name']} | ${r['price']} | {', '.join(r['dietary_tags'] or [])}")

    print("\n" + "=" * 60)
    print("2. Category stats")
    for r in gq.get_category_stats():
        print(f"  {r['category']:<35} {r['product_count']:>4} products | avg ${r['avg_price']}")

    print("\n" + "=" * 60)
    print("3. Top 5 brands")
    for r in gq.get_top_brands(limit=5):
        print(f"  {r['brand']:<30} {r['product_count']:>3} products | avg ${r['avg_price']}")

    print("\n" + "=" * 60)
    print("4. Dietary tag distribution")
    for r in gq.get_dietary_tag_stats():
        print(f"  {r['tag']:<20} {r['product_count']:>4} products")

    print("\n" + "=" * 60)
    print("5. Allergen-safe products (no peanuts, no tree nuts) in Snacks")
    results = gq.get_allergen_safe_products(
        allergens=["peanuts", "tree nuts"],
        category="Snacks & Candy",
        limit=3
    )
    for r in results:
        print(f"  {r['item_name']} | ${r['price']}")

    gq.close()