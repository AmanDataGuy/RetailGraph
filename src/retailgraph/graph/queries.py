"""
# queries.py - fixed Cypher queries over the Neo4j product graph

- GraphQueries: product filter plus category/tag/brand stats.
- allergen_terms / allergen_exclusion: shared allergen filter, also used by hybrid_search.py.
- Used by the agent's analytics node, the API routes and scripts/run_benchmark.py.
"""

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

_DRIVER = None


# The one Neo4j driver (connection pool) for the whole process; agent, search and API all share it.
def get_driver():
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    return _DRIVER

# The graph holds 47 raw allergen spellings ("peanut"/"peanuts", "milk"/"dairy",
# "almonds", "tree nuts", ...). Umbrella words a shopper may use, mapped to the
# raw words that mean the same thing. Anything else matches by substring.
_DAIRY    = ["milk", "dairy", "butter", "cheese", "cream", "whey"]
_TREE_NUT = ["tree nut", "almond", "cashew", "walnut", "hazelnut", "pecan", "pistachio", "macadamia"]
_ALLERGEN_GROUPS = {
    "dairy": _DAIRY, "milk": _DAIRY,
    "tree nut": _TREE_NUT, "nut": _TREE_NUT + ["nut"],
    "gluten": ["gluten", "wheat"],
}


# Turn one allergen word from a user into the substrings to exclude.
# Over-excluding is the safe direction ("nut" also drops "coconut").
def allergen_terms(raw: str) -> list[str]:
    word = (raw or "").strip().lower()
    if word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]   # "peanuts" -> "peanut", so it also matches "peanut"
    return _ALLERGEN_GROUPS.get(word, [word] if word else [])


# Cypher condition: drop product p if any of its allergens contains one of the terms in $<param>.
def allergen_exclusion(param: str) -> str:
    return (
        f"NOT EXISTS {{ MATCH (p)-[:CONTAINS_ALLERGEN]->(a:Allergen) "
        f"WHERE any(t IN ${param} WHERE a.name CONTAINS t) }}"
    )


# Runs the fixed Cypher queries on the shared driver.
class GraphQueries:

    def __init__(self):
        self.driver = get_driver()

    def _run(self, cypher: str, params: dict | None = None) -> list[dict]:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, params)
            return [dict(r) for r in result]

    # Filter products by category, brand, tags, allergens and price; best quality first.
    def get_products(
        self,
        tags: list[str] = None,
        category: str = None,
        max_price: float = None,
        min_price: float = None,
        exclude_allergens: list[str] = None,
        exclude_tags: list[str] = None,
        brand: str = None,
        limit: int = 20,
    ) -> list[dict]:
        match_clauses = ["MATCH (p:Product)"]
        where_clauses = []
        params        = {"limit": limit}

        if category:
            match_clauses.append("MATCH (p)-[:BELONGS_TO]->(c:Category {name: $category})")
            params["category"] = category

        if brand:
            match_clauses.append("MATCH (p)-[:MADE_BY]->(br:Brand)")
            where_clauses.append("toLower(br.name) = toLower($brand)")
            params["brand"] = brand

        for i, tag in enumerate(tags or []):
            match_clauses.append(f"MATCH (p)-[:HAS_TAG]->(:DietaryTag {{name: $tag_{i}}})")
            params[f"tag_{i}"] = tag

        for i, allergen in enumerate(exclude_allergens or []):
            where_clauses.append(allergen_exclusion(f"excl_{i}"))
            params[f"excl_{i}"] = allergen_terms(allergen)

        for i, tag in enumerate(exclude_tags or []):
            where_clauses.append(
                f"NOT EXISTS {{ MATCH (p)-[:HAS_TAG]->(:DietaryTag {{name: $excl_tag_{i}}}) }}"
            )
            params[f"excl_tag_{i}"] = tag

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

    # Product count and price range per category, largest first.
    def get_category_stats(self) -> list[dict]:
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

    # How many products carry each dietary tag.
    def get_dietary_tag_stats(self) -> list[dict]:
        cypher = """
MATCH (p:Product)-[:HAS_TAG]->(t:DietaryTag)
RETURN t.name    AS tag,
       count(p)  AS product_count
ORDER BY product_count DESC
"""
        return self._run(cypher)

    # Brands with the most products.
    def get_top_brands(self, limit: int = 20) -> list[dict]:
        cypher = """
MATCH (p:Product)-[:MADE_BY]->(b:Brand)
RETURN b.name     AS brand,
       count(p)   AS product_count,
       round(avg(p.price), 2) AS avg_price
ORDER BY product_count DESC
LIMIT $limit
"""
        return self._run(cypher, {"limit": limit})

    # Total distinct brands (get_top_brands is capped by its LIMIT, so it can't give this).
    def count_brands(self) -> int:
        result = self._run("MATCH (b:Brand) RETURN count(DISTINCT b) AS total")
        return result[0]["total"] if result else 0

    # No-op: the driver is shared and lives as long as the process.
    def close(self):
        pass

