"""
src/graph/hybrid_search.py

GraphRAG hybrid search — combines Qdrant (semantic) + Neo4j (structured).

Two search modes:
  1. Semantic-first  — Qdrant finds candidates → Neo4j verifies constraints
  2. Filter-first    — Neo4j filters exactly   → Qdrant re-ranks by similarity

Usage:
    from src.graph.hybrid_search import HybridSearch

    hs = HybridSearch()
    results = hs.search("vegan protein bar under $10 with no nuts")
    results = hs.search("organic tea", max_price=15.0, dietary_tags=["organic"])
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "retailgraph-products"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

QDRANT_CANDIDATE_POOL = 50
NEO4J_CANDIDATE_POOL  = 100


class HybridSearch:
    """
    GraphRAG hybrid search.

    Internally decides which path to use based on query type:
      - Has hard constraints (price/tags/allergens) → filter-first
      - Pure semantic query                         → semantic-first
      - Both                                        → semantic-first with constraint verification
    """

    def __init__(self):
        self.model  = SentenceTransformer(EMBEDDING_MODEL)
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        print("✅ HybridSearch initialized (Neo4j + Qdrant)")

    # ── Public Interface ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        category: str = None,
        max_price: float = None,
        min_price: float = None,
        dietary_tags: list[str] = None,
        exclude_allergens: list[str] = None,
        brand: str = None,
    ) -> list[dict]:
        has_constraints = any([
            category, max_price, min_price,
            dietary_tags, exclude_allergens, brand
        ])

        if has_constraints:
            return self._filter_first(
                query, top_k, category, max_price, min_price,
                dietary_tags, exclude_allergens, brand
            )
        else:
            return self._semantic_first(query, top_k)

    # ── Semantic-First Path ───────────────────────────────────────────────────

    def _semantic_first(self, query: str, top_k: int) -> list[dict]:
        """Qdrant semantic search → enrich with Neo4j relationship data."""
        vector = self.model.encode(query).tolist()
        qdrant_results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points

        if not qdrant_results:
            return []

        product_ids = [r.payload.get("product_id") or r.payload.get("sample_id")
                       for r in qdrant_results]
        scores      = {(r.payload.get("product_id") or r.payload.get("sample_id")): r.score
                       for r in qdrant_results}

        enriched = self._enrich_from_neo4j(product_ids)
        return self._merge_results(enriched, scores, top_k)

    # ── Filter-First Path ─────────────────────────────────────────────────────

    def _filter_first(
        self, query, top_k, category, max_price, min_price,
        dietary_tags, exclude_allergens, brand
    ) -> list[dict]:
        """Neo4j exact filtering → Qdrant semantic re-ranking."""
        neo4j_results = self._neo4j_filter(
            category, max_price, min_price,
            dietary_tags, exclude_allergens, brand,
            limit=NEO4J_CANDIDATE_POOL
        )

        if not neo4j_results:
            return []

        vector        = self.model.encode(query).tolist()
        qdrant_scores = self._score_by_vector(vector, neo4j_results)

        for r in neo4j_results:
            pid = r["product_id"]
            r["semantic_score"] = qdrant_scores.get(pid, 0.0)
            r["hybrid_score"]   = round(
                0.6 * r["semantic_score"] + 0.4 * (r.get("quality_score", 50) / 100),
                3
            )

        neo4j_results.sort(key=lambda x: -x["hybrid_score"])
        return neo4j_results[:top_k]

    # ── Neo4j Filter ──────────────────────────────────────────────────────────

    def _neo4j_filter(
        self, category, max_price, min_price,
        dietary_tags, exclude_allergens, brand, limit
    ) -> list[dict]:
        """Build and run a dynamic Cypher filter query."""

        match_clauses = ["MATCH (p:Product)"]
        where_clauses = []
        params        = {"limit": limit}

        if category:
            match_clauses.append("MATCH (p)-[:BELONGS_TO]->(c:Category {name: $category})")
            params["category"] = category

        if brand:
            match_clauses.append("MATCH (p)-[:MADE_BY]->(b:Brand {name: $brand})")
            params["brand"] = brand

        if dietary_tags:
            for i, tag in enumerate(dietary_tags):
                match_clauses.append(f"MATCH (p)-[:HAS_TAG]->(:DietaryTag {{name: $tag_{i}}})")
                params[f"tag_{i}"] = tag

        if exclude_allergens:
            for i, allergen in enumerate(exclude_allergens):
                where_clauses.append(
                    f"NOT EXISTS {{ MATCH (p)-[:CONTAINS_ALLERGEN]->(:Allergen {{name: $excl_{i}}}) }}"
                )
                params[f"excl_{i}"] = allergen

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
OPTIONAL MATCH (p)-[:MADE_BY]->(brand)
OPTIONAL MATCH (p)-[:BELONGS_TO]->(cat)
RETURN p.product_id      AS product_id,
       p.item_name       AS item_name,
       brand.name        AS brand,
       cat.name          AS category,
       p.price           AS price,
       p.image_url       AS image_url,
       p.dietary_tags    AS dietary_tags,
       p.allergen_list   AS allergen_list,
       p.quality_score   AS quality_score
LIMIT $limit
"""

        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, params)
            return [dict(r) for r in result]

    # ── Qdrant Re-ranking ─────────────────────────────────────────────────────

    def _score_by_vector(self, vector: list[float], products: list[dict]) -> dict[str, float]:
        """Score a small set of products by vector similarity."""
        scores = {}
        for product in products:
            pid = product.get("product_id")
            if not pid:
                continue
            try:
                results = self.qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vector,
                    query_filter=Filter(must=[
                        FieldCondition(key="product_id", match=MatchValue(value=pid))
                    ]),
                    limit=1,
                    with_payload=False,
                ).points
                scores[pid] = results[0].score if results else 0.5
            except Exception:
                scores[pid] = 0.5
        return scores

    # ── Neo4j Enrichment ──────────────────────────────────────────────────────

    def _enrich_from_neo4j(self, product_ids: list[str]) -> list[dict]:
        """Fetch full relationship data + image_url from Neo4j for a list of product IDs."""
        cypher = """
UNWIND $ids AS pid
MATCH (p:Product {product_id: pid})
OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
RETURN p.product_id    AS product_id,
       p.item_name     AS item_name,
       b.name          AS brand,
       c.name          AS category,
       p.price         AS price,
       p.image_url     AS image_url,
       p.dietary_tags  AS dietary_tags,
       p.allergen_list AS allergen_list,
       p.quality_score AS quality_score
"""
        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, {"ids": product_ids})
            return [dict(r) for r in result]

    # ── Merge & Rank ──────────────────────────────────────────────────────────

    def _merge_results(
        self, enriched: list[dict], scores: dict[str, float], top_k: int
    ) -> list[dict]:
        """Combine Neo4j data with Qdrant scores into final ranked list."""
        for r in enriched:
            pid        = r.get("product_id")
            sem_score  = scores.get(pid, 0.0)
            qual_score = (r.get("quality_score") or 50) / 100
            r["semantic_score"] = round(sem_score, 3)
            r["hybrid_score"]   = round(0.7 * sem_score + 0.3 * qual_score, 3)

        enriched.sort(key=lambda x: -x["hybrid_score"])
        return enriched[:top_k]

    def close(self):
        self.driver.close()


# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hs = HybridSearch()

    tests = [
        {
            "desc":  "Pure semantic — open-ended",
            "query": "organic green tea",
            "kwargs": {}
        },
        {
            "desc":  "Filter-first — vegan snacks under $10",
            "query": "healthy snack",
            "kwargs": {"dietary_tags": ["vegan"], "max_price": 10.0}
        },
        {
            "desc":  "Filter-first — gluten-free condiments no allergens",
            "query": "sauce for cooking",
            "kwargs": {
                "category": "Condiments & Sauces",
                "dietary_tags": ["gluten-free"],
            }
        },
    ]

    for t in tests:
        print(f"\n{'='*60}")
        print(f"Query: '{t['query']}'")
        print(f"Mode:  {t['desc']}")
        print(f"Filters: {t['kwargs']}")
        print("-" * 60)
        results = hs.search(t["query"], top_k=5, **t["kwargs"])
        if not results:
            print("  No results found.")
        for r in results:
            tags = ", ".join(r.get("dietary_tags") or []) or "none"
            print(f"  [{r['hybrid_score']}] {r['item_name']}")
            print(f"         Brand: {r['brand']} | ${r['price']} | {r['category']}")
            print(f"         Image: {r.get('image_url', 'none')}")

    hs.close()