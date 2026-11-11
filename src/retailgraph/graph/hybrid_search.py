"""
# hybrid_search.py - GraphRAG product search over Qdrant (meaning) + Neo4j (facts)

- No filters: semantic-first (Qdrant top-k, then Neo4j fills in brand/category).
- Any filter: filter-first (Neo4j exact filter, then Qdrant ranks the survivors).
- HybridSearch is used by the agent nodes, the API routes and scripts/run_benchmark.py.
"""

import logging
import os

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from retailgraph.graph.queries import NEO4J_DATABASE, allergen_exclusion, allergen_terms, get_driver

load_dotenv()
log = logging.getLogger("retailgraph.hybrid_search")

QDRANT_URL     = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "retailgraph-products"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Filter-first ranks every product that passes the filters; a smaller cap would cut an
# arbitrary slice (no ORDER BY) before ranking.
# shortcut: fine for the 2,160-product catalog; filter inside Qdrant past ~50k products.
NEO4J_CANDIDATE_POOL  = 5000


# Process-wide lazy singletons: the model and clients are expensive to build, so each is
# created once and shared by every HybridSearch() instead of reloading per request.
_MODEL  = None
_QDRANT = None


# Shared embedding model (fastembed/ONNX: same 384-dim vectors as torch, about half the RAM).
def _get_model() -> TextEmbedding:
    global _MODEL
    if _MODEL is None:
        _MODEL = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _MODEL


# Shared Qdrant client.
def _get_qdrant() -> QdrantClient:
    global _QDRANT
    if _QDRANT is None:
        _QDRANT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _QDRANT


# Product search: filter-first when any constraint is given, otherwise semantic-first.
class HybridSearch:

    def __init__(self):
        self.model  = _get_model()
        self.qdrant = _get_qdrant()
        self.driver = get_driver()

    # Search products for a query, optionally constrained by category, price, tags, allergens, brand.
    def search(
        self,
        query: str,
        top_k: int = 10,
        category: str = None,
        max_price: float = None,
        min_price: float = None,
        dietary_tags: list[str] = None,
        exclude_allergens: list[str] = None,
        exclude_tags: list[str] = None,
        brand: str = None,
    ) -> list[dict]:
        if any([category, dietary_tags, exclude_allergens, exclude_tags, brand]) or max_price is not None or min_price is not None:
            return self._filter_first(
                query, top_k, category, max_price, min_price,
                dietary_tags, exclude_allergens, exclude_tags, brand
            )
        return self._semantic_first(query, top_k)

    # Qdrant top-k by meaning, then Neo4j adds brand/category/price.
    def _semantic_first(self, query: str, top_k: int) -> list[dict]:
        vector = list(self.model.embed([query]))[0].tolist()
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
        scores = dict(zip(product_ids, (r.score for r in qdrant_results), strict=True))
        return self._merge_results(self._enrich_from_neo4j(product_ids), scores, top_k)

    # Neo4j exact filter, then rank by 0.6 * similarity + 0.4 * quality.
    def _filter_first(
        self, query, top_k, category, max_price, min_price,
        dietary_tags, exclude_allergens, exclude_tags, brand
    ) -> list[dict]:
        neo4j_results = self._neo4j_filter(
            category, max_price, min_price,
            dietary_tags, exclude_allergens, exclude_tags, brand,
            limit=NEO4J_CANDIDATE_POOL
        )

        if not neo4j_results:
            return []

        vector        = list(self.model.embed([query]))[0].tolist()
        qdrant_scores = self._score_by_vector(vector, neo4j_results)

        for r in neo4j_results:
            pid = r["product_id"]
            r["semantic_score"] = qdrant_scores.get(pid, 0.0)
            r["hybrid_score"]   = round(
                0.6 * r["semantic_score"] + 0.4 * ((r.get("quality_score") or 50) / 100),
                3
            )

        neo4j_results.sort(key=lambda x: -x["hybrid_score"])
        return neo4j_results[:top_k]

    # Build and run the Cypher filter query for the given constraints.
    def _neo4j_filter(
        self, category, max_price, min_price,
        dietary_tags, exclude_allergens, exclude_tags, brand, limit
    ) -> list[dict]:
        match_clauses = ["MATCH (p:Product)"]
        where_clauses = []
        params        = {"limit": limit}

        if category:
            match_clauses.append("MATCH (p)-[:BELONGS_TO]->(c:Category {name: $category})")
            params["category"] = category

        if brand:
            match_clauses.append("MATCH (p)-[:MADE_BY]->(b:Brand)")
            where_clauses.append("toLower(b.name) = toLower($brand)")
            params["brand"] = brand

        for i, tag in enumerate(dietary_tags or []):
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

    # Similarity score per product id in one batched Qdrant call (MatchAny over the candidates).
    def _score_by_vector(self, vector: list[float], products: list[dict]) -> dict[str, float]:
        pids = [p.get("product_id") for p in products if p.get("product_id")]
        if not pids:
            return {}

        scores = {pid: 0.5 for pid in pids}  # fallback for any pid Qdrant doesn't have
        try:
            results = self.qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                query_filter=Filter(must=[
                    FieldCondition(key="product_id", match=MatchAny(any=pids))
                ]),
                limit=len(pids),
                with_payload=["product_id"],
            ).points
            for r in results:
                pid = (r.payload or {}).get("product_id")
                if pid:
                    scores[pid] = r.score
        except Exception as e:
            # keep the 0.5 fallback scores, but say so: ranking is now arbitrary
            log.warning(f"Qdrant re-ranking failed, using flat scores: {e}")
        return scores

    # Fetch brand, category, price and tags from Neo4j for the given product ids.
    def _enrich_from_neo4j(self, product_ids: list[str]) -> list[dict]:
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

    # Rank enriched products by 0.7 * similarity + 0.3 * quality.
    def _merge_results(
        self, enriched: list[dict], scores: dict[str, float], top_k: int
    ) -> list[dict]:
        for r in enriched:
            pid        = r.get("product_id")
            sem_score  = scores.get(pid, 0.0)
            qual_score = (r.get("quality_score") or 50) / 100
            r["semantic_score"] = round(sem_score, 3)
            r["hybrid_score"]   = round(0.7 * sem_score + 0.3 * qual_score, 3)

        enriched.sort(key=lambda x: -x["hybrid_score"])
        return enriched[:top_k]
