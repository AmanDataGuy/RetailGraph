"""
POST /search  — direct filtered search, bypasses agent
GET  /products/{product_id} — single product with graph context
"""

import logging
from fastapi import APIRouter, HTTPException

from src.api.models import (
    SearchRequest, SearchResponse, ProductResult,
    ProductDetail,
)

router = APIRouter()
log    = logging.getLogger("retailgraph.api.search")


# ── POST /search ───────────────────────────────────────────────────────────

@router.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_products(request: SearchRequest):
    """
    Direct product search — bypasses the LangGraph agent for speed.

    Use this when you have structured filters and don't need a natural
    language answer. Combines Qdrant semantic search with Neo4j constraints.

    **Examples:**
    - Semantic only: `{"query": "spicy hot sauce"}`
    - Filter only:   `{"category": "Beverages", "dietary_tags": ["vegan"], "max_price": 5}`
    - Combined:      `{"query": "protein bar", "dietary_tags": ["gluten-free"], "max_price": 10}`
    """
    from src.graph.hybrid_search import HybridSearch
    from src.graph.queries import GraphQueries

    log.info(f"POST /search | query='{request.query}' filters={request.dietary_tags}")

    # Build filter kwargs
    kwargs = {}
    if request.category:
        kwargs["category"] = request.category
    if request.max_price is not None:
        kwargs["max_price"] = request.max_price
    if request.dietary_tags:
        kwargs["dietary_tags"] = request.dietary_tags

    try:
        if request.query:
            # Semantic + optional filters → hybrid search
            hs      = HybridSearch()
            raw     = hs.search(request.query, top_k=request.top_k, **kwargs)
            search_type = "hybrid" if kwargs else "semantic"

        else:
            # Filter only → pure Cypher via GraphQueries
            gq  = GraphQueries()
            raw = gq.get_products(
                dietary_tags      = request.dietary_tags or None,
                category          = request.category,
                max_price         = request.max_price,
                min_price         = request.min_price,
                exclude_allergens = request.exclude_allergens or None,
                limit             = request.top_k,
            )
            gq.close()
            search_type = "filter"

    except Exception as e:
        log.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    for r in raw:
        try:
            results.append(ProductResult(
                item_name      = r.get("item_name") or r.get("p.item_name", ""),
                price          = r.get("price") or r.get("p.price"),
                brand          = r.get("brand"),
                category       = r.get("category"),
                quantity_value = r.get("quantity_value"),
                quantity_unit  = r.get("quantity_unit"),
                dietary_tags   = r.get("dietary_tags") or [],
                allergen_list  = r.get("allergen_list") or [],
                hybrid_score   = r.get("hybrid_score"),
            ))
        except Exception:
            continue

    return SearchResponse(
        result_count = len(results),
        results      = results,
        search_type  = search_type,
    )


# ── GET /products/{product_id} ─────────────────────────────────────────────

@router.get("/products/{product_id}", response_model=ProductDetail, tags=["Products"])
async def get_product(product_id: str):
    """
    Fetch a single product by ID with full graph context.

    Returns the product's properties plus similar products
    (products sharing the same category and dietary tags).
    """
    import os
    from neo4j import GraphDatabase
    from dotenv import load_dotenv
    load_dotenv()

    log.info(f"GET /products/{product_id}")

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:

            # Main product query
            result = session.run("""
                MATCH (p:Product {product_id: $pid})
                OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
                OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
                OPTIONAL MATCH (p)-[:HAS_TAG]->(t:DietaryTag)
                OPTIONAL MATCH (p)-[:CONTAINS_ALLERGEN]->(a:Allergen)
                RETURN p, b.name AS brand, c.name AS category,
                       collect(DISTINCT t.name) AS tags,
                       collect(DISTINCT a.name) AS allergens
            """, pid=product_id)

            record = result.single()
            if not record:
                raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

            p = record["p"]

            # Similar products (same category, shared tags)
            sim_result = session.run("""
                MATCH (p:Product {product_id: $pid})-[:BELONGS_TO]->(c:Category)
                MATCH (other:Product)-[:BELONGS_TO]->(c)
                WHERE other.product_id <> $pid
                OPTIONAL MATCH (other)-[:MADE_BY]->(b:Brand)
                RETURN other.item_name AS item_name,
                       other.price     AS price,
                       b.name          AS brand
                LIMIT 5
            """, pid=product_id)

            similar = [dict(r) for r in sim_result]

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Product lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        driver.close()

    return ProductDetail(
        product_id     = product_id,
        item_name      = p.get("item_name", ""),
        price          = p.get("price"),
        brand          = record["brand"],
        category       = record["category"],
        quantity_value = p.get("quantity_value"),
        quantity_unit  = p.get("quantity_unit"),
        dietary_tags   = record["tags"] or [],
        allergen_list  = record["allergens"] or [],
        quality_score  = p.get("quality_score"),
        similar_products = similar,
    )