"""
# analytics.py - GET /analytics (graph statistics) and GET /health (service connectivity)

- Plain `def` handlers on purpose: the DB clients are blocking, so Starlette runs them in its
  thread pool instead of blocking the event loop.
"""

import logging
import os

from fastapi import APIRouter, HTTPException

from retailgraph.api.models import (
    AnalyticsResponse,
    BrandStat,
    CategoryStat,
    HealthResponse,
    TagStat,
)

router = APIRouter()
log    = logging.getLogger("retailgraph.api.analytics")


# Return product/brand/category totals plus per-category, brand and tag stats.
@router.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics():
    """
    Aggregate statistics across the full knowledge graph.

    Returns:
    - Product, brand, and category totals
    - Per-category product count and average price
    - Top 20 brands by product count
    - Dietary tag distribution
    """
    from retailgraph.graph.queries import GraphQueries

    log.info("GET /analytics")

    try:
        gq = GraphQueries()

        categories_raw = gq.get_category_stats()
        brands_raw     = gq.get_top_brands(limit=20)
        tags_raw       = gq.get_dietary_tag_stats()

        total_products   = sum(r.get("product_count", 0) for r in categories_raw)
        total_brands     = gq.count_brands()
        total_categories = len(categories_raw)

        gq.close()

    except Exception as e:
        log.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Analytics error — please try again.") from e

    categories = [
        CategoryStat(
            name          = r.get("category") or r.get("c.name", "Unknown"),
            product_count = r.get("product_count", 0),
            avg_price     = round(r.get("avg_price", 0) or 0, 2),
        )
        for r in categories_raw
    ]

    brands = [
        BrandStat(
            name          = r.get("brand") or r.get("b.name", "Unknown"),
            product_count = r.get("product_count", 0),
        )
        for r in brands_raw
    ]

    tags = [
        TagStat(
            name          = r.get("tag") or r.get("t.name", "Unknown"),
            product_count = r.get("product_count", 0),
        )
        for r in tags_raw
    ]

    return AnalyticsResponse(
        total_products   = total_products,
        total_brands     = total_brands,
        total_categories = total_categories,
        categories       = categories,
        top_brands       = brands,
        dietary_tags     = tags,
    )


# Report whether Neo4j, Qdrant and a Groq key are available.
@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """
    Check connectivity to all backend services.
    Returns status for Neo4j, Qdrant, and Groq.
    """
    from dotenv import load_dotenv
    load_dotenv()

    neo4j_ok = qdrant_ok = False

    try:
        from retailgraph.graph.queries import get_driver
        with get_driver().session(database=os.getenv("NEO4J_DATABASE")) as s:
            s.run("RETURN 1")
        neo4j_ok = True
    except Exception as e:
        log.warning(f"Neo4j health check failed: {e}")

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url     = os.getenv("QDRANT_URL"),
            api_key = os.getenv("QDRANT_API_KEY"),
        )
        client.get_collection("retailgraph-products")
        qdrant_ok = True
    except Exception as e:
        log.warning(f"Qdrant health check failed: {e}")

    # Groq: only check a key exists; a real call would spend quota on every 30s health probe.
    groq_ok = bool(os.getenv("GROQ_API_KEY"))

    return HealthResponse(
        status = "ok" if all([neo4j_ok, qdrant_ok, groq_ok]) else "degraded",
        neo4j  = neo4j_ok,
        qdrant = qdrant_ok,
        groq   = groq_ok,
    )