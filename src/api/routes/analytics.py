"""
GET /analytics — category, brand, dietary tag statistics
GET /health    — connection status for all services
"""

import os
import logging
from fastapi import APIRouter, HTTPException

from src.api.models import (
    AnalyticsResponse, CategoryStat, BrandStat, TagStat,
    HealthResponse,
)

router = APIRouter()
log    = logging.getLogger("retailgraph.api.analytics")


# ── GET /analytics ─────────────────────────────────────────────────────────

@router.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics():
    """
    Aggregate statistics across the full knowledge graph.

    Returns:
    - Product, brand, and category totals
    - Per-category product count and average price
    - Top 20 brands by product count
    - Dietary tag distribution
    """
    from src.graph.queries import GraphQueries

    log.info("GET /analytics")

    try:
        gq = GraphQueries()

        categories_raw = gq.get_category_stats()
        brands_raw     = gq.get_top_brands(limit=20)
        tags_raw       = gq.get_dietary_tag_stats()

        # Totals
        total_products   = sum(r.get("product_count", 0) for r in categories_raw)
        total_brands     = len(brands_raw)
        total_categories = len(categories_raw)

        gq.close()

    except Exception as e:
        log.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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


# ── GET /health ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check connectivity to all backend services.
    Returns status for Neo4j, Qdrant, and Groq.
    """
    from dotenv import load_dotenv
    load_dotenv()

    neo4j_ok  = False
    qdrant_ok = False
    groq_ok   = False

    # Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as s:
            s.run("RETURN 1")
        driver.close()
        neo4j_ok = True
    except Exception as e:
        log.warning(f"Neo4j health check failed: {e}")

    # Qdrant
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

    # Groq
    try:
        from src.agent.llm import generate
        result = generate(system="Say OK.", prompt="ping")
        groq_ok = bool(result)
    except Exception as e:
        log.warning(f"Groq health check failed: {e}")

    status = "ok" if all([neo4j_ok, qdrant_ok, groq_ok]) else "degraded"

    return HealthResponse(
        status = status,
        neo4j  = neo4j_ok,
        qdrant = qdrant_ok,
        groq   = groq_ok,
    )