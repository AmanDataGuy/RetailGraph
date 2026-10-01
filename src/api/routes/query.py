"""
POST /query
Sends a natural language query through the LangGraph agent.
Returns the answer, results, Cypher used, and latency.
"""

import time
import logging
from fastapi import APIRouter, HTTPException, Request

from src.api.models import QueryRequest, QueryResponse, ProductResult
from src.agent.graph import run_query
from src.api.limiter import limiter

router = APIRouter()
log    = logging.getLogger("retailgraph.api.query")


# 20/minute per IP: each call spends a real Groq token budget (1-3 LLM
# calls). Tighter than /search since /search never touches an LLM.
#
# Plain `def`, not `async def`: run_query() is fully synchronous (blocking
# neo4j/groq/qdrant SDK calls, no awaits). An `async def` handler with no
# `await` runs on the single event-loop thread and blocks it for the whole
# request — under concurrent load, requests serialize instead of running in
# parallel (confirmed: 17/20 concurrent requests timed out before this fix).
# A sync `def` handler lets Starlette dispatch it to its thread pool instead.
@router.post("/query", response_model=QueryResponse, tags=["Agent"])
@limiter.limit("20/minute")
def query_agent(request: Request, body: QueryRequest):
    """
    Run a natural language query through the RetailGraph LangGraph agent.

    The agent will:
    1. Extract intent and entities from your query
    2. Route to the appropriate search path (Cypher / GraphRAG / analytics)
    3. Execute against Neo4j and/or Qdrant
    4. Return a plain-English answer with supporting results

    **Example queries:**
    - `show me vegan snacks under $10`
    - `find something similar to sriracha sauce`
    - `gluten-free beverages with no nuts`
    - `which category has the most products`
    - `tell me about McCormick products`
    """
    log.info(f"POST /query | query='{body.query}'")
    start = time.perf_counter()

    try:
        state = run_query(body.query)
    except Exception as e:
        log.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail="Agent error — please try again.")

    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    # Normalise raw_results → list[ProductResult]
    raw     = state.get("raw_results") or []
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
                image_url      = r.get("image_url"),
            ))
        except Exception:
            continue

    return QueryResponse(
        query        = body.query,
        answer       = state.get("answer") or "No answer generated.",
        intent       = state.get("intent"),
        route        = state.get("route"),
        result_count = state.get("result_count", len(results)),
        total_matches = state.get("total_count"),
        results      = results,
        cypher_used  = state.get("cypher_used"),
        latency_ms   = latency_ms,
    )