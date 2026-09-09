"""
POST /query
Sends a natural language query through the LangGraph agent.
Returns the answer, results, Cypher used, and latency.
"""

import time
import logging
from fastapi import APIRouter, HTTPException

from src.api.models import QueryRequest, QueryResponse, ProductResult
from src.agent.graph import run_query
from src.api.request_log import log_request

router = APIRouter()
log    = logging.getLogger("retailgraph.api.query")


@router.post("/query", response_model=QueryResponse, tags=["Agent"])
async def query_agent(request: QueryRequest):
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
    log.info(f"POST /query | query='{request.query}'")
    start = time.perf_counter()

    try:
        state = run_query(request.query)
    except Exception as e:
        log.error(f"Agent error: {e}")
        log_request(
            endpoint="/query", query=request.query, success=False, error=str(e),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
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

    log_request(
        endpoint="/query", query=request.query, intent=state.get("intent"),
        route=state.get("route"), result_count=state.get("result_count", len(results)),
        latency_ms=latency_ms, success=True,
    )

    return QueryResponse(
        query        = request.query,
        answer       = state.get("answer") or "No answer generated.",
        intent       = state.get("intent"),
        route        = state.get("route"),
        result_count = state.get("result_count", len(results)),
        results      = results,
        cypher_used  = state.get("cypher_used"),
        latency_ms   = latency_ms,
    )