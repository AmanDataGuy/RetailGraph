"""
# query.py - POST /query: runs a plain-English question through the LangGraph agent

- Returns the answer, results, Cypher used and latency. Calls agent/graph.py run_query().
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from retailgraph.agent.graph import run_query
from retailgraph.api.limiter import limiter
from retailgraph.api.models import ProductResult, QueryRequest, QueryResponse

router = APIRouter()
log    = logging.getLogger("retailgraph.api.query")


# Answer a question with the agent. 20/min per IP since each call spends 1-3 Groq LLM calls;
# plain `def` so the blocking agent runs in Starlette's thread pool, not on the event loop.
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
        raise HTTPException(status_code=500, detail="Agent error — please try again.") from e

    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    results = []
    for r in state.get("raw_results") or []:
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