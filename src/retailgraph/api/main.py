"""
# main.py - the FastAPI app: CORS, rate limiting, routers and startup warm-up

- Endpoints under /v1: POST /query, POST /search, GET /products/{id}, GET /analytics, GET /health.
- Run with: uvicorn retailgraph.api.main:app --reload --port 8000 (docs at /docs).
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from retailgraph.api.limiter import limiter
from retailgraph.api.routes.analytics import router as analytics_router
from retailgraph.api.routes.query import router as query_router
from retailgraph.api.routes.search import router as search_router

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(name)s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("retailgraph.api")



# On startup, load the embedding model once so the first real search isn't slow.
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("RetailGraph API starting up...")
    log.info("Warming up HybridSearch (fastembed + Qdrant + Neo4j)...")
    try:
        from retailgraph.graph.hybrid_search import HybridSearch
        t0 = time.perf_counter()
        HybridSearch()
        log.info(f"Warm-up complete in {time.perf_counter() - t0:.1f}s.")
    except Exception as e:
        log.error(f"Warm-up failed (will retry lazily on first request): {e}")
    yield
    log.info("RetailGraph API shutting down.")


app = FastAPI(
    title       = "RetailGraph API",
    description = (
        "Multimodal entity extraction and knowledge graph platform for grocery product catalogs. "
        "Query 2,160 products using natural language — powered by LangGraph + Neo4j + Qdrant."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Open CORS for the demo; credentials stay off (browsers reject "*" with credentials, and there is no auth).
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(query_router,     prefix="/v1")
app.include_router(search_router,    prefix="/v1")
app.include_router(analytics_router, prefix="/v1")

# List the API's endpoints.
@app.get("/", tags=["System"])
async def root():
    return {
        "name":        "RetailGraph API",
        "version":     "1.0.0",
        "docs":        "/docs",
        "endpoints": {
            "query":     "POST /v1/query",
            "search":    "POST /v1/search",
            "products":  "GET  /v1/products/{product_id}",
            "analytics": "GET  /v1/analytics",
            "health":    "GET  /v1/health",
        },
    }