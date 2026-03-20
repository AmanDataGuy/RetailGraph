"""
RetailGraph — FastAPI Application
Production-grade REST API for the RetailGraph knowledge graph platform.

Endpoints:
    POST /v1/query          — NL query → LangGraph agent → answer
    POST /v1/search         — direct filtered search (no agent)
    GET  /v1/products/{id}  — single product with graph context
    GET  /v1/analytics      — category + brand + tag statistics
    GET  /v1/health         — service connectivity check

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Interactive docs:
    http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.query     import router as query_router
from src.api.routes.search    import router as search_router
from src.api.routes.analytics import router as analytics_router

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(name)s  %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("retailgraph.api")


# ── Lifespan: startup / shutdown ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("RetailGraph API starting up...")
    log.info("Neo4j, Qdrant, and Groq connections will be made on first request.")
    yield
    log.info("RetailGraph API shutting down.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "RetailGraph API",
    description = (
        "Multimodal entity extraction and knowledge graph platform for grocery product catalogs. "
        "Query 2,160 products using natural language — powered by LangGraph + Neo4j + Qdrant."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS — allow all origins for portfolio/demo ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(query_router,     prefix="/v1")
app.include_router(search_router,    prefix="/v1")
app.include_router(analytics_router, prefix="/v1")

# ── Root ───────────────────────────────────────────────────────────────────
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