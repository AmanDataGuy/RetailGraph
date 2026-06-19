"""
RetailGraph — API Request & Response Models
Pydantic v2 models that define the JSON contract for every endpoint.
FastAPI uses these for automatic validation, serialization, and docs.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Shared product model ───────────────────────────────────────────────────

class ProductResult(BaseModel):
    """A single product returned in search/query results."""
    item_name:      str
    price:          Optional[float]  = None
    brand:          Optional[str]    = None
    category:       Optional[str]    = None
    quantity_value: Optional[float]  = None
    quantity_unit:  Optional[str]    = None
    dietary_tags:   list[str]        = []
    allergen_list:  list[str]        = []
    quality_score:  Optional[float]  = None
    hybrid_score:   Optional[float]  = None
    image_url:      Optional[str]    = None


# ── POST /query ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Natural language query sent to the LangGraph agent."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["show me vegan snacks under $10"],
    )

class QueryResponse(BaseModel):
    """Full agent response including answer, results, and transparency info."""
    query:        str
    answer:       str
    intent:       Optional[str]       = None
    route:        Optional[str]       = None
    result_count: int                 = 0
    results:      list[ProductResult] = []
    cypher_used:  Optional[str]       = None
    latency_ms:   Optional[float]     = None


# ── POST /search ───────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Direct filtered search — bypasses the agent, hits Neo4j/Qdrant directly."""
    query:             Optional[str]   = None
    category:          Optional[str]   = None
    dietary_tags:      list[str]       = []
    exclude_allergens: list[str]       = []
    max_price:         Optional[float] = None
    min_price:         Optional[float] = None
    brand:             Optional[str]   = None
    top_k:             int             = Field(default=10, ge=1, le=50)

class SearchResponse(BaseModel):
    result_count: int
    results:      list[ProductResult]
    search_type:  str


# ── GET /products/{product_id} ─────────────────────────────────────────────

class ProductDetail(BaseModel):
    """Full product detail including graph relationships."""
    product_id:       str
    item_name:        str
    price:            Optional[float]  = None
    brand:            Optional[str]    = None
    category:         Optional[str]    = None
    quantity_value:   Optional[float]  = None
    quantity_unit:    Optional[str]    = None
    dietary_tags:     list[str]        = []
    allergen_list:    list[str]        = []
    quality_score:    Optional[float]  = None
    image_url:        Optional[str]    = None
    similar_products: list[dict]       = []


# ── GET /analytics ─────────────────────────────────────────────────────────

class CategoryStat(BaseModel):
    name:          str
    product_count: int
    avg_price:     Optional[float] = None

class BrandStat(BaseModel):
    name:          str
    product_count: int

class TagStat(BaseModel):
    name:          str
    product_count: int

class AnalyticsResponse(BaseModel):
    total_products:   int
    total_brands:     int
    total_categories: int
    categories:       list[CategoryStat]
    top_brands:       list[BrandStat]
    dietary_tags:     list[TagStat]


# ── GET /health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    neo4j:  bool
    qdrant: bool
    groq:   bool