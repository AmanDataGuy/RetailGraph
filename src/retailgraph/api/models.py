"""
# models.py - Pydantic request/response models for every API endpoint

- FastAPI uses them for validation, serialization and the /docs schema.
- Imported by api/routes/*.py.
"""


from pydantic import BaseModel, Field


class ProductResult(BaseModel):
    """A single product returned in search/query results."""
    item_name:      str
    price:          float | None  = None
    brand:          str | None    = None
    category:       str | None    = None
    quantity_value: float | None  = None
    quantity_unit:  str | None    = None
    dietary_tags:   list[str]        = []
    allergen_list:  list[str]        = []
    quality_score:  float | None  = None
    hybrid_score:   float | None  = None
    image_url:      str | None    = None


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
    intent:       str | None       = None
    route:        str | None       = None
    result_count: int                 = 0
    total_matches: int | None      = None
    results:      list[ProductResult] = []
    cypher_used:  str | None       = None
    latency_ms:   float | None     = None


class SearchRequest(BaseModel):
    """Direct filtered search — bypasses the agent, hits Neo4j/Qdrant directly."""
    query:             str | None   = None
    category:          str | None   = None
    dietary_tags:      list[str]       = []
    exclude_allergens: list[str]       = []
    max_price:         float | None = None
    min_price:         float | None = None
    brand:             str | None   = None
    top_k:             int             = Field(default=10, ge=1, le=50)


# POST /search response.
class SearchResponse(BaseModel):
    result_count: int
    results:      list[ProductResult]
    search_type:  str


class ProductDetail(BaseModel):
    """Full product detail including graph relationships."""
    product_id:       str
    item_name:        str
    price:            float | None  = None
    brand:            str | None    = None
    category:         str | None    = None
    quantity_value:   float | None  = None
    quantity_unit:    str | None    = None
    dietary_tags:     list[str]        = []
    allergen_list:    list[str]        = []
    quality_score:    float | None  = None
    image_url:        str | None    = None
    similar_products: list[dict]       = []


# Product count and average price for one category.
class CategoryStat(BaseModel):
    name:          str
    product_count: int
    avg_price:     float | None = None


# Product count for one brand.
class BrandStat(BaseModel):
    name:          str
    product_count: int


# Product count for one dietary tag.
class TagStat(BaseModel):
    name:          str
    product_count: int


# GET /analytics response: totals plus per-category, brand and tag stats.
class AnalyticsResponse(BaseModel):
    total_products:   int
    total_brands:     int
    total_categories: int
    categories:       list[CategoryStat]
    top_brands:       list[BrandStat]
    dietary_tags:     list[TagStat]


# GET /health response: which backend services are reachable.
class HealthResponse(BaseModel):
    status: str
    neo4j:  bool
    qdrant: bool
    groq:   bool