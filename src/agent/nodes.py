"""
RetailGraph — Agent Nodes
Six node functions that power the LangGraph state machine.
Each node receives the full AgentState and returns a partial dict to update it.

Node flow:
    extract_intent → [route_query] → build_cypher OR hybrid_search OR analytics
                                          ↓
                                    execute_query
                                          ↓
                                    format_answer → END
"""

import json
import logging
import re
from typing import Any

from src.agent.state import AgentState
from src.agent.llm import generate, generate_json

log = logging.getLogger("retailgraph.nodes")

# ── Allowed vocabulary ──────────────────────────────────────────────────────
# Verified directly against the live Neo4j graph (MATCH (c:Category) / MATCH
# (t:DietaryTag), 2026-08-21) rather than any code-side prompt text — an
# earlier pass here matched src/extraction/extractor.py's current
# SYSTEM_PROMPT, which turned out NOT to match what's actually in the graph
# (e.g. "Personal Care" not "Personal Care & Beauty", "Bakery & Bread" not
# "Bread & Bakery", no "Breakfast & Cereal"/"Oils & Vinegars"/"Non-Food" at
# all) — the model's real output on the live catalog didn't follow that
# prompt's vocabulary exactly, so the prompt text isn't reliable ground
# truth. The live graph is. Re-verify with the same query if the catalog is
# ever re-ingested.
ALLOWED_CATEGORIES = {
    "Snacks & Candy", "Coffee & Tea", "Condiments & Sauces", "Beverages",
    "Spices & Seasonings", "Grains, Beans & Legumes", "Unknown",
    "Supplements & Health", "Personal Care", "Meat & Seafood",
    "Fruits & Vegetables", "Baby & Kids", "Pet Supplies", "Frozen Foods",
    "Dairy & Eggs", "Bakery & Bread", "Household & Cleaning",
}

# "keto-friendly" (not "keto"), "vegetarian", "halal", "low-sodium", and
# "egg-free" are all real, live tag data with real products — none were in
# any prior version of this list. "paleo", "allergen-free", "cruelty-free"
# were previously assumed present (they're in extractor.py's SYSTEM_PROMPT)
# but have zero matching products in the live graph — dropped as dead
# weight, not because they're wrong values, just because they'd always
# return 0 results today.
ALLOWED_TAGS = {
    "gluten-free", "kosher", "non-GMO", "vegan", "organic", "dairy-free",
    "sugar-free", "keto-friendly", "high-protein", "caffeine-free",
    "nut-free", "soy-free", "vegetarian", "halal", "low-sodium",
    "egg-free", "low-calorie",
}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — extract_intent
# Parses the raw user query into structured intent + entities.
# ══════════════════════════════════════════════════════════════════════════════

INTENT_SYSTEM = """You are a query parser for a grocery product knowledge graph.

Extract the user's search intent and entities from their query.

Return ONLY valid JSON with these exact keys:
{
  "intent": "filter" | "semantic" | "hybrid" | "lookup" | "analytics",
  "category": string or null,
  "dietary_tags": [list of strings] or [],
  "exclude_allergens": [list of strings] or [],
  "max_price": number or null,
  "min_price": number or null,
  "brand": string or null,
  "semantic_query": string or null
}

Intent definitions:
- "filter": user wants products matching hard constraints (tags, price, allergens, category)
- "semantic": user wants products similar to something ("like sriracha", "similar to X")
- "hybrid": both similarity AND constraints ("vegan snacks similar to protein bars")
- "lookup": user asks about a specific product or brand ("tell me about Heinz")
- "analytics": user wants aggregate info ("which category has most products", "top brands")

Allowed categories (use exact spelling):
Snacks & Candy, Coffee & Tea, Condiments & Sauces, Beverages,
Spices & Seasonings, Grains, Beans & Legumes, Supplements & Health,
Personal Care, Meat & Seafood, Fruits & Vegetables, Baby & Kids,
Pet Supplies, Frozen Foods, Dairy & Eggs, Bakery & Bread, Household & Cleaning

Allowed dietary_tags (use exact spelling):
gluten-free, kosher, non-GMO, vegan, organic, dairy-free, sugar-free,
keto-friendly, high-protein, caffeine-free, nut-free, soy-free,
vegetarian, halal, low-sodium, egg-free, low-calorie

For semantic_query: extract the core product concept the user is describing."""


def extract_intent(state: AgentState) -> dict:
    """
    Node 1: Parse query → intent + entities.
    Calls Groq with JSON mode.
    """
    query = state["query"]
    log.info(f"[Node 1] extract_intent | query='{query}'")

    result = generate_json(
        system=INTENT_SYSTEM,
        prompt=f"Parse this grocery search query: {query}",
    )

    # Normalise: keep only tags that are in allowed vocab
    tags = result.get("dietary_tags", []) or []
    tags = [t.lower() for t in tags if t.lower() in ALLOWED_TAGS]

    allergens = result.get("exclude_allergens", []) or []
    allergens = [a.lower() for a in allergens]

    category = result.get("category")
    if category and category not in ALLOWED_CATEGORIES:
        # fuzzy fallback: try case-insensitive match
        matched = next(
            (c for c in ALLOWED_CATEGORIES if c.lower() == category.lower()), None
        )
        category = matched  # None if no match

    entities = {
        "category":          category,
        "dietary_tags":      tags,
        "exclude_allergens": allergens,
        "max_price":         result.get("max_price"),
        "min_price":         result.get("min_price"),
        "brand":             result.get("brand"),
        "semantic_query":    result.get("semantic_query") or query,
    }

    intent = result.get("intent", "filter")
    log.info(f"[Node 1] intent={intent} | entities={entities}")

    return {"intent": intent, "entities": entities}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER — route_query (used as conditional edge function)
# Returns the name of the next node based on intent.
# ══════════════════════════════════════════════════════════════════════════════

def route_query(state: AgentState) -> str:
    """
    Conditional edge: decides which node runs after extract_intent.
    Returns node name as string.
    """
    intent = state.get("intent", "filter")

    if intent == "analytics":
        return "analytics"
    elif intent in ("semantic", "hybrid"):
        return "graphrag"
    else:
        # filter, lookup → Cypher path
        return "cypher"


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3a — build_cypher
# Generates a Cypher query from extracted entities.
# ══════════════════════════════════════════════════════════════════════════════

CYPHER_SYSTEM = """You are a Neo4j Cypher query generator for a grocery product knowledge graph.

Graph schema:
  Nodes:         (:Product), (:Brand), (:Category), (:DietaryTag), (:Allergen)
  Relationships: (Product)-[:BELONGS_TO]->(Category)
                 (Product)-[:MADE_BY]->(Brand)
                 (Product)-[:HAS_TAG]->(DietaryTag)
                 (Product)-[:CONTAINS_ALLERGEN]->(Allergen)

Product properties: item_name, price, quantity_value, quantity_unit,
                    quality_score, extraction_confidence, image_url

Generate a single valid Cypher query. Return ONLY valid JSON:
{
  "cypher": "MATCH ... RETURN ...",
  "explanation": "one sentence explaining what this query does"
}

Rules:
- Always RETURN: p.item_name, p.price, p.quantity_value, p.quantity_unit,
                  p.image_url, b.name AS brand, c.name AS category
- Use OPTIONAL MATCH for Brand (some products have no brand)
- Use WHERE NOT EXISTS for allergen exclusions
- Add LIMIT 10 unless the query is for analytics
- Use case-insensitive string matching where possible: toLower()
- Never use APOC procedures"""


def build_cypher(state: AgentState) -> dict:
    """
    Node 3a: Build Cypher query from intent + entities.
    Uses pre-built templates first; falls back to LLM generation.
    """
    entities = state.get("entities", {}) or {}
    query    = state["query"]

    log.info(f"[Node 3a] build_cypher | entities={entities}")

    # ── Try pre-built templates first (faster, free, reliable) ────────────
    template = _try_template(entities)

    if template:
        cypher, params = template
        log.info("[Node 3a] Using pre-built template")
        return {
            "cypher_query": cypher,
            "cypher_params": params,
            "cypher_valid": True,
            "route": "cypher",
        }

    # ── Fall back to LLM-generated Cypher ─────────────────────────────────
    retries = state.get("cypher_retries", 0)
    prev_error = state.get("cypher_error")

    prompt = (
        f"User query: {query}\n"
        f"Extracted entities: {json.dumps(entities)}\n"
        "Generate the Cypher query."
    )
    if retries and prev_error:
        # Feed the previous failure back in so a retry can actually produce a
        # different query, instead of regenerating the same one deterministically.
        prompt += (
            f"\n\nYour previous attempt failed with this error:\n{prev_error}\n"
            "Fix the query so it avoids this error."
        )

    result = generate_json(system=CYPHER_SYSTEM, prompt=prompt)
    cypher = result.get("cypher", "")

    if not cypher:
        return {
            "cypher_query": None,
            "cypher_params": {},
            "cypher_valid": False,
            "cypher_error": "LLM returned empty Cypher",
            "route": "cypher",
        }

    # Basic validation
    valid, error = _validate_cypher(cypher)
    log.info(f"[Node 3a] LLM cypher valid={valid} | cypher={cypher[:80]}...")

    return {
        "cypher_query": cypher,
        "cypher_params": {},
        "cypher_valid": valid,
        "cypher_error": error if not valid else None,
        "route": "cypher",
    }


def _try_template(entities: dict) -> tuple[str, dict] | None:
    """
    Returns a (cypher, params) pair if entities match a known pattern.
    All user-derived values (brand, category, tags, allergens) are passed as
    bound $params rather than interpolated into the query string — an
    f-string here would let a value like "Reese's" break the query, or let
    a crafted entity value inject arbitrary Cypher.
    Returns None if no template matches → falls back to LLM.
    """
    tags      = entities.get("dietary_tags", [])
    category  = entities.get("category")
    max_p     = entities.get("max_price")
    min_p     = entities.get("min_price")
    brand     = entities.get("brand")
    allergens = entities.get("exclude_allergens", [])

    # Brand lookup
    if brand and not tags and not category:
        cypher = (
            "MATCH (p:Product)-[:MADE_BY]->(b:Brand) "
            "WHERE toLower(b.name) = toLower($brand) "
            "OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category) "
            "RETURN p.item_name AS item_name, p.price AS price, "
            "p.quantity_value AS quantity_value, p.quantity_unit AS quantity_unit, "
            "p.image_url AS image_url, "
            "b.name AS brand, c.name AS category "
            "ORDER BY p.price ASC LIMIT 10"
        )
        return cypher, {"brand": brand}

    # Build dynamic MATCH + WHERE
    match_clauses = ["MATCH (p:Product)"]
    where_clauses = []
    params: dict = {}
    return_clause = (
        "RETURN p.item_name AS item_name, p.price AS price, "
        "p.quantity_value AS quantity_value, p.quantity_unit AS quantity_unit, "
        "p.image_url AS image_url, "
        "b.name AS brand, c.name AS category "
        "ORDER BY p.price ASC LIMIT 10"
    )

    if category:
        match_clauses.append("MATCH (p)-[:BELONGS_TO]->(c:Category {name: $category})")
        params["category"] = category
    else:
        match_clauses.append("OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)")

    match_clauses.append("OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)")

    for i, tag in enumerate(tags):
        match_clauses.append(f"MATCH (p)-[:HAS_TAG]->(:DietaryTag {{name: $tag_{i}}})")
        params[f"tag_{i}"] = tag

    if max_p is not None:
        where_clauses.append("p.price <= $max_price")
        params["max_price"] = max_p
    if min_p is not None:
        where_clauses.append("p.price >= $min_price")
        params["min_price"] = min_p

    for i, allergen in enumerate(allergens):
        where_clauses.append(
            f"NOT EXISTS {{ MATCH (p)-[:CONTAINS_ALLERGEN]->(:Allergen {{name: $excl_{i}}}) }}"
        )
        params[f"excl_{i}"] = allergen

    cypher = "\n".join(match_clauses)
    if where_clauses:
        cypher += "\nWHERE " + " AND ".join(where_clauses)
    cypher += "\n" + return_clause

    # Only return template if at least one constraint was applied
    if tags or category or max_p or min_p or allergens or brand:
        return cypher, params

    return None  # no constraints → let LLM handle it


def _validate_cypher(cypher: str) -> tuple[bool, str | None]:
    """Basic Cypher validation — catches the most common LLM mistakes."""
    cypher_upper = cypher.upper()

    if "MATCH" not in cypher_upper:
        return False, "Missing MATCH clause"
    if "RETURN" not in cypher_upper:
        return False, "Missing RETURN clause"
    if "DELETE" in cypher_upper or "DETACH" in cypher_upper:
        return False, "Destructive operations not allowed"
    if "CREATE" in cypher_upper or "MERGE" in cypher_upper:
        return False, "Write operations not allowed"

    return True, None


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3b — hybrid_search_node
# Calls src/graph/hybrid_search.py for semantic + GraphRAG queries.
# ══════════════════════════════════════════════════════════════════════════════

def hybrid_search_node(state: AgentState) -> dict:
    """
    Node 3b: GraphRAG path — semantic + constraint search via Qdrant + Neo4j.
    """
    from src.graph.hybrid_search import HybridSearch

    entities  = state.get("entities", {}) or {}
    sem_query = entities.get("semantic_query") or state["query"]

    log.info(f"[Node 3b] hybrid_search | query='{sem_query}'")

    hs = HybridSearch()

    # Build filter kwargs from entities
    kwargs: dict[str, Any] = {}
    if entities.get("category"):
        kwargs["category"] = entities["category"]
    if entities.get("max_price") is not None:
        kwargs["max_price"] = entities["max_price"]
    if entities.get("dietary_tags"):
        kwargs["dietary_tags"] = entities["dietary_tags"]

    results = hs.search(sem_query, top_k=10, **kwargs)

    # Normalise to list of dicts
    raw = []
    for r in results:
        raw.append({
            "item_name":    r.get("item_name", ""),
            "price":        r.get("price"),
            "brand":        r.get("brand"),
            "category":     r.get("category"),
            "dietary_tags": r.get("dietary_tags", []),
            "hybrid_score": round(r.get("hybrid_score", 0), 3),
            "image_url":    r.get("image_url"),
        })

    log.info(f"[Node 3b] returned {len(raw)} results")
    return {
        "raw_results":  raw,
        "result_count": len(raw),
        "route":        "graphrag",
        "cypher_used":  "GraphRAG (Qdrant semantic + Neo4j constraints)",
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3c — analytics_node
# Handles aggregate queries using pre-built Cypher from queries.py
# ══════════════════════════════════════════════════════════════════════════════

def analytics_node(state: AgentState) -> dict:
    """
    Node 3c: Analytics path — runs aggregate Cypher queries.
    """
    from src.graph.queries import GraphQueries

    query = state["query"].lower()
    log.info(f"[Node 3c] analytics | query='{query}'")

    gq = GraphQueries()

    if "brand" in query:
        results = gq.get_top_brands(limit=10)
        cypher  = "MATCH (p:Product)-[:MADE_BY]->(b:Brand) RETURN b.name, count(p) ORDER BY count(p) DESC LIMIT 10"
    elif "tag" in query or "dietary" in query:
        results = gq.get_dietary_tag_stats()
        cypher  = "MATCH (p:Product)-[:HAS_TAG]->(t:DietaryTag) RETURN t.name, count(p) ORDER BY count(p) DESC"
    else:
        results = gq.get_category_stats()
        cypher  = "MATCH (p:Product)-[:BELONGS_TO]->(c:Category) RETURN c.name, count(p), avg(p.price) ORDER BY count(p) DESC"

    gq.close()

    return {
        "raw_results":  results,
        "result_count": len(results),
        "route":        "analytics",
        "cypher_used":  cypher,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — execute_query
# Runs the Cypher query against Neo4j. Only used on the Cypher path.
# ══════════════════════════════════════════════════════════════════════════════

# Process-wide singleton — mirrors src/graph/hybrid_search.py. Opening a new
# driver (and its connection pool) on every query was the same per-request
# reconnect cost that hybrid_search.py was fixed to avoid.
_DRIVER = None


def _get_driver():
    global _DRIVER
    if _DRIVER is None:
        import os
        from neo4j import GraphDatabase
        from dotenv import load_dotenv
        load_dotenv()
        _DRIVER = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
        )
    return _DRIVER


def execute_query(state: AgentState) -> dict:
    """
    Node 4: Execute Cypher against Neo4j and return raw results.
    Handles retry logic — if cypher_valid is False, bumps retry counter.
    """
    cypher  = state.get("cypher_query")
    params  = state.get("cypher_params") or {}
    valid   = state.get("cypher_valid", False)
    retries = state.get("cypher_retries", 0)

    log.info(f"[Node 4] execute_query | valid={valid} | retries={retries}")

    # If Cypher is invalid and we have retries left → signal retry
    if not valid:
        if retries < 2:
            return {"cypher_retries": retries + 1}
        else:
            return {
                "raw_results":  [],
                "result_count": 0,
                "error":        f"Cypher generation failed after {retries} retries: {state.get('cypher_error')}",
            }

    import os

    driver = _get_driver()

    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            result = session.run(cypher, params)
            records = [dict(r) for r in result]

        log.info(f"[Node 4] Neo4j returned {len(records)} records")
        return {
            "raw_results":  records,
            "result_count": len(records),
            "cypher_used":  cypher,
        }

    except Exception as e:
        log.error(f"[Node 4] Neo4j error: {e}")
        return {
            "raw_results":  [],
            "result_count": 0,
            "error":        str(e),
            "cypher_valid": False,
            "cypher_error": str(e),
            "cypher_retries": retries + 1,
        }


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — format_answer
# Converts raw results into a clean plain-English answer via Groq.
# ══════════════════════════════════════════════════════════════════════════════

FORMAT_SYSTEM = """You are a helpful grocery product assistant.

The user asked a question and a knowledge graph returned results.
Write a clear, friendly answer in 2-4 sentences.

Rules:
- Lead with the count: "Found X products matching your search."
- Mention the top 2-3 results with name and price
- If no results: say so and suggest relaxing the filters
- Keep it concise — no bullet points, no markdown
- Never make up products that aren't in the results"""


def format_answer(state: AgentState) -> dict:
    """
    Node 5: Format raw results into plain-English answer via Groq.
    """
    query   = state["query"]
    results = state.get("raw_results") or []
    count   = state.get("result_count", 0)
    error   = state.get("error")

    log.info(f"[Node 5] format_answer | results={count} | error={error}")

    # Handle error state
    if error and not results:
        return {
            "answer": (
                f"I wasn't able to complete that search. {error}. "
                "Try rephrasing your query or relaxing the filters."
            )
        }

    # Handle empty results
    if not results:
        return {
            "answer": (
                f"No products found matching '{query}'. "
                "Try broadening your search — remove some filters or use a different category."
            )
        }

    # Detect analytics vs product results
    first        = results[0] if results else {}
    is_analytics = "item_name" not in first and "p.item_name" not in first

    top = results[:5]

    if is_analytics:
        results_text = "\n".join(
            " | ".join(f"{k}: {v}" for k, v in r.items())
            for r in top
        )
        prompt = (
            f"User query: {query}\n"
            f"Analytics results ({count} rows):\n{results_text}\n\n"
            "Answer the user's question directly using this data. "
            "Be specific — mention actual names and numbers from the results."
        )
    else:
        results_text = "\n".join(
            f"- {r.get('item_name') or r.get('p.item_name', 'Unknown')} | "
            f"${r.get('price') or r.get('p.price', 'N/A')} | "
            f"Brand: {r.get('brand') or 'unknown'} | "
            f"Category: {r.get('category') or 'unknown'}"
            for r in top
        )
        prompt = (
            f"User query: {query}\n"
            f"Total results found: {count}\n"
            f"Top results:\n{results_text}\n\n"
            "Write a helpful answer mentioning product names and prices."
        )

    answer = generate(system=FORMAT_SYSTEM, prompt=prompt)
    log.info(f"[Node 5] answer generated ({len(answer)} chars)")

    return {"answer": answer}