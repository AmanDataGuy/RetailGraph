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
- Never use APOC procedures
- CRITICAL: put any WHERE clause immediately after the first (non-optional)
  MATCH, BEFORE any OPTIONAL MATCH clauses. A WHERE placed after an OPTIONAL
  MATCH only filters that optional pattern — it will NOT exclude non-matching
  rows, since OPTIONAL MATCH never fails a row. Correct order:
    MATCH (p:Product)
    WHERE toLower(p.item_name) CONTAINS toLower('...')
    OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)
    OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
    RETURN ..."""


def build_cypher(state: AgentState) -> dict:
    """
    Node 3a: Build Cypher query from intent + entities.
    Uses pre-built templates first; falls back to LLM generation for "lookup"
    queries only (see the no-entities branch below for why).
    """
    entities = state.get("entities", {}) or {}
    query    = state["query"]
    intent   = state.get("intent")

    log.info(f"[Node 3a] build_cypher | entities={entities}")

    # ── Try pre-built templates first (faster, free, reliable) ────────────
    template = _try_template(entities)

    if template:
        cypher, params, count_cypher = template
        log.info("[Node 3a] Using pre-built template")
        return {
            "cypher_query": cypher,
            "cypher_params": params,
            "cypher_count_query": count_cypher,
            "cypher_valid": True,
            "route": "cypher",
        }

    # "filter" intent means hard constraints (tags/price/category/brand/
    # allergens). If none were extracted, there is nothing to filter on —
    # asking the LLM to invent a Cypher query here previously fabricated an
    # arbitrary top-10-by-price result set for queries with no real signal
    # (e.g. gibberish input), instead of the zero matches those queries
    # actually have. "lookup" intent (a specific product/brand by name)
    # still needs the LLM fallback since `entities` has no product-name
    # field to template against.
    if intent != "lookup":
        log.info("[Node 3a] No entities extracted and intent != lookup — no query to run")
        return {
            "cypher_query": None,
            "cypher_params": {},
            "cypher_count_query": None,
            "cypher_valid": True,
            "raw_results": [],
            "result_count": 0,
            "total_count": 0,
            "route": "cypher",
        }

    # ── Fall back to LLM-generated Cypher (lookup-by-name only) ────────────
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
            "cypher_count_query": None,
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
        "cypher_count_query": None,   # LLM-generated Cypher has no matching count query — count-aware phrasing doesn't apply here
        "cypher_valid": valid,
        "cypher_error": error if not valid else None,
        "route": "cypher",
    }


def _try_template(entities: dict) -> tuple[str, dict, str] | None:
    """
    Returns a (cypher, params, count_cypher) triple if entities match a known
    pattern. `count_cypher` mirrors the same MATCH/WHERE clauses but returns
    the true match count (no LIMIT) — build_cypher/execute_query use it to
    tell "10 shown" apart from "10 of 29 total", instead of the LIMIT-capped
    result count being reported to the user as if it were the total.

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

    # Brand lookup — only when brand is the SOLE constraint. This branch
    # builds a query that filters by brand alone; if price/allergen
    # constraints are also present (e.g. "products from McCormick under
    # $10"), taking this shortcut would silently drop them entirely. Any
    # other combination falls through to the general dynamic builder below,
    # which handles brand correctly alongside every other filter.
    if (
        brand and not tags and not category
        and max_p is None and min_p is None and not allergens
    ):
        match_where = (
            "MATCH (p:Product)-[:MADE_BY]->(b:Brand) "
            "WHERE toLower(b.name) = toLower($brand) "
        )
        cypher = (
            match_where +
            "OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category) "
            "RETURN p.item_name AS item_name, p.price AS price, "
            "p.quantity_value AS quantity_value, p.quantity_unit AS quantity_unit, "
            "p.image_url AS image_url, "
            "b.name AS brand, c.name AS category "
            "ORDER BY p.price ASC LIMIT 10"
        )
        count_cypher = match_where + "RETURN count(DISTINCT p) AS total"
        return cypher, {"brand": brand}, count_cypher

    # Build dynamic MATCH + WHERE.
    #
    # CRITICAL ordering constraint, confirmed live against real Neo4j: a
    # WHERE clause binds to whichever read clause (MATCH/OPTIONAL MATCH)
    # immediately precedes it — regardless of which variables it actually
    # references. If that clause is an OPTIONAL MATCH, the WHERE becomes
    # part of that optional pattern (a failed condition just nulls the
    # optional binding) instead of a row-level filter, so it silently stops
    # excluding non-matching rows. This affected even `p.price <= $max_price`
    # (a condition that only touches the mandatorily-matched `p`) whenever it
    # was placed after an OPTIONAL MATCH for category/brand — i.e. every
    # price-only or allergen-only filter with no brand/tags/category was
    # silently returning unfiltered results. Fix: ALL mandatory MATCH
    # clauses go first, then the single WHERE block, then OPTIONAL MATCH
    # clauses (added only for display enrichment, never for filtering) last.
    mandatory_matches = ["MATCH (p:Product)"]
    optional_matches: list[str] = []
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
        mandatory_matches.append("MATCH (p)-[:BELONGS_TO]->(c:Category {name: $category})")
        params["category"] = category
    else:
        optional_matches.append("OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)")

    if brand:
        mandatory_matches.append("MATCH (p)-[:MADE_BY]->(b:Brand)")
        where_clauses.append("toLower(b.name) = toLower($brand)")
        params["brand"] = brand
    else:
        optional_matches.append("OPTIONAL MATCH (p)-[:MADE_BY]->(b:Brand)")

    for i, tag in enumerate(tags):
        mandatory_matches.append(f"MATCH (p)-[:HAS_TAG]->(:DietaryTag {{name: $tag_{i}}})")
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

    match_block    = "\n".join(mandatory_matches)
    where_block    = ("\nWHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    optional_block = ("\n" + "\n".join(optional_matches)) if optional_matches else ""

    cypher = match_block + where_block + optional_block + "\n" + return_clause
    count_cypher = match_block + where_block + "\nRETURN count(DISTINCT p) AS total"

    # Only return template if at least one constraint was applied
    if tags or category or max_p or min_p or allergens or brand:
        return cypher, params, count_cypher

    return None  # no constraints → build_cypher decides what to do


_MATCH_CLAUSE_RE = re.compile(r"\b(OPTIONAL\s+MATCH|MATCH)\b", re.IGNORECASE)
_WHERE_CLAUSE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)


def _where_scoped_to_optional_match(cypher: str) -> bool:
    """
    True if any WHERE in the query immediately follows an OPTIONAL MATCH
    rather than a plain MATCH. Confirmed live against real Neo4j: Cypher
    scopes WHERE to whichever read clause directly precedes it, regardless
    of which variables the WHERE actually references — if that clause is
    OPTIONAL MATCH, the condition becomes part of the optional pattern (a
    failed check just nulls the optional binding) instead of a row filter,
    so it silently stops excluding non-matching rows.
    """
    clauses = [
        (m.start(), "OPTIONAL" if m.group(1).upper().startswith("OPTIONAL") else "MATCH")
        for m in _MATCH_CLAUSE_RE.finditer(cypher)
    ]
    for wm in _WHERE_CLAUSE_RE.finditer(cypher):
        preceding = [c for c in clauses if c[0] < wm.start()]
        if preceding and preceding[-1][1] == "OPTIONAL":
            return True
    return False


def _has_no_filter_condition(cypher: str) -> bool:
    """
    True if the query filters nothing at all — no WHERE clause, and no
    inline {...} property constraint on any non-optional MATCH. Confirmed
    live: for "lookup" queries with no identifiable search term, the LLM
    sometimes generates exactly this — just the base Product match plus
    OPTIONAL MATCH enrichment clauses, no filter anywhere — which silently
    returns an arbitrary, unconstrained slice of the catalog instead of
    recognizing there's nothing to search for.
    """
    if _WHERE_CLAUSE_RE.search(cypher):
        return False

    markers = list(_MATCH_CLAUSE_RE.finditer(cypher))
    for i, m in enumerate(markers):
        is_optional = m.group(1).upper().startswith("OPTIONAL")
        if is_optional:
            continue
        end = markers[i + 1].start() if i + 1 < len(markers) else len(cypher)
        clause_text = cypher[m.start():end]
        if re.search(r"\{[^}]*:", clause_text):
            return False
    return True


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
    if _has_no_filter_condition(cypher):
        return False, (
            "Query has no WHERE clause and no inline property constraint on "
            "any non-optional MATCH, so it filters nothing and would return "
            "an arbitrary slice of the catalog. If there is no identifiable "
            "search term in the user's query, return an empty cypher string "
            "instead of an unconstrained one."
        )
    if _where_scoped_to_optional_match(cypher):
        return False, (
            "WHERE immediately follows an OPTIONAL MATCH, which scopes the "
            "filter to that optional pattern only — it will not exclude "
            "non-matching rows. Move WHERE immediately after the first "
            "(non-optional) MATCH, before any OPTIONAL MATCH clauses."
        )

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
    cypher       = state.get("cypher_query")
    params       = state.get("cypher_params") or {}
    count_cypher = state.get("cypher_count_query")
    valid        = state.get("cypher_valid", False)
    retries      = state.get("cypher_retries", 0)

    log.info(f"[Node 4] execute_query | valid={valid} | retries={retries}")

    # build_cypher already decided there's nothing to search for (no
    # entities extracted, non-lookup intent) — raw_results/result_count are
    # already set to empty in state; nothing to run against Neo4j.
    # LangGraph requires every node to write at least one state key, so this
    # re-states build_cypher's values rather than returning {}.
    if cypher is None:
        log.info("[Node 4] No cypher to run — passing through build_cypher's empty result")
        return {
            "raw_results":  state.get("raw_results", []),
            "result_count": state.get("result_count", 0),
            "total_count":  state.get("total_count", 0),
        }

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

            total_count = None
            if count_cypher:
                count_record = session.run(count_cypher, params).single()
                total_count = count_record["total"] if count_record else len(records)

        log.info(f"[Node 4] Neo4j returned {len(records)} records (total_count={total_count})")
        return {
            "raw_results":  records,
            "result_count": len(records),
            "total_count":  total_count,
            "cypher_used":  cypher,
        }

    except Exception as e:
        log.error(f"[Node 4] Neo4j error: {e}")
        return {
            "raw_results":  [],
            "result_count": 0,
            "total_count":  None,
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

The user asked a question and a knowledge graph returned matching products.
Write a clear, friendly answer in 2-4 sentences.

Rules:
- Lead with the exact count given to you in the prompt (e.g. "Found 29 matching products, showing the top 10" if a total is given and it's larger than what's shown, otherwise "Found 10 matching products"). Never invent your own count or assume the number of results shown is the total.
- Mention the top 2-3 results with name and price
- If no results: say so and suggest relaxing the filters
- Keep it concise — no bullet points, no markdown
- Never make up products that aren't in the results"""

# Analytics answers describe aggregate rows (categories/brands/tags), not a
# list of matching products — the product-search framing above ("Found X
# matching your search") doesn't fit an aggregation and was found to read as
# off-topic (an eval run scored several factually-correct analytics answers
# 0.0 on correctness, all using that phrasing — see evaluation/deepeval_results.json).
ANALYTICS_FORMAT_SYSTEM = """You are a helpful grocery product assistant.

The user asked an aggregate question (a count, average, or ranking) and a
knowledge graph returned the aggregate data. Write a clear, direct answer in
2-4 sentences.

Rules:
- Answer the question directly with the top result first — do NOT say "Found X results matching your search" or similar; this is aggregate data, not a list of matching products.
- Cite exact names and numbers from the data given to you
- Mention 2-3 more rows of context if useful
- Keep it concise — no bullet points, no markdown
- Never make up numbers or names that aren't in the data"""


def format_answer(state: AgentState) -> dict:
    """
    Node 5: Format raw results into plain-English answer via Groq.
    """
    query       = state["query"]
    results     = state.get("raw_results") or []
    count       = state.get("result_count", 0)
    total_count = state.get("total_count")
    error       = state.get("error")

    log.info(f"[Node 5] format_answer | results={count} | error={error}")

    # Handle error state. The real error is already logged above (and
    # server-side in execute_query/build_cypher) — it's an internal Cypher
    # validation/execution detail (e.g. "no filter condition"), not
    # something a user asking a product question should see verbatim.
    if error and not results:
        return {
            "answer": (
                f"I wasn't able to complete that search for '{query}'. "
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
        answer = generate(system=ANALYTICS_FORMAT_SYSTEM, prompt=prompt)
    else:
        results_text = "\n".join(
            f"- {r.get('item_name') or r.get('p.item_name', 'Unknown')} | "
            f"${r.get('price') or r.get('p.price', 'N/A')} | "
            f"Brand: {r.get('brand') or 'unknown'} | "
            f"Category: {r.get('category') or 'unknown'}"
            for r in top
        )
        if total_count is not None and total_count > count:
            count_line = f"Total matching products: {total_count} (showing top {count})"
        else:
            count_line = f"Total results found: {count}"
        prompt = (
            f"User query: {query}\n"
            f"{count_line}\n"
            f"Top results:\n{results_text}\n\n"
            "Write a helpful answer mentioning product names and prices."
        )
        answer = generate(system=FORMAT_SYSTEM, prompt=prompt)

    log.info(f"[Node 5] answer generated ({len(answer)} chars)")

    return {"answer": answer}