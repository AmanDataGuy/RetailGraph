"""
# nodes.py - the LangGraph agent's node functions and their prompts

- extract_intent, route_query, build_cypher, hybrid_search_node, analytics_node, execute_query, format_answer.
- Each node takes the AgentState and returns a partial dict to merge into it; wired up by agent/graph.py.
"""

import json
import logging
import os
import re

from retailgraph.agent.llm import generate, generate_json
from retailgraph.agent.state import AgentState
from retailgraph.graph.queries import allergen_exclusion, allergen_terms

log = logging.getLogger("retailgraph.nodes")

# Category and tag names exactly as stored in the live Neo4j graph (re-check if the catalog is re-ingested).
ALLOWED_CATEGORIES = {
    "Snacks & Candy", "Coffee & Tea", "Condiments & Sauces", "Beverages",
    "Spices & Seasonings", "Grains, Beans & Legumes", "Unknown",
    "Supplements & Health", "Personal Care", "Meat & Seafood",
    "Fruits & Vegetables", "Baby & Kids", "Pet Supplies", "Frozen Foods",
    "Dairy & Eggs", "Bakery & Bread", "Household & Cleaning",
}

ALLOWED_TAGS = {
    "gluten-free", "kosher", "non-GMO", "vegan", "organic", "dairy-free",
    "sugar-free", "keto-friendly", "high-protein", "caffeine-free",
    "nut-free", "soy-free", "vegetarian", "halal", "low-sodium",
    "egg-free", "low-calorie",
}
_TAG_BY_LOWER = {t.lower(): t for t in ALLOWED_TAGS}


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

Words like "style", "like", "similar", "alternatives" always mean "semantic" (or
"hybrid" if there is also a tag/price/allergen constraint), never "filter".
Set category whenever the query names a product type: snacks/chips/bars/candy -> Snacks & Candy, coffee/tea -> Coffee & Tea,
drinks/juice/soda -> Beverages, sauces/condiments -> Condiments & Sauces,
spices/seasoning -> Spices & Seasonings, vitamins/protein powder -> Supplements & Health.

For semantic_query: extract the core product concept the user is describing."""


# Parse the user's query into an intent and filter entities, keeping only known tags/categories.
def extract_intent(state: AgentState) -> dict:
    query = state["query"]
    log.info(f"[Node 1] extract_intent | query='{query}'")

    result = generate_json(
        system=INTENT_SYSTEM,
        prompt=f"Parse this grocery search query: {query}",
    )

    # Case-insensitive lookup so "non-gmo" still maps to the graph's "non-GMO".
    tags = [_TAG_BY_LOWER[t.lower()] for t in result.get("dietary_tags") or [] if t.lower() in _TAG_BY_LOWER]

    category = result.get("category")
    if category and category not in ALLOWED_CATEGORIES:
        category = next((c for c in ALLOWED_CATEGORIES if c.lower() == category.lower()), None)

    entities = {
        "category":          category,
        "dietary_tags":      tags,
        "exclude_allergens": [a.lower() for a in result.get("exclude_allergens") or []],
        "max_price":         result.get("max_price"),
        "min_price":         result.get("min_price"),
        "brand":             result.get("brand"),
        "semantic_query":    result.get("semantic_query") or query,
    }

    intent = result.get("intent", "filter")
    log.info(f"[Node 1] intent={intent} | entities={entities}")

    return {"intent": intent, "entities": entities}


# Pick the next node from the intent: analytics, graphrag (semantic/hybrid) or cypher (everything else).
def route_query(state: AgentState) -> str:
    intent = state.get("intent", "filter")
    if intent == "analytics":
        return "analytics"
    if intent in ("semantic", "hybrid"):
        return "graphrag"
    return "cypher"


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


# Build the Cypher query: a safe template when filters exist, else LLM-written Cypher for "lookup" only.
def build_cypher(state: AgentState) -> dict:
    entities = state.get("entities", {}) or {}
    query    = state["query"]
    intent   = state.get("intent")

    log.info(f"[Node 3a] build_cypher | entities={entities}")

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

    # No filters and not a by-name lookup: nothing to search for. Letting the LLM invent a
    # query here returns an arbitrary slice of the catalog instead of zero matches.
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

    retries = state.get("cypher_retries", 0)
    prev_error = state.get("cypher_error")

    prompt = (
        f"User query: {query}\n"
        f"Extracted entities: {json.dumps(entities)}\n"
        "Generate the Cypher query."
    )
    if retries and prev_error:
        # Feed the error back so a deterministic retry can produce a different query.
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

    valid, error = _validate_cypher(cypher)
    log.info(f"[Node 3a] LLM cypher valid={valid} | cypher={cypher[:80]}...")

    return {
        "cypher_query": cypher,
        "cypher_params": {},
        "cypher_count_query": None,   # no count query for LLM-written Cypher
        "cypher_valid": valid,
        "cypher_error": error if not valid else None,
        "route": "cypher",
    }


_RETURN_PRODUCTS = (
    "RETURN p.item_name AS item_name, p.price AS price, "
    "p.quantity_value AS quantity_value, p.quantity_unit AS quantity_unit, "
    "p.image_url AS image_url, "
    "b.name AS brand, c.name AS category "
    "ORDER BY p.price ASC LIMIT 10"
)


# Build (cypher, params, count_cypher) from the entities, or None when there are no filters.
# Every user value is a bound $param, never interpolated, so "Reese's" can't break or inject Cypher.
# count_cypher shares the MATCH/WHERE but returns the full match count, so answers can say "10 of 29".
def _try_template(entities: dict) -> tuple[str, dict, str] | None:
    tags      = entities.get("dietary_tags", [])
    category  = entities.get("category")
    max_p     = entities.get("max_price")
    min_p     = entities.get("min_price")
    brand     = entities.get("brand")
    allergens = entities.get("exclude_allergens", [])

    # Brand-only shortcut; any other filter combined with brand uses the general builder below.
    if (
        brand and not tags and not category
        and max_p is None and min_p is None and not allergens
    ):
        match_where = (
            "MATCH (p:Product)-[:MADE_BY]->(b:Brand) "
            "WHERE toLower(b.name) = toLower($brand) "
        )
        cypher = match_where + "OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category) " + _RETURN_PRODUCTS
        count_cypher = match_where + "RETURN count(DISTINCT p) AS total"
        return cypher, {"brand": brand}, count_cypher

    # Order matters: Neo4j scopes a WHERE to the clause right before it, and after an OPTIONAL MATCH
    # it stops filtering rows. So: mandatory MATCHes, then one WHERE, then display-only OPTIONAL MATCHes.
    mandatory_matches = ["MATCH (p:Product)"]
    optional_matches: list[str] = []
    where_clauses = []
    params: dict = {}

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
        where_clauses.append(allergen_exclusion(f"excl_{i}"))
        params[f"excl_{i}"] = allergen_terms(allergen)

    match_block    = "\n".join(mandatory_matches)
    where_block    = ("\nWHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    optional_block = ("\n" + "\n".join(optional_matches)) if optional_matches else ""

    cypher = match_block + where_block + optional_block + "\n" + _RETURN_PRODUCTS
    count_cypher = match_block + where_block + "\nRETURN count(DISTINCT p) AS total"

    if tags or category or max_p is not None or min_p is not None or allergens or brand:
        return cypher, params, count_cypher

    return None


_MATCH_CLAUSE_RE = re.compile(r"\b(OPTIONAL\s+MATCH|MATCH)\b", re.IGNORECASE)
# Clauses that write or call procedures. execute_query also runs in a read transaction (second lock).
_WRITE_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|FOREACH|LOAD\s+CSV)\b",
    re.IGNORECASE,
)
_WHERE_CLAUSE_RE = re.compile(r"\bWHERE\b", re.IGNORECASE)


# True if any WHERE directly follows an OPTIONAL MATCH, where it silently stops filtering rows.
def _where_scoped_to_optional_match(cypher: str) -> bool:
    clauses = [
        (m.start(), "OPTIONAL" if m.group(1).upper().startswith("OPTIONAL") else "MATCH")
        for m in _MATCH_CLAUSE_RE.finditer(cypher)
    ]
    for wm in _WHERE_CLAUSE_RE.finditer(cypher):
        preceding = [c for c in clauses if c[0] < wm.start()]
        if preceding and preceding[-1][1] == "OPTIONAL":
            return True
    return False


# True if the query has no WHERE and no inline {prop: ...} on a non-optional MATCH, so it filters nothing.
def _has_no_filter_condition(cypher: str) -> bool:
    if _WHERE_CLAUSE_RE.search(cypher):
        return False

    markers = list(_MATCH_CLAUSE_RE.finditer(cypher))
    for i, m in enumerate(markers):
        if m.group(1).upper().startswith("OPTIONAL"):
            continue
        end = markers[i + 1].start() if i + 1 < len(markers) else len(cypher)
        if re.search(r"\{[^}]*:", cypher[m.start():end]):
            return False
    return True


# Reject LLM Cypher with common mistakes; returns (valid, error message fed back on retry).
def _validate_cypher(cypher: str) -> tuple[bool, str | None]:
    cypher_upper = cypher.upper()

    if "MATCH" not in cypher_upper:
        return False, "Missing MATCH clause"
    if "RETURN" not in cypher_upper:
        return False, "Missing RETURN clause"
    if _WRITE_RE.search(cypher):
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


# Semantic + constraint search through Qdrant and Neo4j (the "graphrag" route).
def hybrid_search_node(state: AgentState) -> dict:
    from retailgraph.graph.hybrid_search import HybridSearch

    entities  = state.get("entities", {}) or {}
    sem_query = entities.get("semantic_query") or state["query"]

    log.info(f"[Node 3b] hybrid_search | query='{sem_query}'")

    kwargs = {k: entities[k] for k in ("category", "dietary_tags", "exclude_allergens", "brand") if entities.get(k)}
    kwargs.update({k: entities[k] for k in ("max_price", "min_price") if entities.get(k) is not None})

    results = HybridSearch().search(sem_query, top_k=10, **kwargs)
    raw = [{
        "item_name":    r.get("item_name", ""),
        "price":        r.get("price"),
        "brand":        r.get("brand"),
        "category":     r.get("category"),
        "dietary_tags": r.get("dietary_tags", []),
        "hybrid_score": round(r.get("hybrid_score", 0), 3),
        "image_url":    r.get("image_url"),
    } for r in results]

    log.info(f"[Node 3b] returned {len(raw)} results")
    return {
        "raw_results":  raw,
        "result_count": len(raw),
        "route":        "graphrag",
        "cypher_used":  "GraphRAG (Qdrant semantic + Neo4j constraints)",
    }


# Answer aggregate questions (top brands, tag counts, category stats) with fixed queries.
def analytics_node(state: AgentState) -> dict:
    from retailgraph.graph.queries import GraphQueries

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


# The process-wide Neo4j driver lives in graph/queries.py.
from retailgraph.graph.queries import get_driver as _get_driver  # noqa: E402


# Run the Cypher in a read transaction; on invalid Cypher or a Neo4j error, bump the retry counter.
def execute_query(state: AgentState) -> dict:
    cypher       = state.get("cypher_query")
    params       = state.get("cypher_params") or {}
    count_cypher = state.get("cypher_count_query")
    valid        = state.get("cypher_valid", False)
    retries      = state.get("cypher_retries", 0)

    log.info(f"[Node 4] execute_query | valid={valid} | retries={retries}")

    # Invalid (or empty LLM) Cypher: bump the retry counter. On the last attempt also set the
    # error, since the router in graph.py sends us to format_answer next.
    if not valid:
        out = {"cypher_retries": retries + 1}
        if retries + 1 >= 2:
            out.update(raw_results=[], result_count=0,
                       error=f"Cypher generation failed after {retries + 1} tries: {state.get('cypher_error')}")
        return out

    # build_cypher found nothing to search. LangGraph needs every node to write a key,
    # so re-state its empty result instead of returning {}.
    if cypher is None:
        log.info("[Node 4] No cypher to run — passing through build_cypher's empty result")
        return {
            "raw_results":  state.get("raw_results", []),
            "result_count": state.get("result_count", 0),
            "total_count":  state.get("total_count", 0),
        }

    driver = _get_driver()

    try:
        # Read transaction: Neo4j itself refuses any write the validator missed.
        def _read(tx, q):
            return [dict(r) for r in tx.run(q, params)]

        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            records = session.execute_read(_read, cypher)

            total_count = None
            if count_cypher:
                count_rows = session.execute_read(_read, count_cypher)
                total_count = count_rows[0]["total"] if count_rows else len(records)

        log.info(f"[Node 4] Neo4j returned {len(records)} records (total_count={total_count})")
        return {
            "raw_results":  records,
            "result_count": len(records),
            "total_count":  total_count,
            "cypher_used":  cypher,
            "error":        None,   # clear an error left by an earlier failed try
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


FORMAT_SYSTEM = """You are a helpful grocery product assistant.

The user asked a question and a knowledge graph returned matching products.
Write a clear, friendly answer in 2-4 sentences.

Rules:
- Lead with the exact count given to you in the prompt (e.g. "Found 29 matching products, showing the top 10" if a total is given and it's larger than what's shown, otherwise "Found 10 matching products"). Never invent your own count or assume the number of results shown is the total.
- Mention the top 2-3 results with name and price
- If no results: say so and suggest relaxing the filters
- Keep it concise — no bullet points, no markdown
- Never make up products that aren't in the results
- Never say a product is free of an allergen or "safe" for an allergy: allergen
  data is missing for most products, so say it was filtered on known allergen data only"""

# Aggregate rows are not "matching products", so analytics answers get their own framing.
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


# Turn the results into a short plain-English answer with the LLM.
def format_answer(state: AgentState) -> dict:
    query       = state["query"]
    results     = state.get("raw_results") or []
    count       = state.get("result_count", 0)
    total_count = state.get("total_count")
    error       = state.get("error")

    log.info(f"[Node 5] format_answer | results={count} | error={error}")

    # Internal Cypher errors are logged, not shown to the user.
    if error and not results:
        return {
            "answer": (
                f"I wasn't able to complete that search for '{query}'. "
                "Try rephrasing your query or relaxing the filters."
            )
        }

    if not results:
        return {
            "answer": (
                f"No products found matching '{query}'. "
                "Try broadening your search — remove some filters or use a different category."
            )
        }

    first        = results[0]
    is_analytics = "item_name" not in first and "p.item_name" not in first
    top          = results[:5]

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