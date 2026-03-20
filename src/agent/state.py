"""
RetailGraph — Agent State
The single TypedDict that flows through every node in the LangGraph.
Every node reads from state and returns a partial dict to update it.
"""

from typing import TypedDict, Optional, Literal


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str                          # raw user query, never modified

    # ── Intent extraction (Node 1) ────────────────────────────────────────
    intent: Optional[Literal[
        "filter",       # hard constraints: tags, price, allergens, category
        "semantic",     # similarity search: "something like sriracha"
        "hybrid",       # both: "vegan snacks similar to protein bars"
        "lookup",       # specific product or brand: "tell me about Heinz"
        "analytics",    # aggregate: "which category has the most products"
    ]]
    entities: Optional[dict]            # extracted: category, tags, price, brand, etc.

    # ── Query routing (conditional edge) ──────────────────────────────────
    route: Optional[Literal[
        "cypher",       # → build_cypher node
        "graphrag",     # → hybrid_search node
        "analytics",    # → analytics node
    ]]

    # ── Cypher path (Node 3a) ─────────────────────────────────────────────
    cypher_query: Optional[str]         # generated Cypher string
    cypher_valid: Optional[bool]        # passed validation?
    cypher_error: Optional[str]         # validation error if any
    cypher_retries: int                 # retry counter (max 2)

    # ── Execution (Node 4) ────────────────────────────────────────────────
    raw_results: Optional[list]         # list of product dicts from Neo4j/Qdrant
    result_count: int                   # how many results returned

    # ── Answer formatting (Node 5) ────────────────────────────────────────
    answer: Optional[str]               # final plain-English answer
    cypher_used: Optional[str]          # Cypher shown to user for transparency

    # ── Error handling ────────────────────────────────────────────────────
    error: Optional[str]                # set if something went wrong


def make_initial_state(query: str) -> AgentState:
    """
    Returns a fresh AgentState for a new query.
    All optional fields start as None, counters start at 0.
    """
    return AgentState(
        query=query,
        intent=None,
        entities=None,
        route=None,
        cypher_query=None,
        cypher_valid=None,
        cypher_error=None,
        cypher_retries=0,
        raw_results=None,
        result_count=0,
        answer=None,
        cypher_used=None,
        error=None,
    )