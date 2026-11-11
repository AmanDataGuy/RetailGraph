"""
# state.py - the AgentState dict that flows through every LangGraph node

- Each node reads the state and returns a partial dict to update it.
- Used by agent/nodes.py and agent/graph.py; make_initial_state() starts a fresh query.
"""

from typing import Literal, TypedDict


# Everything the agent knows about one query, from raw input to final answer.
class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str                          # raw user query, never modified

    # ── Intent extraction (Node 1) ────────────────────────────────────────
    intent: Literal["filter", "semantic", "hybrid", "lookup", "analytics"] | None
    entities: dict | None            # extracted: category, tags, price, brand, etc.

    # ── Query routing (conditional edge) ──────────────────────────────────
    route: Literal["cypher", "graphrag", "analytics"] | None

    # ── Cypher path (Node 3a) ─────────────────────────────────────────────
    cypher_query: str | None         # generated Cypher string (uses $bound params, never inlined values)
    cypher_params: dict | None       # bound parameters for cypher_query
    cypher_count_query: str | None   # same MATCH/WHERE, returns count(DISTINCT p); templates only
    cypher_valid: bool | None        # passed validation?
    cypher_error: str | None         # validation error if any
    cypher_retries: int                 # retry counter (max 2)

    # ── Execution (Node 4) ────────────────────────────────────────────────
    raw_results: list | None         # list of product dicts from Neo4j/Qdrant
    result_count: int                   # how many results returned (capped by LIMIT)
    total_count: int | None          # true match count before LIMIT, when known (Cypher template path only)

    # ── Answer formatting (Node 5) ────────────────────────────────────────
    answer: str | None               # final plain-English answer
    cypher_used: str | None          # Cypher shown to user for transparency

    # ── Error handling ────────────────────────────────────────────────────
    error: str | None                # set if something went wrong

    # ── Guardrails ─────────────────────────────────────────────────────────
    blocked:  bool                      # True if check_input() rejected the query
    grounded: bool | None            # advisory output check from guardrails.check_output()


# A fresh state for a new query: optional fields None, counters 0.
def make_initial_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        intent=None,
        entities=None,
        route=None,
        cypher_query=None,
        cypher_params=None,
        cypher_count_query=None,
        cypher_valid=None,
        cypher_error=None,
        cypher_retries=0,
        raw_results=None,
        result_count=0,
        total_count=None,
        answer=None,
        cypher_used=None,
        error=None,
        blocked=False,
        grounded=None,
    )