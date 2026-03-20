"""
RetailGraph — LangGraph Agent Graph
Wires all nodes into a StateGraph with conditional routing and retry logic.

Usage:
    from src.agent.graph import build_graph, run_query

    result = run_query("show me vegan snacks under $10")
    print(result["answer"])
    print(result["cypher_used"])
"""

import logging
from langgraph.graph import StateGraph, START, END

from src.agent.state import AgentState, make_initial_state
from src.agent.nodes import (
    extract_intent,
    route_query,
    build_cypher,
    hybrid_search_node,
    analytics_node,
    execute_query,
    format_answer,
)

log = logging.getLogger("retailgraph.graph")


def _should_retry_cypher(state: AgentState) -> str:
    """
    Conditional edge after execute_query on the Cypher path.
    If Cypher failed and retries remain → go back to build_cypher.
    Otherwise → format_answer.
    """
    valid   = state.get("cypher_valid", True)
    retries = state.get("cypher_retries", 0)

    if not valid and retries < 2:
        log.info(f"[Router] Cypher invalid — retry {retries}/2")
        return "build_cypher"

    return "format_answer"


def build_graph() -> StateGraph:
    """
    Builds and compiles the RetailGraph LangGraph agent.

    Graph structure:
        START → extract_intent → [route_query]
                                      ├─ "cypher"    → build_cypher → execute_query → [retry?] → format_answer
                                      ├─ "graphrag"  → hybrid_search_node → format_answer
                                      └─ "analytics" → analytics_node → format_answer
                                                                              ↓
                                                                             END
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    graph.add_node("extract_intent",     extract_intent)
    graph.add_node("build_cypher",       build_cypher)
    graph.add_node("hybrid_search_node", hybrid_search_node)
    graph.add_node("analytics_node",     analytics_node)
    graph.add_node("execute_query",      execute_query)
    graph.add_node("format_answer",      format_answer)

    # ── Entry point ────────────────────────────────────────────────────────
    graph.add_edge(START, "extract_intent")

    # ── Conditional routing after intent extraction ────────────────────────
    graph.add_conditional_edges(
        "extract_intent",
        route_query,
        {
            "cypher":    "build_cypher",
            "graphrag":  "hybrid_search_node",
            "analytics": "analytics_node",
        },
    )

    # ── Cypher path: build → execute → [retry or format] ──────────────────
    graph.add_edge("build_cypher", "execute_query")

    graph.add_conditional_edges(
        "execute_query",
        _should_retry_cypher,
        {
            "build_cypher":  "build_cypher",   # retry
            "format_answer": "format_answer",  # done
        },
    )

    # ── GraphRAG + analytics paths go straight to format ──────────────────
    graph.add_edge("hybrid_search_node", "format_answer")
    graph.add_edge("analytics_node",     "format_answer")

    # ── Final node → END ──────────────────────────────────────────────────
    graph.add_edge("format_answer", END)

    return graph.compile()


# ── Public interface ───────────────────────────────────────────────────────

def run_query(query: str) -> AgentState:
    """
    Run a single natural language query through the agent.

    Args:
        query: Plain English product search query

    Returns:
        Final AgentState with answer, cypher_used, raw_results, etc.
    """
    agent = build_graph()
    initial = make_initial_state(query)
    log.info(f"Running query: '{query}'")
    result = agent.invoke(initial)
    return result


# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    test_queries = [
        "show me vegan snacks under $10",
        "find something similar to sriracha sauce",
        "which category has the most products",
        "gluten-free beverages with no allergens",
    ]

    for q in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {q}")
        print("=" * 60)

        result = run_query(q)

        print(f"Intent:   {result.get('intent')}")
        print(f"Route:    {result.get('route')}")
        print(f"Results:  {result.get('result_count', 0)}")
        print(f"Cypher:   {result.get('cypher_used', 'N/A')[:80]}")
        print(f"\nAnswer:\n{result.get('answer')}")

        if result.get("error"):
            print(f"\n⚠️  Error: {result.get('error')}")