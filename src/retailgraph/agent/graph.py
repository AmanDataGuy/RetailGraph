"""
# graph.py - wires the agent nodes into a LangGraph state machine

- extract_intent -> cypher | graphrag | analytics path -> format_answer, with up to 2 Cypher retries.
- run_query() is the entry point, called by api/routes/query.py and eval/collect_results.py.
"""

import logging

from langgraph.graph import END, START, StateGraph

from retailgraph.agent.guardrails import check_input, check_output
from retailgraph.agent.nodes import (
    analytics_node,
    build_cypher,
    execute_query,
    extract_intent,
    format_answer,
    hybrid_search_node,
    route_query,
)
from retailgraph.agent.state import AgentState, make_initial_state

log = logging.getLogger("retailgraph.graph")


# After running Cypher: go back and rebuild it if it failed and retries remain.
def _should_retry_cypher(state: AgentState) -> str:
    retries = state.get("cypher_retries", 0)
    if not state.get("cypher_valid", True) and retries < 2:
        log.info(f"[Router] Cypher invalid — retry {retries}/2")
        return "build_cypher"
    return "format_answer"


# Build and compile the agent graph.
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    for name, fn in [
        ("extract_intent", extract_intent),
        ("build_cypher", build_cypher),
        ("hybrid_search_node", hybrid_search_node),
        ("analytics_node", analytics_node),
        ("execute_query", execute_query),
        ("format_answer", format_answer),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "extract_intent")
    graph.add_conditional_edges("extract_intent", route_query, {
        "cypher": "build_cypher",
        "graphrag": "hybrid_search_node",
        "analytics": "analytics_node",
    })
    graph.add_edge("build_cypher", "execute_query")
    graph.add_conditional_edges("execute_query", _should_retry_cypher, {
        "build_cypher": "build_cypher",
        "format_answer": "format_answer",
    })
    graph.add_edge("hybrid_search_node", "format_answer")
    graph.add_edge("analytics_node", "format_answer")
    graph.add_edge("format_answer", END)
    return graph.compile()


# Compiled once at import; the graph is stateless so every request can share it.
_COMPILED_GRAPH = build_graph()


# Run one plain-English query through the guardrails and the agent; returns the final state.
def run_query(query: str) -> AgentState:
    guard = check_input(query)
    if guard["blocked"]:
        log.warning(f"Blocked query ({','.join(guard['categories'])}): '{query}'")
        state = make_initial_state(query)
        state["blocked"] = True
        state["answer"] = guard["reason"]
        return state

    log.info(f"Running query: '{query}'")
    result = _COMPILED_GRAPH.invoke(make_initial_state(query))

    result["grounded"] = check_output(result.get("answer") or "", result.get("raw_results") or [])
    if not result["grounded"]:
        log.warning(f"Ungrounded answer for query: '{query}'")
    return result
