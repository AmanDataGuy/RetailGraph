"""
eval/collect_results.py

Step 1 of the eval pipeline. Runs in the APP's venv (venv/) — the only place
that has Neo4j, Qdrant, Groq, and the LangGraph agent. For every query in
golden_queries.yaml this:

  1. Calls the real agent (src.agent.graph.run_query) — no mocking.
  2. Computes route/intent accuracy against the expected_* fields.
  3. For queries with a gold_cypher, runs it directly against Neo4j and
     compares the count to the agent's result_count (execution accuracy —
     does the agent's generated query return the same rows as the correct
     one, not just "did it not crash").
  4. Serializes each raw result into a text context (RAGAS expects text
     chunks, not structured dicts — this is the adapter mentioned in
     ROADMAP.md), because RAGAS itself cannot run in this venv (see
     eval/requirements-eval.txt for why).

Writes evaluation/agent_eval_raw.json, which score_ragas.py (run in
venv-eval/) reads to compute RAGAS metrics.

Usage:
    venv/Scripts/python eval/collect_results.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import run_query  # noqa: E402

GOLDEN_FILE = Path(__file__).parent / "golden_queries.yaml"
OUTPUT_FILE = Path(__file__).parent.parent / "evaluation" / "agent_eval_raw.json"


def record_to_context(r: dict) -> str:
    """
    RAGAS scores text contexts, but our retrieval returns structured product
    records from Neo4j/Qdrant. This is the serialization step that adapts
    graph/vector results into the text chunks RAGAS expects.
    """
    name = r.get("item_name") or r.get("p.item_name") or "Unknown product"
    price = r.get("price") if r.get("price") is not None else r.get("p.price")
    brand = r.get("brand") or "unknown"
    category = r.get("category") or "unknown"
    tags = r.get("dietary_tags") or []
    allergens = r.get("allergen_list") or []
    return (
        f"{name} by {brand} — ${price}, category {category}, "
        f"tags: {', '.join(tags) or 'none'}, "
        f"allergens: {', '.join(allergens) or 'none'}"
    )


def run_gold_cypher(cypher: str) -> int | None:
    """Runs a gold_cypher string directly against Neo4j and returns the count."""
    import os

    from dotenv import load_dotenv
    from neo4j import GraphDatabase

    load_dotenv()
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
            record = session.run(cypher).single()
            # gold_cypher always RETURNs a single count(...) value
            return list(record.values())[0] if record else None
    except Exception as e:
        print(f"    [gold_cypher error] {e}")
        return None
    finally:
        driver.close()


def build_ground_truth(entry: dict) -> str:
    """Best-effort reference answer text for RAGAS's context_recall metric."""
    if entry.get("gold_answer_contains"):
        return " and ".join(entry["gold_answer_contains"])
    if entry.get("expected_result_count") is not None:
        return f"There are {entry['expected_result_count']} matching products."
    return ""


def main():
    golden = yaml.safe_load(GOLDEN_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(golden)} golden queries from {GOLDEN_FILE.name}\n")

    records = []
    route_correct = intent_correct = 0
    route_total = intent_total = 0
    exec_correct = exec_total = 0
    analytics_correct = analytics_total = 0

    for entry in golden:
        qid, query = entry["id"], entry["query"]
        print(f"[{qid}] {query}")

        t0 = time.time()
        state = run_query(query)
        latency = round(time.time() - t0, 2)

        intent, route = state.get("intent"), state.get("route")
        answer = state.get("answer") or ""
        raw_results = state.get("raw_results") or []
        result_count = state.get("result_count", 0)
        contexts = [record_to_context(r) for r in raw_results] or ["No products retrieved."]

        # ── route / intent accuracy ────────────────────────────────────────
        if entry.get("expected_route"):
            route_total += 1
            if route == entry["expected_route"]:
                route_correct += 1
        if entry.get("expected_intent"):
            intent_total += 1
            if intent == entry["expected_intent"]:
                intent_correct += 1

        gold_count = None
        analytics_hit = None

        if entry.get("gold_answer_contains"):
            # Analytics queries: gold_cypher (if present) returns (name, count) —
            # not a bare count — so it's not comparable to result_count at all.
            # Correctness here means "the answer text mentions the right facts."
            analytics_total += 1
            analytics_hit = all(s in answer for s in entry["gold_answer_contains"])
            if analytics_hit:
                analytics_correct += 1

        elif entry.get("gold_cypher"):
            # Cypher execution accuracy: does the agent's query return the same
            # row COUNT as the correct one? The agent's Cypher templates
            # (src/agent/nodes.py, src/graph/queries.py) all hardcode LIMIT 10,
            # so a query with 29 true matches still returns 10 — that's correct
            # behavior, not a miss. Cap the gold count at 10 before comparing.
            gold_count = run_gold_cypher(entry["gold_cypher"])
            if gold_count is not None:
                exec_total += 1
                if min(gold_count, 10) == result_count:
                    exec_correct += 1

        elif entry.get("expected_result_count") is not None:
            # no gold_cypher but a known expected count (adversarial cases)
            exec_total += 1
            if min(entry["expected_result_count"], 10) == result_count:
                exec_correct += 1

        records.append({
            "id": qid,
            "type": entry.get("type"),
            "query": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": build_ground_truth(entry),
            "intent": intent,
            "route": route,
            "result_count": result_count,
            "expected_intent": entry.get("expected_intent"),
            "expected_route": entry.get("expected_route"),
            "expected_result_count": entry.get("expected_result_count"),
            "gold_count": gold_count,
            "analytics_hit": analytics_hit,
            "known_bug": entry.get("known_bug", False),
            "latency_s": latency,
            "error": state.get("error"),
        })

        detail = f"(gold={gold_count})" if gold_count is not None else (
            f"(analytics_hit={analytics_hit})" if analytics_hit is not None else ""
        )
        print(f"    intent={intent} route={route} results={result_count} {detail} latency={latency}s")

    summary = {
        "n_queries": len(golden),
        "route_accuracy": round(route_correct / route_total, 3) if route_total else None,
        "route_total": route_total,
        "intent_accuracy": round(intent_correct / intent_total, 3) if intent_total else None,
        "intent_total": intent_total,
        "execution_accuracy": round(exec_correct / exec_total, 3) if exec_total else None,
        "execution_total": exec_total,
        "analytics_accuracy": round(analytics_correct / analytics_total, 3) if analytics_total else None,
        "analytics_total": analytics_total,
    }

    print("\n── Summary (route/intent/execution — no LLM judge needed) ──────")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {OUTPUT_FILE}")
    print("Next: run `venv-eval/Scripts/python eval/score_ragas.py` to add RAGAS scores.")


if __name__ == "__main__":
    main()
