"""
# collect_results.py - runs the real agent on every golden query and scores it

- Scores route/intent accuracy, execution accuracy (vs gold_cypher counts) and analytics answers.
- Writes evaluation/agent_eval_raw.json. Run: `python eval/collect_results.py` (needs Neo4j, Qdrant, Groq).
"""

import json
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

from retailgraph.agent.graph import run_query

GOLDEN_FILE = Path(__file__).parent / "golden_queries.yaml"
OUTPUT_FILE = Path(__file__).parent.parent / "evaluation" / "agent_eval_raw.json"


# Run a gold_cypher directly against Neo4j and return its single count value (None on error).
def run_gold_cypher(cypher: str) -> int | None:
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


# Run every golden query, score it, print the summary and write the JSON report.
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
        result_count = state.get("result_count", 0)

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
            # Analytics: gold_cypher returns (name, count), not a bare count, so score the answer text.
            analytics_total += 1
            analytics_hit = all(s in answer for s in entry["gold_answer_contains"])
            if analytics_hit:
                analytics_correct += 1

        elif entry.get("gold_cypher"):
            # Execution accuracy: compare with the agent's true total (before LIMIT 10) when it
            # reports one, else with the shown rows, capping the gold count at 10.
            gold_count = run_gold_cypher(entry["gold_cypher"])
            if gold_count is not None:
                exec_total += 1
                total = state.get("total_count")
                if (gold_count == total) if total is not None else (min(gold_count, 10) == result_count):
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
            "intent": intent,
            "route": route,
            "result_count": result_count,
            "expected_intent": entry.get("expected_intent"),
            "expected_route": entry.get("expected_route"),
            "expected_result_count": entry.get("expected_result_count"),
            "gold_count": gold_count,
            "total_count": state.get("total_count"),
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


if __name__ == "__main__":
    main()
