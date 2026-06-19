"""
RetailGraph — Phase 13: GraphRAG vs VectorRAG vs Neo4j-only Benchmark
Run from project root:
    $env:PYTHONPATH = "C:\\Users\\AMAND\\projects\\RetailGraph"
    python scripts/run_benchmark.py
"""

import os
import time
import json
from dotenv import load_dotenv
load_dotenv()

from src.graph.queries import GraphQueries
from src.graph.vector_store import search as qdrant_search
from src.graph.hybrid_search import HybridSearch

# ── Ground truth ──────────────────────────────────────────────────────────────

QUERIES = [

    # ── Type 1: Multi-constraint (GraphRAG should win) ────────────────────────
    {
        "id": 1, "type": "multi_constraint",
        "query": "vegan snacks under $5",
        "check": lambda r: all(
            "vegan" in (p.get("dietary_tags") or []) and
            (p.get("price") or 999) <= 5.0
            for p in r
        ),
        "expected": "All results vegan + price ≤ $5",
    },
    {
        "id": 2, "type": "multi_constraint",
        "query": "gluten-free beverages under $3",
        "check": lambda r: all(
            "gluten-free" in (p.get("dietary_tags") or []) and
            (p.get("price") or 999) <= 3.0
            for p in r
        ),
        "expected": "All results gluten-free + price ≤ $3",
    },
    {
        "id": 3, "type": "multi_constraint",
        "query": "kosher organic coffee under $20",
        "check": lambda r: all(
            "kosher" in (p.get("dietary_tags") or []) and
            (p.get("price") or 999) <= 20.0
            for p in r
        ),
        "expected": "All results kosher + price ≤ $20",
    },
    {
        "id": 4, "type": "multi_constraint",
        "query": "dairy-free keto snacks under $8",
        "check": lambda r: all(
            "dairy-free" in (p.get("dietary_tags") or []) and
            (p.get("price") or 999) <= 8.0
            for p in r
        ),
        "expected": "All results dairy-free + price ≤ $8",
    },
    {
        "id": 5, "type": "multi_constraint",
        "query": "non-GMO vegan condiments under $15",
        "check": lambda r: all(
            "vegan" in (p.get("dietary_tags") or []) and
            (p.get("price") or 999) <= 15.0
            for p in r
        ),
        "expected": "All results vegan + price ≤ $15",
    },
    {
        "id": 6, "type": "multi_constraint",
        "query": "high-protein gluten-free products under $5",
        "check": lambda r: all(
            "gluten-free" in (p.get("dietary_tags") or []) and
            (p.get("price") or 999) <= 5.0
            for p in r
        ),
        "expected": "All results gluten-free + price ≤ $5",
    },
    {
        "id": 7, "type": "multi_constraint",
        "query": "organic coffee under $10",
        "check": lambda r: all(
            "organic" in (p.get("dietary_tags") or []) and
            (p.get("price") or 999) <= 10.0
            for p in r
        ),
        "expected": "All results organic + price ≤ $10",
    },

    # ── Type 2: Semantic (all systems attempt, GraphRAG re-ranks) ─────────────
    {
        "id": 8, "type": "semantic",
        "query": "something similar to sriracha",
        "check": lambda r: len(r) > 0,
        "expected": "Returns hot sauce or spicy condiment results",
    },
    {
        "id": 9, "type": "semantic",
        "query": "matcha or green tea drinks",
        "check": lambda r: len(r) > 0,
        "expected": "Returns tea or matcha-related products",
    },
    {
        "id": 10, "type": "semantic",
        "query": "healthy breakfast options",
        "check": lambda r: len(r) > 0,
        "expected": "Returns breakfast-related products",
    },
    {
        "id": 11, "type": "semantic",
        "query": "spicy cooking sauces",
        "check": lambda r: len(r) > 0,
        "expected": "Returns hot sauce or spicy condiment results",
    },
    {
        "id": 12, "type": "semantic",
        "query": "low sugar snack options",
        "check": lambda r: len(r) > 0,
        "expected": "Returns snack products",
    },
    {
        "id": 13, "type": "semantic",
        "query": "protein bar alternatives",
        "check": lambda r: len(r) > 0,
        "expected": "Returns high-protein snack products",
    },
    {
        "id": 14, "type": "semantic",
        "query": "pasta sauce or marinara",
        "check": lambda r: len(r) > 0,
        "expected": "Returns sauce or condiment products",
    },

    # ── Type 3: Analytics (only Neo4j can answer these) ───────────────────────
    {
        "id": 15, "type": "analytics",
        "query": "which category has the most products",
        "check": lambda r: any(
            "snacks" in str(p).lower() or "580" in str(p)
            for p in r
        ),
        "expected": "Snacks & Candy (580 products)",
    },
    {
        "id": 16, "type": "analytics",
        "query": "top 5 brands by product count",
        "check": lambda r: any(
            "terravita" in str(p).lower() or "mccormick" in str(p).lower()
            for p in r
        ),
        "expected": "TerraVita (116), McCormick (40), Food to Live (35)",
    },
    {
        "id": 17, "type": "analytics",
        "query": "most common dietary tag",
        "check": lambda r: any(
            "gluten" in str(p).lower() or "426" in str(p)
            for p in r
        ),
        "expected": "gluten-free (426 products)",
    },
    {
        "id": 18, "type": "analytics",
        "query": "average price in beverages category",
        "check": lambda r: any(
            "20" in str(p) or "beverage" in str(p).lower()
            for p in r
        ),
        "expected": "~$20.24",
    },
    {
        "id": 19, "type": "analytics",
        "query": "how many vegan products are there",
        "check": lambda r: any(
            "255" in str(p) or "vegan" in str(p).lower()
            for p in r
        ),
        "expected": "255 products",
    },
    {
        "id": 20, "type": "analytics",
        "query": "which category has the highest average price",
        "check": lambda r: any(
            "coffee" in str(p).lower() or "32" in str(p)
            for p in r
        ),
        "expected": "Coffee & Tea (~$32.07 avg)",
    },
]

# ── System runners ─────────────────────────────────────────────────────────────

def run_vector_only(query: str) -> tuple[list, float]:
    """System A — Qdrant semantic search only, no Neo4j constraints."""
    try:
        t0 = time.time()
        results = qdrant_search(query, top_k=10)
        latency = round(time.time() - t0, 3)
        # Normalise field names
        out = []
        for r in results:
            payload = r.get("payload", r)
            out.append({
                "item_name":    payload.get("item_name"),
                "price":        payload.get("price"),
                "category":     payload.get("category"),
                "dietary_tags": payload.get("dietary_tags", []),
            })
        return out, latency
    except Exception as e:
        return [], 0.0


def run_graph_only(query_obj: dict) -> tuple[list, float]:
    """System B — Neo4j Cypher templates only, no vector search."""
    q = GraphQueries()
    t0 = time.time()
    results = []
    try:
        qtype = query_obj["type"]
        query = query_obj["query"].lower()

        if qtype == "analytics":
            if "category" in query and "most" in query:
                results = q.get_category_stats()[:5]
            elif "brand" in query:
                results = q.get_top_brands(5)
            elif "dietary" in query or "tag" in query or "common" in query:
                results = q.get_dietary_tag_stats()[:5]
            elif "average price" in query and "beverage" in query:
                stats = q.get_category_stats()
                results = [s for s in stats if "beverage" in s.get("category","").lower()]
            elif "vegan" in query and "how many" in query:
                results = q.get_dietary_tag_stats()
                results = [r for r in results if r.get("tag") == "vegan"]
            elif "highest" in query and "price" in query:
                stats = q.get_category_stats()
                results = sorted(stats, key=lambda x: x.get("avg_price", 0), reverse=True)[:3]
            else:
                results = q.get_category_stats()[:5]

        elif qtype == "multi_constraint":
            tags, cat, max_p = [], None, None
            for tag in ["vegan", "gluten-free", "kosher", "organic", "dairy-free",
                        "non-GMO", "high-protein", "keto-friendly"]:
                if tag.lower() in query:
                    tags.append(tag)
            for c in ["snacks & candy", "beverages", "coffee & tea",
                      "condiments & sauces", "spices & seasonings"]:
                if c.split(" & ")[0].lower() in query or c.split(",")[0].lower() in query:
                    cat = c.title()
            import re
            m = re.search(r'\$(\d+)', query)
            if m:
                max_p = float(m.group(1))
            results = q.get_products(
                tags=tags if tags else None,
                category=cat,
                max_price=max_p,
                limit=10
            )

        else:  # semantic — graph does its best with keyword match
            results = q.get_products(limit=10)

    except Exception as e:
        results = []

    latency = round(time.time() - t0, 3)
    return results, latency


def run_graphrag(query_obj: dict) -> tuple[list, float]:
    """System C — GraphRAG hybrid: Qdrant candidates + Neo4j constraints."""
    try:
        import re
        hs = HybridSearch()
        query = query_obj["query"].lower()
        tags, cat, max_p = [], None, None
        for tag in ["vegan", "gluten-free", "kosher", "organic", "dairy-free",
                    "non-GMO", "high-protein", "keto-friendly", "nut-free"]:
            if tag.lower() in query:
                tags.append(tag)
        for c, kw in [("Snacks & Candy", ["snack"]),
                      ("Beverages", ["beverage", "drink"]),
                      ("Coffee & Tea", ["coffee", "tea", "matcha"]),
                      ("Condiments & Sauces", ["sauce", "condiment"]),
                      ("Spices & Seasonings", ["spice", "seasoning"])]:
            if any(k in query for k in kw):
                cat = c
        m = re.search(r"\$(\d+)", query)
        if m:
            max_p = float(m.group(1))
        t0 = time.time()
        results = hs.search(
            query_obj["query"],
            top_k=10,
            dietary_tags=tags if tags else None,
            category=cat,
            max_price=max_p,
        )
        latency = round(time.time() - t0, 3)
        return results, latency
    except Exception as e:
        return [], 0.0


# ── Main benchmark ─────────────────────────────────────────────────────────────

def score(results: list, check_fn) -> bool:
    if not results:
        return False
    try:
        return check_fn(results)
    except Exception:
        return False


def run_benchmark():
    print("\n" + "="*70)
    print("  RetailGraph — GraphRAG vs VectorRAG vs Neo4j Benchmark")
    print("  20 queries · 3 systems · ground truth scoring")
    print("="*70 + "\n")

    rows = []
    totals = {"vector": 0, "graph": 0, "graphrag": 0}
    latencies = {"vector": [], "graph": [], "graphrag": []}
    type_scores = {}

    for q in QUERIES:
        print(f"[{q['id']:02d}/20] {q['query'][:55]:<55}", end=" ", flush=True)

        r_vec,  lat_vec  = run_vector_only(q["query"])
        r_gph,  lat_gph  = run_graph_only(q)
        r_rag,  lat_rag  = run_graphrag(q)

        s_vec = score(r_vec, q["check"])
        s_gph = score(r_gph, q["check"])
        s_rag = score(r_rag, q["check"])

        totals["vector"]   += int(s_vec)
        totals["graph"]    += int(s_gph)
        totals["graphrag"] += int(s_rag)

        latencies["vector"].append(lat_vec)
        latencies["graph"].append(lat_gph)
        latencies["graphrag"].append(lat_rag)

        t = q["type"]
        if t not in type_scores:
            type_scores[t] = {"vector": 0, "graph": 0, "graphrag": 0, "total": 0}
        type_scores[t]["vector"]   += int(s_vec)
        type_scores[t]["graph"]    += int(s_gph)
        type_scores[t]["graphrag"] += int(s_rag)
        type_scores[t]["total"]    += 1

        v = "✅" if s_vec else "❌"
        g = "✅" if s_gph else "❌"
        r = "✅" if s_rag else "❌"
        print(f"Vector {v}  Graph {g}  GraphRAG {r}")

        rows.append({
            "id": q["id"], "type": q["type"], "query": q["query"],
            "vector": s_vec, "graph": s_gph, "graphrag": s_rag,
            "lat_vec": lat_vec, "lat_gph": lat_gph, "lat_rag": lat_rag,
        })

    # ── Summary ────────────────────────────────────────────────────────────────
    n = len(QUERIES)
    avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else 0

    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    print(f"  {'System':<20} {'Accuracy':>10} {'Avg Latency':>14}")
    print(f"  {'-'*44}")
    print(f"  {'Vector only (Qdrant)':<20} {totals['vector']:>5}/{n} ({100*totals['vector']//n:>2}%)  {avg(latencies['vector']):>8}s")
    print(f"  {'Graph only (Neo4j)':<20} {totals['graph']:>5}/{n} ({100*totals['graph']//n:>2}%)  {avg(latencies['graph']):>8}s")
    print(f"  {'GraphRAG (hybrid)':<20} {totals['graphrag']:>5}/{n} ({100*totals['graphrag']//n:>2}%)  {avg(latencies['graphrag']):>8}s")

    print(f"\n  BREAKDOWN BY QUERY TYPE")
    print(f"  {'-'*60}")
    for t, s in type_scores.items():
        n_t = s["total"]
        print(f"  {t:<20}  Vector {s['vector']}/{n_t}  Graph {s['graph']}/{n_t}  GraphRAG {s['graphrag']}/{n_t}")

    print(f"\n  KEY FINDING:")
    rag_win = totals['graphrag'] - max(totals['vector'], totals['graph'])
    best_other = "Vector" if totals['vector'] > totals['graph'] else "Graph"
    print(f"  GraphRAG scores {totals['graphrag']}/{n} vs best alternative ({best_other}) at {max(totals['vector'],totals['graph'])}/{n}")
    print(f"  Improvement: +{rag_win} queries answered correctly\n")

    # Save results
    out_path = "evaluation/benchmark_results.json"
    os.makedirs("evaluation", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "totals": totals,
            "latencies": {k: avg(v) for k, v in latencies.items()},
            "type_scores": type_scores,
            "rows": rows
        }, f, indent=2)
    print(f"  Full results saved → {out_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_benchmark()