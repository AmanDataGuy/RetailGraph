"""
RetailGraph — GraphRAG vs VectorRAG vs Neo4j-only Benchmark
Run from project root:
    $env:PYTHONPATH = "C:\\Users\\AMAND\\projects\\RetailGraph"
    python scripts/run_benchmark.py
"""

import os
import re
import time
import json
from dotenv import load_dotenv
load_dotenv()

from src.graph.queries import GraphQueries
from src.graph.vector_store import search as qdrant_search
from src.graph.hybrid_search import HybridSearch

# ── Shared constraint parser ────────────────────────────────────────────────
# Used by all three systems so none of them gets an unfair advantage from
# constraint-extraction logic another system doesn't have. Tag list is the
# live, verified DietaryTag vocabulary (MATCH (t:DietaryTag), 2026-08-21) —
# same source of truth as src/agent/nodes.py's ALLOWED_TAGS.
_TAGS = ["vegan", "gluten-free", "kosher", "organic", "dairy-free",
         "non-GMO", "high-protein", "keto-friendly", "nut-free",
         "soy-free", "caffeine-free", "sugar-free", "vegetarian", "halal"]
_CATEGORY_KEYWORDS = [
    ("Snacks & Candy", ["snack"]),
    ("Beverages", ["beverage", "drink"]),
    ("Coffee & Tea", ["coffee", "tea", "matcha"]),
    ("Condiments & Sauces", ["sauce", "condiment"]),
    ("Spices & Seasonings", ["spice", "seasoning"]),
    ("Supplements & Health", ["supplement", "vitamin"]),
    ("Meat & Seafood", ["meat", "seafood", "jerky"]),
]
_NEGATION_PATTERNS = ["not ", "excluding ", "without ", "no "]


def parse_constraints(query: str) -> tuple[list, str | None, float | None, float | None, list]:
    """
    Extract dietary tags, category, min/max price, and excluded tags from a
    query string. Returns (tags, category, min_price, max_price, exclude_tags).

    A tag is treated as EXCLUDED if one of _NEGATION_PATTERNS immediately
    precedes it in the lowercased query text (e.g. "not vegan", "excluding
    nuts") — otherwise it's a required (positive) constraint.
    """
    q = query.lower()

    tags, exclude_tags = [], []
    for tag in _TAGS:
        idx = q.find(tag.lower())
        if idx == -1:
            continue
        preceding = q[:idx]
        if any(preceding.rstrip().endswith(neg.strip()) for neg in _NEGATION_PATTERNS):
            exclude_tags.append(tag)
        else:
            tags.append(tag)

    cat = None
    for name, keywords in _CATEGORY_KEYWORDS:
        if any(k in q for k in keywords):
            cat = name
            break

    min_p = max_p = None
    m = re.search(r"between\s*\$?(\d+)\s*(?:and|-|to)\s*\$?(\d+)", q)
    if m:
        min_p, max_p = float(m.group(1)), float(m.group(2))
    else:
        m = re.search(r"(?:over|above|more than)\s*\$(\d+)", q)
        if m:
            min_p = float(m.group(1))
        m = re.search(r"\$(\d+)", q)
        if m and max_p is None and min_p is None:
            max_p = float(m.group(1))

    return tags, cat, min_p, max_p, exclude_tags

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

    # ── Type 4: Compound (4+ simultaneous constraints) ─────────────────────────
    {
        "id": 21, "type": "compound",
        "query": "organic gluten-free coffee between $10 and $25",
        "check": lambda r: all(
            "organic" in (p.get("dietary_tags") or []) and
            "gluten-free" in (p.get("dietary_tags") or []) and
            10.0 <= (p.get("price") or -1) <= 25.0
            for p in r
        ),
        "expected": "All results organic + gluten-free + $10-$25",
    },
    {
        "id": 22, "type": "compound",
        "query": "dairy-free soy-free supplements over $20",
        "check": lambda r: all(
            "dairy-free" in (p.get("dietary_tags") or []) and
            "soy-free" in (p.get("dietary_tags") or []) and
            (p.get("price") or -1) >= 20.0
            for p in r
        ),
        "expected": "All results dairy-free + soy-free + price ≥ $20",
    },

    # ── Type 5: Negation (exclude a tag, not just allergens) ───────────────────
    {
        "id": 23, "type": "negation",
        "query": "kosher snacks under $5 that are not vegan",
        "check": lambda r: all(
            "kosher" in (p.get("dietary_tags") or []) and
            "vegan" not in (p.get("dietary_tags") or []) and
            (p.get("price") or 999) <= 5.0
            for p in r
        ),
        "expected": "All results kosher + NOT vegan + price ≤ $5",
    },
    {
        "id": 24, "type": "negation",
        "query": "gluten-free snacks that are not sugar-free",
        "check": lambda r: all(
            "gluten-free" in (p.get("dietary_tags") or []) and
            "sugar-free" not in (p.get("dietary_tags") or [])
            for p in r
        ),
        "expected": "All results gluten-free + NOT sugar-free",
    },

    # ── Type 6: Price range (harder than a single "under $X") ──────────────────
    {
        "id": 25, "type": "price_range",
        "query": "beverages between $15 and $30",
        "check": lambda r: all(
            15.0 <= (p.get("price") or -1) <= 30.0
            for p in r
        ),
        "expected": "All results price between $15 and $30",
    },
    {
        "id": 26, "type": "price_range",
        "query": "spices and seasonings over $10",
        "check": lambda r: all(
            (p.get("price") or -1) >= 10.0
            for p in r
        ),
        "expected": "All results price ≥ $10",
    },

    # ── Type 7: Subtle semantic (no literal category/tag keyword in the query —
    #    requires genuine embedding understanding, not keyword matching) ───────
    {
        "id": 27, "type": "subtle_semantic",
        "query": "something to spice up a bland dinner",
        "check": lambda r: len(r) > 0,
        "expected": "Returns spice/seasoning or condiment products",
    },
    {
        "id": 28, "type": "subtle_semantic",
        "query": "a warm drink for a cold morning",
        "check": lambda r: len(r) > 0,
        "expected": "Returns coffee/tea products",
    },

    # ── Type 8: Multi-hop (the one thing vector search structurally cannot do,
    #    and the current GraphRAG implementation doesn't attempt either — every
    #    other query type here is single-hop, exactly where vector search is
    #    competitive; this type tests the actual differentiator a knowledge
    #    graph is supposed to provide) ──────────────────────────────────────────
    {
        "id": 29, "type": "multi_hop",
        "query": "brands that sell products in both Coffee & Tea and Snacks & Candy",
        "check": lambda r: len(r) > 0 and all(
            "brand" in p and "item_name" not in p for p in r
        ),
        "expected": "Real brand names, from 2-hop Brand-Product-Category reasoning",
    },
    {
        "id": 30, "type": "multi_hop",
        "query": "vegan brands that also make gluten-free products",
        "check": lambda r: len(r) > 0 and all(
            "brand" in p and "item_name" not in p for p in r
        ),
        "expected": "Real brand names, from 2-hop Brand-Product-DietaryTag reasoning",
    },
]

# ── System runners ─────────────────────────────────────────────────────────────

def run_vector_only(query_obj: dict, hs: HybridSearch) -> tuple[list, float]:
    """
    System A — Qdrant semantic search WITH the same payload filters (tags,
    category, price) the other two systems get. Previously this called
    qdrant_search(query, top_k=10) — missing the required `client`/`model`
    positional args, so every call raised TypeError and was silently
    swallowed by the bare except below, and even if it hadn't crashed, no
    constraint filters were ever applied. Both are fixed here: real
    client/model are passed, and filters are parsed the same way GraphRAG's
    are so this is a fair "vector + payload filter" baseline, not a
    constraint-blind unfiltered search that fails by construction.
    """
    tags, cat, min_p, max_p, excl = parse_constraints(query_obj["query"])
    try:
        t0 = time.time()
        results = qdrant_search(
            query_obj["query"], hs.qdrant, hs.model, top_k=10,
            category=cat, max_price=max_p, min_price=min_p,
            dietary_tags=tags if tags else None,
            exclude_tags=excl if excl else None,
        )
        latency = round(time.time() - t0, 3)
        out = [{
            "item_name":    r.get("item_name"),
            "price":        r.get("price"),
            "category":     r.get("category"),
            "dietary_tags": r.get("dietary_tags") or [],
        } for r in results]
        return out, latency
    except Exception as e:
        print(f"\n    [vector_only error] {query_obj['query']!r}: {e}")
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

        elif qtype in ("multi_constraint", "compound", "negation", "price_range"):
            tags, cat, min_p, max_p, excl = parse_constraints(query_obj["query"])
            results = q.get_products(
                tags=tags if tags else None,
                category=cat,
                min_price=min_p,
                max_price=max_p,
                exclude_tags=excl if excl else None,
                limit=10
            )

        elif qtype == "multi_hop":
            # Genuine 2-hop Cypher — the one thing vector search and the
            # current GraphRAG implementation both structurally can't do.
            if "coffee" in query and "snacks" in query:
                cypher = """
MATCH (b:Brand)<-[:MADE_BY]-(:Product)-[:BELONGS_TO]->(:Category {name: 'Coffee & Tea'})
MATCH (b)<-[:MADE_BY]-(:Product)-[:BELONGS_TO]->(:Category {name: 'Snacks & Candy'})
RETURN DISTINCT b.name AS brand
LIMIT 10
"""
            else:  # "vegan brands that also make gluten-free products"
                cypher = """
MATCH (b:Brand)<-[:MADE_BY]-(:Product)-[:HAS_TAG]->(:DietaryTag {name: 'vegan'})
MATCH (b)<-[:MADE_BY]-(:Product)-[:HAS_TAG]->(:DietaryTag {name: 'gluten-free'})
RETURN DISTINCT b.name AS brand
LIMIT 10
"""
            results = q._run(cypher, {})

        else:  # semantic, subtle_semantic — graph does its best with keyword match
            results = q.get_products(limit=10)

    except Exception as e:
        print(f"\n    [graph_only error] {query_obj['query']!r}: {e}")
        results = []

    latency = round(time.time() - t0, 3)
    return results, latency


def run_graphrag(query_obj: dict, hs: HybridSearch) -> tuple[list, float]:
    """System C — GraphRAG hybrid: Qdrant candidates + Neo4j constraints."""
    tags, cat, min_p, max_p, excl = parse_constraints(query_obj["query"])
    try:
        t0 = time.time()
        results = hs.search(
            query_obj["query"],
            top_k=10,
            dietary_tags=tags if tags else None,
            category=cat,
            min_price=min_p,
            max_price=max_p,
            exclude_tags=excl if excl else None,
        )
        latency = round(time.time() - t0, 3)
        return results, latency
    except Exception as e:
        print(f"\n    [graphrag error] {query_obj['query']!r}: {e}")
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
    n_queries = len(QUERIES)
    print("\n" + "="*70)
    print("  RetailGraph — GraphRAG vs VectorRAG vs Neo4j Benchmark")
    print(f"  {n_queries} queries · 3 systems · ground truth scoring")
    print("="*70 + "\n")

    hs = HybridSearch()  # shared across vector-only and graphrag — one client/model load

    rows = []
    totals = {"vector": 0, "graph": 0, "graphrag": 0}
    latencies = {"vector": [], "graph": [], "graphrag": []}
    type_scores = {}

    for q in QUERIES:
        print(f"[{q['id']:02d}/{n_queries}] {q['query'][:55]:<55}", end=" ", flush=True)

        r_vec,  lat_vec  = run_vector_only(q, hs)
        r_gph,  lat_gph  = run_graph_only(q)
        r_rag,  lat_rag  = run_graphrag(q, hs)

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