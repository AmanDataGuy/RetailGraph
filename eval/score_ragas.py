"""
eval/score_ragas.py

Step 2 of the eval pipeline. Runs in venv-eval/ (NOT the app's venv — see
eval/requirements-eval.txt for why RAGAS needs a separate environment).

Reads evaluation/agent_eval_raw.json (written by eval/collect_results.py,
which drives the real agent) and scores it with RAGAS:

  - faithfulness       — is the answer grounded in the retrieved contexts?
  - answer_relevancy   — does the answer address the question?
  - context_precision  — are the retrieved contexts actually relevant?
  - context_recall     — (only for rows with a ground_truth) did retrieval
                          find what it should have?

No OpenAI key needed: the judge LLM is Groq (langchain-groq -> ChatGroq,
the same openai/gpt-oss-120b the agent itself uses — llama-3.3-70b-versatile
was deprecated by Groq on 2026-08-16, see src/agent/llm.py), and embeddings
are the same local all-MiniLM-L6-v2 model src/graph/hybrid_search.py already
loads — via langchain_community's HuggingFaceEmbeddings wrapper.

Usage:
    venv-eval/Scripts/python eval/score_ragas.py
"""

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

INPUT_FILE = ROOT / "evaluation" / "agent_eval_raw.json"
OUTPUT_FILE = ROOT / "evaluation" / "ragas_results.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _pick_working_key() -> str:
    """
    RAGAS's internal batching (ragas.evaluate) catches per-job errors
    (RateLimitError included) and marks them failed rather than surfacing
    them here, so mid-run key rotation like src/agent/llm.py's isn't
    possible through this interface. Instead: try each configured
    GROQ_API_KEY* env var with a 1-token ping before starting the real
    run, and use the first one that isn't already rate-limited. This
    directly fixes the recurring failure mode where key #1 is already
    exhausted from earlier testing the same day.
    """
    from groq import Groq, RateLimitError

    keys = [
        k for k in [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
            os.getenv("GROQ_API_KEY_4"),
            os.getenv("GROQ_API_KEY_5"),
        ] if k
    ]
    if not keys:
        raise SystemExit("No GROQ_API_KEY configured — check your .env")

    for i, key in enumerate(keys):
        try:
            Groq(api_key=key).chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            print(f"Using Groq key #{i + 1}/{len(keys)} for RAGAS judge calls.")
            return key
        except RateLimitError:
            print(f"Groq key #{i + 1}/{len(keys)} already rate-limited, trying next...")
            continue

    raise SystemExit(f"All {len(keys)} configured Groq key(s) are currently rate-limited.")


def build_dataset(records: list[dict]) -> Dataset:
    return Dataset.from_dict({
        "question": [r["query"] for r in records],
        "answer": [r["answer"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"{INPUT_FILE} not found. Run this first, in the APP venv:\n"
            f"  venv/Scripts/python eval/collect_results.py"
        )

    raw = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    records = raw["records"]
    print(f"Loaded {len(records)} agent runs from {INPUT_FILE.name}")

    groq_key = _pick_working_key()

    llm = LangchainLLMWrapper(ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=groq_key))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    # context_recall requires a non-empty ground_truth; only score it where we
    # actually built one (filter/brand/analytics queries with a known answer).
    with_gt = [r for r in records if r["ground_truth"]]
    without_gt = [r for r in records if not r["ground_truth"]]

    results_by_id = {}

    if with_gt:
        print(f"\nScoring {len(with_gt)} queries with all 4 metrics (have ground_truth)...")
        ds = build_dataset(with_gt)
        scored = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        df = scored.to_pandas()
        for i, r in enumerate(with_gt):
            results_by_id[r["id"]] = df.iloc[i].to_dict()

    if without_gt:
        print(f"\nScoring {len(without_gt)} queries with 3 metrics (no ground_truth — skipping context_recall)...")
        ds = build_dataset(without_gt)
        scored = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
        )
        df = scored.to_pandas()
        for i, r in enumerate(without_gt):
            results_by_id[r["id"]] = df.iloc[i].to_dict()

    # ── Aggregate ────────────────────────────────────────────────────────────
    def avg(metric):
        vals = [v[metric] for v in results_by_id.values() if metric in v and v[metric] == v[metric]]  # NaN-safe
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "n_scored": len(results_by_id),
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
    }

    print("\n── RAGAS Summary ─────────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {
        "summary": summary,
        "agent_summary": raw["summary"],
        "per_query": [
            {**{k: v for k, v in r.items() if k != "contexts"}, **results_by_id.get(r["id"], {})}
            for r in records
        ],
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
