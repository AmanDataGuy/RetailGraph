"""
eval/score_deepeval.py

Step 2b of the eval pipeline (alongside score_ragas.py) — runs in venv-eval/.
Reads evaluation/agent_eval_raw.json (written by eval/collect_results.py) and
scores it with DeepEval:

  - HallucinationMetric — does the answer contradict/invent beyond the
                           retrieved context? The live equivalent of this is
                           src/agent/guardrails.py's check_output(), which is
                           free but only a substring heuristic; this is the
                           slower, LLM-judged version, run offline.
  - ToxicityMetric       — offline equivalent of guardrails.py's keyword
                            toxicity check, judged by an LLM instead of a
                            fixed wordlist.
  - BiasMetric           — not covered anywhere else in this repo.
  - GEval (correctness)  — custom rubric: does the answer actually address
                            the query, grounded in the given context?

Judge LLM is Groq (same openai/gpt-oss-120b the agent itself uses), via a
~20-line DeepEvalBaseLLM subclass — no LangChain wrapper needed, unlike
score_ragas.py, since DeepEval's custom-model interface is just
generate()/a_generate()/get_model_name().

Each metric runs with async_mode=False, one test case at a time — RAGAS's
default concurrent batching is what caused its still-unresolved timeout
issue (see EVAL_REPORT.md §6); this deliberately trades speed for not
repeating that failure mode.

Usage:
    venv-eval/Scripts/python eval/score_deepeval.py
"""

import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from groq import Groq, RateLimitError
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, ToxicityMetric, BiasMetric, GEval
from deepeval.test_case import LLMTestCaseParams

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

INPUT_FILE  = ROOT / "evaluation" / "agent_eval_raw.json"
OUTPUT_FILE = ROOT / "evaluation" / "deepeval_results.json"

MODEL = "openai/gpt-oss-120b"


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """
    DeepEval judge-model adapter around the raw Groq SDK — rotates across
    up to 4 keys on RateLimitError, same pattern as src/agent/llm.py. A
    first run of this script used a single key and burned its entire daily
    quota (TPD) by query 29/30 (see EVAL_REPORT.md) — this fixes that for
    the eval script specifically, mirroring the live-app fix.
    """

    def __init__(self, api_keys: list[str], model: str = MODEL):
        self._clients = [Groq(api_key=k) for k in api_keys]
        self._idx = 0
        super().__init__(model=model)

    def load_model(self):
        return self._clients  # generate() indexes into this directly

    def generate(self, prompt: str) -> str:
        last_error = None
        for _ in range(len(self._clients)):
            try:
                resp = self._clients[self._idx].chat.completions.create(
                    model=self.name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return resp.choices[0].message.content
            except RateLimitError as e:
                last_error = e
                print(f"    judge key #{self._idx + 1}/{len(self._clients)} rate-limited; rotating.")
                self._idx = (self._idx + 1) % len(self._clients)
        raise last_error

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.name


def build_metrics(judge: GroqDeepEvalLLM) -> dict:
    common = dict(model=judge, async_mode=False, include_reason=False)
    return {
        "hallucination": HallucinationMetric(**common),
        "toxicity":      ToxicityMetric(**common),
        "bias":          BiasMetric(**common),
        "correctness":   GEval(
            name="Correctness",
            criteria=(
                "Determine if the actual output correctly and helpfully "
                "answers the input query, using only information present "
                "in the given context."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            model=judge,
            async_mode=False,
        ),
    }


def main():
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"{INPUT_FILE} not found. Run this first, in the APP venv:\n"
            f"  venv/Scripts/python eval/collect_results.py"
        )

    raw = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    records = raw["records"]
    print(f"Loaded {len(records)} agent runs from {INPUT_FILE.name}")

    groq_keys = [
        k for k in [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        ] if k
    ]
    if not groq_keys:
        raise SystemExit("GROQ_API_KEY not set — check your .env")
    print(f"Judge key pool: {len(groq_keys)} key(s) configured")

    judge   = GroqDeepEvalLLM(api_keys=groq_keys)
    metrics = build_metrics(judge)

    results_by_id = {}
    for i, r in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {r['id']}: {r['query'][:60]}")
        test_case = LLMTestCase(
            input=r["query"],
            actual_output=r["answer"],
            context=r["contexts"] or ["(no retrieval context — 0 results)"],
        )
        scores = {}
        for name, metric in metrics.items():
            try:
                metric.measure(test_case)
                scores[name] = round(metric.score, 3) if metric.score is not None else None
            except Exception as e:
                print(f"    {name} failed: {type(e).__name__}: {e}")
                scores[name] = None
        results_by_id[r["id"]] = scores

    # ── Aggregate ────────────────────────────────────────────────────────────
    def avg(metric):
        vals = [v[metric] for v in results_by_id.values() if v.get(metric) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "n_scored":      len(results_by_id),
        "hallucination": avg("hallucination"),
        "toxicity":      avg("toxicity"),
        "bias":          avg("bias"),
        "correctness":   avg("correctness"),
    }

    print("\n── DeepEval Summary ──────────────────────────────────────")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output = {
        "summary":       summary,
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
