"""
training/llm_judge.py

LLM-as-Judge evaluation for RetailGraph extraction quality.
Uses GPT-4o-mini to score extractions semantically (not just exact string match).

Reads:  evaluation/eval_results_r3  (678 examples from evaluate.py)
Writes: evaluation/judge_results.json
Logs:   MLflow (alongside field accuracy)

Cost:   ~$0.50 for all 678 examples
Time:   ~15-20 minutes

Usage:
    python training/llm_judge.py
    python training/llm_judge.py --max-samples 20   # quick test
    python training/llm_judge.py --fields category dietary_tags
"""

import os
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_FILE    = Path("evaluation/eval_results_r3")
OUTPUT_FILE  = Path("evaluation/judge_results.json")
MODEL        = "gpt-4o-mini"
SLEEP_SEC    = 0.3   # between calls to avoid rate limits

# Only judge these fields — the ones below 95% exact match
JUDGE_FIELDS = ["category", "dietary_tags", "allergen_list"]

SYSTEM_PROMPT = """You are evaluating a grocery product extraction model.
Your job is to score how correct the model's prediction is compared to the ground truth.

Score each field on a 1-5 scale:
5 = Perfectly correct — exact match or trivially equivalent (e.g. "fl oz" vs "Fl Oz")
4 = Minor difference — same meaning, slightly different format or wording
3 = Partially correct — right category/concept but missing detail or too broad
2 = Wrong but related — in the right ballpark but incorrect
1 = Completely wrong — wrong field, hallucinated, or missing entirely

For list fields (dietary_tags, allergen_list):
- Score based on how well the predicted list matches the ground truth list
- Missing items and extra items both reduce the score
- Empty list when ground truth is empty = 5

Return ONLY valid JSON, no explanation, no markdown:
{"score": <1-5>, "reason": "<one sentence>"}"""


def score_field(
    client: OpenAI,
    field: str,
    ground_truth,
    prediction,
    product_listing: str,
) -> dict:
    """Ask GPT-4o-mini to score a single field."""
    prompt = f"""Product listing: {product_listing[:300]}

Field: {field}
Ground truth: {json.dumps(ground_truth)}
Model prediction: {json.dumps(prediction)}

Score this prediction."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=100,
    )

    text = response.choices[0].message.content.strip()

    # Strip markdown if present
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        result = json.loads(text)
        return {
            "score":  int(result.get("score", 3)),
            "reason": result.get("reason", ""),
        }
    except Exception:
        return {"score": 3, "reason": "parse error"}


def get_product_listing(example: dict) -> str:
    """Extract product listing text from eval result."""
    pred = example.get("prediction", {})
    return pred.get("item_name", "") or f"Example {example['example_index']}"


def run_judge(
    client: OpenAI,
    results: list[dict],
    fields: list[str],
    max_samples: int = None,
) -> list[dict]:
    """Run LLM-as-Judge on all examples."""

    if max_samples:
        results = results[:max_samples]

    total = len(results)
    judged = []
    errors = 0

    print(f"Judging {total} examples on fields: {fields}")
    print(f"Model: {MODEL} · Est. cost: ~${total * len(fields) * 0.0001:.2f}\n")

    for i, example in enumerate(results):
        pred   = example.get("prediction", {})
        gt     = example.get("ground_truth", {})
        listing = get_product_listing(example)

        field_scores = {}

        for field in fields:
            gt_val   = gt.get(field)
            pred_val = pred.get(field)

            # Skip if both are None/empty — no point judging
            if not gt_val and not pred_val:
                field_scores[field] = {"score": 5, "reason": "both empty"}
                continue

            # If exact match already — skip API call, save credits
            exact_score = example.get("field_scores", {}).get(field, 0)
            if exact_score == 1.0:
                field_scores[field] = {"score": 5, "reason": "exact match"}
                continue

            try:
                result = score_field(client, field, gt_val, pred_val, listing)
                field_scores[field] = result
                time.sleep(SLEEP_SEC)
            except Exception as e:
                field_scores[field] = {"score": 3, "reason": f"error: {e}"}
                errors += 1

        judged.append({
            "example_index": example["example_index"],
            "product":       listing[:80],
            "exact_overall": example.get("overall_score", 0),
            "field_scores":  field_scores,
            "predictions":   {f: pred.get(f) for f in fields},
            "ground_truths": {f: gt.get(f) for f in fields},
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            avg_scores = {}
            for f in fields:
                scores = [
                    ex["field_scores"][f]["score"]
                    for ex in judged
                    if f in ex["field_scores"]
                ]
                avg_scores[f] = round(sum(scores) / len(scores), 2) if scores else 0
            print(f"  [{i+1}/{total}] running avg: {avg_scores} | errors: {errors}")

    return judged


def compute_summary(judged: list[dict], fields: list[str]) -> dict:
    """Compute final summary stats."""
    summary = {}

    for field in fields:
        scores = [
            ex["field_scores"][field]["score"]
            for ex in judged
            if field in ex["field_scores"]
        ]
        if not scores:
            continue

        exact_matches = sum(1 for ex in judged if ex["field_scores"].get(field, {}).get("score") == 5)
        false_negatives = sum(
            1 for ex in judged
            if ex["field_scores"].get(field, {}).get("score") == 5
            and ex.get("predictions", {}).get(field) != ex.get("ground_truths", {}).get(field)
        )

        summary[field] = {
            "avg_judge_score":    round(sum(scores) / len(scores), 3),
            "pct_score_4_or_5":   round(sum(1 for s in scores if s >= 4) / len(scores) * 100, 1),
            "pct_score_3":        round(sum(1 for s in scores if s == 3) / len(scores) * 100, 1),
            "pct_score_1_or_2":   round(sum(1 for s in scores if s <= 2) / len(scores) * 100, 1),
            "false_negatives":    false_negatives,
        }

    # Overall semantic accuracy (score >= 4 across all judged fields)
    all_high = sum(
        1 for ex in judged
        if all(
            ex["field_scores"].get(f, {}).get("score", 0) >= 4
            for f in fields
            if f in ex["field_scores"]
        )
    )
    summary["overall_semantic_accuracy"] = round(all_high / len(judged) * 100, 1)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fields", nargs="+", default=JUDGE_FIELDS)
    args = parser.parse_args()

    # Load eval results
    print(f"Loading eval results from {EVAL_FILE}...")
    with open(EVAL_FILE, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    print(f"Loaded {len(results)} examples")
    print(f"Existing field accuracy: {data['field_accuracy']}\n")

    # Run judge
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    judged = run_judge(client, results, args.fields, args.max_samples)

    # Summary
    summary = compute_summary(judged, args.fields)

    print("\n── LLM-as-Judge Results ──────────────────────────────────")
    print(f"Examples judged: {len(judged)}")
    print(f"Overall semantic accuracy (score ≥ 4): {summary['overall_semantic_accuracy']}%")
    print()
    for field, stats in summary.items():
        if field == "overall_semantic_accuracy":
            continue
        exact = data["field_accuracy"].get(field, 0)
        print(f"  {field}")
        print(f"    Exact match:      {exact*100:.1f}%")
        print(f"    Judge avg score:  {stats['avg_judge_score']}/5")
        print(f"    Score 4-5:        {stats['pct_score_4_or_5']}%")
        print(f"    Score 1-2:        {stats['pct_score_1_or_2']}%")
        print(f"    False negatives:  {stats['false_negatives']} (correct but failed exact match)")
        print()
    print("──────────────────────────────────────────────────────────")

    # Save
    output = {
        "timestamp":    data["timestamp"],
        "judge_model":  MODEL,
        "fields_judged": args.fields,
        "examples_judged": len(judged),
        "existing_field_accuracy": data["field_accuracy"],
        "summary":      summary,
        "results":      judged,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()