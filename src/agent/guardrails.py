"""
RetailGraph — input/output guardrails.

Deliberately regex/keyword-based, not a transformer classifier: this project
already stripped torch/sentence-transformers out of the deploy image once to
fit Render's free-tier memory cap (see EVAL_REPORT.md) — a toxicity/PII model
would undo that. These checks run on every request, so they need to be cheap
and dependency-free.

Input checks (block before the query ever reaches the LLM):
    - PII: email, phone, credit-card-shaped, SSN-shaped patterns
    - Prompt injection / jailbreak phrasing
    - Toxicity: a small profanity/slur keyword list

Output check (advisory, not blocking — see check_output docstring):
    - Groundedness: does the answer actually reference a real result?
"""
import re

_PII_PATTERNS = {
    "email":       re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone":       re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore (all|the)?\s*(previous|prior|above)\s*instructions",
        r"disregard (all|the)?\s*(previous|prior|above)",
        r"you are now\b",
        r"system prompt",
        r"reveal (your|the) (prompt|instructions)",
        r"developer mode",
        r"jailbreak",
        r"act as (an?|the)\b",
    ]
]

# Small, deliberately short list — enough to demonstrate the pattern, not a
# production moderation system. Real deployments should use a maintained
# wordlist or an LLM-judge pass (see eval/score_deepeval.py's ToxicityMetric
# for the offline-eval equivalent of this check).
_TOXIC_WORDS = {"fuck", "shit", "bitch", "asshole", "nigger", "faggot", "cunt"}


def check_input(query: str) -> dict:
    """
    Returns {"blocked": bool, "reason": str | None, "categories": list[str]}.
    Categories present when blocked: any of "pii", "prompt_injection", "toxicity".
    """
    categories = []

    if any(p.search(query) for p in _PII_PATTERNS.values()):
        categories.append("pii")

    if any(p.search(query) for p in _INJECTION_PATTERNS):
        categories.append("prompt_injection")

    words = set(re.findall(r"[a-z']+", query.lower()))
    if words & _TOXIC_WORDS:
        categories.append("toxicity")

    if not categories:
        return {"blocked": False, "reason": None, "categories": []}

    reason = {
        "pii":              "Please don't include personal information (email, phone, card numbers) in a product search.",
        "prompt_injection": "That request looks like it's trying to change how I operate rather than search products.",
        "toxicity":         "Please rephrase your search without offensive language.",
    }[categories[0]]

    return {"blocked": True, "reason": reason, "categories": categories}


def check_output(answer: str, raw_results: list[dict]) -> bool:
    """
    Advisory groundedness check, not blocking.

    Deliberately not enforced (i.e. never swaps out the answer): a false
    positive here would silently replace a correct answer with a generic
    refusal, which is worse for a demo app than an unflagged edge case.
    Callers should log/surface `grounded=False`, not act on it — it's a
    signal for eval/monitoring, not a filter. A real hallucination check
    belongs in eval/score_deepeval.py's HallucinationMetric, which can
    afford an LLM-judge call; this one has to be free and instant since it
    runs on every live request.

    Heuristic: if there are results, at least one result's item_name should
    appear (case-insensitive substring) somewhere in the answer text.
    """
    if not raw_results:
        return True  # nothing to ground against — "no results" answers pass trivially

    answer_lower = answer.lower()
    names = [
        (r.get("item_name") or r.get("p.item_name") or "").lower()
        for r in raw_results
    ]
    return any(name and name in answer_lower for name in names)
