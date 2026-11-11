"""
# guardrails.py - cheap regex/keyword checks on agent input and output

- check_input() blocks PII, prompt injection and toxic words before any LLM call.
- check_output() is an advisory groundedness flag. Both are called by agent/graph.py run_query().
- Regex, not a classifier model: runs on every request and must stay dependency-free and small in memory.
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

# Short demo list, not a production moderation system.
_TOXIC_WORDS = {"fuck", "shit", "bitch", "asshole", "nigger", "faggot", "cunt"}

# Prefix match catches inflections ("fucking"); the leading \b avoids mid-word hits ("Scunthorpe").
_TOXIC_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _TOXIC_WORDS) + r")\w*\b",
    re.IGNORECASE,
)


# Decide whether a query must be blocked; returns {"blocked", "reason", "categories"}.
def check_input(query: str) -> dict:
    categories = []

    if any(p.search(query) for p in _PII_PATTERNS.values()):
        categories.append("pii")

    if any(p.search(query) for p in _INJECTION_PATTERNS):
        categories.append("prompt_injection")

    if _TOXIC_PATTERN.search(query):
        categories.append("toxicity")

    if not categories:
        return {"blocked": False, "reason": None, "categories": []}

    reason = {
        "pii":              "Please don't include personal information (email, phone, card numbers) in a product search.",
        "prompt_injection": "That request looks like it's trying to change how I operate rather than search products.",
        "toxicity":         "Please rephrase your search without offensive language.",
    }[categories[0]]

    return {"blocked": True, "reason": reason, "categories": categories}


# True if the answer names at least one result; advisory only, so a false alarm never hides a good answer.
def check_output(answer: str, raw_results: list[dict]) -> bool:
    if not raw_results:
        return True  # nothing to ground against — "no results" answers pass trivially

    answer_lower = answer.lower()
    names = ((r.get("item_name") or r.get("p.item_name") or "").lower() for r in raw_results)
    return any(name and name in answer_lower for name in names)
