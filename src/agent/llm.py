"""
RetailGraph — LLM wrapper (Groq API)
Model: llama-3.3-70b-versatile
All agent nodes call generate() or generate_json() from here.
"""

import os
import re
import json
import logging
from typing import Optional
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("retailgraph.llm")

# ── Client pool — rotates across multiple Groq API keys ─────────────────────
# A concurrency stress test (16 parallel requests) hit Groq's per-minute
# token limit (TPM: 8000 on a single free-tier key) and failed 2/16 requests
# outright; a later all-keys-exhausted run hit the per-day (TPD) cap across
# every key. GROQ_API_KEY is required; any GROQ_API_KEY_<N> is an optional
# extra key, tried in order once the current one 429s. Scanned dynamically
# (not hardcoded to _1/_2/_3) and sorted numerically — the numbering doesn't
# have to be contiguous (e.g. keys added as _2.._8 with no _1 still all get
# picked up). The index persists across calls so an exhausted key isn't
# retried on every request.
_numbered_keys = sorted(
    (int(m.group(1)), v)
    for k, v in os.environ.items()
    if (m := re.fullmatch(r"GROQ_API_KEY_(\d+)", k)) and v
)
_API_KEYS = [k for k in [os.getenv("GROQ_API_KEY"), *(v for _, v in _numbered_keys)] if k]
if not _API_KEYS:
    raise RuntimeError("No GROQ_API_KEY configured — set at least GROQ_API_KEY in .env")

_clients = [Groq(api_key=k) for k in _API_KEYS]
_current_key_index = 0

log.info(f"Groq key pool: {len(_clients)} key(s) configured")


def _call_with_rotation(make_request):
    """Try the current key; on RateLimitError, advance to the next key and
    retry, up to one full pass over all configured keys."""
    global _current_key_index
    last_error = None
    for _ in range(len(_clients)):
        client = _clients[_current_key_index]
        try:
            return make_request(client)
        except RateLimitError as e:
            last_error = e
            log.warning(
                f"Groq key #{_current_key_index + 1}/{len(_clients)} rate-limited; rotating."
            )
            _current_key_index = (_current_key_index + 1) % len(_clients)
    raise last_error

# llama-3.3-70b-versatile was deprecated by Groq on 2026-08-16 (returns 404
# model_not_found as of this fix). Groq's own docs recommend gpt-oss-120b or
# qwen3.6-27b as replacements; gpt-oss-120b confirmed compatible with the
# json_object response_format mode used in generate_json() below.
# https://console.groq.com/docs/deprecations
MODEL    = "openai/gpt-oss-120b"
MAX_TOKENS = 512
TEMPERATURE = 0.1   # near-deterministic for structured outputs


def generate(system: str, prompt: str, temperature: float = TEMPERATURE) -> str:
    """
    Raw text generation. Used for answer formatting (Node 5).

    Args:
        system:      System prompt string
        prompt:      User message string
        temperature: Sampling temperature (default 0.1)

    Returns:
        Model response as stripped string
    """
    response = _call_with_rotation(lambda client: client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=temperature,
    ))
    return response.choices[0].message.content.strip()


def generate_json(system: str, prompt: str) -> dict:
    """
    JSON-mode generation. Used for intent extraction and Cypher generation.
    Groq's JSON mode guarantees valid JSON output — no parsing errors.

    Args:
        system: System prompt (must instruct model to return JSON)
        prompt: User message

    Returns:
        Parsed dict. Returns {} on any failure.
    """
    try:
        response = _call_with_rotation(lambda client: client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,   # fully deterministic for JSON
            response_format={"type": "json_object"},
        ))
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)

    except Exception as e:
        log.error(f"generate_json failed: {e}")
        return {}


# ── Smoke test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing generate()...")
    answer = generate(
        system="You are a helpful assistant.",
        prompt="Say hello in one sentence.",
    )
    print(f"  Response: {answer}")

    print("\nTesting generate_json()...")
    result = generate_json(
        system=(
            "You extract grocery query intent. "
            "Return JSON with keys: intent, category, dietary_tags, max_price, brand."
        ),
        prompt="show me vegan gluten-free snacks under $8",
    )
    print(f"  Parsed JSON: {json.dumps(result, indent=2)}")

    print("\n✅ LLM wrapper working.")