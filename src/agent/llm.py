"""
RetailGraph — LLM wrapper (Groq API)
Model: see MODEL below (openai/gpt-oss-120b as of the deprecation fix).
All agent nodes call generate() or generate_json() from here.
Both are @traceable — with LANGCHAIN_TRACING_V2=true in .env, each call
shows up as a named LLM span in LangSmith with prompt/response/latency.
"""

import os
import json
import logging
from typing import Optional
from groq import Groq, RateLimitError
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

log = logging.getLogger("retailgraph.llm")

# ── Client pool — rotates across multiple Groq API keys ─────────────────────
# GROQ_API_KEY is required; GROQ_API_KEY_2..GROQ_API_KEY_5 are optional extra
# keys (e.g. separate free-tier accounts) used as fallback once the current
# key hits a 429 RateLimitError (daily-token-limit exhaustion is the common
# case here — see eval/score_ragas.py's known quota issue). The rotation
# remembers which key is currently "live" across calls, so once key N is
# exhausted, every later call goes straight to key N+1 instead of re-trying
# the dead key first each time.
_API_KEYS = [
    k for k in [
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
        os.getenv("GROQ_API_KEY_4"),
        os.getenv("GROQ_API_KEY_5"),
    ] if k
]
if not _API_KEYS:
    raise RuntimeError("No GROQ_API_KEY configured — set at least GROQ_API_KEY in .env")

_clients = [Groq(api_key=k) for k in _API_KEYS]
_current_key_index = 0

log.info(f"Groq key pool: {len(_clients)} key(s) configured")


def _call_with_rotation(make_request):
    """
    Runs make_request(client) against the current key; on RateLimitError,
    advances to the next key in the pool and retries, up to one full pass
    over all configured keys. Raises the last error if every key is
    exhausted.
    """
    global _current_key_index
    last_error = None
    for attempt in range(len(_clients)):
        client = _clients[_current_key_index]
        try:
            return make_request(client)
        except RateLimitError as e:
            last_error = e
            log.warning(
                f"Groq key #{_current_key_index + 1}/{len(_clients)} rate-limited "
                f"({e}); rotating to next key."
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


@traceable(name="groq_generate", run_type="llm")
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


@traceable(name="groq_generate_json", run_type="llm")
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