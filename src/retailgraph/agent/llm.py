"""
# llm.py - thin Groq wrapper (openai/gpt-oss-120b) with API-key rotation

- generate() for plain text, generate_json() for JSON-mode output.
- Called by agent/nodes.py. Rotates through GROQ_API_KEY and any GROQ_API_KEY_<N> on rate limits.
"""

import json
import logging
import os
import re

from dotenv import load_dotenv
from groq import Groq, RateLimitError

load_dotenv()

log = logging.getLogger("retailgraph.llm")

# Built on first use, not at import, so tests and CI can import the agent without any key set.
# The index persists across calls so an exhausted key isn't retried on every request.
_clients: list[Groq] = []
_current_key_index = 0


# Build one Groq client per configured key (GROQ_API_KEY first, then _<N> keys in numeric order).
def _get_clients() -> list[Groq]:
    if not _clients:
        numbered = sorted(
            (int(m.group(1)), v)
            for k, v in os.environ.items()
            if (m := re.fullmatch(r"GROQ_API_KEY_(\d+)", k)) and v
        )
        keys = [k for k in [os.getenv("GROQ_API_KEY"), *(v for _, v in numbered)] if k]
        if not keys:
            raise RuntimeError("No GROQ_API_KEY configured — set at least GROQ_API_KEY in .env")
        _clients.extend(Groq(api_key=k) for k in keys)
        log.info(f"Groq key pool: {len(_clients)} key(s) configured")
    return _clients


# Run a request on the current key; on a rate limit, move to the next key (one full pass at most).
def _call_with_rotation(make_request):
    global _current_key_index
    _get_clients()
    last_error = None
    for _ in range(len(_clients)):
        try:
            return make_request(_clients[_current_key_index])
        except RateLimitError as e:
            last_error = e
            log.warning(f"Groq key #{_current_key_index + 1}/{len(_clients)} rate-limited; rotating.")
            _current_key_index = (_current_key_index + 1) % len(_clients)
    raise last_error


MODEL = "openai/gpt-oss-120b"
# gpt-oss reasoning tokens count against this cap, so keep effort low and leave headroom,
# or JSON comes back cut off / empty.
MAX_TOKENS = 1024
REASONING_EFFORT = "low"
TEMPERATURE = 0.1
# gpt-oss emits typographic spaces/hyphens ("gluten‑free"); map them to plain ones so names match the catalog.
_PLAIN = str.maketrans({" ": " ", " ": " ", " ": " ", "‐": "-", "‑": "-"})


# Send one system + user message pair to Groq and return the raw response.
def _chat(system: str, prompt: str, temperature: float, **extra):
    return _call_with_rotation(lambda client: client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        reasoning_effort=REASONING_EFFORT,
        **extra,
    ))


# Generate plain text (used to write the final answer).
def generate(system: str, prompt: str, temperature: float = TEMPERATURE) -> str:
    response = _chat(system, prompt, temperature)
    return (response.choices[0].message.content or "").translate(_PLAIN).strip()


# Generate a JSON object (intent, Cypher); returns {} on any failure.
def generate_json(system: str, prompt: str) -> dict:
    try:
        response = _chat(system, prompt, 0.0, response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        log.error(f"generate_json failed: {e}")
        return {}
