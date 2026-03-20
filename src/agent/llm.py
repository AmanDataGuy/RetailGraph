"""
RetailGraph — LLM wrapper (Groq API)
Model: llama-3.3-70b-versatile
All agent nodes call generate() or generate_json() from here.
"""

import os
import json
import logging
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("retailgraph.llm")

# ── Client ─────────────────────────────────────────────────────────────────
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL    = "llama-3.3-70b-versatile"
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
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=temperature,
    )
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
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,   # fully deterministic for JSON
            response_format={"type": "json_object"},
        )
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