"""
Patch script — fixes two bugs in training/generate_visual_pairs.py:

BUG 1: GPT-4o returns null for dietary_tags / allergen_list
        → Pydantic fails because we changed defaults to list[str]
        FIX: Normalize nulls to [] before calling validate_extraction

BUG 2: build_retry_messages appends to messages that contain image_url blocks
        → On retry, GPT-4o receives malformed conversation history
        FIX: Build retry messages manually, keeping image in first user turn only
"""

from pathlib import Path

path = Path("training/generate_visual_pairs.py")
content = path.read_text(encoding="utf-8")

OLD = '''def extract_visual_product(
    client: OpenAI,
    sample_id: int,
    catalog_content: str,
    price: Optional[float],
    image_b64: str,
) -> tuple[bool, Optional[dict], list[dict]]:
    """
    Run GPT-4o vision extraction with up to MAX_RETRIES attempts.

    On failure, builds a retry message with the specific error description
    so GPT-4o can correct itself — same retry pattern as generate_pairs.py.

    Args:
        client:          OpenAI client
        sample_id:       product ID
        catalog_content: raw catalog text
        price:           product price
        image_b64:       base64 image

    Returns:
        (success, entity_dict, messages_used)
        success=True  → entity_dict has validated ProductEntity fields
        success=False → entity_dict is None
        messages_used → the messages list for building the final pair
    """
    messages   = build_visual_messages(sample_id, catalog_content, price, image_b64)
    raw_output = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = call_gpt4o_vision(client, messages)
        except Exception as e:
            logger.warning(
                "[%s] GPT-4o API error attempt %d: %s",
                sample_id, attempt, e
            )
            time.sleep(5)
            continue

        result = validate_extraction(raw_output, str(sample_id), attempt=attempt)

        if result.success:
            return True, result.entity.model_dump(), messages

        logger.debug(
            "[%s] attempt %d failed at step \'%s\': %s",
            sample_id, attempt, result.step, result.error
        )

        # Build retry message with error feedback (same as generate_pairs.py)
        if attempt < MAX_RETRIES:
            messages = build_retry_messages(messages, raw_output, result.error or "")
            time.sleep(SLEEP_SECONDS)

    return False, None, messages'''

NEW = '''def _fix_null_lists(raw_output: str) -> str:
    """
    Normalize null dietary_tags / allergen_list to empty lists before validation.

    GPT-4o frequently returns null for these fields when no tags are present.
    Our Pydantic schema now requires list[str] (not Optional) so null fails
    validation. This pre-processing step fixes it transparently.

    Only modifies these two fields — everything else is untouched.
    Returns original string unchanged if JSON parsing fails.
    """
    try:
        parsed = json.loads(raw_output)
        if parsed.get("dietary_tags") is None:
            parsed["dietary_tags"] = []
        if parsed.get("allergen_list") is None:
            parsed["allergen_list"] = []
        return json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, AttributeError):
        return raw_output


def extract_visual_product(
    client: OpenAI,
    sample_id: int,
    catalog_content: str,
    price: Optional[float],
    image_b64: str,
) -> tuple[bool, Optional[dict], list[dict]]:
    """
    Run GPT-4o vision extraction with up to MAX_RETRIES attempts.

    On failure, appends the error as a plain-text user message so GPT-4o
    can correct its output. The image is only sent in the first user turn —
    subsequent retry turns are text-only to avoid API errors with mixed
    content types in conversation history.

    Args:
        client:          OpenAI client
        sample_id:       product ID
        catalog_content: raw catalog text
        price:           product price
        image_b64:       base64 image

    Returns:
        (success, entity_dict, messages_used)
        success=True  → entity_dict has validated ProductEntity fields
        success=False → entity_dict is None
        messages_used → the messages list (text-only, safe for JSONL storage)
    """
    messages   = build_visual_messages(sample_id, catalog_content, price, image_b64)
    raw_output = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = call_gpt4o_vision(client, messages)
        except Exception as e:
            logger.warning(
                "[%s] GPT-4o API error attempt %d: %s",
                sample_id, attempt, e
            )
            time.sleep(5)
            continue

        # Normalize null lists before Pydantic validation —
        # GPT-4o returns null when no tags/allergens present,
        # but our schema requires [] not null.
        raw_output = _fix_null_lists(raw_output)

        result = validate_extraction(raw_output, str(sample_id), attempt=attempt)

        if result.success:
            return True, result.entity.model_dump(), messages

        logger.debug(
            "[%s] attempt %d failed at step \'%s\': %s",
            sample_id, attempt, result.step, result.error
        )

        # Retry: append failed output + error as plain text turns.
        # Do NOT use build_retry_messages here — it appends to the original
        # messages list which contains image_url blocks. Mixing image_url
        # and plain-text content in later turns causes GPT-4o API errors.
        if attempt < MAX_RETRIES:
            messages = messages + [
                {
                    "role": "assistant",
                    "content": raw_output,
                },
                {
                    "role": "user",
                    "content": (
                        f"That output failed validation:\\n{result.error}\\n\\n"
                        "Fix the issue and return ONLY the corrected JSON object. "
                        "No explanation, no markdown, no backticks."
                    ),
                },
            ]
            time.sleep(SLEEP_SECONDS)

    return False, None, messages'''

if OLD in content:
    content = content.replace(OLD, NEW)
    path.write_text(content, encoding="utf-8")
    print("✅ Patched successfully")
else:
    print("❌ Could not find target function — check for whitespace differences")