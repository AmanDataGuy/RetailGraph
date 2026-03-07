from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from src.extraction.normalizer import normalize_product
from src.extraction.schemas import ProductEntity

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
MIN_CONFIDENCE = 0.6
FAILED_CSV = Path("data/raw/failed.csv")
FAILED_CSV_FIELDS = [
    "timestamp", "product_id", "attempt", "step", "error"
]


# ── Result dataclass ──────────────────────────────────────────────────────────
class ValidationResult:
    def __init__(
        self,
        success: bool,
        entity: Optional[ProductEntity] = None,
        error: Optional[str] = None,
        step: Optional[str] = None,
        attempt: int = 1,
    ):
        self.success = success
        self.entity = entity
        self.error = error
        self.step = step
        self.attempt = attempt

    def __repr__(self):
        if self.success:
            return f"ValidationResult(success=True, product_id={self.entity.product_id})"
        return f"ValidationResult(success=False, step={self.step}, error={self.error[:80]})"


# ── Failure logger ────────────────────────────────────────────────────────────
def _log_failure(product_id: str, attempt: int, step: str, error: str) -> None:
    """Append a failed extraction to data/raw/failed.csv."""
    FAILED_CSV.parent.mkdir(parents=True, exist_ok=True)

    write_header = not FAILED_CSV.exists()

    with open(FAILED_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FAILED_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "product_id": product_id,
            "attempt": attempt,
            "step": step,
            "error": error[:500],  # cap error length
        })


# ── Format error for model feedback ──────────────────────────────────────────
def _format_error_for_retry(step: str, error: str) -> str:
    """
    Format a validation error into a clear message the model can
    understand and use to correct its output on retry.
    """
    if step == "json_parse":
        return (
            f"Your response could not be parsed as JSON.\n"
            f"Error: {error}\n"
            f"Please respond with valid JSON only. No markdown, no backticks."
        )
    elif step == "pydantic":
        return (
            f"Your JSON was parsed but failed schema validation.\n"
            f"Errors:\n{error}\n"
            f"Fix these fields and try again."
        )
    elif step == "confidence":
        return (
            f"Extraction confidence is too low (below {MIN_CONFIDENCE}).\n"
            f"Please re-examine the product carefully and extract with more certainty."
        )
    return f"Validation failed at step '{step}': {error}"


# ── Format Pydantic errors ────────────────────────────────────────────────────
def _format_pydantic_errors(exc: ValidationError) -> str:
    """
    Convert Pydantic ValidationError into a clean readable string
    that tells the model exactly which fields are wrong and why.
    """
    lines = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err["loc"])
        msg = err["msg"]
        lines.append(f"  - {field}: {msg}")
    return "\n".join(lines)


# ── Core validator ────────────────────────────────────────────────────────────
def validate_extraction(
    raw_json: str,
    product_id: str,
    attempt: int = 1,
) -> ValidationResult:
    """
    Run a single validation attempt on raw model JSON output.

    Steps:
    1. JSON parse
    2. Normalize
    3. Pydantic validation
    4. Confidence check

    Returns ValidationResult with success=True and entity,
    or success=False with error message and step name.
    """

    # ── Step 1: JSON parse ────────────────────────────────────────────────────
    try:
        # strip markdown code fences if model included them
        clean_json = raw_json.strip()
        if clean_json.startswith("```"):
            lines = clean_json.split("\n")
            clean_json = "\n".join(lines[1:-1])

        parsed = json.loads(clean_json)

    except json.JSONDecodeError as e:
        error = str(e)
        logger.warning(f"[{product_id}] attempt {attempt} — JSON parse failed: {error}")
        return ValidationResult(
            success=False,
            error=_format_error_for_retry("json_parse", error),
            step="json_parse",
            attempt=attempt,
        )

    # ── Step 2: Normalize ─────────────────────────────────────────────────────
    parsed["product_id"] = product_id
    normalized = normalize_product(parsed)

    # ── Step 3: Pydantic validation ───────────────────────────────────────────
    try:
        entity = ProductEntity(**normalized)
        entity.assign_content_tier()
        entity.compute_quality_score()

    except ValidationError as e:
        error = _format_pydantic_errors(e)
        logger.warning(f"[{product_id}] attempt {attempt} — Pydantic failed:\n{error}")
        return ValidationResult(
            success=False,
            error=_format_error_for_retry("pydantic", error),
            step="pydantic",
            attempt=attempt,
        )

    # ── Step 4: Confidence check ──────────────────────────────────────────────
    if entity.extraction_confidence < MIN_CONFIDENCE:
        error = (
            f"extraction_confidence={entity.extraction_confidence} "
            f"is below minimum {MIN_CONFIDENCE}"
        )
        logger.warning(f"[{product_id}] attempt {attempt} — low confidence: {error}")
        return ValidationResult(
            success=False,
            error=_format_error_for_retry("confidence", error),
            step="confidence",
            attempt=attempt,
        )

    logger.info(
        f"[{product_id}] attempt {attempt} — "
        f"PASSED (score={entity.quality_score}, tier={entity.content_tier})"
    )
    return ValidationResult(success=True, entity=entity, attempt=attempt)


# ── Retry wrapper ─────────────────────────────────────────────────────────────
def validate_with_retry(
    raw_json: str,
    product_id: str,
    retry_callback=None,
) -> ValidationResult:
    """
    Validate with up to MAX_RETRIES attempts.

    retry_callback: optional function(product_id, error_message, attempt) -> str
        Called when validation fails, should return new raw_json for retry.
        If None, validation fails immediately after first failure.

    On final failure, logs to failed.csv and returns last ValidationResult.
    """
    result = validate_extraction(raw_json, product_id, attempt=1)

    if result.success:
        return result

    # retry loop
    current_json = raw_json
    for attempt in range(2, MAX_RETRIES + 1):
        if retry_callback is None:
            break

        # get new model output with error feedback
        try:
            current_json = retry_callback(
                product_id,
                result.error,
                attempt,
            )
        except Exception as e:
            logger.error(f"[{product_id}] retry_callback failed: {e}")
            break

        result = validate_extraction(current_json, product_id, attempt=attempt)

        if result.success:
            return result

    # all attempts exhausted — log failure
    _log_failure(
        product_id=product_id,
        attempt=result.attempt,
        step=result.step or "unknown",
        error=result.error or "unknown error",
    )
    logger.error(
        f"[{product_id}] FAILED after {result.attempt} attempts "
        f"at step '{result.step}'"
    )

    return result