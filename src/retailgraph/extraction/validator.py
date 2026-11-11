"""
# validator.py - checks one raw model/teacher JSON reply and turns it into a ProductEntity

- Steps: parse JSON, normalize, Pydantic-validate, confidence check. Failures carry a retry message for the model.
- Used by training/generate_pairs.py and training/generate_visual_pairs.py.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pydantic import ValidationError

from retailgraph.extraction.normalizer import normalize_product
from retailgraph.extraction.schemas import ProductEntity

logger = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.6


# Outcome of one validation attempt: the entity on success, else the failed step and error text.
@dataclass(eq=False, repr=False)
class ValidationResult:
    success: bool
    entity: ProductEntity | None = None
    error: str | None = None
    step: str | None = None
    attempt: int = 1

    def __repr__(self):
        if self.success:
            return f"ValidationResult(success=True, product_id={self.entity.product_id})"
        return f"ValidationResult(success=False, step={self.step}, error={self.error[:80]})"


# Turn a failed step into feedback the model can act on when it retries.
def _format_error_for_retry(step: str, error: str) -> str:
    if step == "json_parse":
        return (
            f"Your response could not be parsed as JSON.\n"
            f"Error: {error}\n"
            f"Please respond with valid JSON only. No markdown, no backticks."
        )
    if step == "pydantic":
        return (
            f"Your JSON was parsed but failed schema validation.\n"
            f"Errors:\n{error}\n"
            f"Fix these fields and try again."
        )
    if step == "confidence":
        return (
            f"Extraction confidence is too low (below {MIN_CONFIDENCE}).\n"
            f"Please re-examine the product carefully and extract with more certainty."
        )
    return f"Validation failed at step '{step}': {error}"


# List each Pydantic error as "  - field -> path: message".
def _format_pydantic_errors(exc: ValidationError) -> str:
    return "\n".join(
        f"  - {' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
    )


# Parse, normalize and validate one raw JSON reply; returns a ValidationResult.
def validate_extraction(raw_json: str, product_id: str, attempt: int = 1) -> ValidationResult:
    try:
        clean_json = raw_json.strip()
        if clean_json.startswith("```"):  # drop a markdown code fence
            clean_json = "\n".join(clean_json.split("\n")[1:-1])
        parsed = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.warning(f"[{product_id}] attempt {attempt} — JSON parse failed: {e}")
        return ValidationResult(False, error=_format_error_for_retry("json_parse", str(e)),
                                step="json_parse", attempt=attempt)

    parsed["product_id"] = product_id
    try:
        entity = ProductEntity(**normalize_product(parsed))
        entity.assign_content_tier()
        entity.compute_quality_score()
    except ValidationError as e:
        error = _format_pydantic_errors(e)
        logger.warning(f"[{product_id}] attempt {attempt} — Pydantic failed:\n{error}")
        return ValidationResult(False, error=_format_error_for_retry("pydantic", error),
                                step="pydantic", attempt=attempt)

    if entity.extraction_confidence < MIN_CONFIDENCE:
        error = f"extraction_confidence={entity.extraction_confidence} is below minimum {MIN_CONFIDENCE}"
        logger.warning(f"[{product_id}] attempt {attempt} — low confidence: {error}")
        return ValidationResult(False, error=_format_error_for_retry("confidence", error),
                                step="confidence", attempt=attempt)

    logger.info(
        f"[{product_id}] attempt {attempt} — PASSED (score={entity.quality_score}, tier={entity.content_tier})"
    )
    return ValidationResult(True, entity=entity, attempt=attempt)
