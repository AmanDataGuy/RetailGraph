import csv
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.extraction.validator import (
    ValidationResult,
    validate_extraction,
    validate_with_retry,
    _log_failure,
    _format_pydantic_errors,
    _format_error_for_retry,
    FAILED_CSV,
    MAX_RETRIES,
    MIN_CONFIDENCE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_valid_json(overrides: dict = {}) -> str:
    base = {
        "item_name": "Smucker's Natural Peanut Butter 16oz",
        "price": 9.99,
        "quantity_value": 16.0,
        "quantity_unit": "oz",
        "category": "Nuts & Seeds",
        "dietary_tags": ["organic"],
        "allergen_list": ["peanut"],
        "extraction_confidence": 0.92,
    }
    base.update(overrides)
    return json.dumps(base)


# ── ValidationResult Tests ────────────────────────────────────────────────────
class TestValidationResult:

    def test_success_repr(self):
        from src.extraction.schemas import ProductEntity
        entity = ProductEntity(product_id="1", item_name="Test", price=9.99)
        result = ValidationResult(success=True, entity=entity)
        assert "success=True" in repr(result)

    def test_failure_repr(self):
        result = ValidationResult(
            success=False, error="bad field", step="pydantic"
        )
        assert "success=False" in repr(result)
        assert "pydantic" in repr(result)


# ── Format Error Tests ────────────────────────────────────────────────────────
class TestFormatError:

    def test_json_parse_error_message(self):
        msg = _format_error_for_retry("json_parse", "Expecting value")
        assert "JSON" in msg
        assert "backticks" in msg

    def test_pydantic_error_message(self):
        msg = _format_error_for_retry("pydantic", "quantity_unit: invalid")
        assert "schema validation" in msg
        assert "quantity_unit" in msg

    def test_confidence_error_message(self):
        msg = _format_error_for_retry("confidence", "0.4 below 0.6")
        assert "confidence" in msg.lower()

    def test_unknown_step_message(self):
        msg = _format_error_for_retry("unknown_step", "something broke")
        assert "unknown_step" in msg


# ── JSON Parse Step Tests ─────────────────────────────────────────────────────
class TestJsonParse:

    def test_valid_json_passes(self):
        result = validate_extraction(make_valid_json(), "123")
        assert result.success is True

    def test_invalid_json_fails(self):
        result = validate_extraction("this is not json", "123")
        assert result.success is False
        assert result.step == "json_parse"

    def test_empty_string_fails(self):
        result = validate_extraction("", "123")
        assert result.success is False
        assert result.step == "json_parse"

    def test_markdown_fences_stripped(self):
        raw = "```json\n" + make_valid_json() + "\n```"
        result = validate_extraction(raw, "123")
        assert result.success is True

    def test_markdown_fences_no_lang_stripped(self):
        raw = "```\n" + make_valid_json() + "\n```"
        result = validate_extraction(raw, "123")
        assert result.success is True

    def test_partial_json_fails(self):
        result = validate_extraction('{"item_name": "test"', "123")
        assert result.success is False
        assert result.step == "json_parse"


# ── Pydantic Validation Step Tests ────────────────────────────────────────────
class TestPydanticValidation:

    def test_invalid_category_fails(self):
        # negative price — normalizer doesn't touch price, schema rejects it
        raw = make_valid_json({"price": -99.0})
        result = validate_extraction(raw, "123")
        assert result.success is False
        assert result.step == "pydantic"

    def test_invalid_unit_fails(self):
        # item_name too short — clean_item_name returns None, required field missing
        raw = make_valid_json({"item_name": "X"})
        result = validate_extraction(raw, "123")
        assert result.success is False
        assert result.step == "pydantic"

    # ← FIX 2: was testing quality_score (not validated by normalizer),
    #   now uses extraction_confidence > 1.0 which Pydantic rejects (le=1.0)
    def test_invalid_dietary_tag_fails(self):
        # extraction_confidence > 1.0 — normalizer passes it through, schema rejects it
        raw = make_valid_json({"extraction_confidence": 2.0})
        result = validate_extraction(raw, "123")
        assert result.success is False
        assert result.step == "pydantic"

    def test_negative_price_fails(self):
        raw = make_valid_json({"price": -1.0})
        result = validate_extraction(raw, "123")
        assert result.success is False
        assert result.step == "pydantic"

    def test_error_message_contains_field_name(self):
        raw = make_valid_json({"price": -5.0})
        result = validate_extraction(raw, "123")
        assert result.success is False
        assert "price" in result.error

    def test_missing_price_fails(self):
        data = json.loads(make_valid_json())
        del data["price"]
        result = validate_extraction(json.dumps(data), "123")
        assert result.success is False


# ── Confidence Check Step Tests ───────────────────────────────────────────────
class TestConfidenceCheck:

    def test_high_confidence_passes(self):
        raw = make_valid_json({"extraction_confidence": 0.95})
        result = validate_extraction(raw, "123")
        assert result.success is True

    def test_exact_minimum_confidence_passes(self):
        raw = make_valid_json({"extraction_confidence": MIN_CONFIDENCE})
        result = validate_extraction(raw, "123")
        assert result.success is True

    def test_below_minimum_confidence_fails(self):
        raw = make_valid_json({"extraction_confidence": 0.4})
        result = validate_extraction(raw, "123")
        assert result.success is False
        assert result.step == "confidence"

    def test_zero_confidence_fails(self):
        raw = make_valid_json({"extraction_confidence": 0.0})
        result = validate_extraction(raw, "123")
        assert result.success is False
        assert result.step == "confidence"

    def test_confidence_error_message_helpful(self):
        raw = make_valid_json({"extraction_confidence": 0.3})
        result = validate_extraction(raw, "123")
        assert "confidence" in result.error.lower()


# ── Successful Validation Tests ───────────────────────────────────────────────
class TestSuccessfulValidation:

    def test_entity_returned_on_success(self):
        result = validate_extraction(make_valid_json(), "123")
        assert result.entity is not None
        assert result.entity.product_id == "123"

    def test_product_id_injected(self):
        result = validate_extraction(make_valid_json(), "999")
        assert result.entity.product_id == "999"

    def test_content_tier_assigned(self):
        result = validate_extraction(make_valid_json(), "123")
        assert result.entity.content_tier is not None

    def test_quality_score_computed(self):
        result = validate_extraction(make_valid_json(), "123")
        assert result.entity.quality_score > 0

    def test_attempt_number_recorded(self):
        result = validate_extraction(make_valid_json(), "123", attempt=2)
        assert result.attempt == 2

    def test_unit_normalized_before_validation(self):
        raw = make_valid_json({"quantity_unit": "Ounce"})
        result = validate_extraction(raw, "123")
        assert result.success is True
        assert result.entity.quantity_unit == "oz"

    def test_dietary_tags_normalized(self):
        raw = make_valid_json({"dietary_tags": ["gluten free", "non gmo"]})
        result = validate_extraction(raw, "123")
        assert result.success is True
        assert "gluten-free" in result.entity.dietary_tags
        assert "non-GMO" in result.entity.dietary_tags


# ── Failure Logging Tests ─────────────────────────────────────────────────────
class TestFailureLogging:

    def test_failure_logged_to_csv(self, tmp_path):
        test_csv = tmp_path / "failed.csv"
        with patch("src.extraction.validator.FAILED_CSV", test_csv):
            _log_failure("123", 3, "pydantic", "bad category")
            assert test_csv.exists()
            rows = list(csv.DictReader(open(test_csv, encoding="utf-8")))
            assert len(rows) == 1
            assert rows[0]["product_id"] == "123"
            assert rows[0]["step"] == "pydantic"

    def test_csv_header_written_once(self, tmp_path):
        test_csv = tmp_path / "failed.csv"
        with patch("src.extraction.validator.FAILED_CSV", test_csv):
            _log_failure("1", 1, "json_parse", "error1")
            _log_failure("2", 2, "pydantic", "error2")
            lines = open(test_csv, encoding="utf-8").readlines()
            assert len(lines) == 3

    def test_long_error_truncated(self, tmp_path):
        test_csv = tmp_path / "failed.csv"
        long_error = "x" * 1000
        with patch("src.extraction.validator.FAILED_CSV", test_csv):
            _log_failure("123", 1, "pydantic", long_error)
            rows = list(csv.DictReader(open(test_csv, encoding="utf-8")))
            assert len(rows[0]["error"]) <= 500

    def test_timestamp_present(self, tmp_path):
        test_csv = tmp_path / "failed.csv"
        with patch("src.extraction.validator.FAILED_CSV", test_csv):
            _log_failure("123", 1, "pydantic", "error")
            rows = list(csv.DictReader(open(test_csv, encoding="utf-8")))
            assert rows[0]["timestamp"] != ""


# ── Retry Logic Tests ─────────────────────────────────────────────────────────
class TestRetryLogic:

    def test_success_on_first_attempt_no_retry(self):
        call_count = {"n": 0}

        def callback(pid, error, attempt):
            call_count["n"] += 1
            return make_valid_json()

        result = validate_with_retry(make_valid_json(), "123", callback)
        assert result.success is True
        assert call_count["n"] == 0

    def test_retry_called_on_failure(self):
        call_count = {"n": 0}

        def callback(pid, error, attempt):
            call_count["n"] += 1
            return make_valid_json()  # return valid on retry

        bad_json = make_valid_json({"price": -1.0})
        result = validate_with_retry(bad_json, "123", callback)
        assert result.success is True
        assert call_count["n"] == 1

    def test_max_retries_respected(self, tmp_path):
        with patch("src.extraction.validator.FAILED_CSV", tmp_path / "failed.csv"):
            call_count = {"n": 0}

            def callback(pid, error, attempt):
                call_count["n"] += 1
                return make_valid_json({"price": -1.0})  # always invalid

            bad_json = make_valid_json({"price": -1.0})
            result = validate_with_retry(bad_json, "123", callback)
            assert result.success is False
            assert call_count["n"] == MAX_RETRIES - 1

    def test_failure_logged_after_max_retries(self, tmp_path):
        test_csv = tmp_path / "failed.csv"
        with patch("src.extraction.validator.FAILED_CSV", test_csv):

            def callback(pid, error, attempt):
                return make_valid_json({"price": -1.0})

            bad_json = make_valid_json({"price": -1.0})
            validate_with_retry(bad_json, "123", callback)
            assert test_csv.exists()

    def test_no_callback_fails_immediately(self, tmp_path):
        with patch("src.extraction.validator.FAILED_CSV", tmp_path / "failed.csv"):
            bad_json = make_valid_json({"price": -1.0})
            result = validate_with_retry(bad_json, "123", retry_callback=None)
            assert result.success is False

    def test_success_on_second_attempt(self, tmp_path):
        with patch("src.extraction.validator.FAILED_CSV", tmp_path / "failed.csv"):
            attempts = {"n": 0}

            def callback(pid, error, attempt):
                attempts["n"] += 1
                if attempts["n"] == 1:
                    return make_valid_json({"price": -1.0})  # still bad
                return make_valid_json()  # good on second retry

            bad_json = make_valid_json({"price": -1.0})
            result = validate_with_retry(bad_json, "123", callback)
            assert result.success is True