# tests/unit/test_prompt_templates.py
"""
Unit tests for src/extraction/prompt_templates.py.

Testing strategy:
    - We test the structure and content of every prompt builder function.
    - We do NOT call any LLM API in these tests — all tests are pure Python.
    - We verify the prompts contain the correct controlled vocabulary,
      required fields, and structural properties.
    - We test retry message construction separately since that is critical
      for the validate_with_retry() flow in validator.py.
"""

import json
import pytest

from src.extraction.prompt_templates import (
    ALLOWED_CATEGORIES,
    ALLOWED_DIETARY_TAGS,
    CANONICAL_UNITS,
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_system_prompt,
    build_user_prompt,
    build_few_shot_messages,
    build_full_messages,
    build_retry_messages,
    estimate_tokens,
)


# ── ALLOWED_CATEGORIES Tests ──────────────────────────────────────────────────

class TestAllowedCategories:

    def test_has_eleven_categories(self):
        # Must match CATEGORY_NAMES in label_model.py — 11 categories
        assert len(ALLOWED_CATEGORIES) == 11

    def test_contains_beverages(self):
        assert "Beverages" in ALLOWED_CATEGORIES

    def test_contains_coffee_and_tea(self):
        assert "Coffee & Tea" in ALLOWED_CATEGORIES

    def test_contains_spices(self):
        assert "Spices & Seasonings" in ALLOWED_CATEGORIES

    def test_contains_grains(self):
        assert "Grains, Beans & Legumes" in ALLOWED_CATEGORIES

    def test_contains_protein_bars(self):
        assert "Protein Bars & Snacks" in ALLOWED_CATEGORIES

    def test_all_categories_are_strings(self):
        assert all(isinstance(c, str) for c in ALLOWED_CATEGORIES)

    def test_no_duplicate_categories(self):
        assert len(ALLOWED_CATEGORIES) == len(set(ALLOWED_CATEGORIES))


# ── ALLOWED_DIETARY_TAGS Tests ────────────────────────────────────────────────

class TestAllowedDietaryTags:

    def test_has_fifteen_tags(self):
        # Must match DIETARY_TAGS in label_model.py — 15 tags
        assert len(ALLOWED_DIETARY_TAGS) == 15

    def test_contains_organic(self):
        assert "organic" in ALLOWED_DIETARY_TAGS

    def test_contains_kosher(self):
        assert "kosher" in ALLOWED_DIETARY_TAGS

    def test_contains_gluten_free(self):
        assert "gluten-free" in ALLOWED_DIETARY_TAGS

    def test_contains_non_gmo(self):
        assert "non-GMO" in ALLOWED_DIETARY_TAGS

    def test_contains_vegan(self):
        assert "vegan" in ALLOWED_DIETARY_TAGS

    def test_all_lowercase_or_correct_case(self):
        for tag in ALLOWED_DIETARY_TAGS:
            if tag != "non-GMO":
                assert tag == tag.lower(), f"Tag '{tag}' should be lowercase"

    def test_no_duplicate_tags(self):
        assert len(ALLOWED_DIETARY_TAGS) == len(set(ALLOWED_DIETARY_TAGS))


# ── CANONICAL_UNITS Tests ─────────────────────────────────────────────────────

class TestCanonicalUnits:

    def test_contains_oz(self):
        assert "oz" in CANONICAL_UNITS

    def test_contains_fl_oz(self):
        assert "fl oz" in CANONICAL_UNITS

    def test_contains_ct(self):
        assert "ct" in CANONICAL_UNITS

    def test_contains_lb(self):
        assert "lb" in CANONICAL_UNITS

    def test_no_uppercase_units(self):
        for unit in CANONICAL_UNITS:
            assert unit == unit.lower(), f"Unit '{unit}' should be lowercase"


# ── SYSTEM_PROMPT Tests ───────────────────────────────────────────────────────

class TestSystemPrompt:

    def test_system_prompt_is_string(self):
        assert isinstance(SYSTEM_PROMPT, str)

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_contains_all_required_fields(self):
        required_fields = [
            "item_name", "price", "quantity_value", "quantity_unit",
            "category", "dietary_tags", "allergen_list", "extraction_confidence"
        ]
        for field in required_fields:
            assert field in SYSTEM_PROMPT, f"Field '{field}' missing from system prompt"

    def test_contains_canonical_units(self):
        for unit in ["oz", "fl oz", "ct", "lb"]:
            assert unit in SYSTEM_PROMPT

    def test_contains_all_allowed_categories(self):
        for category in ALLOWED_CATEGORIES:
            assert category in SYSTEM_PROMPT, f"Category '{category}' missing from system prompt"

    def test_contains_all_dietary_tags(self):
        for tag in ALLOWED_DIETARY_TAGS:
            assert tag in SYSTEM_PROMPT, f"Tag '{tag}' missing from system prompt"

    def test_no_json_instruction(self):
        assert "ONLY" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT

    def test_no_markdown_instruction(self):
        assert "markdown" in SYSTEM_PROMPT.lower() or "backtick" in SYSTEM_PROMPT.lower()

    def test_contains_confidence_calibration(self):
        assert "extraction_confidence" in SYSTEM_PROMPT
        assert "0.90" in SYSTEM_PROMPT or "0.95" in SYSTEM_PROMPT

    def test_never_output_1_instruction(self):
        assert "1.0" in SYSTEM_PROMPT


# ── build_system_prompt Tests ─────────────────────────────────────────────────

class TestBuildSystemPrompt:

    def test_returns_string(self):
        result = build_system_prompt()
        assert isinstance(result, str)

    def test_no_leading_trailing_whitespace(self):
        result = build_system_prompt()
        assert result == result.strip()

    def test_same_content_as_system_prompt_constant(self):
        result = build_system_prompt()
        assert result == SYSTEM_PROMPT.strip()


# ── build_user_prompt Tests ───────────────────────────────────────────────────

class TestBuildUserPrompt:

    def test_returns_string(self):
        result = build_user_prompt("Item Name: Test Product\nValue: 1")
        assert isinstance(result, str)

    def test_contains_catalog_content(self):
        catalog = "Item Name: McCormick Garlic Powder\nValue: 3"
        result = build_user_prompt(catalog)
        assert "McCormick Garlic Powder" in result

    def test_appends_price_when_provided(self):
        result = build_user_prompt("Item Name: Test Product", price=4.99)
        assert "4.99" in result
        assert "Price:" in result

    def test_does_not_duplicate_price_if_already_present(self):
        catalog = "Item Name: Test Product\nPrice: 4.99"
        result = build_user_prompt(catalog, price=4.99)
        assert result.count("Price:") == 1

    def test_no_price_appended_when_none(self):
        result = build_user_prompt("Item Name: Test Product")
        assert "Price:" not in result

    def test_contains_extraction_instruction(self):
        result = build_user_prompt("Item Name: Test Product")
        assert "JSON" in result

    def test_strips_whitespace_from_catalog_content(self):
        result = build_user_prompt("  Item Name: Test Product  ")
        assert not result.startswith(" ")


# ── FEW_SHOT_EXAMPLES Tests ───────────────────────────────────────────────────

class TestFewShotExamples:

    def test_has_at_least_five_examples(self):
        assert len(FEW_SHOT_EXAMPLES) >= 5

    def test_each_example_has_input_and_output(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            assert "input" in ex, f"Example {i} missing 'input'"
            assert "output" in ex, f"Example {i} missing 'output'"

    def test_each_output_is_valid_json(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            try:
                parsed = json.loads(ex["output"])
                assert isinstance(parsed, dict), f"Example {i} output is not a dict"
            except json.JSONDecodeError as e:
                pytest.fail(f"Example {i} output is not valid JSON: {e}")

    def test_each_output_has_required_fields(self):
        required = [
            "item_name", "price", "quantity_value", "quantity_unit",
            "category", "dietary_tags", "allergen_list", "extraction_confidence"
        ]
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            parsed = json.loads(ex["output"])
            for field in required:
                assert field in parsed, f"Example {i} missing field '{field}'"

    def test_each_output_uses_canonical_unit(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            parsed = json.loads(ex["output"])
            assert parsed["quantity_unit"] in CANONICAL_UNITS, (
                f"Example {i} has non-canonical unit '{parsed['quantity_unit']}'"
            )

    def test_each_output_uses_allowed_category(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            parsed = json.loads(ex["output"])
            assert parsed["category"] in ALLOWED_CATEGORIES, (
                f"Example {i} has invalid category '{parsed['category']}'"
            )

    def test_each_output_dietary_tags_are_allowed(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            parsed = json.loads(ex["output"])
            for tag in parsed["dietary_tags"]:
                assert tag in ALLOWED_DIETARY_TAGS, (
                    f"Example {i} has invalid dietary tag '{tag}'"
                )

    def test_confidence_never_1_0(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            parsed = json.loads(ex["output"])
            assert parsed["extraction_confidence"] < 1.0, (
                f"Example {i} has confidence = 1.0 which teaches bad calibration"
            )

    def test_confidence_always_above_0_4(self):
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            parsed = json.loads(ex["output"])
            assert parsed["extraction_confidence"] >= 0.4, (
                f"Example {i} has confidence < 0.4"
            )

    def test_sparse_example_has_lower_confidence(self):
        low_confidence_examples = [
            ex for ex in FEW_SHOT_EXAMPLES
            if json.loads(ex["output"])["extraction_confidence"] < 0.85
        ]
        assert len(low_confidence_examples) >= 1, (
            "Need at least one sparse/low-confidence example to teach calibration"
        )

    def test_has_hinglish_example(self):
        hinglish_keywords = ["masala", "atta", "dal", "besan", "chawal", "garam"]
        has_hinglish = any(
            any(kw in ex["input"].lower() for kw in hinglish_keywords)
            for ex in FEW_SHOT_EXAMPLES
        )
        assert has_hinglish, "Need at least one Hinglish example for Indian products"

    def test_has_allergen_example(self):
        has_allergen = any(
            json.loads(ex["output"])["allergen_list"] != []
            for ex in FEW_SHOT_EXAMPLES
        )
        assert has_allergen, "Need at least one example with allergens"

    def test_item_name_does_not_contain_trailing_size(self):
        import re
        for i, ex in enumerate(FEW_SHOT_EXAMPLES):
            parsed = json.loads(ex["output"])
            item_name = parsed["item_name"]
            if re.search(r'\d+oz$|\d+lb$|\d+g$|\d+ Count$', item_name, re.IGNORECASE):
                pytest.fail(
                    f"Example {i} item_name '{item_name}' ends with size — "
                    f"size should be in quantity_value/quantity_unit instead"
                )


# ── build_few_shot_messages Tests ─────────────────────────────────────────────

class TestBuildFewShotMessages:

    def test_returns_list(self):
        result = build_few_shot_messages()
        assert isinstance(result, list)

    def test_alternates_user_and_assistant(self):
        result = build_few_shot_messages()
        for i, msg in enumerate(result):
            if i % 2 == 0:
                assert msg["role"] == "user"
            else:
                assert msg["role"] == "assistant"

    def test_length_is_double_examples(self):
        result = build_few_shot_messages()
        assert len(result) == 2 * len(FEW_SHOT_EXAMPLES)

    def test_each_message_has_role_and_content(self):
        result = build_few_shot_messages()
        for msg in result:
            assert "role" in msg
            assert "content" in msg
            assert len(msg["content"]) > 0

    def test_assistant_messages_contain_json(self):
        result = build_few_shot_messages()
        assistant_messages = [m for m in result if m["role"] == "assistant"]
        for msg in assistant_messages:
            try:
                json.loads(msg["content"])
            except json.JSONDecodeError:
                pytest.fail(f"Assistant message is not valid JSON: {msg['content'][:100]}")


# ── build_full_messages Tests ─────────────────────────────────────────────────

class TestBuildFullMessages:

    def test_returns_list(self):
        result = build_full_messages("Item Name: Test Product\nValue: 1")
        assert isinstance(result, list)

    def test_first_message_is_system(self):
        result = build_full_messages("Item Name: Test Product\nValue: 1")
        assert result[0]["role"] == "system"

    def test_last_message_is_user(self):
        result = build_full_messages("Item Name: Test Product\nValue: 1")
        assert result[-1]["role"] == "user"

    def test_last_message_contains_catalog_content(self):
        result = build_full_messages("Item Name: Tropicana Orange Juice")
        assert "Tropicana Orange Juice" in result[-1]["content"]

    def test_with_few_shot_has_more_messages_than_without(self):
        with_fs    = build_full_messages("Item Name: Test", include_few_shot=True)
        without_fs = build_full_messages("Item Name: Test", include_few_shot=False)
        assert len(with_fs) > len(without_fs)

    def test_without_few_shot_has_exactly_two_messages(self):
        result = build_full_messages("Item Name: Test", include_few_shot=False)
        assert len(result) == 2

    def test_with_few_shot_message_count(self):
        result = build_full_messages("Item Name: Test", include_few_shot=True)
        expected = 1 + 2 * len(FEW_SHOT_EXAMPLES) + 1
        assert len(result) == expected

    def test_price_appended_to_last_message(self):
        result = build_full_messages("Item Name: Test Product", price=9.99)
        assert "9.99" in result[-1]["content"]


# ── build_retry_messages Tests ────────────────────────────────────────────────

class TestBuildRetryMessages:

    def _base_messages(self):
        return build_full_messages("Item Name: Test Product\nValue: 1", include_few_shot=False)

    def test_returns_list(self):
        base = self._base_messages()
        result = build_retry_messages(base, '{"broken": json}', "JSON parse error")
        assert isinstance(result, list)

    def test_longer_than_original(self):
        base   = self._base_messages()
        result = build_retry_messages(base, '{"broken": json}', "error")
        assert len(result) > len(base)

    def test_appends_two_messages(self):
        base   = self._base_messages()
        result = build_retry_messages(base, '{"broken": json}', "error")
        assert len(result) == len(base) + 2

    def test_new_second_to_last_is_assistant(self):
        base   = self._base_messages()
        result = build_retry_messages(base, '{"broken": json}', "error")
        assert result[-2]["role"] == "assistant"

    def test_new_last_message_is_user(self):
        base   = self._base_messages()
        result = build_retry_messages(base, '{"broken": json}', "error")
        assert result[-1]["role"] == "user"

    def test_failed_output_in_assistant_message(self):
        base   = self._base_messages()
        failed = '{"item_name": "Test", "price": -1.0}'
        result = build_retry_messages(base, failed, "negative price")
        assert failed in result[-2]["content"]

    def test_error_description_in_user_message(self):
        base   = self._base_messages()
        error  = "quantity_unit 'Ounce' is invalid — use 'oz'"
        result = build_retry_messages(base, "{}", error)
        assert error in result[-1]["content"]

    def test_retry_message_asks_for_json_only(self):
        base   = self._base_messages()
        result = build_retry_messages(base, "{}", "some error")
        assert "JSON" in result[-1]["content"]

    def test_does_not_mutate_original_messages(self):
        base         = self._base_messages()
        original_len = len(base)
        build_retry_messages(base, "{}", "error")
        assert len(base) == original_len


# ── estimate_tokens Tests ─────────────────────────────────────────────────────

class TestEstimateTokens:

    def test_returns_integer(self):
        messages = [{"role": "user", "content": "hello world"}]
        result = estimate_tokens(messages)
        assert isinstance(result, int)

    def test_more_content_means_more_tokens(self):
        short = [{"role": "user", "content": "hi"}]
        long  = [{"role": "user", "content": "hello world this is a much longer message"}]
        assert estimate_tokens(long) > estimate_tokens(short)

    def test_empty_messages_returns_zero(self):
        assert estimate_tokens([]) == 0

    def test_multiple_messages_summed(self):
        messages = [
            {"role": "system",    "content": "a" * 100},
            {"role": "user",      "content": "b" * 100},
            {"role": "assistant", "content": "c" * 100},
        ]
        result = estimate_tokens(messages)
        assert result == 75  # 300 chars / 4

    def test_full_prompt_estimate_reasonable(self):
        messages = build_full_messages(
            "Item Name: Test Product\nValue: 1",
            include_few_shot=True
        )
        tokens = estimate_tokens(messages)
        assert tokens < 5000, f"Prompt is too long: {tokens} estimated tokens"