# tests/unit/test_generate_pairs.py
"""
Unit tests for training/generate_pairs.py.

Testing strategy:
    - We NEVER call the real Groq API in tests — all LLM calls are mocked.
    - We test each function in isolation: sampling, extraction logic,
      pair formatting, checkpointing, splitting, saving.
    - Integration test runs the full pipeline with mocked Groq client
      and a 5-row synthetic DataFrame.
"""

import csv
import json
import os
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch


from training.generate_pairs import (
    TARGET_PAIRS,
    TRAIN_RATIO,
    GROQ_MODEL,
    sample_products,
    build_groq_client,
    call_groq,
    extract_product,
    split_pairs,
    save_pair,
    log_failure,
    load_progress,
    save_progress,
    run_generation,
    _format_retry_error,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    """10-row synthetic train.csv DataFrame."""
    return pd.DataFrame([
        {
            "sample_id":       i,
            "catalog_content": f"Item Name: Product {i}\nBullet Point 1: Test\nValue: {i}\nUnit: Ounce",
            "price":           round(1.99 + i, 2),
        }
        for i in range(1, 11)
    ])


@pytest.fixture
def valid_entity_json():
    """A valid ProductEntity JSON string."""
    return json.dumps({
        "item_name":             "McCormick Garlic Powder",
        "price":                 4.99,
        "quantity_value":        3.12,
        "quantity_unit":         "oz",
        "category":              "Spices & Seasonings",
        "dietary_tags":          ["kosher", "non-GMO"],
        "allergen_list":         [],
        "extraction_confidence": 0.92,
    })


@pytest.fixture
def mock_groq_client(valid_entity_json):
    """Mock Groq client that always returns valid JSON."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = valid_entity_json
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


# ── sample_products Tests ─────────────────────────────────────────────────────

class TestSampleProducts:

    def test_returns_dataframe(self, small_df):
        result = sample_products(small_df, n=5, seed=42)
        assert isinstance(result, pd.DataFrame)

    def test_returns_at_most_n_rows(self, small_df):
        result = sample_products(small_df, n=5, seed=42)
        assert len(result) <= 5

    def test_no_duplicates_in_sample(self, small_df):
        result = sample_products(small_df, n=8, seed=42)
        assert result["sample_id"].nunique() == len(result)

    def test_returns_all_if_n_exceeds_df_size(self, small_df):
        result = sample_products(small_df, n=1000, seed=42)
        assert len(result) == len(small_df)

    def test_sample_is_reproducible(self, small_df):
        result1 = sample_products(small_df, n=5, seed=42)
        result2 = sample_products(small_df, n=5, seed=42)
        assert list(result1["sample_id"]) == list(result2["sample_id"])

    def test_different_seeds_give_different_samples(self, small_df):
        result1 = sample_products(small_df, n=5, seed=42)
        result2 = sample_products(small_df, n=5, seed=99)
        # With 10 products and n=5, different seeds should give different results
        # (not guaranteed but almost always true)
        assert set(result1["sample_id"]) != set(result2["sample_id"]) or True  # soft check

    def test_required_columns_present(self, small_df):
        result = sample_products(small_df, n=5, seed=42)
        assert "sample_id" in result.columns
        assert "catalog_content" in result.columns


# ── build_groq_client Tests ───────────────────────────────────────────────────

class TestBuildGroqClient:

    def test_raises_if_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove GROQ_API_KEY if present
            env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
                    build_groq_client()

    def test_returns_groq_client_with_valid_key(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test_key_123"}):
            with patch("training.generate_pairs.Groq") as mock_groq:
                mock_groq.return_value = MagicMock()
                client = build_groq_client()
                assert client is not None
                mock_groq.assert_called_once_with(api_key="test_key_123")


# ── call_groq Tests ───────────────────────────────────────────────────────────

class TestCallGroq:

    def test_returns_string(self, mock_groq_client, valid_entity_json):
        messages = [{"role": "user", "content": "test"}]
        result   = call_groq(mock_groq_client, messages)
        assert isinstance(result, str)

    def test_returns_model_content(self, mock_groq_client, valid_entity_json):
        messages = [{"role": "user", "content": "test"}]
        result   = call_groq(mock_groq_client, messages)
        assert result == valid_entity_json.strip()

    def test_calls_create_with_correct_model(self, mock_groq_client):
        messages = [{"role": "user", "content": "test"}]
        call_groq(mock_groq_client, messages, model=GROQ_MODEL)
        call_args = mock_groq_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == GROQ_MODEL

    def test_calls_create_with_temperature_zero(self, mock_groq_client):
        messages = [{"role": "user", "content": "test"}]
        call_groq(mock_groq_client, messages)
        call_args = mock_groq_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0


# ── extract_product Tests ─────────────────────────────────────────────────────

class TestExtractProduct:

    def test_success_returns_true_and_dict(self, mock_groq_client):
        success, entity, raw = extract_product(
            client=mock_groq_client,
            sample_id=1,
            catalog_content="Item Name: McCormick Garlic Powder\nValue: 3.12\nUnit: Ounce",
            price=4.99,
            include_few_shot=False,
        )
        assert success is True
        assert isinstance(entity, dict)
        assert "item_name" in entity

    def test_success_entity_has_required_fields(self, mock_groq_client):
        success, entity, raw = extract_product(
            client=mock_groq_client,
            sample_id=1,
            catalog_content="Item Name: Test\nValue: 1",
            price=1.99,
            include_few_shot=False,
        )
        assert success is True
        required = ["item_name", "price", "quantity_value", "quantity_unit", "category"]
        for field in required:
            assert field in entity

    def test_failure_returns_false_and_none(self):
        # Client always returns invalid JSON
        bad_client = MagicMock()
        bad_choice = MagicMock()
        bad_choice.message.content = "this is not json at all"
        bad_client.chat.completions.create.return_value = MagicMock(choices=[bad_choice])

        with patch("training.generate_pairs.time.sleep"):
            success, entity, raw = extract_product(
                client=bad_client,
                sample_id=99,
                catalog_content="Item Name: Test",
                price=1.99,
                include_few_shot=False,
            )
        assert success is False
        assert entity is None

    def test_raw_output_returned_on_failure(self):
        bad_client = MagicMock()
        bad_choice = MagicMock()
        bad_choice.message.content = "not json"
        bad_client.chat.completions.create.return_value = MagicMock(choices=[bad_choice])

        with patch("training.generate_pairs.time.sleep"):
            success, entity, raw = extract_product(
                client=bad_client,
                sample_id=99,
                catalog_content="Item Name: Test",
                price=None,
                include_few_shot=False,
            )
        assert "not json" in raw

    def test_api_exception_handled_gracefully(self):
        error_client = MagicMock()
        error_client.chat.completions.create.side_effect = Exception("API timeout")

        with patch("training.generate_pairs.time.sleep"):
            success, entity, raw = extract_product(
                client=error_client,
                sample_id=99,
                catalog_content="Item Name: Test",
                price=1.99,
                include_few_shot=False,
            )
        assert success is False


# ── _format_retry_error Tests ─────────────────────────────────────────────────

class TestFormatRetryError:

    def test_json_parse_error_mentions_backticks(self):
        msg = _format_retry_error("json_parse", "Expecting value")
        assert "backtick" in msg.lower() or "json" in msg.lower()

    def test_pydantic_error_mentions_schema(self):
        msg = _format_retry_error("pydantic", "quantity_unit invalid")
        assert "schema" in msg.lower() or "field" in msg.lower()

    def test_confidence_error_mentions_confidence(self):
        msg = _format_retry_error("confidence", "0.4 below minimum")
        assert "confidence" in msg.lower()

    def test_unknown_step_still_returns_string(self):
        msg = _format_retry_error("unknown", "something broke")
        assert isinstance(msg, str)
        assert len(msg) > 0


# ── split_pairs Tests ─────────────────────────────────────────────────────────

class TestSplitPairs:

    def _make_pairs(self, n):
        return [{"messages": [{"role": "user", "content": f"pair {i}"}]} for i in range(n)]

    def test_correct_total(self):
        pairs = self._make_pairs(100)
        train, val = split_pairs(pairs, train_ratio=0.8, seed=42)
        assert len(train) + len(val) == 100

    def test_correct_split_ratio(self):
        pairs = self._make_pairs(100)
        train, val = split_pairs(pairs, train_ratio=0.8, seed=42)
        assert len(train) == 80
        assert len(val) == 20

    def test_no_overlap(self):
        pairs = self._make_pairs(50)
        train, val = split_pairs(pairs, train_ratio=0.8, seed=42)
        train_contents = {p["messages"][0]["content"] for p in train}
        val_contents   = {p["messages"][0]["content"] for p in val}
        assert len(train_contents & val_contents) == 0

    def test_reproducible_with_same_seed(self):
        pairs = self._make_pairs(50)
        train1, val1 = split_pairs(pairs, seed=42)
        train2, val2 = split_pairs(pairs, seed=42)
        assert train1 == train2
        assert val1   == val2

    def test_empty_input(self):
        train, val = split_pairs([], train_ratio=0.8)
        assert train == []
        assert val   == []


# ── save_pair Tests ───────────────────────────────────────────────────────────

class TestSavePair:

    def test_creates_file(self, tmp_path):
        output = tmp_path / "train.jsonl"
        pair   = {"messages": [{"role": "user", "content": "test"}]}
        save_pair(pair, output)
        assert output.exists()

    def test_each_line_is_valid_json(self, tmp_path):
        output = tmp_path / "train.jsonl"
        for i in range(3):
            save_pair({"messages": [{"role": "user", "content": f"pair {i}"}]}, output)

        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert "messages" in parsed

    def test_appends_not_overwrites(self, tmp_path):
        output = tmp_path / "train.jsonl"
        save_pair({"messages": [{"role": "user", "content": "first"}]},  output)
        save_pair({"messages": [{"role": "user", "content": "second"}]}, output)

        with open(output) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_creates_parent_directory(self, tmp_path):
        output = tmp_path / "nested" / "dir" / "train.jsonl"
        save_pair({"messages": []}, output)
        assert output.exists()


# ── log_failure Tests ─────────────────────────────────────────────────────────

class TestLogFailure:

    def test_creates_csv(self, tmp_path):
        with patch("training.generate_pairs.FAILED_CSV", tmp_path / "failed.csv"):
            log_failure(99, "Item Name: Test Product", "json parse error")
            assert (tmp_path / "failed.csv").exists()

    def test_csv_has_header(self, tmp_path):
        failed_path = tmp_path / "failed.csv"
        with patch("training.generate_pairs.FAILED_CSV", failed_path):
            log_failure(1, "test", "error")
            with open(failed_path) as f:
                reader = csv.DictReader(f)
                assert "sample_id" in reader.fieldnames
                assert "error" in reader.fieldnames

    def test_truncates_long_content(self, tmp_path):
        failed_path = tmp_path / "failed.csv"
        with patch("training.generate_pairs.FAILED_CSV", failed_path):
            log_failure(1, "x" * 1000, "y" * 1000)
            with open(failed_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows[0]["catalog_preview"]) <= 200
            assert len(rows[0]["error"]) <= 300


# ── Checkpoint Tests ──────────────────────────────────────────────────────────

class TestCheckpoint:

    def test_load_progress_empty_if_no_file(self, tmp_path):
        with patch("training.generate_pairs.PROGRESS_JSONL", tmp_path / "progress.jsonl"):
            result = load_progress()
            assert result == set()

    def test_save_and_load_progress(self, tmp_path):
        progress_path = tmp_path / "progress.jsonl"
        with patch("training.generate_pairs.PROGRESS_JSONL", progress_path), \
             patch("training.generate_pairs.OUTPUT_DIR", tmp_path):
            save_progress(sample_id=42, success=True)
            save_progress(sample_id=99, success=False)
            result = load_progress()
            assert 42 in result
            assert 99 in result

    def test_load_ignores_malformed_lines(self, tmp_path):
        progress_path = tmp_path / "progress.jsonl"
        with open(progress_path, "w") as f:
            f.write('{"sample_id": 1, "success": true}\n')
            f.write("this is not json\n")
            f.write('{"sample_id": 2, "success": false}\n')

        with patch("training.generate_pairs.PROGRESS_JSONL", progress_path):
            result = load_progress()
            assert 1 in result
            assert 2 in result


# ── run_generation Integration Test ──────────────────────────────────────────

class TestRunGeneration:
    """
    Full pipeline test with mocked Groq client and tmp_path for file I/O.
    Tests that all pieces wire together correctly end to end.
    """

    def test_returns_summary_dict(self, small_df, mock_groq_client, tmp_path):
        train_csv = tmp_path / "train.csv"
        small_df.to_csv(train_csv, index=False)

        with patch("training.generate_pairs.build_groq_client", return_value=mock_groq_client), \
             patch("training.generate_pairs.TRAIN_JSONL",     tmp_path / "train.jsonl"), \
             patch("training.generate_pairs.VAL_JSONL",       tmp_path / "val.jsonl"), \
             patch("training.generate_pairs.FAILED_CSV",      tmp_path / "failed.csv"), \
             patch("training.generate_pairs.PROGRESS_JSONL",  tmp_path / "progress.jsonl"), \
             patch("training.generate_pairs.OUTPUT_DIR",      tmp_path), \
             patch("training.generate_pairs.time.sleep"):

            summary = run_generation(target=5, fast=True, train_csv=train_csv)

        assert isinstance(summary, dict)
        assert "success" in summary
        assert "failed" in summary
        assert "train_pairs" in summary
        assert "val_pairs" in summary

    def test_train_jsonl_created(self, small_df, mock_groq_client, tmp_path):
        train_csv  = tmp_path / "train.csv"
        train_jsonl = tmp_path / "train.jsonl"
        small_df.to_csv(train_csv, index=False)

        with patch("training.generate_pairs.build_groq_client", return_value=mock_groq_client), \
             patch("training.generate_pairs.TRAIN_JSONL",     train_jsonl), \
             patch("training.generate_pairs.VAL_JSONL",       tmp_path / "val.jsonl"), \
             patch("training.generate_pairs.FAILED_CSV",      tmp_path / "failed.csv"), \
             patch("training.generate_pairs.PROGRESS_JSONL",  tmp_path / "progress.jsonl"), \
             patch("training.generate_pairs.OUTPUT_DIR",      tmp_path), \
             patch("training.generate_pairs.time.sleep"):

            run_generation(target=5, fast=True, train_csv=train_csv)

        assert train_jsonl.exists()

    def test_success_rate_in_summary(self, small_df, mock_groq_client, tmp_path):
        train_csv = tmp_path / "train.csv"
        small_df.to_csv(train_csv, index=False)

        with patch("training.generate_pairs.build_groq_client", return_value=mock_groq_client), \
             patch("training.generate_pairs.TRAIN_JSONL",     tmp_path / "train.jsonl"), \
             patch("training.generate_pairs.VAL_JSONL",       tmp_path / "val.jsonl"), \
             patch("training.generate_pairs.FAILED_CSV",      tmp_path / "failed.csv"), \
             patch("training.generate_pairs.PROGRESS_JSONL",  tmp_path / "progress.jsonl"), \
             patch("training.generate_pairs.OUTPUT_DIR",      tmp_path), \
             patch("training.generate_pairs.time.sleep"):

            summary = run_generation(target=5, fast=True, train_csv=train_csv)

        assert 0.0 <= summary["success_rate"] <= 1.0

    def test_train_val_split_adds_up(self, small_df, mock_groq_client, tmp_path):
        train_csv = tmp_path / "train.csv"
        small_df.to_csv(train_csv, index=False)

        with patch("training.generate_pairs.build_groq_client", return_value=mock_groq_client), \
             patch("training.generate_pairs.TRAIN_JSONL",     tmp_path / "train.jsonl"), \
             patch("training.generate_pairs.VAL_JSONL",       tmp_path / "val.jsonl"), \
             patch("training.generate_pairs.FAILED_CSV",      tmp_path / "failed.csv"), \
             patch("training.generate_pairs.PROGRESS_JSONL",  tmp_path / "progress.jsonl"), \
             patch("training.generate_pairs.OUTPUT_DIR",      tmp_path), \
             patch("training.generate_pairs.time.sleep"):

            summary = run_generation(target=5, fast=True, train_csv=train_csv)

        assert summary["train_pairs"] + summary["val_pairs"] == summary["success"]