# tests/unit/test_generate_synthetic.py
"""
Unit tests for training/generate_synthetic.py.

Testing strategy:
    - All tests are pure Python — no LLM calls, no file I/O unless using tmp_path.
    - We test template correctness, quota logic, pair structure,
      JSON validity, and the full pipeline end to end.
"""

import csv
import json
import random
import pytest
from pathlib import Path
from unittest.mock import patch

from training.generate_synthetic import (
    UNDERREPRESENTED_CATEGORIES,
    CATEGORY_TEMPLATES,
    MAX_SYNTHETIC_RATIO,
    TRAIN_RATIO,
    compute_synthetic_quota,
    generate_synthetic_pairs,
    save_synthetic_pairs,
    run_synthetic_generation,
    _generate_catalog_content,
    _generate_bullets,
)
from src.extraction.prompt_templates import ALLOWED_CATEGORIES, ALLOWED_DIETARY_TAGS, CANONICAL_UNITS


# ── CATEGORY_TEMPLATES Tests ──────────────────────────────────────────────────

class TestCategoryTemplates:

    def test_all_underrepresented_categories_have_templates(self):
        for cat in UNDERREPRESENTED_CATEGORIES:
            assert cat in CATEGORY_TEMPLATES, f"No template for '{cat}'"

    def test_each_template_has_required_keys(self):
        required = ["brands", "products", "flavors", "sizes", "tags",
                    "allergens", "price_range", "confidence_range"]
        for cat, tmpl in CATEGORY_TEMPLATES.items():
            for key in required:
                assert key in tmpl, f"Template '{cat}' missing key '{key}'"

    def test_all_template_categories_are_allowed(self):
        for cat in CATEGORY_TEMPLATES:
            assert cat in ALLOWED_CATEGORIES, f"'{cat}' not in ALLOWED_CATEGORIES"

    def test_all_template_tags_are_allowed(self):
        for cat, tmpl in CATEGORY_TEMPLATES.items():
            for tag in tmpl["tags"]:
                assert tag in ALLOWED_DIETARY_TAGS, (
                    f"Template '{cat}' has invalid tag '{tag}'"
                )

    def test_all_template_units_are_canonical(self):
        for cat, tmpl in CATEGORY_TEMPLATES.items():
            for qty_value, qty_unit in tmpl["sizes"]:
                assert qty_unit in CANONICAL_UNITS, (
                    f"Template '{cat}' has non-canonical unit '{qty_unit}'"
                )

    def test_price_range_is_valid(self):
        for cat, tmpl in CATEGORY_TEMPLATES.items():
            lo, hi = tmpl["price_range"]
            assert lo > 0
            assert hi > lo

    def test_confidence_range_is_valid(self):
        for cat, tmpl in CATEGORY_TEMPLATES.items():
            lo, hi = tmpl["confidence_range"]
            assert lo >= 0.4
            assert hi < 1.0
            assert hi > lo


# ── _generate_catalog_content Tests ──────────────────────────────────────────

class TestGenerateCatalogContent:

    @pytest.mark.parametrize("category", UNDERREPRESENTED_CATEGORIES)
    def test_returns_tuple_for_each_category(self, category):
        rng = random.Random(42)
        catalog, entity = _generate_catalog_content(category, rng)
        assert isinstance(catalog, str)
        assert isinstance(entity, dict)

    @pytest.mark.parametrize("category", UNDERREPRESENTED_CATEGORIES)
    def test_catalog_contains_item_name(self, category):
        rng = random.Random(42)
        catalog, entity = _generate_catalog_content(category, rng)
        assert "Item Name:" in catalog

    @pytest.mark.parametrize("category", UNDERREPRESENTED_CATEGORIES)
    def test_catalog_contains_value_and_unit(self, category):
        rng = random.Random(42)
        catalog, entity = _generate_catalog_content(category, rng)
        assert "Value:" in catalog
        assert "Unit:" in catalog

    @pytest.mark.parametrize("category", UNDERREPRESENTED_CATEGORIES)
    def test_entity_has_required_fields(self, category):
        rng = random.Random(42)
        _, entity = _generate_catalog_content(category, rng)
        required = [
            "item_name", "price", "quantity_value", "quantity_unit",
            "category", "dietary_tags", "allergen_list", "extraction_confidence"
        ]
        for field in required:
            assert field in entity

    @pytest.mark.parametrize("category", UNDERREPRESENTED_CATEGORIES)
    def test_entity_category_matches_input(self, category):
        rng = random.Random(42)
        _, entity = _generate_catalog_content(category, rng)
        assert entity["category"] == category

    @pytest.mark.parametrize("category", UNDERREPRESENTED_CATEGORIES)
    def test_entity_unit_is_canonical(self, category):
        rng = random.Random(42)
        _, entity = _generate_catalog_content(category, rng)
        assert entity["quantity_unit"] in CANONICAL_UNITS

    @pytest.mark.parametrize("category", UNDERREPRESENTED_CATEGORIES)
    def test_entity_tags_are_allowed(self, category):
        rng = random.Random(42)
        _, entity = _generate_catalog_content(category, rng)
        for tag in entity["dietary_tags"]:
            assert tag in ALLOWED_DIETARY_TAGS

    @pytest.mark.parametrize("category", UNDERREPRESENTED_CATEGORIES)
    def test_confidence_never_1_0(self, category):
        rng = random.Random(42)
        for _ in range(10):
            _, entity = _generate_catalog_content(category, rng)
            assert entity["extraction_confidence"] < 1.0

    @pytest.mark.parametrize("category", UNDERREPRESENTED_CATEGORIES)
    def test_price_positive(self, category):
        rng = random.Random(42)
        _, entity = _generate_catalog_content(category, rng)
        assert entity["price"] > 0

    def test_different_seeds_give_different_products(self):
        rng1 = random.Random(1)
        rng2 = random.Random(2)
        cat = UNDERREPRESENTED_CATEGORIES[0]
        _, entity1 = _generate_catalog_content(cat, rng1)
        _, entity2 = _generate_catalog_content(cat, rng2)
        # At minimum the item names or prices should differ
        assert entity1["item_name"] != entity2["item_name"] or entity1["price"] != entity2["price"]


# ── _generate_bullets Tests ───────────────────────────────────────────────────

class TestGenerateBullets:

    def test_returns_list(self):
        rng = random.Random(42)
        result = _generate_bullets("Protein Bars & Snacks", ["gluten-free"], [], rng)
        assert isinstance(result, list)

    def test_at_most_four_bullets(self):
        rng = random.Random(42)
        result = _generate_bullets(
            "Protein Bars & Snacks",
            ["gluten-free", "vegan", "kosher", "non-GMO", "high-protein"],
            ["milk", "eggs"],
            rng
        )
        assert len(result) <= 4

    def test_allergen_bullet_included_when_present(self):
        rng = random.Random(42)
        result = _generate_bullets("Supplements & Health", [], ["milk", "soy"], rng)
        allergen_bullets = [b for b in result if "Contains:" in b]
        assert len(allergen_bullets) == 1

    def test_no_allergen_bullet_when_empty(self):
        rng = random.Random(42)
        result = _generate_bullets("Protein Bars & Snacks", ["gluten-free"], [], rng)
        allergen_bullets = [b for b in result if "Contains:" in b]
        assert len(allergen_bullets) == 0

    def test_all_bullets_are_strings(self):
        rng = random.Random(42)
        result = _generate_bullets("Supplements & Health", ["organic"], ["soy"], rng)
        assert all(isinstance(b, str) for b in result)


# ── compute_synthetic_quota Tests ─────────────────────────────────────────────

class TestComputeSyntheticQuota:

    def test_returns_dict(self):
        result = compute_synthetic_quota(real_pairs=3000)
        assert isinstance(result, dict)

    def test_total_does_not_exceed_cap(self):
        result = compute_synthetic_quota(real_pairs=3000, max_ratio=0.20)
        total  = sum(result.values())
        cap    = int(3000 * 0.20)
        assert total <= cap

    def test_all_target_categories_present(self):
        result = compute_synthetic_quota(real_pairs=3000)
        for cat in UNDERREPRESENTED_CATEGORIES:
            assert cat in result

    def test_each_category_gets_at_least_50(self):
        result = compute_synthetic_quota(real_pairs=3000)
        for cat, n in result.items():
            assert n >= 50, f"'{cat}' got only {n} — minimum is 50"

    def test_custom_categories(self):
        custom = ["Protein Bars & Snacks"]
        result = compute_synthetic_quota(real_pairs=1000, target_categories=custom)
        assert "Protein Bars & Snacks" in result
        assert len(result) == 1

    def test_zero_real_pairs_handled(self):
        result = compute_synthetic_quota(real_pairs=0)
        # Should return minimum quotas without crashing
        assert isinstance(result, dict)


# ── generate_synthetic_pairs Tests ───────────────────────────────────────────

class TestGenerateSyntheticPairs:

    def test_returns_list(self):
        quotas = {"Protein Bars & Snacks": 5}
        result = generate_synthetic_pairs(quotas, seed=42)
        assert isinstance(result, list)

    def test_correct_count(self):
        quotas = {"Protein Bars & Snacks": 10, "Supplements & Health": 5}
        result = generate_synthetic_pairs(quotas, seed=42)
        assert len(result) == 15

    def test_each_pair_has_messages_key(self):
        quotas = {"Protein Bars & Snacks": 3}
        result = generate_synthetic_pairs(quotas, seed=42)
        for pair in result:
            assert "messages" in pair

    def test_each_pair_messages_has_three_parts(self):
        quotas = {"Protein Bars & Snacks": 3}
        result = generate_synthetic_pairs(quotas, seed=42)
        for pair in result:
            # system + user + assistant
            assert len(pair["messages"]) == 3

    def test_assistant_message_is_valid_json(self):
        quotas = {"Protein Bars & Snacks": 5}
        result = generate_synthetic_pairs(quotas, seed=42)
        for pair in result:
            assistant = pair["messages"][-1]
            assert assistant["role"] == "assistant"
            entity = json.loads(assistant["content"])
            assert "item_name" in entity

    def test_synthetic_flag_set(self):
        quotas = {"Supplements & Health": 3}
        result = generate_synthetic_pairs(quotas, seed=42)
        for pair in result:
            assert pair.get("synthetic") is True

    def test_category_field_set(self):
        quotas = {"Supplements & Health": 3}
        result = generate_synthetic_pairs(quotas, seed=42)
        for pair in result:
            assert pair.get("category") == "Supplements & Health"

    def test_unknown_category_skipped_gracefully(self):
        quotas = {"NonExistentCategory": 5}
        result = generate_synthetic_pairs(quotas, seed=42)
        assert result == []

    def test_reproducible_with_same_seed(self):
        quotas = {"Protein Bars & Snacks": 5}
        r1 = generate_synthetic_pairs(quotas, seed=42)
        r2 = generate_synthetic_pairs(quotas, seed=42)
        assert [p["messages"][-1]["content"] for p in r1] == \
               [p["messages"][-1]["content"] for p in r2]


# ── save_synthetic_pairs Tests ────────────────────────────────────────────────

class TestSaveSyntheticPairs:

    def _make_pairs(self, n, category="Protein Bars & Snacks"):
        rng = random.Random(42)
        pairs = []
        for _ in range(n):
            catalog, entity = _generate_catalog_content(category, rng)
            pairs.append({
                "messages": [
                    {"role": "system",    "content": "system prompt"},
                    {"role": "user",      "content": catalog},
                    {"role": "assistant", "content": json.dumps(entity)},
                ],
                "synthetic": True,
                "category":  category,
            })
        return pairs

    def test_appends_to_train_jsonl(self, tmp_path):
        pairs = self._make_pairs(10)
        with patch("training.generate_synthetic.TRAIN_JSONL",        tmp_path / "train.jsonl"), \
             patch("training.generate_synthetic.VAL_JSONL",          tmp_path / "val.jsonl"), \
             patch("training.generate_synthetic.SYNTHETIC_MANIFEST", tmp_path / "manifest.csv"), \
             patch("training.generate_synthetic.OUTPUT_DIR",         tmp_path):
            n_train, n_val = save_synthetic_pairs(pairs, train_ratio=0.8, seed=42)

        assert (tmp_path / "train.jsonl").exists()
        assert n_train == 8

    def test_appends_to_val_jsonl(self, tmp_path):
        pairs = self._make_pairs(10)
        with patch("training.generate_synthetic.TRAIN_JSONL",        tmp_path / "train.jsonl"), \
             patch("training.generate_synthetic.VAL_JSONL",          tmp_path / "val.jsonl"), \
             patch("training.generate_synthetic.SYNTHETIC_MANIFEST", tmp_path / "manifest.csv"), \
             patch("training.generate_synthetic.OUTPUT_DIR",         tmp_path):
            n_train, n_val = save_synthetic_pairs(pairs, train_ratio=0.8, seed=42)

        assert (tmp_path / "val.jsonl").exists()
        assert n_val == 2

    def test_train_val_sum_equals_total(self, tmp_path):
        pairs = self._make_pairs(20)
        with patch("training.generate_synthetic.TRAIN_JSONL",        tmp_path / "train.jsonl"), \
             patch("training.generate_synthetic.VAL_JSONL",          tmp_path / "val.jsonl"), \
             patch("training.generate_synthetic.SYNTHETIC_MANIFEST", tmp_path / "manifest.csv"), \
             patch("training.generate_synthetic.OUTPUT_DIR",         tmp_path):
            n_train, n_val = save_synthetic_pairs(pairs, seed=42)

        assert n_train + n_val == 20

    def test_manifest_created(self, tmp_path):
        pairs = self._make_pairs(5)
        manifest = tmp_path / "manifest.csv"
        with patch("training.generate_synthetic.TRAIN_JSONL",        tmp_path / "train.jsonl"), \
             patch("training.generate_synthetic.VAL_JSONL",          tmp_path / "val.jsonl"), \
             patch("training.generate_synthetic.SYNTHETIC_MANIFEST", manifest), \
             patch("training.generate_synthetic.OUTPUT_DIR",         tmp_path):
            save_synthetic_pairs(pairs, seed=42)

        assert manifest.exists()

    def test_manifest_has_correct_columns(self, tmp_path):
        pairs = self._make_pairs(5)
        manifest = tmp_path / "manifest.csv"
        with patch("training.generate_synthetic.TRAIN_JSONL",        tmp_path / "train.jsonl"), \
             patch("training.generate_synthetic.VAL_JSONL",          tmp_path / "val.jsonl"), \
             patch("training.generate_synthetic.SYNTHETIC_MANIFEST", manifest), \
             patch("training.generate_synthetic.OUTPUT_DIR",         tmp_path):
            save_synthetic_pairs(pairs, seed=42)

        with open(manifest) as f:
            reader = csv.DictReader(f)
            assert "category" in reader.fieldnames
            assert "split" in reader.fieldnames
            assert "item_name" in reader.fieldnames

    def test_saved_pairs_strip_synthetic_flag(self, tmp_path):
        pairs = self._make_pairs(5)
        with patch("training.generate_synthetic.TRAIN_JSONL",        tmp_path / "train.jsonl"), \
             patch("training.generate_synthetic.VAL_JSONL",          tmp_path / "val.jsonl"), \
             patch("training.generate_synthetic.SYNTHETIC_MANIFEST", tmp_path / "manifest.csv"), \
             patch("training.generate_synthetic.OUTPUT_DIR",         tmp_path):
            save_synthetic_pairs(pairs, seed=42)

        with open(tmp_path / "train.jsonl") as f:
            for line in f:
                saved = json.loads(line)
                assert "synthetic" not in saved
                assert "category" not in saved


# ── run_synthetic_generation Integration Test ─────────────────────────────────

class TestRunSyntheticGeneration:

    def test_returns_summary_dict(self, tmp_path):
        with patch("training.generate_synthetic.TRAIN_JSONL",        tmp_path / "train.jsonl"), \
             patch("training.generate_synthetic.VAL_JSONL",          tmp_path / "val.jsonl"), \
             patch("training.generate_synthetic.SYNTHETIC_MANIFEST", tmp_path / "manifest.csv"), \
             patch("training.generate_synthetic.OUTPUT_DIR",         tmp_path):
            summary = run_synthetic_generation(
                real_pairs=100,
                target_categories=["Protein Bars & Snacks"],
                seed=42,
            )

        assert isinstance(summary, dict)
        assert "total_generated" in summary
        assert "train_pairs" in summary
        assert "val_pairs" in summary

    def test_total_does_not_exceed_cap(self, tmp_path):
        real_pairs = 100
        with patch("training.generate_synthetic.TRAIN_JSONL",        tmp_path / "train.jsonl"), \
             patch("training.generate_synthetic.VAL_JSONL",          tmp_path / "val.jsonl"), \
             patch("training.generate_synthetic.SYNTHETIC_MANIFEST", tmp_path / "manifest.csv"), \
             patch("training.generate_synthetic.OUTPUT_DIR",         tmp_path):
            summary = run_synthetic_generation(
                real_pairs=real_pairs,
                target_categories=["Protein Bars & Snacks"],
                seed=42,
            )

        cap = int(real_pairs * MAX_SYNTHETIC_RATIO)
        assert summary["total_generated"] <= cap

    def test_train_val_split_adds_up(self, tmp_path):
        with patch("training.generate_synthetic.TRAIN_JSONL",        tmp_path / "train.jsonl"), \
             patch("training.generate_synthetic.VAL_JSONL",          tmp_path / "val.jsonl"), \
             patch("training.generate_synthetic.SYNTHETIC_MANIFEST", tmp_path / "manifest.csv"), \
             patch("training.generate_synthetic.OUTPUT_DIR",         tmp_path):
            summary = run_synthetic_generation(
                real_pairs=100,
                target_categories=["Supplements & Health"],
                seed=42,
            )

        assert summary["train_pairs"] + summary["val_pairs"] == summary["total_generated"]