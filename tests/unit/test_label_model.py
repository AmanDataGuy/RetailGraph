# tests/unit/test_label_model.py
"""
Unit tests for label_model.py.

Testing strategy:
    - We never run the full LabelModel.fit() in unit tests — it's slow
      and non-deterministic. Instead we mock it and test the logic
      around it: threshold application, result building, CSV output.
    - apply_*_lfs() functions ARE tested with real data because they
      just call Snorkel's applier — deterministic and fast.
    - We use a small synthetic dataframe (5–10 rows) so tests run
      in milliseconds.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.extraction.weak_supervision.label_model import (
    AUTO_ACCEPT_THRESHOLD,
    AUTO_REJECT_THRESHOLD,
    CATEGORY_NAMES,
    DIETARY_TAGS,
    LFStats,
    WeakLabelResult,
    _get_status,
    apply_category_lfs,
    apply_dietary_lfs,
    analyze_lfs,
    build_results,
    save_labeled_output,
    run_label_pipeline,
)
from src.extraction.weak_supervision.labeling_functions import (
    CATEGORY_LFS,
    DIETARY_TAG_LFS,
    CAT_COFFEE_TEA,
    POSITIVE,
    ABSTAIN,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    """
    5-row synthetic dataframe that mimics train.csv structure.
    Each row is crafted to trigger specific LFs so we can assert
    the applier works correctly without loading real data.
    """
    return pd.DataFrame([
        {
            "sample_id": 1,
            "catalog_content": "Item Name: Starbucks Pike Place Ground Coffee 12oz\nBullet Point 1: Rich and balanced",
            "price": 12.99,
            "image_link": "https://images.amazon.com/coffee.jpg",
        },
        {
            "sample_id": 2,
            "catalog_content": "Item Name: Organic Valley Whole Milk 1 Gallon\nBullet Point 1: USDA Organic certified",
            "price": 7.49,
            "image_link": "https://images.amazon.com/milk.jpg",
        },
        {
            "sample_id": 3,
            "catalog_content": "Item Name: Bob's Red Mill Gluten-Free Oats 32oz\nBullet Point 1: Certified Gluten-Free",
            "price": 5.99,
            "image_link": "https://images.amazon.com/oats.jpg",
        },
        {
            "sample_id": 4,
            "catalog_content": "Item Name: Rani Garam Masala Indian Spice Blend 3oz",
            "price": 8.99,
            "image_link": "https://images.amazon.com/masala.jpg",
        },
        {
            "sample_id": 5,
            "catalog_content": "Item Name: Quest Nutrition Protein Bar Chocolate Chip",
            "price": 24.99,
            "image_link": "https://images.amazon.com/bar.jpg",
        },
    ])


@pytest.fixture
def mock_category_probs():
    """
    Fake category probability matrix — 5 rows, 11 categories.
    Row 0 is confident Coffee & Tea (CAT_COFFEE_TEA = 2).
    Row 1 is uncertain (review zone).
    Rows 2–4 are below threshold (auto_reject).
    """
    probs = np.full((5, 11), 0.03)  # low baseline everywhere
    # Row 0 — confident Coffee & Tea
    probs[0, CAT_COFFEE_TEA] = 0.90
    # Row 1 — uncertain, review zone
    probs[1, 0] = 0.65
    # Rows 2–4 stay in auto_reject territory
    return probs


@pytest.fixture
def mock_dietary_models_and_probs():
    """
    Fake dietary model outputs for 2 tags (organic, gluten-free).
    Row 1 (Organic Valley) gets high organic probability.
    Row 2 (Bob's Red Mill) gets high gluten-free probability.
    """
    # Shape (5, 2) — [prob_negative, prob_positive] per row
    organic_probs = np.array([
        [0.95, 0.05],  # row 0 — not organic
        [0.05, 0.95],  # row 1 — organic (Organic Valley)
        [0.80, 0.20],  # row 2 — review
        [0.92, 0.08],  # row 3 — not organic
        [0.88, 0.12],  # row 4 — not organic
    ])
    gluten_free_probs = np.array([
        [0.90, 0.10],
        [0.85, 0.15],
        [0.05, 0.95],  # row 2 — gluten-free (Bob's Red Mill)
        [0.92, 0.08],
        [0.88, 0.12],
    ])
    # We don't need real LabelModel objects for result-building tests
    mock_model = MagicMock()
    return {
        "organic":     (mock_model, organic_probs),
        "gluten-free": (mock_model, gluten_free_probs),
    }


# ── _get_status Tests ─────────────────────────────────────────────────────────

class TestGetStatus:
    """
    _get_status is the core threshold logic.
    Test every boundary — at threshold, above, below.
    """

    def test_above_accept_threshold(self):
        assert _get_status(0.90) == "auto_accept"

    def test_exactly_at_accept_threshold(self):
        # Boundary condition — exactly 0.85 should auto_accept
        assert _get_status(AUTO_ACCEPT_THRESHOLD) == "auto_accept"

    def test_in_review_zone(self):
        assert _get_status(0.70) == "review"

    def test_exactly_at_reject_threshold(self):
        # Exactly 0.50 — boundary, should be review not reject
        assert _get_status(AUTO_REJECT_THRESHOLD) == "review"

    def test_below_reject_threshold(self):
        assert _get_status(0.30) == "auto_reject"

    def test_zero_probability(self):
        assert _get_status(0.0) == "auto_reject"

    def test_one_probability(self):
        assert _get_status(1.0) == "auto_accept"

    def test_just_below_accept(self):
        # 0.849 — should be review, not auto_accept
        assert _get_status(0.849) == "review"

    def test_just_above_reject(self):
        # 0.501 — should be review, not auto_reject
        assert _get_status(0.501) == "review"


# ── apply_category_lfs Tests ──────────────────────────────────────────────────

class TestApplyCategoryLfs:
    """
    Tests that the LF applier produces a correctly shaped matrix.
    We don't test individual LF votes here — that's done in
    test_labeling_functions.py. We only test the matrix structure.
    """

    def test_output_shape(self, small_df):
        L = apply_category_lfs(small_df)
        # Rows = number of products, cols = number of category LFs
        assert L.shape == (len(small_df), len(CATEGORY_LFS))

    def test_output_is_numpy_array(self, small_df):
        L = apply_category_lfs(small_df)
        assert isinstance(L, np.ndarray)

    def test_values_are_valid_labels(self, small_df):
        # Every cell must be either -1 (ABSTAIN) or a valid category int
        L = apply_category_lfs(small_df)
        valid_values = set([-1] + list(CATEGORY_NAMES.keys()))
        unique_values = set(L.flatten().tolist())
        assert unique_values.issubset(valid_values)

    def test_coffee_row_gets_coffee_vote(self, small_df):
        # Row 0 is a Starbucks coffee product — at least one LF should
        # vote CAT_COFFEE_TEA for it
        L = apply_category_lfs(small_df)
        row_0_votes = L[0]
        assert CAT_COFFEE_TEA in row_0_votes


# ── apply_dietary_lfs Tests ───────────────────────────────────────────────────

class TestApplyDietaryLfs:

    def test_output_shape(self, small_df):
        L = apply_dietary_lfs(small_df)
        assert L.shape == (len(small_df), len(DIETARY_TAG_LFS))

    def test_output_is_numpy_array(self, small_df):
        L = apply_dietary_lfs(small_df)
        assert isinstance(L, np.ndarray)

    def test_values_are_valid_labels(self, small_df):
        L = apply_dietary_lfs(small_df)
        valid = {-1, 0, 1}
        assert set(L.flatten().tolist()).issubset(valid)

    def test_organic_row_gets_positive_vote(self, small_df):
        # Row 1 is "Organic Valley" — organic LF should vote POSITIVE (1)
        L = apply_dietary_lfs(small_df)
        row_1_votes = L[1]
        assert POSITIVE in row_1_votes

    def test_gluten_free_row_gets_positive_vote(self, small_df):
        # Row 2 is Bob's Red Mill Gluten-Free — should get POSITIVE vote
        L = apply_dietary_lfs(small_df)
        row_2_votes = L[2]
        assert POSITIVE in row_2_votes


# ── analyze_lfs Tests ─────────────────────────────────────────────────────────

class TestAnalyzeLfs:

    def test_returns_lf_stats(self, small_df):
        L = apply_category_lfs(small_df)
        stats = analyze_lfs(L, CATEGORY_LFS, label="category")
        assert isinstance(stats, LFStats)

    def test_n_rows_correct(self, small_df):
        L = apply_category_lfs(small_df)
        stats = analyze_lfs(L, CATEGORY_LFS, label="category")
        assert stats.n_rows == len(small_df)

    def test_n_lfs_correct(self, small_df):
        L = apply_category_lfs(small_df)
        stats = analyze_lfs(L, CATEGORY_LFS, label="category")
        assert stats.n_lfs == len(CATEGORY_LFS)

    def test_coverage_keys_match_lf_names(self, small_df):
        L = apply_category_lfs(small_df)
        stats = analyze_lfs(L, CATEGORY_LFS, label="category")
        lf_names = [lf.name for lf in CATEGORY_LFS]
        assert set(stats.coverage.keys()) == set(lf_names)

    def test_coverage_values_between_0_and_1(self, small_df):
        L = apply_category_lfs(small_df)
        stats = analyze_lfs(L, CATEGORY_LFS, label="category")
        for v in stats.coverage.values():
            assert 0.0 <= v <= 1.0


# ── build_results Tests ───────────────────────────────────────────────────────

class TestBuildResults:

    def test_returns_one_result_per_row(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        assert len(results) == len(small_df)

    def test_result_type_is_weak_label_result(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        assert all(isinstance(r, WeakLabelResult) for r in results)

    def test_sample_ids_match_dataframe(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        expected_ids = small_df["sample_id"].tolist()
        result_ids   = [r.sample_id for r in results]
        assert result_ids == expected_ids

    def test_confident_row_gets_auto_accept(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        # Row 0 has category_prob=0.90 — should be auto_accept
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        assert results[0].category_status == "auto_accept"

    def test_uncertain_row_gets_review(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        # Row 1 has category_prob=0.65 — should be review
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        assert results[1].category_status == "review"

    def test_low_confidence_row_gets_auto_reject(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        # Rows 2–4 have max prob ~0.35 — should be auto_reject
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        assert results[2].category_status == "auto_reject"

    def test_organic_tag_accepted_for_organic_row(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        # Row 1 (Organic Valley) has organic prob=0.95 — should be in dietary_tags
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        assert "organic" in results[1].dietary_tags

    def test_tag_probs_all_present(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        # Every tag in the models dict should appear in tag_probs
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        for tag in mock_dietary_models_and_probs.keys():
            assert tag in results[0].tag_probs

    def test_category_name_resolved_correctly(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        # Row 0 argmax is CAT_COFFEE_TEA — category should be "Coffee & Tea"
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        assert results[0].category == "Coffee & Tea"

    def test_category_prob_rounded_to_4_decimals(
        self, small_df, mock_category_probs, mock_dietary_models_and_probs
    ):
        results = build_results(
            small_df, mock_category_probs, mock_dietary_models_and_probs
        )
        for r in results:
            assert r.category_prob == round(r.category_prob, 4)


# ── save_labeled_output Tests ─────────────────────────────────────────────────

class TestSaveLabeledOutput:

    def _make_results(self):
        return [
            WeakLabelResult(
                sample_id=1,
                category="Coffee & Tea",
                category_prob=0.92,
                category_status="auto_accept",
                dietary_tags=["organic"],
                tag_probs={"organic": 0.92, "vegan": 0.30},
                tag_statuses={"organic": "auto_accept", "vegan": "auto_reject"},
            ),
            WeakLabelResult(
                sample_id=2,
                category="Beverages",
                category_prob=0.60,
                category_status="review",
                dietary_tags=[],
                tag_probs={"organic": 0.55},
                tag_statuses={"organic": "review"},
            ),
        ]

    def test_csv_created(self, tmp_path):
        results = self._make_results()
        with patch(
            "src.extraction.weak_supervision.label_model.OUTPUT_PATH",
            tmp_path / "weak_labels.csv"
        ):
            save_labeled_output(results)
            assert (tmp_path / "weak_labels.csv").exists()

    def test_csv_has_correct_row_count(self, tmp_path):
        results = self._make_results()
        with patch(
            "src.extraction.weak_supervision.label_model.OUTPUT_PATH",
            tmp_path / "weak_labels.csv"
        ):
            save_labeled_output(results)
            df = pd.read_csv(tmp_path / "weak_labels.csv")
            assert len(df) == 2

    def test_csv_has_required_columns(self, tmp_path):
        results = self._make_results()
        with patch(
            "src.extraction.weak_supervision.label_model.OUTPUT_PATH",
            tmp_path / "weak_labels.csv"
        ):
            save_labeled_output(results)
            df = pd.read_csv(tmp_path / "weak_labels.csv")
            required = {
                "sample_id", "category", "category_prob",
                "category_status", "dietary_tags", "tag_probs", "tag_statuses"
            }
            assert required.issubset(set(df.columns))

    def test_dietary_tags_pipe_separated(self, tmp_path):
        results = self._make_results()
        with patch(
            "src.extraction.weak_supervision.label_model.OUTPUT_PATH",
            tmp_path / "weak_labels.csv"
        ):
            save_labeled_output(results)
            df = pd.read_csv(tmp_path / "weak_labels.csv")
            # Row 0 has dietary_tags=["organic"] — should be "organic" in CSV
            assert df.loc[0, "dietary_tags"] == "organic"

    def test_tag_probs_valid_json(self, tmp_path):
        results = self._make_results()
        with patch(
            "src.extraction.weak_supervision.label_model.OUTPUT_PATH",
            tmp_path / "weak_labels.csv"
        ):
            save_labeled_output(results)
            df = pd.read_csv(tmp_path / "weak_labels.csv")
            # tag_probs column should be parseable JSON
            parsed = json.loads(df.loc[0, "tag_probs"])
            assert isinstance(parsed, dict)

    def test_empty_dietary_tags_saved_as_empty_string(self, tmp_path):
        results = self._make_results()
        with patch(
            "src.extraction.weak_supervision.label_model.OUTPUT_PATH",
            tmp_path / "weak_labels.csv"
        ):
            save_labeled_output(results)
            df = pd.read_csv(tmp_path / "weak_labels.csv")
            # pandas reads empty strings from CSV as NaN — both are acceptable
            val = df.loc[1, "dietary_tags"]
            assert val == "" or pd.isna(val)


# ── run_label_pipeline integration test ──────────────────────────────────────

class TestRunLabelPipeline:
    """
    Light integration test — we mock the slow LabelModel.fit() calls
    but test that the full pipeline wires together correctly and
    returns the right number of results.
    """

    def test_returns_one_result_per_row(self, small_df, tmp_path):
        # Mock LabelModel so we don't actually train during tests.
        # We only care that the pipeline runs end to end without crashing.
        mock_probs_cat = np.full((len(small_df), 11), 1/11)
        mock_probs_diet = np.full((len(small_df), 2), 0.5)

        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = [
            mock_probs_cat,   # called once for category model
        ] + [mock_probs_diet] * len(DIETARY_TAGS)  # called once per tag

        with patch(
            "src.extraction.weak_supervision.label_model.LabelModel",
            return_value=mock_model
        ), patch(
            "src.extraction.weak_supervision.label_model.OUTPUT_PATH",
            tmp_path / "weak_labels.csv"
        ):
            results = run_label_pipeline(small_df)
            assert len(results) == len(small_df)

    def test_output_csv_created(self, small_df, tmp_path):
        mock_probs_cat  = np.full((len(small_df), 11), 1/11)
        mock_probs_diet = np.full((len(small_df), 2), 0.5)

        mock_model = MagicMock()
        mock_model.predict_proba.side_effect = (
            [mock_probs_cat] + [mock_probs_diet] * len(DIETARY_TAGS)
        )

        with patch(
            "src.extraction.weak_supervision.label_model.LabelModel",
            return_value=mock_model
        ), patch(
            "src.extraction.weak_supervision.label_model.OUTPUT_PATH",
            tmp_path / "weak_labels.csv"
        ):
            run_label_pipeline(small_df)
            assert (tmp_path / "weak_labels.csv").exists()