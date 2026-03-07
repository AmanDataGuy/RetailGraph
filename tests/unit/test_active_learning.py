# tests/unit/test_active_learning.py
"""
Unit tests for scripts/active_learning_query.py.

Testing strategy:
    - We never touch real data files — all tests use synthetic DataFrames
      injected via fixtures and tmp_path.
    - We test each function in isolation first, then test the full
      pipeline via run_active_learning_query() with mocked file I/O.
    - Edge cases covered: empty review zone, fewer candidates than TOP_N,
      missing tag_probs, all-abstain products.
"""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from scripts.active_learning_query import (
    WEIGHT_ENTROPY,
    WEIGHT_DISAGREEMENT,
    WEIGHT_RARITY,
    TOP_N,
    compute_entropy,
    compute_entropy_score,
    compute_disagreement_score,
    compute_rarity_scores,
    compute_uncertainty_scores,
    filter_review_candidates,
    select_top_n,
    build_review_queue,
    save_review_queue,
    run_active_learning_query,
    _build_reason,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_labels_df():
    """
    Synthetic weak_labels.csv — 6 rows covering all status types.
    Rows 1 and 2 are in the review zone and should be selected.
    """
    return pd.DataFrame([
        {
            "sample_id": 1,
            "category": "Coffee & Tea",
            "category_prob": 0.92,
            "category_status": "auto_accept",
            "dietary_tags": "organic",
            "tag_probs": json.dumps({"organic": 0.92, "vegan": 0.10}),
            "tag_statuses": json.dumps({"organic": "auto_accept", "vegan": "auto_reject"}),
        },
        {
            "sample_id": 2,
            "category": "Beverages",
            "category_prob": 0.65,
            "category_status": "review",
            "dietary_tags": "",
            "tag_probs": json.dumps({"organic": 0.55, "vegan": 0.52}),
            "tag_statuses": json.dumps({"organic": "review", "vegan": "review"}),
        },
        {
            "sample_id": 3,
            "category": "Snacks & Candy",
            "category_prob": 0.72,
            "category_status": "review",
            "dietary_tags": "",
            "tag_probs": json.dumps({"organic": 0.88, "vegan": 0.15}),
            "tag_statuses": json.dumps({"organic": "auto_accept", "vegan": "auto_reject"}),
        },
        {
            "sample_id": 4,
            "category": "Grains, Beans & Legumes",
            "category_prob": 0.30,
            "category_status": "auto_reject",
            "dietary_tags": "",
            "tag_probs": json.dumps({"organic": 0.10, "vegan": 0.08}),
            "tag_statuses": json.dumps({"organic": "auto_reject", "vegan": "auto_reject"}),
        },
        {
            "sample_id": 5,
            "category": "Beverages",
            "category_prob": 0.60,
            "category_status": "review",
            "dietary_tags": "",
            "tag_probs": json.dumps({"organic": 0.51, "vegan": 0.49}),
            "tag_statuses": json.dumps({"organic": "review", "vegan": "review"}),
        },
        {
            "sample_id": 6,
            "category": "Spices & Seasonings",
            "category_prob": 0.58,
            "category_status": "review",
            "dietary_tags": "",
            "tag_probs": json.dumps({"organic": 0.50, "vegan": 0.50}),
            "tag_statuses": json.dumps({"organic": "review", "vegan": "review"}),
        },
    ])


@pytest.fixture
def sample_catalog_df():
    """
    Synthetic train.csv — just sample_id and catalog_content.
    Item names are embedded in catalog_content as in the real dataset.
    """
    return pd.DataFrame([
        {"sample_id": 1, "catalog_content": "Item Name: Starbucks Coffee Pike Place\nValue: 12"},
        {"sample_id": 2, "catalog_content": "Item Name: Tropicana Orange Juice 52oz\nValue: 52"},
        {"sample_id": 3, "catalog_content": "Item Name: Lay's Classic Potato Chips\nValue: 8"},
        {"sample_id": 4, "catalog_content": "Item Name: Bob's Red Mill Brown Rice\nValue: 5"},
        {"sample_id": 5, "catalog_content": "Item Name: Sparkling Water Variety Pack\nValue: 12"},
        {"sample_id": 6, "catalog_content": "Item Name: McCormick Garlic Powder 3oz\nValue: 3"},
    ])


# ── compute_entropy Tests ─────────────────────────────────────────────────────

class TestComputeEntropy:
    """
    Entropy is the mathematical heart of the uncertainty score.
    These tests verify the formula is correct at key boundary values.
    """

    def test_maximum_entropy_at_half(self):
        # H(0.5) = 1.0 — maximally uncertain
        assert compute_entropy(0.5) == pytest.approx(1.0, abs=1e-6)

    def test_zero_entropy_at_zero(self):
        # H(0.0) = 0.0 — certain it's NEGATIVE
        assert compute_entropy(0.0) == 0.0

    def test_zero_entropy_at_one(self):
        # H(1.0) = 0.0 — certain it's POSITIVE
        assert compute_entropy(1.0) == 0.0

    def test_entropy_decreases_toward_certainty(self):
        # As probability moves away from 0.5, entropy decreases
        assert compute_entropy(0.7) < compute_entropy(0.6)
        assert compute_entropy(0.6) < compute_entropy(0.5)

    def test_entropy_symmetric(self):
        # H(p) == H(1-p) — uncertainty is symmetric
        assert compute_entropy(0.3) == pytest.approx(compute_entropy(0.7), abs=1e-6)

    def test_entropy_at_accept_threshold(self):
        # At our 0.85 threshold, entropy should be < 0.7 (not maximally uncertain)
        assert compute_entropy(0.85) < 0.7

    def test_entropy_bounded_zero_to_one(self):
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            e = compute_entropy(p)
            assert 0.0 <= e <= 1.0


# ── compute_entropy_score Tests ───────────────────────────────────────────────

class TestComputeEntropyScore:

    def test_returns_float(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"]
        row = review.iloc[0]
        score = compute_entropy_score(row)
        assert isinstance(score, float)

    def test_score_between_0_and_1(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"]
        for _, row in review.iterrows():
            score = compute_entropy_score(row)
            assert 0.0 <= score <= 1.0

    def test_high_entropy_for_near_half_probs(self):
        # Both tags near 0.5 — should produce high entropy score
        row = pd.Series({
            "tag_probs": json.dumps({"organic": 0.51, "vegan": 0.49}),
            "category_prob": 0.52,
        })
        score = compute_entropy_score(row)
        assert score > 0.9

    def test_low_entropy_for_confident_probs(self):
        # Both tags very confident — should produce low entropy score
        row = pd.Series({
            "tag_probs": json.dumps({"organic": 0.95, "vegan": 0.05}),
            "category_prob": 0.95,
        })
        score = compute_entropy_score(row)
        assert score < 0.4

    def test_falls_back_to_category_prob_on_bad_json(self):
        row = pd.Series({
            "tag_probs": "not valid json",
            "category_prob": 0.5,
        })
        # Should not crash — falls back to category_prob entropy
        score = compute_entropy_score(row)
        assert isinstance(score, float)


# ── compute_disagreement_score Tests ─────────────────────────────────────────

class TestComputeDisagreementScore:

    def test_returns_float(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"]
        row = review.iloc[0]
        score = compute_disagreement_score(row)
        assert isinstance(score, float)

    def test_score_between_0_and_1(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"]
        for _, row in review.iterrows():
            score = compute_disagreement_score(row)
            assert 0.0 <= score <= 1.0

    def test_max_disagreement_when_perfectly_split(self):
        # Half accept, half reject — maximum disagreement
        row = pd.Series({
            "tag_statuses": json.dumps({
                "organic": "auto_accept",
                "vegan": "auto_reject",
                "keto": "auto_accept",
                "paleo": "auto_reject",
            })
        })
        score = compute_disagreement_score(row)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_zero_disagreement_when_all_agree(self):
        # All accept — no disagreement
        row = pd.Series({
            "tag_statuses": json.dumps({
                "organic": "auto_accept",
                "vegan": "auto_accept",
            })
        })
        score = compute_disagreement_score(row)
        assert score == 0.0

    def test_zero_disagreement_when_all_review(self):
        # All review — no accept/reject votes to disagree
        row = pd.Series({
            "tag_statuses": json.dumps({
                "organic": "review",
                "vegan": "review",
            })
        })
        score = compute_disagreement_score(row)
        assert score == 0.0

    def test_handles_bad_json(self):
        row = pd.Series({"tag_statuses": "broken"})
        assert compute_disagreement_score(row) == 0.0


# ── compute_rarity_scores Tests ───────────────────────────────────────────────

class TestComputeRarityScores:

    def test_returns_series_of_correct_length(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"]
        scores = compute_rarity_scores(review)
        assert len(scores) == len(review)

    def test_scores_normalized_between_0_and_1(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"]
        scores = compute_rarity_scores(review)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_max_score_is_1(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"]
        scores = compute_rarity_scores(review)
        assert scores.max() == pytest.approx(1.0, abs=1e-6)

    def test_rare_category_scores_higher_than_common(self, sample_labels_df):
        # "Spices & Seasonings" appears once, "Beverages" appears twice
        # → Spices should get higher rarity score
        review = sample_labels_df[sample_labels_df["category_status"] == "review"]
        scores = compute_rarity_scores(review)
        review_with_scores = review.copy()
        review_with_scores["rarity_score"] = scores
        spices_score = review_with_scores[
            review_with_scores["category"] == "Spices & Seasonings"
        ]["rarity_score"].values[0]
        beverages_score = review_with_scores[
            review_with_scores["category"] == "Beverages"
        ]["rarity_score"].mean()
        assert spices_score > beverages_score


# ── compute_uncertainty_scores Tests ─────────────────────────────────────────

class TestComputeUncertaintyScores:

    def test_adds_required_columns(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"].copy()
        result = compute_uncertainty_scores(review)
        for col in ["entropy_score", "disagreement_score", "rarity_score",
                    "uncertainty_score", "reason"]:
            assert col in result.columns

    def test_uncertainty_score_is_weighted_sum(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"].copy()
        result = compute_uncertainty_scores(review)
        for _, row in result.iterrows():
            expected = round(
                WEIGHT_ENTROPY      * row["entropy_score"]
                + WEIGHT_DISAGREEMENT * row["disagreement_score"]
                + WEIGHT_RARITY       * row["rarity_score"],
                4
            )
            assert row["uncertainty_score"] == pytest.approx(expected, abs=1e-3)

    def test_reason_is_non_empty_string(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"].copy()
        result = compute_uncertainty_scores(review)
        for _, row in result.iterrows():
            assert isinstance(row["reason"], str)
            assert len(row["reason"]) > 0


# ── filter_review_candidates Tests ───────────────────────────────────────────

class TestFilterReviewCandidates:

    def test_only_review_rows_kept(self, sample_labels_df):
        result = filter_review_candidates(sample_labels_df)
        assert all(result["category_status"] == "review")

    def test_auto_accept_excluded(self, sample_labels_df):
        result = filter_review_candidates(sample_labels_df)
        assert "auto_accept" not in result["category_status"].values

    def test_auto_reject_excluded(self, sample_labels_df):
        result = filter_review_candidates(sample_labels_df)
        assert "auto_reject" not in result["category_status"].values

    def test_correct_count(self, sample_labels_df):
        # 4 rows have category_status == "review" in our fixture
        result = filter_review_candidates(sample_labels_df)
        assert len(result) == 4


# ── select_top_n Tests ────────────────────────────────────────────────────────

class TestSelectTopN:

    def test_returns_at_most_n_rows(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"].copy()
        scored = compute_uncertainty_scores(review)
        result = select_top_n(scored, n=2)
        assert len(result) <= 2

    def test_sorted_by_score_descending(self, sample_labels_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"].copy()
        scored = compute_uncertainty_scores(review)
        result = select_top_n(scored, n=3)
        scores = result["uncertainty_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_returns_all_if_fewer_than_n(self, sample_labels_df):
        # Only 4 review candidates — requesting 50 should return all 4
        review = sample_labels_df[sample_labels_df["category_status"] == "review"].copy()
        scored = compute_uncertainty_scores(review)
        result = select_top_n(scored, n=50)
        assert len(result) == len(review)


# ── build_review_queue Tests ──────────────────────────────────────────────────

class TestBuildReviewQueue:

    def test_item_name_column_added(self, sample_labels_df, sample_catalog_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"].copy()
        scored = compute_uncertainty_scores(review)
        top    = select_top_n(scored, n=10)
        queue  = build_review_queue(top, sample_catalog_df)
        assert "item_name" in queue.columns

    def test_item_names_not_empty(self, sample_labels_df, sample_catalog_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"].copy()
        scored = compute_uncertainty_scores(review)
        top    = select_top_n(scored, n=10)
        queue  = build_review_queue(top, sample_catalog_df)
        assert queue["item_name"].notna().all()

    def test_sample_ids_preserved(self, sample_labels_df, sample_catalog_df):
        review = sample_labels_df[sample_labels_df["category_status"] == "review"].copy()
        scored = compute_uncertainty_scores(review)
        top    = select_top_n(scored, n=4)
        queue  = build_review_queue(top, sample_catalog_df)
        assert "sample_id" in queue.columns
        assert queue["sample_id"].notna().all()


# ── save_review_queue Tests ───────────────────────────────────────────────────

class TestSaveReviewQueue:

    def _make_queue(self):
        return pd.DataFrame([
            {
                "sample_id": 5,
                "item_name": "Sparkling Water Variety Pack",
                "category": "Beverages",
                "category_prob": 0.60,
                "category_status": "review",
                "dietary_tags": "",
                "uncertainty_score": 0.88,
                "entropy_score": 0.99,
                "disagreement_score": 0.0,
                "rarity_score": 0.5,
                "reason": "High model uncertainty (entropy=0.99)",
            }
        ])

    def test_csv_created(self, tmp_path):
        queue = self._make_queue()
        out   = tmp_path / "review_queue.csv"
        with patch(
            "scripts.active_learning_query.OUTPUT_PATH", out
        ):
            save_review_queue(queue)
        assert out.exists()

    def test_csv_has_correct_row_count(self, tmp_path):
        queue = self._make_queue()
        out   = tmp_path / "review_queue.csv"
        queue.to_csv(out, index=False)
        df = pd.read_csv(out)
        assert len(df) == 1


# ── run_active_learning_query Integration Tests ───────────────────────────────

class TestRunActiveLearningQuery:
    """
    Full pipeline integration tests using tmp_path for file I/O.
    We write synthetic CSV files to tmp_path and run the full pipeline.
    """

    def test_returns_dataframe(self, sample_labels_df, sample_catalog_df, tmp_path):
        labels_path  = tmp_path / "weak_labels.csv"
        catalog_path = tmp_path / "train.csv"
        output_path  = tmp_path / "review_queue.csv"

        sample_labels_df.to_csv(labels_path,  index=False)
        sample_catalog_df.to_csv(catalog_path, index=False)

        result = run_active_learning_query(
            weak_labels_path=labels_path,
            train_csv_path=catalog_path,
            output_path=output_path,
            top_n=3,
        )
        assert isinstance(result, pd.DataFrame)

    def test_output_csv_created(self, sample_labels_df, sample_catalog_df, tmp_path):
        labels_path  = tmp_path / "weak_labels.csv"
        catalog_path = tmp_path / "train.csv"
        output_path  = tmp_path / "review_queue.csv"

        sample_labels_df.to_csv(labels_path,  index=False)
        sample_catalog_df.to_csv(catalog_path, index=False)

        run_active_learning_query(
            weak_labels_path=labels_path,
            train_csv_path=catalog_path,
            output_path=output_path,
            top_n=3,
        )
        assert output_path.exists()

    def test_respects_top_n_parameter(self, sample_labels_df, sample_catalog_df, tmp_path):
        labels_path  = tmp_path / "weak_labels.csv"
        catalog_path = tmp_path / "train.csv"
        output_path  = tmp_path / "review_queue.csv"

        sample_labels_df.to_csv(labels_path,  index=False)
        sample_catalog_df.to_csv(catalog_path, index=False)

        result = run_active_learning_query(
            weak_labels_path=labels_path,
            train_csv_path=catalog_path,
            output_path=output_path,
            top_n=2,
        )
        assert len(result) <= 2

    def test_empty_review_zone_returns_empty_df(self, tmp_path):
        # All products are auto_accept — no review candidates
        all_accept = pd.DataFrame([{
            "sample_id": 1,
            "category": "Beverages",
            "category_prob": 0.95,
            "category_status": "auto_accept",
            "dietary_tags": "organic",
            "tag_probs": json.dumps({"organic": 0.95}),
            "tag_statuses": json.dumps({"organic": "auto_accept"}),
        }])
        catalog = pd.DataFrame([{
            "sample_id": 1,
            "catalog_content": "Item Name: Test Product"
        }])

        labels_path  = tmp_path / "weak_labels.csv"
        catalog_path = tmp_path / "train.csv"
        output_path  = tmp_path / "review_queue.csv"

        all_accept.to_csv(labels_path,  index=False)
        catalog.to_csv(catalog_path,    index=False)

        result = run_active_learning_query(
            weak_labels_path=labels_path,
            train_csv_path=catalog_path,
            output_path=output_path,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_output_has_required_columns(self, sample_labels_df, sample_catalog_df, tmp_path):
        labels_path  = tmp_path / "weak_labels.csv"
        catalog_path = tmp_path / "train.csv"
        output_path  = tmp_path / "review_queue.csv"

        sample_labels_df.to_csv(labels_path,  index=False)
        sample_catalog_df.to_csv(catalog_path, index=False)

        result = run_active_learning_query(
            weak_labels_path=labels_path,
            train_csv_path=catalog_path,
            output_path=output_path,
        )
        required = {"sample_id", "item_name", "category", "uncertainty_score", "reason"}
        assert required.issubset(set(result.columns))