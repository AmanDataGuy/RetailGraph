# src/extraction/weak_supervision/label_model.py
"""
Snorkel LabelModel training and prediction for RetailGraph.

Two separate models are trained:
    1. CategoryLabelModel   — multi-class (11 categories, cardinality=11)
    2. DietaryTagLabelModel — binary per tag (cardinality=2), one model
                              trained per dietary tag independently.

Pipeline:
    raw df (75k rows)
        → apply_category_lfs()    → L_category  (n x 13 matrix)
        → apply_dietary_lfs()     → L_dietary    (n x 18 matrix)
        → train_category_model()  → category probabilities per row
        → train_dietary_models()  → per-tag probabilities per row
        → apply_thresholds()      → auto_accept / review / auto_reject
        → save_labeled_output()   → data/extracted/weak_labels.csv

Why two models and not one?
    Dietary tags are binary (POSITIVE/NEGATIVE, cardinality=2).
    Categories are multi-class (11 classes, cardinality=11).
    Snorkel's LabelModel handles these differently internally —
    mixing them produces incorrect probability estimates.

Why one binary model per dietary tag (not one model for all tags)?
    Each tag is an independent label. A product can be both organic
    AND gluten-free AND vegan simultaneously. Training one model per
    tag lets each model focus on its own LF signals without cross-tag
    interference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from snorkel.labeling import LFAnalysis, PandasLFApplier
from snorkel.labeling.model import LabelModel

from src.extraction.weak_supervision.labeling_functions import (
    # Category integer constants — used to map predictions back to names
    CAT_ABSTAIN,
    CAT_BEVERAGES,
    CAT_BAKING,
    CAT_COFFEE_TEA,
    CAT_CONDIMENTS,
    CAT_GRAINS,
    CAT_NUTS,
    CAT_PERSONAL,
    CAT_PROTEIN_BAR,
    CAT_SNACKS,
    CAT_SPICES,
    CAT_SUPPLEMENTS,
    # The two LF registries built in labeling_functions.py
    CATEGORY_LFS,
    DIETARY_TAG_LFS,
    # Dietary label constants
    ABSTAIN,
    NEGATIVE,
    POSITIVE,
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────
# These thresholds come directly from the Phase 3 spec.
# >0.85  → confident enough to auto-accept for training data
# <0.50  → confident enough to auto-reject (not worth human review)
# 0.50–0.85 → uncertain, route to active learning / human review queue

AUTO_ACCEPT_THRESHOLD = 0.85
AUTO_REJECT_THRESHOLD = 0.50

# Where the final weak labels get saved after the full pipeline runs
OUTPUT_PATH = Path("data/extracted/weak_labels.csv")

# Snorkel training hyperparameters.
# 500 epochs is the standard starting point from Snorkel docs.
# Lower → faster but less accurate. Higher → diminishing returns past ~1000.
N_EPOCHS = 500
LEARNING_RATE = 0.001

# Maps category integer index → human readable name for output CSV.
# Must stay in sync with the CAT_* constants in labeling_functions.py.
CATEGORY_NAMES = {
    CAT_BEVERAGES:   "Beverages",
    CAT_SNACKS:      "Snacks & Candy",
    CAT_COFFEE_TEA:  "Coffee & Tea",
    CAT_CONDIMENTS:  "Condiments & Sauces",
    CAT_GRAINS:      "Grains, Beans & Legumes",
    CAT_BAKING:      "Baking & Cooking",
    CAT_SPICES:      "Spices & Seasonings",
    CAT_SUPPLEMENTS: "Supplements & Health",
    CAT_NUTS:        "Nuts & Seeds",
    CAT_PERSONAL:    "Personal Care & Beauty",
    CAT_PROTEIN_BAR: "Protein Bars & Snacks",
}

# All dietary tags we label — must match retail.yaml dietary_tags_controlled.
# Order matters here — index 0 is used as the POSITIVE class in each
# binary LabelModel (Snorkel requires classes to start from 0).
DIETARY_TAGS = [
    "organic",
    "cruelty-free",
    "kosher",
    "gluten-free",
    "non-GMO",
    "vegan",
    "keto",
    "paleo",
    "dairy-free",
    "sugar-free",
    "nut-free",
    "soy-free",
    "high-protein",
    "low-calorie",
    "caffeine-free",
    "allergen-free",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LFStats:
    """
    Summary statistics from LFAnalysis for one label matrix.

    Stored so callers can inspect LF health without re-running
    the full applier. High conflict rate on an LF means it's
    fighting other LFs — usually a sign the pattern is too broad.

    Attributes:
        coverage   — {lf_name: float} fraction of rows each LF voted on
        conflicts  — {lf_name: float} fraction of rows with conflicts
        n_rows     — total rows in the matrix
        n_lfs      — number of LFs applied
    """
    coverage: dict
    conflicts: dict
    n_rows: int
    n_lfs: int


@dataclass
class WeakLabelResult:
    """
    The complete weak label output for one product row.

    After the pipeline runs, every product gets one of these.
    The active_learning_query.py script in Step 3.3 reads
    tag_probs and category_prob to build the uncertainty score.

    Attributes:
        sample_id       — original dataset ID from train.csv
        category        — predicted category name, or None if all LFs abstained
        category_prob   — confidence score for category (0.0–1.0)
        category_status — "auto_accept" / "review" / "auto_reject"
        dietary_tags    — list of tag names that passed AUTO_ACCEPT_THRESHOLD
        tag_probs       — {tag_name: probability} for every dietary tag
        tag_statuses    — {tag_name: status} for every dietary tag
    """
    sample_id: int
    category: Optional[str]
    category_prob: float
    category_status: str
    dietary_tags: list[str] = field(default_factory=list)
    tag_probs: dict = field(default_factory=dict)
    tag_statuses: dict = field(default_factory=dict)


# ── Step 1 — Apply LFs to build label matrices ────────────────────────────────

def apply_category_lfs(df: pd.DataFrame) -> np.ndarray:
    """
    Apply all category LFs to the dataframe and return the label matrix.

    The label matrix L has shape (n_rows, n_lfs).
    Each cell L[i][j] is what LF j voted for row i:
        CAT_ABSTAIN (-1) — LF has no opinion on this row
        CAT_BEVERAGES (0), CAT_SNACKS (1), ... — LF voted this category

    This matrix is what Snorkel's LabelModel trains on — it never
    sees the raw text, only these votes.

    Args:
        df: Full dataframe with catalog_content, price, image_link columns.

    Returns:
        L_category: numpy array of shape (n_rows, len(CATEGORY_LFS))
    """
    logger.info(
        "Applying %d category LFs to %d rows...",
        len(CATEGORY_LFS), len(df)
    )
    applier = PandasLFApplier(lfs=CATEGORY_LFS)
    L = applier.apply(df)
    logger.info("Category label matrix shape: %s", L.shape)
    return L


def apply_dietary_lfs(df: pd.DataFrame) -> np.ndarray:
    """
    Apply all dietary tag LFs to the dataframe and return the label matrix.

    Shape: (n_rows, len(DIETARY_TAG_LFS)).
    Values: POSITIVE(1) / NEGATIVE(0) / ABSTAIN(-1).

    Note: All dietary LFs are applied together in one pass for efficiency.
    The resulting matrix is then sliced per-tag when training individual
    binary LabelModels in train_dietary_models().

    Args:
        df: Full dataframe.

    Returns:
        L_dietary: numpy array of shape (n_rows, len(DIETARY_TAG_LFS))
    """
    logger.info(
        "Applying %d dietary LFs to %d rows...",
        len(DIETARY_TAG_LFS), len(df)
    )
    applier = PandasLFApplier(lfs=DIETARY_TAG_LFS)
    L = applier.apply(df)
    logger.info("Dietary label matrix shape: %s", L.shape)
    return L


# ── Step 2 — Analyze LF health before training ────────────────────────────────

def analyze_lfs(L: np.ndarray, lfs: list, label: str) -> LFStats:
    """
    Run LFAnalysis on a label matrix and return a summary.

    Always call this before training. If coverage is very low (<5%)
    on most LFs, the LabelModel will have little signal to work with.
    If conflicts are very high (>30%), some LFs are contradicting each
    other — consider tightening their patterns.

    Args:
        L     — label matrix from apply_*_lfs()
        lfs   — the list of LF functions (for column names)
        label — human readable name for logging ("category" or "dietary")

    Returns:
        LFStats dataclass with coverage and conflict dicts.
    """
    analysis = LFAnalysis(L=L, lfs=lfs)
    summary = analysis.lf_summary()

    # Extract coverage and conflict as plain dicts for easy inspection
    coverage  = summary["Coverage"].to_dict()
    conflicts = summary["Conflicts"].to_dict()

    logger.info(
        "\n── LF Analysis: %s ──\n%s",
        label, summary[["Coverage", "Conflicts", "Overlaps"]].to_string()
    )

    return LFStats(
        coverage=coverage,
        conflicts=conflicts,
        n_rows=L.shape[0],
        n_lfs=L.shape[1],
    )


# ── Step 3 — Train models ─────────────────────────────────────────────────────

def train_category_model(L_category: np.ndarray) -> tuple[LabelModel, np.ndarray]:
    """
    Train a multi-class LabelModel on the category label matrix.

    cardinality=11 because we have 11 categories (CAT_BEVERAGES through
    CAT_PROTEIN_BAR). Snorkel's LabelModel learns the accuracy of each LF
    and how they correlate — without needing any ground truth labels.

    After training, predict_proba returns a (n_rows, 11) matrix where
    each row sums to 1.0. We take argmax to get the predicted category
    and max to get the confidence score.

    Args:
        L_category: label matrix from apply_category_lfs()

    Returns:
        model:  trained LabelModel (can be reused for prediction)
        probs:  numpy array of shape (n_rows, 11) — probability per category
    """
    logger.info("Training category LabelModel (cardinality=11)...")

    # cardinality = number of classes the model must choose between.
    # This must match the number of unique non-abstain values in L_category.
    model = LabelModel(cardinality=11, verbose=True)

    model.fit(
        L_train=L_category,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        # seed for reproducibility — same data always produces same model
        seed=42,
    )

    probs = model.predict_proba(L=L_category)
    logger.info("Category model trained. Output shape: %s", probs.shape)
    return model, probs


def train_dietary_models(
    L_dietary: np.ndarray,
) -> dict[str, tuple[LabelModel, np.ndarray]]:
    """
    Train one binary LabelModel per dietary tag.

    Why one model per tag?
        Each tag is independent. Organic and vegan are separate questions.
        A product can have both, either, or neither. Training them separately
        means each model only sees the LFs relevant to its tag, and the
        probability estimate is clean and independent.

    We only train a model for a tag if at least one LF voted on it
    (coverage > 0). Tags with zero coverage get skipped — no signal
    means no useful model.

    Args:
        L_dietary: full dietary label matrix (n_rows, n_dietary_lfs)

    Returns:
        models_and_probs: {tag_name: (trained_model, probs_array)}
        probs_array shape: (n_rows, 2) — [prob_negative, prob_positive]
    """
    logger.info("Training %d binary dietary LabelModels...", len(DIETARY_TAGS))

    # We trained all LFs together but each tag has its own subset of LFs.
    # Map tag index → which columns of L_dietary belong to that tag.
    # The LFs in DIETARY_TAG_LFS are ordered to match DIETARY_TAGS —
    # each tag gets one LF column (some tags have multiple LFs, see below).
    #
    # Actually, all dietary LFs vote on the same binary scale (POSITIVE/
    # NEGATIVE/ABSTAIN) and all contribute signal for ALL tags through
    # the conflict/overlap structure. So we train on the full L_dietary
    # for every tag — Snorkel's model separates the signal internally.

    models_and_probs = {}

    for tag in DIETARY_TAGS:
        # Check if any LF voted on this tag at all.
        # If every cell in L_dietary is ABSTAIN, the model has no signal.
        n_votes = (L_dietary != ABSTAIN).sum()
        if n_votes == 0:
            logger.warning("Tag '%s' has zero LF coverage — skipping.", tag)
            continue

        logger.info("  Training model for tag: %s", tag)

        # cardinality=2 because this is binary: POSITIVE(1) or NEGATIVE(0).
        # Snorkel requires class labels to start from 0, which matches our
        # NEGATIVE=0, POSITIVE=1 constants.
        model = LabelModel(cardinality=2, verbose=False)
        model.fit(
            L_train=L_dietary,
            n_epochs=N_EPOCHS,
            lr=LEARNING_RATE,
            seed=42,
        )

        # probs[:, 1] = probability of POSITIVE for each row
        probs = model.predict_proba(L=L_dietary)
        models_and_probs[tag] = (model, probs)

    logger.info("Trained models for %d tags.", len(models_and_probs))
    return models_and_probs


# ── Step 4 — Apply thresholds ─────────────────────────────────────────────────

def _get_status(prob: float) -> str:
    """
    Convert a probability score into a routing decision.

    This is the core business logic from the Phase 3 spec:
        > 0.85 → auto_accept  — high confidence, use for training
        0.50–0.85 → review    — uncertain, route to human review queue
        < 0.50 → auto_reject  — confident it doesn't apply, discard

    Args:
        prob: float between 0.0 and 1.0

    Returns:
        One of: "auto_accept", "review", "auto_reject"
    """
    if prob >= AUTO_ACCEPT_THRESHOLD:
        return "auto_accept"
    elif prob >= AUTO_REJECT_THRESHOLD:
        return "review"
    else:
        return "auto_reject"


def build_results(
    df: pd.DataFrame,
    category_probs: np.ndarray,
    dietary_models_and_probs: dict[str, tuple[LabelModel, np.ndarray]],
) -> list[WeakLabelResult]:
    """
    Combine category and dietary predictions into WeakLabelResult objects.

    For each row we:
        1. Take argmax of category_probs → predicted category index
        2. Take max of category_probs → confidence score
        3. Apply threshold → routing status
        4. For each dietary tag, take probs[:, 1] (POSITIVE probability)
        5. Apply threshold per tag

    Rows where all category LFs abstained get category=None and
    category_status="auto_reject" — there is no signal for them.

    Args:
        df:                         original dataframe (for sample_id)
        category_probs:             shape (n_rows, 11)
        dietary_models_and_probs:   {tag: (model, probs)} from train_dietary_models()

    Returns:
        List of WeakLabelResult, one per row in df.
    """
    logger.info("Building results for %d rows...", len(df))
    results = []

    for i, row in enumerate(df.itertuples(index=False)):
        # ── Category prediction ───────────────────────────────────────────
        cat_prob_row = category_probs[i]          # shape (11,)
        cat_idx      = int(np.argmax(cat_prob_row))
        cat_prob     = float(np.max(cat_prob_row))
        cat_name     = CATEGORY_NAMES.get(cat_idx)
        cat_status   = _get_status(cat_prob)

        # ── Dietary tag predictions ───────────────────────────────────────
        tag_probs    = {}
        tag_statuses = {}
        accepted_tags = []

        for tag, (_, probs) in dietary_models_and_probs.items():
            # probs[:, 1] is the probability of POSITIVE for every row.
            # We index with [i] to get just this row's probability.
            p = float(probs[i, 1])
            status = _get_status(p)
            tag_probs[tag]    = round(p, 4)
            tag_statuses[tag] = status
            if status == "auto_accept":
                accepted_tags.append(tag)

        results.append(WeakLabelResult(
            sample_id=int(row.sample_id),
            category=cat_name,
            category_prob=round(cat_prob, 4),
            category_status=cat_status,
            dietary_tags=accepted_tags,
            tag_probs=tag_probs,
            tag_statuses=tag_statuses,
        ))

    logger.info("Built %d results.", len(results))
    return results


# ── Step 5 — Save output ──────────────────────────────────────────────────────

def save_labeled_output(results: list[WeakLabelResult]) -> None:
    """
    Save weak label results to data/extracted/weak_labels.csv.

    One row per product. Columns:
        sample_id, category, category_prob, category_status,
        dietary_tags (pipe-separated), tag_probs (JSON string),
        tag_statuses (JSON string)

    The pipe separator for dietary_tags avoids CSV quoting issues
    with commas. active_learning_query.py reads this file directly.

    Args:
        results: list of WeakLabelResult from build_results()
    """
    import json

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            "sample_id":       r.sample_id,
            "category":        r.category or "",
            "category_prob":   r.category_prob,
            "category_status": r.category_status,
            # pipe-separated so CSV parsing stays clean
            "dietary_tags":    "|".join(r.dietary_tags),
            "tag_probs":       json.dumps(r.tag_probs),
            "tag_statuses":    json.dumps(r.tag_statuses),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info("Saved %d labeled rows to %s", len(rows), OUTPUT_PATH)

    # ── Print summary stats ───────────────────────────────────────────────
    # These numbers tell you at a glance how much auto-labeled data
    # you have for training vs how much needs human review.
    cat_counts = out_df["category_status"].value_counts()
    logger.info("\n── Category label distribution ──\n%s", cat_counts.to_string())

    auto_accepted = out_df[out_df["category_status"] == "auto_accept"]
    logger.info(
        "Auto-accepted: %d / %d (%.1f%%)",
        len(auto_accepted), len(out_df),
        100 * len(auto_accepted) / max(len(out_df), 1)
    )


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_label_pipeline(df: pd.DataFrame) -> list[WeakLabelResult]:
    """
    Run the full weak labeling pipeline end to end.

    This is the single entry point called by scripts/run_extraction.py
    and by tests. Every step is logged so you can see exactly where
    time is being spent on large datasets.

    Steps:
        1. Apply LFs to build label matrices
        2. Analyze LF health (log warnings if coverage is low)
        3. Train models
        4. Build results with thresholds applied
        5. Save to CSV

    Args:
        df: dataframe with columns: sample_id, catalog_content,
            image_link, price

    Returns:
        List of WeakLabelResult, one per row.
    """
    # ── Step 1 — Build label matrices ────────────────────────────────────
    L_category = apply_category_lfs(df)
    L_dietary  = apply_dietary_lfs(df)

    # ── Step 2 — Analyze LF health ────────────────────────────────────────
    # We log but don't crash on low coverage — some LFs are intentionally
    # narrow (e.g. lf_hinglish_masala only fires on Indian product names).
    analyze_lfs(L_category, CATEGORY_LFS, label="category")
    analyze_lfs(L_dietary,  DIETARY_TAG_LFS, label="dietary")

    # ── Step 3 — Train models ─────────────────────────────────────────────
    _, category_probs = train_category_model(L_category)
    dietary_results   = train_dietary_models(L_dietary)

    # ── Step 4 — Build results ────────────────────────────────────────────
    results = build_results(df, category_probs, dietary_results)

    # ── Step 5 — Save ─────────────────────────────────────────────────────
    save_labeled_output(results)

    return results