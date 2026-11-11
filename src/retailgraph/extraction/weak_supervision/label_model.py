"""
# label_model.py - trains Snorkel LabelModels on the LF votes and writes data/extracted/weak_labels.csv

- One multi-class model for category, one binary model per dietary tag; probabilities are routed by threshold.
- run_label_pipeline() is the entry point; scripts/tests call it with a train.csv-shaped dataframe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from snorkel.labeling import LFAnalysis, PandasLFApplier
from snorkel.labeling.model import LabelModel

from retailgraph.extraction.weak_supervision.labeling_functions import (
    ABSTAIN,
    CAT_BAKING,
    CAT_BEVERAGES,
    CAT_COFFEE_TEA,
    CAT_CONDIMENTS,
    CAT_GRAINS,
    CAT_NUTS,
    CAT_PERSONAL,
    CAT_PROTEIN_BAR,
    CAT_SNACKS,
    CAT_SPICES,
    CAT_SUPPLEMENTS,
    CATEGORY_LFS,
    DIETARY_TAG_LFS,
)

logger = logging.getLogger(__name__)


# >= 0.85 auto-accept for training, < 0.50 auto-reject, in between goes to human review.
AUTO_ACCEPT_THRESHOLD = 0.85
AUTO_REJECT_THRESHOLD = 0.50
OUTPUT_PATH = Path("data/extracted/weak_labels.csv")
N_EPOCHS = 500
LEARNING_RATE = 0.001

# Category id -> name; must stay in sync with the CAT_* ids in labeling_functions.py.
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

# Dietary tags we label; must match retail.yaml dietary_tags_controlled.
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


# Per-LF coverage and conflict rates from LFAnalysis for one label matrix.
@dataclass
class LFStats:
    coverage: dict
    conflicts: dict
    n_rows: int
    n_lfs: int


# Final weak labels for one product: category and dietary tags with probabilities and routing status.
@dataclass
class WeakLabelResult:
    sample_id: int
    category: str | None
    category_prob: float
    category_status: str
    dietary_tags: list[str] = field(default_factory=list)
    tag_probs: dict = field(default_factory=dict)
    tag_statuses: dict = field(default_factory=dict)


# Run every category LF over df; returns the (rows x LFs) vote matrix.
def apply_category_lfs(df: pd.DataFrame) -> np.ndarray:
    logger.info(
        "Applying %d category LFs to %d rows...",
        len(CATEGORY_LFS), len(df)
    )
    applier = PandasLFApplier(lfs=CATEGORY_LFS)
    L = applier.apply(df)
    logger.info("Category label matrix shape: %s", L.shape)
    return L


# Run every dietary LF over df; returns the (rows x LFs) vote matrix.
def apply_dietary_lfs(df: pd.DataFrame) -> np.ndarray:
    logger.info(
        "Applying %d dietary LFs to %d rows...",
        len(DIETARY_TAG_LFS), len(df)
    )
    applier = PandasLFApplier(lfs=DIETARY_TAG_LFS)
    L = applier.apply(df)
    logger.info("Dietary label matrix shape: %s", L.shape)
    return L


# Log LFAnalysis coverage/conflicts for a vote matrix and return them as LFStats.
def analyze_lfs(L: np.ndarray, lfs: list, label: str) -> LFStats:
    analysis = LFAnalysis(L=L, lfs=lfs)
    summary = analysis.lf_summary()

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


# Fit the 11-class category LabelModel and return it with per-row probabilities.
def train_category_model(L_category: np.ndarray) -> tuple[LabelModel, np.ndarray]:
    logger.info("Training category LabelModel (cardinality=11)...")

    model = LabelModel(cardinality=11, verbose=True)

    model.fit(
        L_train=L_category,
        n_epochs=N_EPOCHS,
        lr=LEARNING_RATE,
        seed=42,
    )

    probs = model.predict_proba(L=L_category)
    logger.info("Category model trained. Output shape: %s", probs.shape)
    return model, probs


# Fit one binary LabelModel per dietary tag; returns {tag: (model, probs)}.
def train_dietary_models(
    L_dietary: np.ndarray,
) -> dict[str, tuple[LabelModel, np.ndarray]]:
    logger.info("Training %d binary dietary LabelModels...", len(DIETARY_TAGS))

    models_and_probs = {}

    for tag in DIETARY_TAGS:
        n_votes = (L_dietary != ABSTAIN).sum()
        if n_votes == 0:
            logger.warning("Tag '%s' has zero LF coverage — skipping.", tag)
            continue

        logger.info("  Training model for tag: %s", tag)

        model = LabelModel(cardinality=2, verbose=False)
        model.fit(
            L_train=L_dietary,
            n_epochs=N_EPOCHS,
            lr=LEARNING_RATE,
            seed=42,
        )

        probs = model.predict_proba(L=L_dietary)
        models_and_probs[tag] = (model, probs)

    logger.info("Trained models for %d tags.", len(models_and_probs))
    return models_and_probs


# Route a probability to auto_accept / review / auto_reject.
def _get_status(prob: float) -> str:
    if prob >= AUTO_ACCEPT_THRESHOLD:
        return "auto_accept"
    elif prob >= AUTO_REJECT_THRESHOLD:
        return "review"
    else:
        return "auto_reject"


# Combine category and dietary probabilities into one WeakLabelResult per row.
def build_results(
    df: pd.DataFrame,
    category_probs: np.ndarray,
    dietary_models_and_probs: dict[str, tuple[LabelModel, np.ndarray]],
) -> list[WeakLabelResult]:
    logger.info("Building results for %d rows...", len(df))
    results = []

    for i, row in enumerate(df.itertuples(index=False)):
        cat_prob_row = category_probs[i]
        cat_idx      = int(np.argmax(cat_prob_row))
        cat_prob     = float(np.max(cat_prob_row))
        cat_name     = CATEGORY_NAMES.get(cat_idx)
        cat_status   = _get_status(cat_prob)

        tag_probs    = {}
        tag_statuses = {}
        accepted_tags = []

        for tag, (_, probs) in dietary_models_and_probs.items():
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


# Write results to OUTPUT_PATH (tags pipe-joined, dicts as JSON) and log the split.
def save_labeled_output(results: list[WeakLabelResult]) -> None:
    import json

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        rows.append({
            "sample_id":       r.sample_id,
            "category":        r.category or "",
            "category_prob":   r.category_prob,
            "category_status": r.category_status,
            "dietary_tags":    "|".join(r.dietary_tags),
            "tag_probs":       json.dumps(r.tag_probs),
            "tag_statuses":    json.dumps(r.tag_statuses),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info("Saved %d labeled rows to %s", len(rows), OUTPUT_PATH)

    cat_counts = out_df["category_status"].value_counts()
    logger.info("\n── Category label distribution ──\n%s", cat_counts.to_string())

    auto_accepted = out_df[out_df["category_status"] == "auto_accept"]
    logger.info(
        "Auto-accepted: %d / %d (%.1f%%)",
        len(auto_accepted), len(out_df),
        100 * len(auto_accepted) / max(len(out_df), 1)
    )


# Full pipeline: apply LFs, analyze, train, build results, save CSV.
def run_label_pipeline(df: pd.DataFrame) -> list[WeakLabelResult]:
    L_category = apply_category_lfs(df)
    L_dietary  = apply_dietary_lfs(df)

    # Low coverage is only logged: some LFs (e.g. Hinglish) are narrow on purpose.
    analyze_lfs(L_category, CATEGORY_LFS, label="category")
    analyze_lfs(L_dietary,  DIETARY_TAG_LFS, label="dietary")

    _, category_probs = train_category_model(L_category)
    dietary_results   = train_dietary_models(L_dietary)

    results = build_results(df, category_probs, dietary_results)

    save_labeled_output(results)

    return results