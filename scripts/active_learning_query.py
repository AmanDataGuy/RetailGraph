# scripts/active_learning_query.py
"""
Active Learning Query Script for RetailGraph.

PURPOSE:
    After weak labeling (label_model.py), some products land in the
    0.50–0.85 "review" zone — the model is uncertain about them.
    We can't afford to review all of them. This script picks the TOP 50
    most VALUABLE products for a human to review.

WHY ACTIVE LEARNING?
    Not all uncertain products are equally valuable to label.
    Labeling a product the model is 51% sure about teaches it more
    than labeling one it's 84% sure about. Active learning formalizes
    this intuition into a score.

HOW WE SCORE UNCERTAINTY (3 signals combined):
    1. Entropy         — how uncertain is the model's probability? (weight: 50%)
                         entropy(0.5) = 1.0 (maximum uncertainty)
                         entropy(0.99) = 0.01 (very certain)

    2. Disagreement    — how much do the LFs disagree with each other? (weight: 30%)
                         If 6 LFs vote POSITIVE and 6 vote NEGATIVE,
                         disagreement = 1.0 (maximum conflict)

    3. Category rarity — how underrepresented is this product's category? (weight: 20%)
                         Labeling 1 product from a rare category teaches
                         more than labeling another from an overrepresented one.

INPUTS:
    data/extracted/weak_labels.csv  — output of label_model.py
    data/raw/train.csv              — original catalog (for item names)

OUTPUT:
    data/extracted/review_queue.csv — top 50 products for human review,
                                      sorted by uncertainty score descending,
                                      with reason column explaining WHY
                                      each product was selected.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging setup ─────────────────────────────────────────────────────────────
# We use module-level logger so every function's log messages are
# prefixed with the script name in the log output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── File paths ────────────────────────────────────────────────────────────────
# All paths relative to project root — run this script from RetailGraph/.

WEAK_LABELS_PATH = Path("data/extracted/weak_labels.csv")
TRAIN_CSV_PATH   = Path("data/raw/train.csv")
OUTPUT_PATH      = Path("data/extracted/review_queue.csv")

# ── Scoring weights ───────────────────────────────────────────────────────────
# These three weights must sum to 1.0.
# Entropy gets the highest weight because it directly measures
# how confused the model is. Disagreement is second because LF
# conflict is a strong signal of genuine ambiguity. Rarity is
# third because it's a dataset-level concern, not a per-product one.

WEIGHT_ENTROPY      = 0.50
WEIGHT_DISAGREEMENT = 0.30
WEIGHT_RARITY       = 0.20

# Number of products to surface for human review per batch.
# 50 is a standard active learning batch size — large enough to be
# useful, small enough for a human to review in one sitting (~2 hours).
TOP_N = 50

# ── Label constants ───────────────────────────────────────────────────────────
# Must match labeling_functions.py
ABSTAIN  = -1
NEGATIVE =  0
POSITIVE =  1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load weak labels and original catalog data.

    We merge these two DataFrames so the output review queue contains
    the actual product name alongside the uncertainty score — otherwise
    a human reviewer would see sample_id numbers with no context.

    Returns:
        labels_df  — weak_labels.csv as DataFrame
        catalog_df — train.csv with sample_id and catalog_content columns
    """
    logger.info("Loading weak labels from %s...", WEAK_LABELS_PATH)
    labels_df = pd.read_csv(WEAK_LABELS_PATH)

    logger.info("Loading catalog from %s...", TRAIN_CSV_PATH)
    # We only need sample_id and catalog_content from train.csv.
    # Loading the full file is fine — pandas is fast for 75k rows.
    catalog_df = pd.read_csv(TRAIN_CSV_PATH, usecols=["sample_id", "catalog_content"])

    logger.info(
        "Loaded %d labeled rows, %d catalog rows.",
        len(labels_df), len(catalog_df)
    )
    return labels_df, catalog_df


def filter_review_candidates(labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only products in the 'review' zone (0.50–0.85 category confidence).

    Why filter to review zone only?
        auto_accept products don't need human review — the model is confident.
        auto_reject products aren't worth reviewing — the model is confident
        they don't belong to any category.
        Only the uncertain middle needs human attention.

    Args:
        labels_df: full weak_labels DataFrame

    Returns:
        Filtered DataFrame with only category_status == "review" rows.
    """
    review_df = labels_df[labels_df["category_status"] == "review"].copy()
    logger.info(
        "Review candidates: %d / %d (%.1f%%)",
        len(review_df),
        len(labels_df),
        100 * len(review_df) / max(len(labels_df), 1),
    )
    return review_df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Compute uncertainty scores
# ══════════════════════════════════════════════════════════════════════════════

def compute_entropy(prob: float) -> float:
    """
    Compute binary entropy for a single probability value.

    Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)

    Properties:
        H(0.5)  = 1.0  — maximum uncertainty, model has no idea
        H(0.0)  = 0.0  — certain it's NEGATIVE
        H(1.0)  = 0.0  — certain it's POSITIVE
        H(0.85) = 0.61 — at our accept threshold, still some uncertainty
        H(0.50) = 1.0  — right at the boundary, maximum confusion

    We use base-2 log so entropy is bounded [0, 1] for binary labels.

    Args:
        prob: float between 0.0 and 1.0

    Returns:
        entropy value between 0.0 and 1.0
    """
    # Guard against log(0) — mathematically undefined, treat as 0
    if prob <= 0.0 or prob >= 1.0:
        return 0.0
    return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)


def compute_entropy_score(row: pd.Series) -> float:
    """
    Compute the mean entropy across all dietary tag probabilities for a row.

    We average entropy over all tags rather than using just category_prob
    because a product might be certain in category but very uncertain
    in dietary tags — those are also valuable to review.

    Args:
        row: one row from the review candidates DataFrame

    Returns:
        Mean entropy score across all dietary tags (0.0–1.0)
    """
    try:
        # tag_probs is stored as a JSON string in the CSV — parse it back
        tag_probs: dict = json.loads(row["tag_probs"])
    except (json.JSONDecodeError, TypeError):
        # If tag_probs is missing or malformed, fall back to category prob
        return compute_entropy(row["category_prob"])

    if not tag_probs:
        return compute_entropy(row["category_prob"])

    # Compute entropy for each tag probability and average them.
    # This gives a holistic uncertainty measure across all labels.
    entropies = [compute_entropy(p) for p in tag_probs.values()]
    return float(np.mean(entropies))


def compute_disagreement_score(row: pd.Series) -> float:
    """
    Compute LF disagreement score from tag_statuses.

    Disagreement measures how much the labeling functions conflict.
    A product where half the LFs voted auto_accept and half voted
    auto_reject is maximally ambiguous — it's the most valuable to label.

    We approximate disagreement from tag_statuses:
        - Count how many tags are "auto_accept" vs "auto_reject"
        - Disagreement = min(accept, reject) / total_voted
        - Range: 0.0 (all agree) to 0.5 (perfectly split)
        - We multiply by 2 to normalize to [0, 1]

    Args:
        row: one row from the review candidates DataFrame

    Returns:
        Disagreement score between 0.0 and 1.0
    """
    try:
        tag_statuses: dict = json.loads(row["tag_statuses"])
    except (json.JSONDecodeError, TypeError):
        return 0.0

    statuses = list(tag_statuses.values())
    if not statuses:
        return 0.0

    n_accept = statuses.count("auto_accept")
    n_reject = statuses.count("auto_reject")
    n_voted  = n_accept + n_reject  # exclude "review" — they're already uncertain

    if n_voted == 0:
        return 0.0

    # min(accept, reject) / total is maximized when accept == reject == 0.5
    # Multiply by 2 to scale to [0, 1]
    return 2 * min(n_accept, n_reject) / n_voted


def compute_rarity_scores(df: pd.DataFrame) -> pd.Series:
    """
    Compute category rarity score for each row.

    Rarity logic:
        A category with 100 products → high rarity score
        A category with 10,000 products → low rarity score

    We use 1 / log(1 + count) so:
        - count=1    → rarity = 1/log(2) ≈ 1.44 (very rare)
        - count=100  → rarity = 1/log(101) ≈ 0.22 (rare)
        - count=1000 → rarity = 1/log(1001) ≈ 0.14 (common)
        - count=10000 → rarity = 1/log(10001) ≈ 0.11 (very common)

    We then normalize to [0, 1] by dividing by the max rarity score
    so all three signals are on the same scale before combining.

    Args:
        df: review candidates DataFrame with "category" column

    Returns:
        pd.Series of rarity scores, one per row, normalized to [0, 1]
    """
    # Count how many products fall into each category
    category_counts = df["category"].value_counts()

    # Map each row's category to its rarity score
    raw_rarity = df["category"].map(
        lambda cat: 1.0 / np.log1p(category_counts.get(cat, 1))
        if pd.notna(cat) else 0.0
    )

    # Normalize to [0, 1] — divide by max so scores are comparable
    max_rarity = raw_rarity.max()
    if max_rarity == 0:
        return pd.Series(np.zeros(len(df)), index=df.index)

    return raw_rarity / max_rarity


def compute_uncertainty_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the final combined uncertainty score for every review candidate.

    Final score = (WEIGHT_ENTROPY * entropy)
                + (WEIGHT_DISAGREEMENT * disagreement)
                + (WEIGHT_RARITY * rarity)

    Higher score = more valuable to label.

    Also computes a human-readable "reason" string explaining WHY
    this product was selected — useful for the reviewer to understand
    what they're looking at.

    Args:
        df: filtered review candidates DataFrame

    Returns:
        df with new columns: entropy_score, disagreement_score,
        rarity_score, uncertainty_score, reason
    """
    logger.info("Computing uncertainty scores for %d candidates...", len(df))

    # ── Compute individual scores ─────────────────────────────────────────
    df["entropy_score"]      = df.apply(compute_entropy_score, axis=1)
    df["disagreement_score"] = df.apply(compute_disagreement_score, axis=1)
    df["rarity_score"]       = compute_rarity_scores(df)

    # ── Combine into final score ──────────────────────────────────────────
    df["uncertainty_score"] = (
        WEIGHT_ENTROPY      * df["entropy_score"]
        + WEIGHT_DISAGREEMENT * df["disagreement_score"]
        + WEIGHT_RARITY       * df["rarity_score"]
    ).round(4)

    # ── Build reason string ───────────────────────────────────────────────
    # The reason tells the reviewer exactly which signal drove the selection.
    # This makes the review queue actionable — reviewer knows what to focus on.
    df["reason"] = df.apply(_build_reason, axis=1)

    logger.info(
        "Uncertainty score stats:\n  mean=%.3f  max=%.3f  min=%.3f",
        df["uncertainty_score"].mean(),
        df["uncertainty_score"].max(),
        df["uncertainty_score"].min(),
    )

    return df


def _build_reason(row: pd.Series) -> str:
    """
    Build a human-readable reason string for why a product was selected.

    We identify the dominant signal — the one that contributed most
    to the uncertainty score — and describe it in plain English.
    This helps the human reviewer focus on the right aspect of the product.

    Args:
        row: one row with entropy_score, disagreement_score, rarity_score

    Returns:
        A plain English reason string, e.g.:
        "High model uncertainty (entropy=0.94); LF disagreement detected"
    """
    parts = []

    # Identify the dominant signal
    scores = {
        "entropy":      row["entropy_score"]      * WEIGHT_ENTROPY,
        "disagreement": row["disagreement_score"] * WEIGHT_DISAGREEMENT,
        "rarity":       row["rarity_score"]       * WEIGHT_RARITY,
    }
    dominant = max(scores, key=scores.get)

    # Always mention the dominant signal first
    if dominant == "entropy":
        parts.append(f"High model uncertainty (entropy={row['entropy_score']:.2f})")
    elif dominant == "disagreement":
        parts.append(f"LF disagreement detected (score={row['disagreement_score']:.2f})")
    else:
        parts.append(f"Underrepresented category (rarity={row['rarity_score']:.2f})")

    # Add secondary signals if they're also notable (above 0.5)
    if dominant != "entropy" and row["entropy_score"] > 0.5:
        parts.append(f"also uncertain (entropy={row['entropy_score']:.2f})")
    if dominant != "disagreement" and row["disagreement_score"] > 0.3:
        parts.append(f"LF conflict present")
    if dominant != "rarity" and row["rarity_score"] > 0.7:
        parts.append(f"rare category")

    return "; ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Select top N and save
# ══════════════════════════════════════════════════════════════════════════════

def select_top_n(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """
    Select the top N most uncertain products for human review.

    We sort by uncertainty_score descending and take the first N rows.
    If fewer than N products are in the review zone, we return all of them
    rather than crashing — this handles small datasets in tests gracefully.

    Args:
        df: DataFrame with uncertainty_score column
        n:  number of products to select (default: TOP_N = 50)

    Returns:
        DataFrame with at most n rows, sorted by uncertainty_score desc.
    """
    sorted_df = df.sort_values("uncertainty_score", ascending=False)
    selected  = sorted_df.head(n)

    logger.info(
        "Selected top %d products (requested %d, available %d).",
        len(selected), n, len(df)
    )
    return selected


def build_review_queue(
    top_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge top uncertain products with catalog data to build the review queue.

    The review queue is what a human actually reads. It needs:
        - The product name (from catalog_df)
        - The predicted category and confidence (from top_df)
        - The uncertainty score and reason (computed above)
        - The dietary tag probabilities (so reviewer can correct them)

    Columns in output:
        sample_id, item_name, category, category_prob, category_status,
        dietary_tags, uncertainty_score, entropy_score, disagreement_score,
        rarity_score, reason

    Args:
        top_df:     selected top-N uncertain products
        catalog_df: original train.csv with catalog_content

    Returns:
        Clean review queue DataFrame ready to save.
    """
    import re

    # ── Extract item_name from catalog_content ────────────────────────────
    # We parse item_name here rather than passing it through label_model.py
    # to keep the weak labeling pipeline lean — it doesn't need item names.
    def extract_item_name(text: str) -> str:
        match = re.search(r'Item Name:\s*(.+?)(?:\n|$)', str(text))
        return match.group(1).strip() if match else str(text)[:80]

    catalog_df = catalog_df.copy()
    catalog_df["item_name"] = catalog_df["catalog_content"].apply(extract_item_name)

    # ── Merge on sample_id ────────────────────────────────────────────────
    merged = top_df.merge(
        catalog_df[["sample_id", "item_name"]],
        on="sample_id",
        how="left",
    )

    # ── Select and order output columns ──────────────────────────────────
    output_cols = [
        "sample_id",
        "item_name",
        "category",
        "category_prob",
        "category_status",
        "dietary_tags",
        "uncertainty_score",
        "entropy_score",
        "disagreement_score",
        "rarity_score",
        "reason",
    ]

    # Only keep columns that exist — tag_probs/tag_statuses are internal
    available = [c for c in output_cols if c in merged.columns]
    return merged[available].reset_index(drop=True)


def save_review_queue(queue_df: pd.DataFrame) -> None:
    """
    Save the review queue to data/extracted/review_queue.csv.

    Also prints a summary table to stdout so you can see at a glance
    what categories dominated the uncertain zone — useful for deciding
    whether to add more LFs for those categories.

    Args:
        queue_df: final review queue DataFrame from build_review_queue()
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    queue_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    logger.info("Saved review queue (%d rows) to %s", len(queue_df), OUTPUT_PATH)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n── Review Queue Summary ─────────────────────────────────────")
    print(f"Total products selected: {len(queue_df)}")
    print(f"\nTop uncertainty scores:")
    print(queue_df[["item_name", "category", "uncertainty_score", "reason"]]
          .head(10)
          .to_string(index=False))

    print(f"\nCategory distribution in review queue:")
    print(queue_df["category"].value_counts().to_string())

    print(f"\nMean scores:")
    print(f"  entropy_score:      {queue_df['entropy_score'].mean():.3f}")
    print(f"  disagreement_score: {queue_df['disagreement_score'].mean():.3f}")
    print(f"  rarity_score:       {queue_df['rarity_score'].mean():.3f}")
    print(f"  uncertainty_score:  {queue_df['uncertainty_score'].mean():.3f}")
    print("─────────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_active_learning_query(
    weak_labels_path: Path = WEAK_LABELS_PATH,
    train_csv_path: Path   = TRAIN_CSV_PATH,
    output_path: Path      = OUTPUT_PATH,
    top_n: int             = TOP_N,
) -> pd.DataFrame:
    """
    Run the full active learning query pipeline end to end.

    This is the single entry point called by __main__ and by tests.
    All file paths and top_n are parameterized so tests can inject
    tmp_path fixtures without touching real data files.

    Steps:
        1. Load weak_labels.csv and train.csv
        2. Filter to review zone candidates
        3. Compute entropy, disagreement, rarity, combined score
        4. Select top N by uncertainty score
        5. Merge with catalog for item names
        6. Save review_queue.csv

    Args:
        weak_labels_path: path to weak_labels.csv
        train_csv_path:   path to train.csv
        output_path:      where to save review_queue.csv
        top_n:            how many products to select

    Returns:
        review queue DataFrame (also saved to output_path)
    """
    logger.info("═" * 60)
    logger.info("Starting active learning query pipeline")
    logger.info("═" * 60)

    # ── Step 1 — Load ─────────────────────────────────────────────────────
    labels_df  = pd.read_csv(weak_labels_path)
    catalog_df = pd.read_csv(train_csv_path, usecols=["sample_id", "catalog_content"])

    # ── Step 2 — Filter to review zone ────────────────────────────────────
    review_df = filter_review_candidates(labels_df)

    if len(review_df) == 0:
        logger.warning(
            "No review candidates found. "
            "All products were auto_accept or auto_reject. "
            "Consider lowering AUTO_ACCEPT_THRESHOLD in label_model.py."
        )
        return pd.DataFrame()

    # ── Step 3 — Compute scores ───────────────────────────────────────────
    scored_df = compute_uncertainty_scores(review_df)

    # ── Step 4 — Select top N ─────────────────────────────────────────────
    top_df = select_top_n(scored_df, n=top_n)

    # ── Step 5 — Build review queue ───────────────────────────────────────
    queue_df = build_review_queue(top_df, catalog_df)

    # ── Step 6 — Save ─────────────────────────────────────────────────────
    # Patch output path if it was overridden (for tests)
    actual_output = output_path
    actual_output.parent.mkdir(parents=True, exist_ok=True)
    queue_df.to_csv(actual_output, index=False, encoding="utf-8")
    logger.info("Saved %d rows to %s", len(queue_df), actual_output)

    save_review_queue(queue_df)

    logger.info("Active learning query complete.")
    return queue_df


# ── Entry point ───────────────────────────────────────────────────────────────
# Run directly: python scripts/active_learning_query.py
# Requires weak_labels.csv to exist (run label_model.py first).

if __name__ == "__main__":
    run_active_learning_query()