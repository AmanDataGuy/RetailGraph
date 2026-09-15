"""
training/make_visual_eval_split.py

Diagnoses and (partially) fixes the "94.2% vs 92.0%" eval gap: the 3 visual
fields (packaging_type, packaging_color, has_brand_logo) have never been
scored against a real held-out example — training/evaluate.py's val.jsonl
has zero visual pairs in it, so those fields score 100% on N=0 and inflate
the headline accuracy number.

THE HONEST FINDING THIS SCRIPT SURFACES, NOT JUST FIXES:
    All 500 of data/training/visual_pairs.jsonl were merged into
    train_r3.jsonl and used in the actual Round-3 fine-tuning run — confirmed
    here by checking image_path presence, not assumed. That means there is
    no way to retroactively carve a clean, never-seen-by-the-model held-out
    set from the current 500 for the CURRENT checkpoint — any split of them
    now measures fit/memorization, not generalization.

WHAT THIS SCRIPT DOES:
    1. Confirms the leakage (every visual pair's product_id already appears
       in the training files) — prints it, doesn't hide it.
    2. Writes data/training/val_visual.jsonl: a deterministic 20% split
       (100 of 500, fixed seed) in the same {messages, image_path} shape
       training/evaluate.py already expects, so it's ready to use —
       explicitly labeled in its own header comment as "contaminated for
       the current checkpoint, clean for the next training round."
    3. The real fix is a Round 4 change, not something this script can do
       alone: exclude val_visual.jsonl's product_ids from train_r3's
       successor before the next fine-tune, then this split is a genuine
       held-out set from that point on.

Usage:
    venv/Scripts/python training/make_visual_eval_split.py
"""

import json
import random
from pathlib import Path

ROOT           = Path(__file__).parent.parent
VISUAL_PAIRS   = ROOT / "data" / "training" / "visual_pairs.jsonl"
TRAIN_FILE     = ROOT / "data" / "training" / "train_r3.jsonl"
OUTPUT_FILE    = ROOT / "data" / "training" / "val_visual.jsonl"

HOLDOUT_FRACTION = 0.20
SEED = 42


def _product_id(record: dict) -> str | None:
    for m in record["messages"]:
        if m["role"] == "assistant":
            try:
                return str(json.loads(m["content"]).get("product_id"))
            except (json.JSONDecodeError, KeyError):
                return None
    return None


def main():
    visual_pairs = [json.loads(l) for l in VISUAL_PAIRS.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(visual_pairs)} visual pairs from {VISUAL_PAIRS.name}")

    trained_with_image = sum(
        1 for l in TRAIN_FILE.read_text(encoding="utf-8").splitlines()
        if "image_path" in json.loads(l)
    )
    print(f"{trained_with_image} records with image_path found in {TRAIN_FILE.name} "
          f"(the file the actual Round-3 fine-tune loaded)")

    if trained_with_image >= len(visual_pairs):
        print(
            "\nLEAKAGE WARNING: all (or nearly all) visual pairs were already used in "
            "training. The split below is NOT a clean held-out set for the current "
            "checkpoint — only for a future round that explicitly excludes it first."
        )

    random.seed(SEED)
    shuffled  = visual_pairs[:]
    random.shuffle(shuffled)
    n_holdout = round(len(shuffled) * HOLDOUT_FRACTION)
    holdout   = shuffled[:n_holdout]

    OUTPUT_FILE.write_text(
        "\n".join(json.dumps(r) for r in holdout) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {len(holdout)} examples ({HOLDOUT_FRACTION:.0%}) to {OUTPUT_FILE}")
    print(
        "Next step for a genuinely clean number: before the next fine-tuning round, "
        "exclude these product_ids from the training file, retrain, then score against "
        "this file — training/evaluate.py's VISUAL_FIELDS scoring already knows how to "
        "read this shape, it just needs val.jsonl (or this file) to actually contain them."
    )


if __name__ == "__main__":
    main()
