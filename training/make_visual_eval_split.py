"""# make_visual_eval_split.py - writes a fixed 20% hold-out of visual_pairs.jsonl for scoring the visual fields

- Input: data/training/visual_pairs.jsonl, train_r3.jsonl. Output: data/training/val_visual.jsonl ({messages, image_path}).
- Why: val.jsonl has no visual pairs, so packaging_type / packaging_color / has_brand_logo score 100% on N=0.
- Caveat: all 500 visual pairs were trained on in Round 3, so this split is only clean for a future round
  that excludes these products first. The script prints that leakage warning.
- Run: python training/make_visual_eval_split.py
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


# Warn about Round-3 leakage, then write the seeded 20% visual hold-out.
def main():
    visual_pairs = [json.loads(line) for line in VISUAL_PAIRS.read_text(encoding="utf-8").splitlines()]
    print(f"Loaded {len(visual_pairs)} visual pairs from {VISUAL_PAIRS.name}")

    trained_with_image = sum(
        1 for line in TRAIN_FILE.read_text(encoding="utf-8").splitlines()
        if "image_path" in json.loads(line)
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
