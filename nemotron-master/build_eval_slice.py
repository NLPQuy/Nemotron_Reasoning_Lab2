"""Build a per-category held-out eval slice (Batch-3 prerequisite for exp23 + all A/B).

Samples up to N real (non-gen_*) rule_found problems per category, capped so no
category loses more than 25% of its solved instances to the slice. Writes:
  - eval_slice.jsonl     : {id, category, prompt, answer}  (answer = train.csv ground truth)
  - eval_slice_ids.txt   : held-out ids, one per line (pass to pack_kaggle_snapshot.py --exclude)

The slice is held OUT of the training snapshot (via --exclude at pack time) so that
measuring the model on it is leak-free. The "measure" step itself runs on GPU:
generate greedy predictions with infer_slice.py on Kaggle, then score with eval_slice.py.

Usage:
    uv run python3 build_eval_slice.py [--n 25] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

PROBLEMS_INDEX = Path(__file__).parent / "problems.jsonl"
TRAIN_CSV = Path(__file__).parent / "train.csv"
SLICE_JSONL = Path(__file__).parent / "eval_slice.jsonl"
SLICE_IDS = Path(__file__).parent / "eval_slice_ids.txt"

MAX_FRACTION = 0.25  # never hold out more than this share of a category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n", type=int, default=25, help="target held-out per category"
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import random

    rng = random.Random(args.seed)

    # rule_found, real (non-gen) problem ids per category
    by_cat: dict[str, list[str]] = defaultdict(list)
    for line in PROBLEMS_INDEX.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["id"].startswith("gen_"):
            continue
        if e.get("status") == "rule_found":
            by_cat[e["category"]].append(e["id"])

    # prompts + answers from train.csv (ground truth)
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    with open(TRAIN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            prompts[row["id"]] = row["prompt"]
            answers[row["id"]] = row["answer"]

    selected: list[dict[str, str]] = []
    per_cat_counts: dict[str, int] = {}
    for cat in sorted(by_cat):
        ids = [pid for pid in by_cat[cat] if pid in prompts]
        cap = min(args.n, max(1, int(len(ids) * MAX_FRACTION)))
        rng.shuffle(ids)
        chosen = ids[:cap]
        per_cat_counts[cat] = len(chosen)
        for pid in chosen:
            selected.append(
                {
                    "id": pid,
                    "category": cat,
                    "prompt": prompts[pid],
                    "answer": answers[pid],
                }
            )

    with open(SLICE_JSONL, "w") as f:
        for rec in selected:
            f.write(json.dumps(rec) + "\n")
    SLICE_IDS.write_text("\n".join(rec["id"] for rec in selected) + "\n")

    print(f"Wrote {len(selected)} held-out problems -> {SLICE_JSONL.name}")
    print(f"Held-out ids -> {SLICE_IDS.name}")
    for cat in sorted(per_cat_counts):
        print(f"  {cat:28} {per_cat_counts[cat]}")
    print()
    print("Next:")
    print("  1) re-pack training snapshot WITHOUT the slice:")
    print(
        "       uv run python3 pack_kaggle_snapshot.py --out kaggle_snapshot --exclude eval_slice_ids.txt"
    )
    print("  2) (on Kaggle) generate greedy predictions: infer_slice.py -> preds.jsonl")
    print("  3) score: uv run python3 eval_slice.py --preds preds.jsonl")


if __name__ == "__main__":
    main()
