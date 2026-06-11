"""Build DPO preference pairs from eval-slice rollouts (Batch-4 exp29).

Reads the multi-sample rollouts from infer_slice.py (run with --n_samples > 1) and the
ground-truth eval slice, then for each problem classifies every sampled completion as
correct/incorrect via the grader's own compare_answer (the source of truth — we never
re-derive answers). A problem yields a pair only when it has BOTH a correct and an
incorrect sample; chosen = a clean correct completion, rejected = an incorrect one.

A held-out subset (default 30 ids) is reserved and NOT used to build pairs, so it can
measure generalisation after DPO. Output is TRL-DPO ready: {id, category, prompt_raw,
chosen, rejected} where prompt_raw is the natural-language prompt (the training script
applies the chat template, matching infer_slice.py / corpus.py).

Usage:
    uv run python3 build_dpo_pairs.py \
        --slice eval_slice.jsonl --preds preds_exp29.jsonl \
        --out dpo_pairs_exp29.jsonl --holdout_n 30 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from reasoning import compare_answer, extract_answer


def _normalize_pred(rec: dict) -> tuple[list[str], list[bool]]:
    """Return (texts, hit_cap_flags) from either the multi-sample or legacy schema."""
    if "outputs" in rec:
        texts = [str(t) for t in rec["outputs"]]
        hit = rec.get("hit_cap")
        if isinstance(hit, list) and len(hit) == len(texts):
            flags = [bool(h) for h in hit]
        else:
            flags = [False] * len(texts)
    else:  # legacy single-output record
        texts = [str(rec.get("output", ""))]
        flags = [bool(rec.get("hit_cap", False))]
    return texts, flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", default="eval_slice.jsonl")
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default="dpo_pairs_exp29.jsonl")
    parser.add_argument(
        "--holdout_n",
        type=int,
        default=30,
        help="ids reserved for post-DPO evaluation (never used to build pairs)",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max_pairs_per_problem",
        type=int,
        default=1,
        help="cap pairs emitted per problem to avoid skew toward easy categories",
    )
    parser.add_argument(
        "--min_pairs",
        type=int,
        default=20,
        help="abort with exit code 1 if fewer than this many pairs are built",
    )
    args = parser.parse_args()

    slice_recs = {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(args.slice).read_text().splitlines()
        if line.strip()
    }
    preds = {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(args.preds).read_text().splitlines()
        if line.strip()
    }

    # Deterministic holdout split: reserve holdout_n ids for evaluation only.
    rng = random.Random(args.seed)
    all_ids = sorted(slice_recs)
    rng.shuffle(all_ids)
    holdout_ids = set(all_ids[: args.holdout_n])
    build_ids = [pid for pid in all_ids if pid not in holdout_ids]

    holdout_path = Path(args.out).with_name("eval_holdout_ids.txt")
    holdout_path.write_text("\n".join(sorted(holdout_ids)) + "\n")

    pairs: list[dict] = []
    per_cat: dict[str, int] = defaultdict(int)
    skipped_no_split = 0
    skipped_no_preds = 0

    for pid in build_ids:
        rec = slice_recs[pid]
        if pid not in preds:
            skipped_no_preds += 1
            continue
        texts, hit_flags = _normalize_pred(preds[pid])
        answer = rec["answer"]

        correct: list[tuple[bool, bool, str]] = []  # (has_box, hit_cap, text)
        incorrect: list[tuple[bool, bool, str]] = []
        for text, hit in zip(texts, hit_flags):
            extracted = extract_answer(text)
            has_box = bool(extracted)
            if has_box and compare_answer(answer, extracted):
                correct.append((has_box, hit, text))
            else:
                incorrect.append((has_box, hit, text))

        if not correct or not incorrect:
            skipped_no_split += 1
            continue

        # chosen: prefer not-truncated, then shortest (cleanest correct trace).
        chosen_sorted = sorted(correct, key=lambda c: (c[1], len(c[2])))
        # rejected: prefer a wrong-but-boxed answer (teaches subtle distinction),
        # then not-truncated, then shortest.
        rejected_sorted = sorted(incorrect, key=lambda c: (not c[0], c[1], len(c[2])))

        n_pairs = min(
            args.max_pairs_per_problem, len(chosen_sorted), len(rejected_sorted)
        )
        for k in range(n_pairs):
            pairs.append(
                {
                    "id": pid,
                    "category": rec["category"],
                    "prompt_raw": rec["prompt"],
                    "chosen": chosen_sorted[k][2],
                    "rejected": rejected_sorted[k][2],
                }
            )
            per_cat[rec["category"]] += 1

    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Built {len(pairs)} DPO pairs -> {args.out}")
    print(f"Held out {len(holdout_ids)} ids for eval -> {holdout_path.name}")
    for cat in sorted(per_cat):
        print(f"  {cat:28} {per_cat[cat]}")
    print(
        f"skipped: {skipped_no_split} (no correct+incorrect split), "
        f"{skipped_no_preds} (no preds)"
    )

    if len(pairs) < args.min_pairs:
        print(
            f"\nABORT: only {len(pairs)} pairs (< --min_pairs={args.min_pairs}). "
            "Rollout pass-rate is unsuitable for DPO; do not train.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("\nNext: uv run python3 train_dpo_trl.py --pairs", args.out)


if __name__ == "__main__":
    main()
