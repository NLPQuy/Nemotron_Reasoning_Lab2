"""Build a cryptarithm eval slice STRATIFIED by deterministic-solver solvability,
for the exp31 §5 falsification: is the model already accurate on the 5-op family
(=> branch A/B saturated) vs the hard unsolved rules (=> branch C headroom)?

Each problem is bucketed by what `reasoning_cryptarithm` does (verified vs the true
answer), and the bucket is written into the `category` field so eval_slice.py reports
per-bucket model accuracy unchanged:
  - cryptarithm_concat   : solver solves it via concatenation
  - cryptarithm_arith    : solver solves it via arithmetic (add/abs_diff/mul)
  - cryptarithm_unsolved : solver can't (rule outside the current op family)

Usage:
    uv run python3 build_cryptarithm_eval.py [--per_bucket 30] [--seed 7]
Writes cryptarithm_eval_slice.jsonl ({id, category=bucket, prompt, answer}).

CAVEAT: the 0.86 adapter trained on the huikang corpus, so solved-bucket numbers may
be partly in-sample (still informative for saturation). Real headroom = unsolved bucket.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from reasoners.cryptarithm import reasoning_cryptarithm
from reasoners.store_types import Problem
from reasoning import compare_answer, extract_answer

HERE = Path(__file__).parent
PROBLEMS_INDEX = HERE / "problems.jsonl"
TRAIN_CSV = HERE / "train.csv"
OUT = HERE / "cryptarithm_eval_slice.jsonl"


def _bucket(prob: Problem) -> str:
    trace = reasoning_cryptarithm(prob)
    if trace is None:
        return "cryptarithm_unsolved"
    pred = extract_answer(trace)
    if not pred or not compare_answer(prob.answer, pred):
        return "cryptarithm_unsolved"
    return "cryptarithm_arith" if "arithmetic operation" in trace else "cryptarithm_concat"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_bucket", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    prompts: dict[str, dict] = {}
    with open(TRAIN_CSV) as f:
        for row in csv.DictReader(f):
            prompts[row["id"]] = row

    crypt_ids = []
    with open(PROBLEMS_INDEX) as f:
        for line in f:
            p = json.loads(line)
            if p.get("category", "").startswith("cryptarithm") and not p["id"].startswith(
                "gen_"
            ):
                crypt_ids.append(p["id"])

    rng = random.Random(args.seed)
    rng.shuffle(crypt_ids)

    buckets: dict[str, list[dict]] = {
        "cryptarithm_concat": [],
        "cryptarithm_arith": [],
        "cryptarithm_unsolved": [],
    }
    need = args.per_bucket
    for pid in crypt_ids:
        if all(len(v) >= need for v in buckets.values()):
            break
        row = prompts.get(pid)
        if not row or not row.get("prompt") or not row.get("answer"):
            continue
        try:
            prob = Problem.load_from_json(pid)
        except Exception:
            continue
        b = _bucket(prob)
        if len(buckets[b]) >= need:
            continue
        buckets[b].append(
            {"id": pid, "category": b, "prompt": row["prompt"], "answer": row["answer"]}
        )

    with open(OUT, "w") as f:
        for b in buckets.values():
            for rec in b:
                f.write(json.dumps(rec) + "\n")

    total = sum(len(v) for v in buckets.values())
    print(f"Wrote {OUT} : {total} problems")
    for name, v in buckets.items():
        print(f"  {name}: {len(v)}")


if __name__ == "__main__":
    main()
