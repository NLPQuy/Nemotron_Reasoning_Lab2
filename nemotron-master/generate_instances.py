"""Append procedurally-generated in-distribution instances (Batch-3 exp20).

For each target category, calls its generator in generators/, then writes each new
Problem to three places so the existing pipeline picks it up unchanged:
  - problems/<pid>.jsonl   (Problem payload, read by reasoning.py via load_from_json)
  - problems.jsonl         (index row; reasoning.py fills in status/submission)
  - train.csv              (id,prompt,answer; corpus.py reads prompt+answer here)

All generated ids are prefixed "gen_". The driver is idempotent: every run first
removes existing gen_* instances, so reruns do not accumulate. Use --clear to remove
them all and restore the baseline.

Usage:
    uv run python3 generate_instances.py --category gravity --n 400 [--seed 20]
    uv run python3 generate_instances.py --clear
Then regenerate + pack:
    uv run python3 reasoning.py
    uv run python3 corpus.py
    uv run python3 pack_kaggle_snapshot.py --out kaggle_snapshot_exp20
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from generators import (
    bit_manipulation_gen,
    cipher_gen,
    cryptarithm_gen,
    equation_numeric_gen,
    gravity_gen,
    numeral_gen,
    unit_conversion_gen,
)
from reasoners.store_types import Problem

PROBLEMS_DIR = Path(__file__).parent / "problems"
PROBLEMS_INDEX = Path(__file__).parent / "problems.jsonl"
REASONING_DIR = Path(__file__).parent / "reasoning"
TRAIN_CSV = Path(__file__).parent / "train.csv"

GEN_PREFIX = "gen_"

GENERATORS = {
    "gravity": gravity_gen.generate,
    "numeral": numeral_gen.generate,
    "unit_conversion": unit_conversion_gen.generate,
    "cryptarithm_deduce": cryptarithm_gen.generate,
    "equation_numeric_deduce": equation_numeric_gen.generate,
    "cipher": cipher_gen.generate,
    "bit_manipulation": bit_manipulation_gen.generate,
}


def clear_generated() -> int:
    """Remove all gen_* instances from problems/, problems.jsonl, train.csv, reasoning/."""
    removed = 0

    for path in PROBLEMS_DIR.glob(f"{GEN_PREFIX}*.jsonl"):
        path.unlink()
        removed += 1
    for path in REASONING_DIR.glob(f"{GEN_PREFIX}*.txt"):
        path.unlink()

    if PROBLEMS_INDEX.exists():
        kept = [
            line
            for line in PROBLEMS_INDEX.read_text().splitlines()
            if line.strip() and not json.loads(line)["id"].startswith(GEN_PREFIX)
        ]
        PROBLEMS_INDEX.write_text("\n".join(kept) + "\n")

    if TRAIN_CSV.exists():
        with open(TRAIN_CSV, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or ["id", "prompt", "answer"]
            rows = [r for r in reader if not r["id"].startswith(GEN_PREFIX)]
        with open(TRAIN_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return removed


def append_problems(problems: list[Problem]) -> None:
    for prob in problems:
        (PROBLEMS_DIR / f"{prob.id}.jsonl").write_text(json.dumps(prob.to_payload()))

    with open(PROBLEMS_INDEX, "a") as f:
        for prob in problems:
            f.write(
                json.dumps(
                    {
                        "id": prob.id,
                        "category": prob.category,
                        "status": "rule_unknown",
                        "submission": "",
                    }
                )
                + "\n"
            )

    with open(TRAIN_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
        for prob in problems:
            writer.writerow(
                {"id": prob.id, "prompt": prob.prompt, "answer": prob.answer}
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=sorted(GENERATORS) + ["all"])
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--clear", action="store_true", help="remove all gen_* instances and exit"
    )
    args = parser.parse_args()

    removed = clear_generated()
    print(f"Cleared {removed} existing gen_* instances.")
    if args.clear:
        return

    if not args.category or args.n <= 0:
        print("Nothing to generate (pass --category and --n > 0).")
        return

    # "all" generates --n instances for every registered category (clearing once above,
    # so categories accumulate instead of wiping each other).
    categories = sorted(GENERATORS) if args.category == "all" else [args.category]
    total = 0
    for cat in categories:
        problems = GENERATORS[cat](args.n, seed=args.seed)
        append_problems(problems)
        total += len(problems)
        print(f"Appended {len(problems)} {cat} instances (gen_*).")
    print(f"Total appended: {total}")
    print("Next: uv run python3 reasoning.py && uv run python3 corpus.py")


if __name__ == "__main__":
    main()
