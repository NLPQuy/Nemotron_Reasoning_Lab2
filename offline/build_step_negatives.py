#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.common.corpus_io import read_rollouts, write_rows  # noqa: E402
from offline.common.tokenize_format import make_example_from_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build step-localized REDI negatives using gold-divergence masks."
    )
    p.add_argument("--rollouts", default="rollouts.jsonl")
    p.add_argument("--gold_corpus", default="corpus/parts/S_solver.jsonl")
    p.add_argument("--out", default="corpus/parts/F_stepneg.jsonl")
    p.add_argument("--lambda_step", type=float, default=0.5)
    p.add_argument(
        "--categories",
        default="bit_manipulation,cipher,equation_numeric_deduce",
    )
    return p.parse_args()


def parse_categories(raw: str) -> set[str]:
    return {c.strip() for c in raw.split(",") if c.strip()}


def completion_from_corpus_row(row: dict) -> list[int]:
    return [int(tok) for tok, mask in zip(row["tokens"], row["mask"]) if mask]


def load_gold(path: str | Path, categories: set[str]) -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for row in read_rollouts(ROOT / path):
        category = str(row.get("category", ""))
        if category not in categories:
            continue
        gold[str(row["problem_id"])] = {
            "category": category,
            "completion_ids": completion_from_corpus_row(row),
        }
    return gold


def first_divergence(gold_ids: list[int], wrong_ids: list[int]) -> int:
    for idx, (gold_tok, wrong_tok) in enumerate(zip(gold_ids, wrong_ids)):
        if gold_tok != wrong_tok:
            return idx
    return min(len(gold_ids), len(wrong_ids))


def bucket_d(index: int) -> str:
    if index < 128:
        return "0000-0127"
    if index < 512:
        return "0128-0511"
    if index < 1024:
        return "0512-1023"
    if index < 2048:
        return "1024-2047"
    if index < 4096:
        return "2048-4095"
    return "4096+"


def make_step_negative(
    rec: dict,
    gold_completion: list[int],
    *,
    weight: float,
) -> tuple[dict | None, int]:
    wrong_completion = list(rec["completion_token_ids"])
    d = first_divergence(gold_completion, wrong_completion)
    row = make_example_from_ids(
        rec["prompt_token_ids"],
        wrong_completion,
        problem_id=str(rec["problem_id"]),
        category=str(rec.get("category", "")),
        weight=weight,
        sign=-1.0,
    )

    prompt_len = min(len(rec["prompt_token_ids"]), len(row["tokens"]))
    completion_len = len(row["tokens"]) - prompt_len
    d = min(d, completion_len)
    if d >= completion_len:
        return None, d

    row["mask"] = [0] * prompt_len + [0] * d + [1] * (completion_len - d)
    if len(row["tokens"]) != len(row["mask"]):
        raise AssertionError(f"{row['problem_id']}: len(tokens) != len(mask)")
    return row, d


def iter_examples(args: argparse.Namespace, stats: Counter) -> Iterator[dict]:
    categories = parse_categories(args.categories)
    gold = load_gold(args.gold_corpus, categories)
    emitted_ids: set[str] = set()

    for rec in read_rollouts(ROOT / args.rollouts):
        category = str(rec.get("category", ""))
        if category not in categories:
            continue
        pid = str(rec["problem_id"])
        if pid in emitted_ids:
            continue
        if pid not in gold:
            stats[f"missing_gold:{category}"] += 1
            continue
        if float(rec.get("reward", 0.0)) > 0.0:
            stats[f"skipped_positive:{category}"] += 1
            continue

        row, d = make_step_negative(
            rec,
            gold[pid]["completion_ids"],
            weight=float(args.lambda_step),
        )
        if row is None:
            stats[f"skipped_no_suffix:{category}"] += 1
            continue

        emitted_ids.add(pid)
        stats["rows"] += 1
        stats[f"cat:{category}"] += 1
        stats[f"d_bucket:{category}:{bucket_d(d)}"] += 1
        stats[f"d_sum:{category}"] += d
        yield row


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    write_rows(ROOT / args.out, iter_examples(args, stats))

    print(f"Wrote {stats['rows']} step-negative rows to {args.out}")
    print(f"lambda_step={args.lambda_step}")
    print("Step-negative rows per category:")
    for key, value in sorted(stats.items()):
        if key.startswith("cat:"):
            category = key[4:]
            avg_d = stats[f"d_sum:{category}"] / value if value else 0.0
            print(f"  {category:28s} rows={value:6d} avg_d={avg_d:8.1f}")
            for bucket_key, bucket_value in sorted(stats.items()):
                prefix = f"d_bucket:{category}:"
                if bucket_key.startswith(prefix):
                    print(f"    {bucket_key.removeprefix(prefix):>9s}: {bucket_value}")
    skipped = {
        key: value
        for key, value in sorted(stats.items())
        if key.startswith("skipped_") or key.startswith("missing_gold:")
    }
    if skipped:
        print("Skipped:")
        for key, value in skipped.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
