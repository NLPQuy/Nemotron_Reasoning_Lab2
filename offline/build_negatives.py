#!/usr/bin/env python3
import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.common.corpus_io import read_rollouts, write_rows  # noqa: E402
from offline.common.tokenize_format import make_example_from_ids  # noqa: E402
from offline.common.verify import format_ok  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build REDI-style negative corpus rows from wrong rollout traces."
    )
    p.add_argument("--rollouts", default="rollouts.jsonl")
    p.add_argument("--out", default="corpus/parts/N_neg.jsonl")
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.8)
    p.add_argument("--max_per_cat", type=int, default=2000)
    p.add_argument("--neg_weighting", action="store_true")
    p.add_argument("--neg_weighting_gamma", type=float, default=1.0)
    return p.parse_args()


def normalized_old_logp(rec: dict) -> float:
    old_logp = rec.get("old_logp")
    if isinstance(old_logp, list) and old_logp:
        return float(sum(float(x) for x in old_logp) / len(old_logp))
    if old_logp is None:
        return 0.0
    denom = max(1, len(rec.get("completion_token_ids") or []))
    return float(old_logp) / denom


def negative_weight(rec: dict, args: argparse.Namespace) -> float:
    weight = float(args.lambda_)
    if args.neg_weighting:
        weight *= math.exp(float(args.neg_weighting_gamma) * normalized_old_logp(rec))
    return weight


def collect_candidates(
    args: argparse.Namespace, stats: Counter
) -> dict[str, list[dict]]:
    candidates: dict[str, list[dict]] = defaultdict(list)

    for rec in read_rollouts(ROOT / args.rollouts):
        category = str(rec.get("category", ""))
        reward = float(rec.get("reward", 0.0))
        if reward > 0.0:
            stats["skipped_positive_reward"] += 1
            continue
        if not format_ok(str(rec.get("text", ""))):
            stats[f"format_bad:{category}"] += 1
            continue

        stats[f"eligible:{category}"] += 1
        if len(candidates[category]) >= args.max_per_cat:
            stats[f"capped:{category}"] += 1
            continue
        candidates[category].append(rec)

    return candidates


def iter_examples(args: argparse.Namespace, stats: Counter) -> Iterator[dict]:
    candidates = collect_candidates(args, stats)
    categories_by_volume = sorted(
        candidates,
        key=lambda cat: (stats[f"eligible:{cat}"], cat),
        reverse=True,
    )

    for category in categories_by_volume:
        for rec in candidates[category]:
            row = make_example_from_ids(
                rec["prompt_token_ids"],
                rec["completion_token_ids"],
                problem_id=str(rec["problem_id"]),
                category=category,
                weight=negative_weight(rec, args),
                sign=-1.0,
            )
            if len(row["tokens"]) != len(row["mask"]):
                raise AssertionError(f"{row['problem_id']}: len(tokens) != len(mask)")
            if not any(row["mask"]):
                stats[f"skipped_no_unmasked:{category}"] += 1
                continue

            stats["rows"] += 1
            stats[f"cat:{category}"] += 1
            yield row


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    write_rows(ROOT / args.out, iter_examples(args, stats))

    print(f"Wrote {stats['rows']} negative rows to {args.out}")
    print(f"lambda={args.lambda_} neg_weighting={args.neg_weighting}")
    print("Negative rows per category:")
    for key, value in sorted(stats.items()):
        if key.startswith("cat:"):
            cat = key[4:]
            eligible = stats[f"eligible:{cat}"]
            capped = stats[f"capped:{cat}"]
            print(
                f"  {cat:28s} kept={value:6d} eligible={eligible:6d} capped={capped:6d}"
            )
    if stats["skipped_positive_reward"]:
        print(f"Skipped positive-reward traces: {stats['skipped_positive_reward']}")
    total_format_bad = sum(v for k, v in stats.items() if k.startswith("format_bad:"))
    if total_format_bad:
        print(f"Skipped format-bad wrong traces: {total_format_bad}")


if __name__ == "__main__":
    main()
