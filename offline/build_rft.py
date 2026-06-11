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
from offline.common.dedup import dedup_keep_diverse  # noqa: E402
from offline.common.tokenize_format import make_example_from_ids  # noqa: E402
from offline.common.verify import compare_answer, extract_answer, format_ok  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build RFT positives from verifier-labeled rollout traces."
    )
    p.add_argument("--rollouts", default="rollouts.jsonl")
    p.add_argument("--out", default="corpus/parts/R_rft.jsonl")
    p.add_argument("--keep_per_id", type=int, default=2)
    p.add_argument("--partial_only", action="store_true")
    p.add_argument(
        "--categories",
        default="",
        help="Comma-separated category allowlist, e.g. bit_manipulation,cipher.",
    )
    p.add_argument("--reward_weight", action="store_true")
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=1.0)
    p.add_argument("--min_weight", type=float, default=0.5)
    p.add_argument("--max_weight", type=float, default=3.0)
    return p.parse_args()


def parse_categories(raw: str) -> set[str]:
    return {c.strip() for c in raw.split(",") if c.strip()}


def is_correct_trace(rec: dict) -> bool:
    text = str(rec.get("text", ""))
    if not format_ok(text):
        return False
    if float(rec.get("reward", 0.0)) <= 0.0:
        return False
    answer = str(rec.get("answer", ""))
    pred = str(rec.get("pred") or extract_answer(text))
    return compare_answer(answer, pred)


def logmeanexp(vals: list[float]) -> float:
    if not vals:
        return float("-inf")
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals) / len(vals))


def load_pass_counts(path: str | Path) -> tuple[dict[str, int], dict[str, list[float]]]:
    pass_counts: Counter[str] = Counter()
    rewards: dict[str, list[float]] = defaultdict(list)

    for rec in read_rollouts(ROOT / path):
        pid = str(rec["problem_id"])
        reward = 1.0 if is_correct_trace(rec) else 0.0
        pass_counts[pid] += int(reward > 0.0)
        rewards[pid].append(reward)

    return dict(pass_counts), rewards


def trace_weight(
    rewards: list[float],
    *,
    beta1: float,
    beta2: float,
    min_weight: float,
    max_weight: float,
) -> float:
    scaled = [r / beta1 for r in rewards]
    value = beta1 * logmeanexp(scaled)
    advantage = 1.0 - value
    weight = math.exp(advantage / beta2)
    return min(max(weight, min_weight), max_weight)


def iter_examples(args: argparse.Namespace, stats: Counter) -> Iterator[dict]:
    allowed_categories = parse_categories(args.categories)
    pass_counts, rewards_by_id = load_pass_counts(args.rollouts)
    candidates: dict[str, list[dict]] = defaultdict(list)

    zero_ids = sum(1 for c in pass_counts.values() if c == 0)
    partial_ids = sum(1 for c in pass_counts.values() if 1 <= c <= 7)
    all_correct_ids = sum(1 for c in pass_counts.values() if c >= 8)
    stats["passcount_zero_ids"] = zero_ids
    stats["passcount_partial_ids"] = partial_ids
    stats["passcount_all_correct_ids"] = all_correct_ids

    for rec in read_rollouts(ROOT / args.rollouts):
        category = str(rec.get("category", ""))
        if allowed_categories and category not in allowed_categories:
            continue

        pid = str(rec["problem_id"])
        pass_count = pass_counts.get(pid, 0)
        if args.partial_only or args.reward_weight:
            if not 1 <= pass_count <= 7:
                continue
        if not is_correct_trace(rec):
            continue

        candidates[pid].append(rec)

    for pid in sorted(candidates):
        rows = candidates[pid]
        kept = dedup_keep_diverse(
            rows,
            k=args.keep_per_id,
            key=lambda r: r["completion_token_ids"],
            thr=0.7,
        )
        weight = 1.0
        if args.reward_weight:
            weight = trace_weight(
                rewards_by_id[pid],
                beta1=args.beta1,
                beta2=args.beta2,
                min_weight=args.min_weight,
                max_weight=args.max_weight,
            )

        for rec in kept:
            text = str(rec.get("text", ""))
            if not text.rstrip().endswith("}"):
                stats["boxed_not_at_text_end"] += 1
            category = str(rec.get("category", ""))
            row = make_example_from_ids(
                rec["prompt_token_ids"],
                rec["completion_token_ids"],
                problem_id=pid,
                category=category,
                weight=weight,
                sign=1.0,
            )
            if len(row["tokens"]) != len(row["mask"]):
                raise AssertionError(f"{pid}: len(tokens) != len(mask)")
            if not any(row["mask"]):
                raise AssertionError(f"{pid}: row has no unmasked tokens")

            stats["rows"] += 1
            stats[f"cat:{category}"] += 1
            stats[f"id:{pid}"] = 1
            yield row


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    write_rows(ROOT / args.out, iter_examples(args, stats))

    n_ids = sum(1 for key in stats if key.startswith("id:"))
    print(f"Wrote {stats['rows']} rows / {n_ids} ids to {args.out}")
    print(
        "Pass-count ids: "
        f"zero={stats['passcount_zero_ids']} "
        f"partial={stats['passcount_partial_ids']} "
        f"all_correct={stats['passcount_all_correct_ids']}"
    )
    if stats["boxed_not_at_text_end"]:
        print(
            f"WARNING: {stats['boxed_not_at_text_end']} source texts did not end in '}}'"
        )
    print("Rows per category:")
    for key, value in sorted(stats.items()):
        if key.startswith("cat:"):
            print(f"  {key[4:]:28s} {value:6d}")


if __name__ == "__main__":
    main()
