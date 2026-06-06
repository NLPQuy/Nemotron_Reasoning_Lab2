#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.common.corpus_io import read_rollouts  # noqa: E402
from offline.common.verify import compare_answer, extract_answer, format_ok  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure pass/yield from offline rollouts")
    p.add_argument("--rollouts", default="rollouts.jsonl")
    p.add_argument("--problems_jsonl", default="nemotron-master/problems.jsonl")
    p.add_argument("--output", default="yield_report.json")
    p.add_argument(
        "--stage",
        default="",
        help="Optional rollout stage filter, e.g. probe or targeted",
    )
    p.add_argument("--max_tokens", type=int, default=7680)
    p.add_argument("--trunc_epsilon", type=int, default=8)
    return p.parse_args()


def load_categories(path: str | Path) -> dict[str, str]:
    categories: dict[str, str] = {}
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            categories[str(rec["id"])] = str(rec["category"])
    return categories


def median_or_none(vals: list[float]) -> float | None:
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return None
    return float(statistics.median(finite))


def pct(n: int, d: int) -> float:
    return n / d if d else 0.0


def main() -> None:
    args = parse_args()
    categories = load_categories(args.problems_jsonl)

    by_problem: dict[str, list[dict]] = defaultdict(list)
    for rec in read_rollouts(args.rollouts):
        if args.stage and rec.get("stage") != args.stage:
            continue
        pid = str(rec["problem_id"])
        rec["category"] = str(rec.get("category") or categories.get(pid, ""))
        by_problem[pid].append(rec)

    cat_stats: dict[str, dict] = {}
    for pid, rows in sorted(by_problem.items()):
        category = rows[0].get("category") or categories.get(pid, "unknown")
        stats = cat_stats.setdefault(
            category,
            {
                "problems": 0,
                "traces": 0,
                "correct": 0,
                "passk_hits": 0,
                "wrong": 0,
                "truncation": 0,
                "format": 0,
                "arithmetic_slip": 0,
                "ppl_approx": [],
            },
        )
        stats["problems"] += 1
        stats["passk_hits"] += int(any(float(r.get("reward", 0.0)) > 0.0 for r in rows))

        for row in rows:
            reward = float(row.get("reward", 0.0))
            text = str(row.get("text", ""))
            completion_ids = row.get("completion_token_ids") or []
            answer = str(row.get("answer", ""))
            pred = str(row.get("pred") or extract_answer(text))
            ok_format = format_ok(text)
            is_correct = bool(reward > 0.0)
            if not is_correct and ok_format and answer:
                is_correct = compare_answer(answer, pred)

            stats["traces"] += 1
            stats["correct"] += int(is_correct)
            if row.get("ppl_approx") is not None:
                stats["ppl_approx"].append(float(row["ppl_approx"]))

            if is_correct:
                continue

            stats["wrong"] += 1
            near_limit = len(completion_ids) >= max(
                0, args.max_tokens - args.trunc_epsilon
            )
            if near_limit or not ok_format:
                stats["truncation"] += 1
            if not ok_format:
                stats["format"] += 1
            if ok_format:
                stats["arithmetic_slip"] += 1

    report = {"categories": {}, "zero_yield_categories": []}
    for category, stats in sorted(cat_stats.items()):
        traces = int(stats["traces"])
        problems = int(stats["problems"])
        wrong = int(stats["wrong"])
        row = {
            "problems": problems,
            "traces": traces,
            "pass@1": pct(int(stats["correct"]), traces),
            "pass@k": pct(int(stats["passk_hits"]), problems),
            "wrong": wrong,
            "truncation_rate_wrong": pct(int(stats["truncation"]), wrong),
            "format_rate_wrong": pct(int(stats["format"]), wrong),
            "arithmetic_slip_rate_wrong": pct(int(stats["arithmetic_slip"]), wrong),
            "ppl_approx_median": median_or_none(stats["ppl_approx"]),
        }
        report["categories"][category] = row
        if row["pass@k"] == 0.0:
            report["zero_yield_categories"].append(category)

    with Path(args.output).open("w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    header = (
        "category",
        "n",
        "pass@1",
        "pass@k",
        "%trunc",
        "%slip",
        "%format",
        "ppl_med",
    )
    print(
        f"{header[0]:28s} {header[1]:>6s} {header[2]:>8s} {header[3]:>8s} "
        f"{header[4]:>8s} {header[5]:>8s} {header[6]:>8s} {header[7]:>10s}"
    )
    for category, row in report["categories"].items():
        ppl_med = row["ppl_approx_median"]
        ppl_text = "n/a" if ppl_med is None else f"{ppl_med:.2f}"
        print(
            f"{category:28s} {row['problems']:6d} "
            f"{row['pass@1']:8.1%} {row['pass@k']:8.1%} "
            f"{row['truncation_rate_wrong']:8.1%} "
            f"{row['arithmetic_slip_rate_wrong']:8.1%} "
            f"{row['format_rate_wrong']:8.1%} {ppl_text:>10s}"
        )

    if report["zero_yield_categories"]:
        cats = ", ".join(report["zero_yield_categories"])
        print(f"Zero-yield categories: {cats}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
