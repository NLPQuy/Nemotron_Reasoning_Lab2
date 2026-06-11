#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.common.corpus_io import read_rollouts, write_rows  # noqa: E402
from offline.common.verify import compare_answer, extract_answer, format_ok  # noqa: E402

ROW_KEYS = ("problem_id", "category", "tokens", "mask", "weight", "sign")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge corpus parts into a trainer-ready JSONL corpus."
    )
    p.add_argument("--parts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_tokens_total", type=int, default=0)
    p.add_argument("--dedup_id", action="store_true")
    p.add_argument("--difficulty_weight", action="store_true")
    p.add_argument("--passcount_json", default="")
    p.add_argument("--dump_passcount_json", default="")
    p.add_argument("--rollouts", default="rollouts.jsonl")
    p.add_argument("--boundary_weight", type=float, default=1.5)
    p.add_argument("--mastered_weight", type=float, default=1.0)
    p.add_argument("--zero_weight", type=float, default=1.0)
    p.add_argument(
        "--drop_category_from_part",
        nargs="*",
        default=[],
        help="Drop categories from one part, e.g. S_solver:bit_manipulation",
    )
    return p.parse_args()


def resolve_part_path(raw: str) -> Path:
    path = Path(raw)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(ROOT / path)
        candidates.append(ROOT / "corpus" / "parts" / path)
        if path.suffix != ".jsonl":
            candidates.append(ROOT / "corpus" / "parts" / f"{raw}.jsonl")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve part path: {raw}")


def source_name(path: Path) -> str:
    stem = path.stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def parse_drop_rules(raw_rules: list[str]) -> dict[str, set[str]]:
    rules: dict[str, set[str]] = defaultdict(set)
    for raw in raw_rules:
        if ":" not in raw:
            raise ValueError(f"Invalid --drop_category_from_part value: {raw}")
        part, categories = raw.split(":", 1)
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid --drop_category_from_part value: {raw}")
        rules[part].update(c.strip() for c in categories.split(",") if c.strip())
    return dict(rules)


def is_correct_rollout(rec: dict) -> bool:
    reward = float(rec.get("reward", 0.0))
    if reward > 0.0:
        return True
    text = str(rec.get("text", ""))
    answer = str(rec.get("answer", ""))
    if not text or not answer or not format_ok(text):
        return False
    pred = str(rec.get("pred") or extract_answer(text))
    return compare_answer(answer, pred)


def dump_pass_counts(rollouts: str | Path, out_path: str | Path) -> dict[str, dict]:
    problems: dict[str, dict] = {}
    for rec in read_rollouts(ROOT / rollouts):
        pid = str(rec["problem_id"])
        item = problems.setdefault(
            pid,
            {
                "category": str(rec.get("category", "")),
                "pass_count": 0,
                "traces": 0,
            },
        )
        item["traces"] += 1
        item["pass_count"] += int(is_correct_rollout(rec))

    for item in problems.values():
        item["pass_count"] = min(int(item["pass_count"]), 8)

    out = ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"problems": problems}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote pass-counts for {len(problems)} ids to {out_path}")
    return problems


def load_pass_counts(path: str | Path) -> dict[str, int]:
    with (ROOT / path).open() as f:
        raw = json.load(f)
    problems = raw.get("problems", raw)

    pass_counts: dict[str, int] = {}
    for pid, value in problems.items():
        if isinstance(value, dict):
            pass_counts[str(pid)] = min(int(value.get("pass_count", 0)), 8)
        else:
            pass_counts[str(pid)] = min(int(value), 8)
    return pass_counts


def difficulty_multiplier(
    pid: str, pass_counts: dict[str, int], args: argparse.Namespace
) -> float:
    count = pass_counts.get(pid)
    if count is None:
        return 1.0
    if 1 <= count <= 7:
        return float(args.boundary_weight)
    if count >= 8:
        return float(args.mastered_weight)
    return float(args.zero_weight)


def clean_row(row: dict) -> dict:
    out = {key: row[key] for key in ROW_KEYS if key in row}
    missing = [key for key in ROW_KEYS if key not in out]
    if missing:
        raise KeyError(f"Corpus row missing keys: {missing}")

    tokens = out["tokens"]
    mask = out["mask"]
    if len(tokens) != len(mask):
        raise AssertionError(f"{out['problem_id']}: len(tokens) != len(mask)")
    out["weight"] = float(out.get("weight", 1.0))
    out["sign"] = float(out.get("sign", 1.0))
    return out


def iter_merged_rows(
    part_paths: list[Path],
    args: argparse.Namespace,
    pass_counts: dict[str, int],
    drop_rules: dict[str, set[str]],
    stats: dict,
) -> Iterator[dict]:
    protected_ids: set[str] = set()
    total_tokens = 0
    hit_cap = False

    for part_index, path in enumerate(part_paths):
        if hit_cap:
            break
        source = source_name(path)
        ids_this_part: set[str] = set()

        for raw_row in read_rollouts(path):
            row = clean_row(raw_row)
            pid = str(row["problem_id"])
            category = str(row["category"])
            stats["read_rows"][source] += 1

            drop_categories = drop_rules.get(path.stem, set()) | drop_rules.get(
                source, set()
            )
            if category in drop_categories:
                stats["skipped_category"][source] += 1
                continue

            ids_this_part.add(pid)

            if args.dedup_id and pid in protected_ids:
                stats["skipped_dedup"][source] += 1
                continue

            if args.difficulty_weight:
                row["weight"] *= difficulty_multiplier(pid, pass_counts, args)

            row_tokens = len(row["tokens"])
            if (
                args.max_tokens_total
                and total_tokens + row_tokens > args.max_tokens_total
            ):
                hit_cap = True
                stats["cap_part"] = source
                stats["cap_part_index"] = part_index
                break

            unmasked = int(sum(row["mask"]))
            total_tokens += row_tokens
            stats["emitted_rows"][source] += 1
            stats["total_tokens"] += row_tokens
            stats["total_unmasked"] += unmasked
            cat = stats["categories"][category]
            cat["rows"] += 1
            cat["tokens"] += row_tokens
            cat["unmasked"] += unmasked
            cat["sources"][source] += 1
            stats["pass_count_hist"][pass_counts.get(pid, -1)] += 1

            yield row

        if args.dedup_id:
            protected_ids.update(ids_this_part)


def print_summary(stats: dict) -> None:
    print(
        f"Emitted {sum(stats['emitted_rows'].values())} rows, "
        f"{stats['total_tokens']:,} tokens, "
        f"{stats['total_unmasked']:,} unmasked tokens"
    )
    if stats.get("cap_part"):
        print(f"Stopped at max_tokens_total while reading source {stats['cap_part']}")

    print("Source summary:")
    total_rows = sum(stats["emitted_rows"].values())
    total_unmasked = max(1, stats["total_unmasked"])
    for source in sorted(stats["read_rows"]):
        emitted = stats["emitted_rows"][source]
        skipped = stats["skipped_dedup"][source]
        skipped_category = stats["skipped_category"][source]
        pct = 100.0 * emitted / total_rows if total_rows else 0.0
        print(
            f"  {source:12s} read={stats['read_rows'][source]:6d} "
            f"emitted={emitted:6d} skipped_dedup={skipped:6d} "
            f"skipped_cat={skipped_category:6d} rows={pct:5.1f}%"
        )

    print("Category summary:")
    print(f"{'category':28s} {'rows':>6s} {'unmasked':>14s} {'%tokens':>8s} sources")
    for category, cat in sorted(stats["categories"].items()):
        pct = 100.0 * int(cat["unmasked"]) / total_unmasked
        source_text = ", ".join(
            f"{src}:{n} ({100.0 * n / cat['rows']:.1f}%)"
            for src, n in sorted(cat["sources"].items())
        )
        print(
            f"{category:28s} {cat['rows']:6d} {cat['unmasked']:14,d} "
            f"{pct:7.2f}% {source_text}"
        )

    if stats["pass_count_hist"]:
        print("Pass-count histogram for emitted rows (-1 = no P0 record):")
        for count, rows in sorted(stats["pass_count_hist"].items()):
            print(f"  {count:2d}: {rows}")


def main() -> None:
    args = parse_args()
    if args.dump_passcount_json:
        dump_pass_counts(args.rollouts, args.dump_passcount_json)
        if args.difficulty_weight and not args.passcount_json:
            args.passcount_json = args.dump_passcount_json

    pass_counts: dict[str, int] = {}
    if args.difficulty_weight:
        if not args.passcount_json:
            raise ValueError("--difficulty_weight requires --passcount_json")
        pass_counts = load_pass_counts(args.passcount_json)
        print(
            f"Loaded pass-counts for {len(pass_counts)} ids from {args.passcount_json}"
        )

    part_paths = [resolve_part_path(part) for part in args.parts]
    print("Parts:")
    for path in part_paths:
        print(f"  {source_name(path):12s} {path}")

    stats = {
        "read_rows": Counter(),
        "emitted_rows": Counter(),
        "skipped_dedup": Counter(),
        "skipped_category": Counter(),
        "categories": defaultdict(
            lambda: {"rows": 0, "tokens": 0, "unmasked": 0, "sources": Counter()}
        ),
        "pass_count_hist": Counter(),
        "total_tokens": 0,
        "total_unmasked": 0,
    }
    drop_rules = parse_drop_rules(args.drop_category_from_part)
    write_rows(
        ROOT / args.out,
        iter_merged_rows(part_paths, args, pass_counts, drop_rules, stats),
    )
    print(f"Wrote {args.out}")
    print_summary(stats)


if __name__ == "__main__":
    main()
