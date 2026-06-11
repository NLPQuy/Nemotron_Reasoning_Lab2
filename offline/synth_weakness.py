#!/usr/bin/env python3
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
NEMOTRON_ROOT = ROOT / "nemotron-master"
for path in (ROOT, NEMOTRON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from offline.common.corpus_io import read_rollouts, write_rows  # noqa: E402
from offline.common.tokenize_format import (  # noqa: E402
    TOKEN_LIMIT,
    load_tokenizers,
    make_example,
)
from offline.common.verify import compare_answer, extract_answer  # noqa: E402
from offline.evolve_solved import (  # noqa: E402
    _rule,
    apply_bit_vector,
    build_bit_prompt,
    infer_bit_vector,
    random_bits,
)
from reasoning import GENERATORS  # noqa: E402
from reasoners.bit_manipulation import (  # noqa: E402
    N_BITS,
    PAIR_FAMILIES,
)
from reasoners.store_types import Example, Problem  # noqa: E402

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
BIT_FAMILIES = (
    "I",
    "NOT",
    "0",
    "1",
    "XOR",
    "OR",
    "AND",
    "AND-NOT",
    "XOR-NOT",
    "OR-NOT",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build weakness-targeted bit-manipulation synth rows."
    )
    p.add_argument("--rollouts", default="rollouts.jsonl")
    p.add_argument("--out", default="corpus/parts/I_weakness.jsonl")
    p.add_argument("--top_k_families", type=int, default=5)
    p.add_argument("--n_per_family", type=int, default=400)
    p.add_argument("--problems_dir", default=str(NEMOTRON_ROOT / "problems"))
    p.add_argument("--seed", type=int, default=4321)
    p.add_argument("--examples", type=int, default=8)
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument(
        "--tokenizer_json_path",
        default=str(NEMOTRON_ROOT / "tokenizer.json"),
    )
    return p.parse_args()


def load_problem(path: Path) -> Problem | None:
    if not path.exists():
        return None
    with path.open() as f:
        line = f.readline().strip()
    if not line:
        return None
    return Problem.from_payload(json.loads(line))


def is_wrong_bit_rollout(rec: dict) -> bool:
    if rec.get("category") != "bit_manipulation":
        return False
    if float(rec.get("reward", 0.0)) > 0.0:
        return False
    answer = str(rec.get("answer", ""))
    pred = str(rec.get("pred") or extract_answer(str(rec.get("text", ""))))
    return bool(answer) and not compare_answer(answer, pred)


def family_set(problem: Problem) -> set[str]:
    vector = infer_bit_vector(problem)
    if vector is None:
        return set()
    return {rule.family for rule in vector if rule.family in BIT_FAMILIES}


def count_wrong_families(
    rollouts_path: str | Path,
    problems_dir: Path,
    stats: Counter,
) -> Counter:
    counts: Counter = Counter()
    cache: dict[str, set[str]] = {}

    for rec in read_rollouts(ROOT / rollouts_path):
        if not is_wrong_bit_rollout(rec):
            continue
        pid = str(rec["problem_id"])
        stats["wrong_bit_rollouts"] += 1
        if pid not in cache:
            problem = load_problem(problems_dir / f"{pid}.jsonl")
            if problem is None:
                cache[pid] = set()
            else:
                cache[pid] = family_set(problem)
        families = cache[pid]
        if not families:
            stats["skipped_no_family"] += 1
            continue
        for family in families:
            counts[family] += 1
        stats["mapped_wrong_rollouts"] += 1

    return counts


def make_family_vector(family: str, rng: random.Random):
    if family not in BIT_FAMILIES:
        raise ValueError(f"Unsupported bit family: {family}")
    if family == "0":
        return [_rule("0", None, None) for _ in range(N_BITS)]
    if family == "1":
        return [_rule("1", None, None) for _ in range(N_BITS)]

    primary_offset = rng.randrange(N_BITS)
    if family in {"I", "NOT"}:
        return [
            _rule(family, (primary_offset + bit) % N_BITS, None)
            for bit in range(N_BITS)
        ]

    if family not in PAIR_FAMILIES:
        raise ValueError(f"Unsupported pair family: {family}")
    secondary_offset = rng.randrange(N_BITS)
    while secondary_offset == primary_offset:
        secondary_offset = rng.randrange(N_BITS)
    return [
        _rule(
            family,
            (primary_offset + bit) % N_BITS,
            (secondary_offset + bit) % N_BITS,
        )
        for bit in range(N_BITS)
    ]


def family_slug(family: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", family).strip("_").lower()


def make_family_problem(
    family: str,
    *,
    idx: int,
    n_examples: int,
    rng: random.Random,
) -> Problem:
    vector = make_family_vector(family, rng)
    seen: set[str] = set()
    examples: list[Example] = []
    while len(examples) < n_examples:
        bits = random_bits(rng)
        if bits in seen:
            continue
        seen.add(bits)
        examples.append(Example(bits, apply_bit_vector(bits, vector)))

    question = random_bits(rng)
    while question in seen:
        question = random_bits(rng)
    answer = apply_bit_vector(question, vector)
    return Problem(
        id=f"synthI_{family_slug(family)}_{idx:05d}",
        category="bit_manipulation",
        examples=examples,
        question=question,
        answer=answer,
        prompt=build_bit_prompt(examples, question),
    )


def solve_verified(problem: Problem) -> str | None:
    generator = GENERATORS.get(problem.category)
    if generator is None:
        return None
    reasoning = generator(problem)
    if reasoning is None:
        return None
    if not compare_answer(problem.answer, extract_answer(reasoning)):
        return None
    return reasoning


def iter_examples(args: argparse.Namespace, stats: Counter) -> Iterator[dict]:
    rng = random.Random(args.seed)
    problems_dir = Path(args.problems_dir)
    family_counts = count_wrong_families(args.rollouts, problems_dir, stats)
    stats["_family_counts"] = family_counts  # type: ignore[assignment]
    top_families = [
        family for family, _ in family_counts.most_common(args.top_k_families)
    ]
    stats["_top_families"] = top_families  # type: ignore[assignment]

    chat_tok, comp_tok = load_tokenizers(args.model_path, args.tokenizer_json_path)

    for family in top_families:
        kept = 0
        attempts = 0
        max_attempts = max(args.n_per_family * 50, 100)
        while kept < args.n_per_family and attempts < max_attempts:
            attempts += 1
            problem = make_family_problem(
                family,
                idx=kept,
                n_examples=args.examples,
                rng=rng,
            )
            stats[f"generated:{family}"] += 1
            reasoning = solve_verified(problem)
            if reasoning is None:
                stats[f"skipped_verify:{family}"] += 1
                continue

            row = make_example(
                problem.prompt,
                reasoning,
                problem.answer,
                chat_tok=chat_tok,
                comp_tok=comp_tok,
                problem_id=problem.id,
                category=problem.category,
                weight=1.0,
                sign=1.0,
            )
            if len(row["tokens"]) >= TOKEN_LIMIT:
                stats[f"skipped_len:{family}"] += 1
                continue
            stats["rows"] += 1
            stats[f"kept:{family}"] += 1
            kept += 1
            yield row

        if kept == 0:
            stats[f"fallback_needed:{family}"] += 1


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    write_rows(ROOT / args.out, iter_examples(args, stats))

    family_counts = stats.get("_family_counts", Counter())
    top_families = stats.get("_top_families", [])
    print(f"Wrote {stats['rows']} weakness-targeted rows to {args.out}")
    print(
        f"wrong_bit_rollouts={stats['wrong_bit_rollouts']} "
        f"mapped={stats['mapped_wrong_rollouts']} "
        f"top_k={args.top_k_families} n_per_family={args.n_per_family}"
    )
    print("Wrong rollout frequency per family:")
    for family, count in family_counts.most_common():
        marker = "*" if family in top_families else " "
        print(f" {marker} {family:8s} {count:8d}")
    print("Selected top families:")
    for family in top_families:
        print(
            f"  {family:8s} generated={stats[f'generated:{family}']:6d} "
            f"kept={stats[f'kept:{family}']:6d} verify=100.0%"
        )

    fallback = {
        key.removeprefix("fallback_needed:"): value
        for key, value in sorted(stats.items())
        if key.startswith("fallback_needed:")
    }
    if fallback:
        print(
            "Fallback needed: generator could not cover these families; use EXP-E uniform:"
        )
        for family, value in fallback.items():
            print(f"  {family}: {value}")

    skipped = {
        key: value for key, value in sorted(stats.items()) if key.startswith("skipped_")
    }
    if skipped:
        print("Skipped:")
        for key, value in skipped.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
