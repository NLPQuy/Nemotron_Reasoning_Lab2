#!/usr/bin/env python3
import argparse
import json
import random
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
    apply_bit_vector,
    build_bit_prompt,
    infer_bit_vector,
    random_bits,
)
from reasoning import GENERATORS  # noqa: E402
from reasoners.store_types import Example, Problem  # noqa: E402

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build short-tier bit-manipulation curriculum rows."
    )
    p.add_argument("--category", default="bit_manipulation")
    p.add_argument("--short_max_cols", type=int, default=6)
    p.add_argument("--variants", type=int, default=5)
    p.add_argument("--out", default="corpus/parts/H_lengthcur.jsonl")
    p.add_argument("--problems_jsonl", default=str(NEMOTRON_ROOT / "problems.jsonl"))
    p.add_argument("--problems_dir", default=str(NEMOTRON_ROOT / "problems"))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--min_cols", type=int, default=3)
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


def load_bit_seeds(args: argparse.Namespace) -> list[Problem]:
    seeds: list[Problem] = []
    problems_dir = Path(args.problems_dir)
    for rec in read_rollouts(args.problems_jsonl):
        if rec.get("category") != args.category:
            continue
        problem = load_problem(problems_dir / f"{rec['id']}.jsonl")
        if problem is not None:
            seeds.append(problem)
    return seeds


def make_short_problem(
    seed: Problem,
    *,
    variant_idx: int,
    n_cols: int,
    rng: random.Random,
) -> Problem | None:
    vector = infer_bit_vector(seed)
    if vector is None:
        return None

    for _attempt in range(100):
        seen: set[str] = set()
        examples: list[Example] = []
        while len(examples) < n_cols:
            bits = random_bits(rng)
            if bits in seen:
                continue
            seen.add(bits)
            examples.append(Example(bits, apply_bit_vector(bits, vector)))

        question = random_bits(rng)
        while question in seen:
            question = random_bits(rng)
        answer = apply_bit_vector(question, vector)
        problem = Problem(
            id=f"evolveH_{seed.id}_{variant_idx:02d}_{n_cols}c",
            category="bit_manipulation",
            examples=examples,
            question=question,
            answer=answer,
            prompt=build_bit_prompt(examples, question),
        )
        return problem
    return None


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
    if args.category != "bit_manipulation":
        raise ValueError(
            "build_length_curriculum currently supports bit_manipulation only"
        )
    if args.short_max_cols < args.min_cols:
        raise ValueError("--short_max_cols must be >= --min_cols")

    rng = random.Random(args.seed)
    seeds = load_bit_seeds(args)
    chat_tok, comp_tok = load_tokenizers(args.model_path, args.tokenizer_json_path)

    for seed in seeds:
        seed_cols = len(seed.examples)
        stats[f"seed_cols:{seed_cols}"] += 1
        if seed_cols > args.short_max_cols:
            stats[f"heldout_cols:{seed_cols}"] += 1
        else:
            stats[f"seed_train_cols:{seed_cols}"] += 1

        for variant_idx in range(args.variants):
            n_cols = rng.randint(args.min_cols, args.short_max_cols)
            stats[f"generated_cols:{n_cols}"] += 1
            problem = make_short_problem(
                seed,
                variant_idx=variant_idx,
                n_cols=n_cols,
                rng=rng,
            )
            stats["generated"] += int(problem is not None)
            if problem is None:
                stats["skipped_no_vector"] += 1
                continue

            reasoning = solve_verified(problem)
            if reasoning is None:
                stats[f"skipped_verify_cols:{n_cols}"] += 1
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
                stats[f"skipped_len_cols:{n_cols}"] += 1
                continue
            stats["rows"] += 1
            stats[f"kept_cols:{n_cols}"] += 1
            yield row


def print_count_table(title: str, stats: Counter, prefix: str) -> None:
    print(title)
    any_row = False
    for key, value in sorted(stats.items()):
        if key.startswith(prefix):
            any_row = True
            print(f"  {key.removeprefix(prefix):>4s}: {value}")
    if not any_row:
        print("  (none)")


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    write_rows(ROOT / args.out, iter_examples(args, stats))

    generated = stats["generated"]
    kept = stats["rows"]
    print(f"Wrote {kept} length-curriculum rows to {args.out}")
    print(
        f"category={args.category} short_max_cols={args.short_max_cols} "
        f"variants={args.variants} generated={generated} kept={kept}"
    )
    print("Solver-verify kept trace rate: 100.0%")
    if generated:
        print(f"Candidate keep rate after solver verify: {kept / generated:.1%}")

    print_count_table(
        "Original bit tiers by example-column height:", stats, "seed_cols:"
    )
    print_count_table("Train short tier generated:", stats, "generated_cols:")
    print_count_table("Train short tier kept:", stats, "kept_cols:")
    print_count_table("Held-out long tier (not emitted):", stats, "heldout_cols:")

    skipped = {
        key: value for key, value in sorted(stats.items()) if key.startswith("skipped_")
    }
    if skipped:
        print("Skipped:")
        for key, value in skipped.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
