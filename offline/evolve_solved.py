#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NEMOTRON_ROOT = ROOT / "nemotron-master"
for path in (ROOT, NEMOTRON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from offline.common.corpus_io import write_rows  # noqa: E402
from offline.common.tokenize_format import load_tokenizers, make_example  # noqa: E402
from offline.common.verify import compare_answer, extract_answer  # noqa: E402
from reasoning import GENERATORS  # noqa: E402
from reasoners.bit_manipulation import (  # noqa: E402
    N_BITS,
    PAIR_FAMILIES,
    RuleCandidate,
    _evaluate_rule,
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
PAIR_HARD_FAMILIES = ("XOR", "OR", "AND", "AND-NOT", "XOR-NOT", "OR-NOT")
EQUATION_OPS = ("addition", "absolute difference", "multiplication", "concatenation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evolve solver-verified problems into corpus rows"
    )
    p.add_argument("--problems_jsonl", default=str(NEMOTRON_ROOT / "problems.jsonl"))
    p.add_argument("--problems_dir", default=str(NEMOTRON_ROOT / "problems"))
    p.add_argument("--output", default=str(ROOT / "corpus_evolved.jsonl"))
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument(
        "--tokenizer_json_path", default=str(NEMOTRON_ROOT / "tokenizer.json")
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_seeds", type=int, default=0)
    p.add_argument("--variants_per_seed", type=int, default=3)
    p.add_argument(
        "--categories",
        default="bit_manipulation,equation_numeric_deduce,equation_numeric_guess",
    )
    p.add_argument("--ops", default="add_operand,longer_chain,nest_op")
    return p.parse_args()


def _rule(family: str, primary: int | None, secondary: int | None) -> RuleCandidate:
    if family == "0":
        return RuleCandidate("0", None, None, "C0")
    if family == "1":
        return RuleCandidate("1", None, None, "C1")
    if family == "I":
        assert primary is not None
        return RuleCandidate("I", primary, None, f"I{primary}")
    if family == "NOT":
        assert primary is not None
        return RuleCandidate("NOT", primary, None, f"NOT{primary}")
    assert family in PAIR_FAMILIES
    assert primary is not None and secondary is not None
    return RuleCandidate(family, primary, secondary, f"{family}{primary}{secondary}")


def _bit_string(value: str) -> str:
    bits = "".join(ch for ch in str(value) if ch in {"0", "1"})
    return bits if len(bits) == N_BITS else ""


def infer_bit_vector(seed: Problem) -> list[RuleCandidate] | None:
    inputs = [_bit_string(ex.input_value) for ex in seed.examples]
    outputs = [_bit_string(ex.output_value) for ex in seed.examples]
    if not inputs or any(not x for x in inputs) or any(not y for y in outputs):
        return None

    vector: list[RuleCandidate] = []
    for bit in range(N_BITS):
        target = [out[bit] for out in outputs]
        candidates: list[RuleCandidate] = []
        for family in BIT_FAMILIES:
            if family in {"0", "1"}:
                candidates.append(_rule(family, None, None))
            elif family in {"I", "NOT"}:
                for primary in range(N_BITS):
                    candidates.append(_rule(family, primary, None))
            else:
                for primary in range(N_BITS):
                    for secondary in range(N_BITS):
                        if primary == secondary:
                            continue
                        candidates.append(_rule(family, primary, secondary))
        match = None
        for candidate in candidates:
            values = [_evaluate_rule(bits, candidate) for bits in inputs]
            if values == target:
                match = candidate
                break
        if match is None:
            return None
        vector.append(match)
    return vector


def apply_bit_vector(bits: str, vector: list[RuleCandidate]) -> str:
    return "".join(_evaluate_rule(bits, rule) for rule in vector)


def mutate_vector(
    vector: list[RuleCandidate],
    op: str,
    rng: random.Random,
) -> list[RuleCandidate]:
    out: list[RuleCandidate] = []
    for bit, rule in enumerate(vector):
        if op == "add_operand":
            primary = rule.primary if rule.primary is not None else bit
            secondary = (primary + rng.choice([1, 2, 3, 5, 7])) % N_BITS
            family = rng.choice(("XOR", "OR", "AND"))
            out.append(_rule(family, primary, secondary))
        elif op == "longer_chain":
            primary = (bit + rng.choice([1, 3, 5])) % N_BITS
            secondary = (bit + rng.choice([2, 4, 7])) % N_BITS
            family = rng.choice(PAIR_HARD_FAMILIES)
            out.append(_rule(family, primary, secondary))
        elif op == "nest_op":
            primary = rule.primary if rule.primary is not None else (bit + 1) % N_BITS
            secondary = (primary + rng.choice([1, 2, 3, 5])) % N_BITS
            family = rng.choice(("AND-NOT", "XOR-NOT", "OR-NOT"))
            out.append(_rule(family, primary, secondary))
        else:
            out.append(rule)
    return out


def random_bits(rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(N_BITS))


def build_bit_prompt(examples: list[Example], question: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
        "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
        "and possibly majority or choice functions.",
        "",
        "Here are some examples of input -> output:",
    ]
    lines.extend(f"{ex.input_value} -> {ex.output_value}" for ex in examples)
    lines.extend(["", f"Now, determine the output for: {question}"])
    return "\n".join(lines)


def _solver_reproduces(problem: Problem) -> bool:
    generator = GENERATORS.get(problem.category)
    if generator is None:
        return False
    reasoning_text = generator(problem)
    if reasoning_text is None:
        return False
    return compare_answer(problem.answer, extract_answer(reasoning_text))


def evolve_bit_manipulation(
    seed: Problem,
    ops: list[str],
    rng: random.Random,
) -> list[Problem]:
    vector = infer_bit_vector(seed)
    if vector is None:
        return []

    evolved: list[Problem] = []
    for idx, op in enumerate(ops):
        new_vector = mutate_vector(vector, op, rng)
        for _attempt in range(50):
            seen: set[str] = set()
            examples: list[Example] = []
            while len(examples) < 8:
                bits = random_bits(rng)
                if bits in seen:
                    continue
                seen.add(bits)
                examples.append(Example(bits, apply_bit_vector(bits, new_vector)))

            question = random_bits(rng)
            while question in seen:
                question = random_bits(rng)
            answer = apply_bit_vector(question, new_vector)
            problem = Problem(
                id=f"{seed.id}__e{idx}_{op}",
                category="bit_manipulation",
                examples=examples,
                question=question,
                answer=answer,
                prompt=build_bit_prompt(examples, question),
            )
            if _solver_reproduces(problem):
                evolved.append(problem)
                break
    return evolved


def _equation_apply(op_name: str, a: int, b: int) -> str:
    if op_name == "addition":
        return str(a + b)
    if op_name == "absolute difference":
        return str(abs(a - b))
    if op_name == "multiplication":
        return str(a * b)
    if op_name == "concatenation":
        return f"{a}{b}"
    raise ValueError(op_name)


def build_equation_prompt(examples: list[Example], question: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:",
    ]
    lines.extend(f"{ex.input_value} = {ex.output_value}" for ex in examples)
    lines.append(f"Now, determine the result for: {question}")
    return "\n".join(lines)


def evolve_equation(seed: Problem, ops: list[str], rng: random.Random) -> list[Problem]:
    raw_ops = re.findall(
        r"\D", " ".join(ex.input_value for ex in seed.examples) + seed.question
    )
    op_char = raw_ops[0] if raw_ops else rng.choice(["+", "-", "*", "/"])
    evolved: list[Problem] = []
    for idx, _op in enumerate(ops):
        op_name = EQUATION_OPS[idx % len(EQUATION_OPS)]
        examples: list[Example] = []
        seen: set[str] = set()
        while len(examples) < 3:
            a = rng.randint(12, 99)
            b = rng.randint(11, 98)
            expr = f"{a}{op_char}{b}"
            if expr in seen:
                continue
            seen.add(expr)
            examples.append(Example(expr, _equation_apply(op_name, a, b)))

        qa = rng.randint(12, 99)
        qb = rng.randint(11, 98)
        question = f"{qa}{op_char}{qb}"
        answer = _equation_apply(op_name, qa, qb)
        category = (
            seed.category
            if seed.category.startswith("equation_numeric")
            else "equation_numeric_deduce"
        )
        evolved.append(
            Problem(
                id=f"{seed.id}__e{idx}_{op_name.replace(' ', '_')}",
                category=category,
                examples=examples,
                question=question,
                answer=answer,
                prompt=build_equation_prompt(examples, question),
            )
        )
    return evolved


def roundtrip_keep(problems: list[Problem]) -> list[tuple[Problem, str]]:
    kept: list[tuple[Problem, str]] = []
    for problem in problems:
        generator = GENERATORS.get(problem.category)
        if generator is None:
            continue
        reasoning_text = generator(problem)
        if reasoning_text is None:
            continue
        if compare_answer(problem.answer, extract_answer(reasoning_text)):
            kept.append((problem, reasoning_text))
    return kept


def load_seed(path: Path) -> Problem | None:
    if not path.exists():
        return None
    with path.open() as f:
        line = f.readline().strip()
    if not line:
        return None
    return Problem.from_payload(json.loads(line))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    categories = {c.strip() for c in args.categories.split(",") if c.strip()}
    ops = [op.strip() for op in args.ops.split(",") if op.strip()]

    seeds: list[dict[str, Any]] = []
    with Path(args.problems_jsonl).open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("category") in categories and rec.get("status") == "rule_found":
                seeds.append(rec)
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]

    chat_tok, comp_tok = load_tokenizers(args.model_path, args.tokenizer_json_path)
    problems_dir = Path(args.problems_dir)

    rows = []
    generated = 0
    kept = 0
    for seed_rec in seeds:
        seed = load_seed(problems_dir / f"{seed_rec['id']}.jsonl")
        if seed is None:
            continue
        seed_ops = ops[: args.variants_per_seed]
        if seed.category == "bit_manipulation":
            candidates = evolve_bit_manipulation(seed, seed_ops, rng)
        elif seed.category.startswith("equation_numeric"):
            candidates = evolve_equation(seed, seed_ops, rng)
        else:
            continue
        generated += len(candidates)
        for problem, reasoning_text in roundtrip_keep(candidates):
            rows.append(
                make_example(
                    problem.prompt,
                    reasoning_text,
                    problem.answer,
                    chat_tok=chat_tok,
                    comp_tok=comp_tok,
                    problem_id=problem.id,
                    category=problem.category,
                    weight=1.0,
                    sign=1.0,
                )
            )
            kept += 1

    write_rows(args.output, rows)
    print(f"seeds={len(seeds)} generated={generated} kept={kept} output={args.output}")


if __name__ == "__main__":
    main()
