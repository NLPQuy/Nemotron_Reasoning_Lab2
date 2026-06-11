"""Procedural in-distribution generator for the `unit_conversion` category (exp28).

Generates linear output = factor * input problems whose parameters are sampled within
the support measured from the real corpus (factor in [0.502, 2.00], inputs 2-dp in
[5.00, 40.00], 3-5 examples). The deterministic solver `reasoning_unit_conversion` is
the source of truth: we run it on each generated Problem and take its boxed output as
the answer, so every instance is `rule_found` by construction (zero label noise).

This module only builds Problem objects + their prompt strings. Appending them to
problems/, problems.jsonl and train.csv is done by ../generate_instances.py.
"""

from __future__ import annotations

import hashlib
import random

from reasoners.store_types import Example, Problem
from reasoners.unit_conversion import reasoning_unit_conversion
from reasoning import extract_answer

FACTOR_MIN, FACTOR_MAX = 0.502, 2.00
IN_MIN, IN_MAX = 5.00, 40.00
N_EXAMPLES_CHOICES = (3, 4, 5, 5)


def _in_value(rng: random.Random) -> str:
    return f"{rng.uniform(IN_MIN, IN_MAX):.2f}"


def _build_prompt(examples: list[Example], question: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret unit conversion is applied to measurements. "
        "For example:"
    ]
    for ex in examples:
        lines.append(f"{ex.input_value} m becomes {ex.output_value}")
    lines.append(f"Now, convert the following measurement: {question} m")
    return "\n".join(lines)


def generate(n: int, seed: int = 28) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        factor = rng.uniform(FACTOR_MIN, FACTOR_MAX)
        n_ex = rng.choice(N_EXAMPLES_CHOICES)

        in_strs: set[str] = set()
        while len(in_strs) < n_ex:
            in_strs.add(_in_value(rng))
        examples = [Example(s, f"{factor * float(s):.2f}") for s in sorted(in_strs)]

        q = _in_value(rng)
        while q in in_strs:
            q = _in_value(rng)

        pid = (
            "gen_" + hashlib.sha256(f"unit_{seed}_{len(out)}".encode()).hexdigest()[:8]
        )
        prob = Problem(
            id=pid,
            category="unit_conversion",
            examples=examples,
            question=q,
            answer="",
        )
        trace = reasoning_unit_conversion(prob)  # solver = source of truth
        if trace is None:
            continue
        answer = extract_answer(trace)
        if not answer:
            continue
        prob.answer = answer
        prob.prompt = _build_prompt(examples, q)
        out.append(prob)
    return out
