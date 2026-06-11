"""Procedural in-distribution generator for the `numeral` category (Batch-4 exp28).

Generates Arabic->Roman conversion problems with question integers sampled in the
measured support [1, 100]. The deterministic solver `reasoning_numeral` is the source
of truth: we run it on each generated Problem and take its boxed output as the answer,
so every instance is `rule_found` by construction (zero label noise). Examples are
distinct integers shown only for prompt realism (the solver ignores them).

This module only builds Problem objects + their prompt strings. Appending them to
problems/, problems.jsonl and train.csv is done by ../generate_instances.py.
"""

from __future__ import annotations

import hashlib
import random

from reasoners.numeral import _to_roman, reasoning_numeral
from reasoners.store_types import Example, Problem
from reasoning import extract_answer

Q_MIN, Q_MAX = 1, 100
N_EXAMPLES_CHOICES = (3, 3, 4, 4, 5)  # weighted toward the modes (3/4)


def _build_prompt(examples: list[Example], question: str) -> str:
    lines = [
        "In Alice's Wonderland, numbers are secretly converted into a different "
        "numeral system. Some examples are given below:"
    ]
    for ex in examples:
        lines.append(f"{ex.input_value} -> {ex.output_value}")
    lines.append(f"Now, write the number {question} in the Wonderland numeral system.")
    return "\n".join(lines)


def generate(n: int, seed: int = 28) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        n_ex = rng.choice(N_EXAMPLES_CHOICES)
        q_int = rng.randint(Q_MIN, Q_MAX)

        ex_ints: set[int] = set()
        while len(ex_ints) < n_ex:
            cand = rng.randint(Q_MIN, Q_MAX)
            if cand != q_int:
                ex_ints.add(cand)
        examples = [Example(str(i), _to_roman(i)) for i in sorted(ex_ints)]

        pid = (
            "gen_"
            + hashlib.sha256(f"numeral_{seed}_{len(out)}".encode()).hexdigest()[:8]
        )
        prob = Problem(
            id=pid,
            category="numeral",
            examples=examples,
            question=str(q_int),
            answer="",
        )
        trace = reasoning_numeral(prob)  # solver = source of truth
        if trace is None:
            continue
        answer = extract_answer(trace)
        if not answer:
            continue
        prob.answer = answer
        prob.prompt = _build_prompt(examples, str(q_int))
        out.append(prob)
    return out
