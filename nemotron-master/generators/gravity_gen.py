"""Procedural in-distribution generator for the `gravity` category (Batch-3 exp20).

Generates new d = 0.5*g*t^2 problems whose parameters are sampled within the support
measured from the real corpus (t in [1.00, 5.00] 2-dp, 3-5 examples, k=d/t^2 in
[2.45, 9.79]). The deterministic solver `reasoning_gravity` is the source of truth:
we run it on each generated Problem and take its boxed output as the answer, so every
instance is `rule_found` by construction (zero label noise). Problems the solver
cannot solve are skipped.

This module only builds Problem objects + their prompt strings. Appending them to
problems/, problems.jsonl and train.csv is done by ../generate_instances.py.
"""

from __future__ import annotations

import hashlib
import random

from reasoners.gravity import reasoning_gravity
from reasoners.store_types import Example, Problem
from reasoning import extract_answer

# Support measured from the 1597 real gravity problems.
T_MIN, T_MAX = 1.00, 5.00
K_MIN, K_MAX = 2.45, 9.79
N_EXAMPLES_CHOICES = (3, 4, 5, 5, 5)  # weighted toward the mode (5)


def _t_value(rng: random.Random) -> str:
    return f"{rng.uniform(T_MIN, T_MAX):.2f}"


def _build_prompt(examples: list[Example], question: str) -> str:
    lines = [
        "In Alice's Wonderland, the gravitational constant has been secretly changed. "
        "Here are some example observations:"
    ]
    for ex in examples:
        lines.append(f"For t = {ex.input_value}s, distance = {ex.output_value} m")
    lines.append(
        f"Now, determine the falling distance for t = {question}s given d = 0.5*g*t^2."
    )
    return "\n".join(lines)


def generate(n: int, seed: int = 20) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        k = rng.uniform(K_MIN, K_MAX)
        n_ex = rng.choice(N_EXAMPLES_CHOICES)

        t_strs: set[str] = set()
        while len(t_strs) < n_ex:
            t_strs.add(_t_value(rng))
        examples = [
            Example(t, f"{k * float(t) * float(t):.2f}") for t in sorted(t_strs)
        ]

        # question t distinct from the example t values
        q = _t_value(rng)
        while q in t_strs:
            q = _t_value(rng)

        pid = (
            "gen_"
            + hashlib.sha256(f"gravity_{seed}_{len(out)}".encode()).hexdigest()[:8]
        )
        prob = Problem(
            id=pid,
            category="gravity",
            examples=examples,
            question=q,
            answer="",
        )
        trace = reasoning_gravity(prob)  # solver = source of truth
        if trace is None:
            continue
        answer = extract_answer(trace)
        if not answer:
            continue
        prob.answer = answer
        prob.prompt = _build_prompt(examples, q)
        out.append(prob)
    return out
