"""Procedural in-distribution generator for `equation_numeric_deduce` (exp28).

Input `a OP b` (two 2-digit numbers joined by a non-digit operator symbol) maps to the
result of a single hidden arithmetic operation keyed by that symbol. We pick a clean
operation that yields non-negative, symbol-free outputs (so the solver's prefix/suffix
machinery stays inert), build consistent examples, and let the deterministic solver
`reasoning_equation_numeric` rediscover the rule. The solver is the source of truth:
its boxed output becomes the answer, verified against the constructed output (zero label
noise; mismatches are skipped).

This module only builds Problem objects + their prompt strings. Appending them is done
by ../generate_instances.py.
"""

from __future__ import annotations

import hashlib
import random

from reasoners.equation_numeric import reasoning_equation_numeric
from reasoners.store_types import Example, Problem
from reasoning import compare_answer, extract_answer

# Operator symbols measured from the corpus, minus '-' (which drives the negative
# prefix/suffix path we deliberately avoid for clean in-distribution instances).
OPERATORS = list('+*/?^`"')
# Operations whose outputs are non-negative and digit-only for positive 2-digit inputs.
OPS = ("addition", "multiplication", "concatenation", "reverse concatenation")
N_EXAMPLES_CHOICES = (4, 5, 5)
LO, HI = 10, 99


def _apply(op: str, a: int, b: int) -> str:
    if op == "addition":
        return str(a + b)
    if op == "multiplication":
        return str(a * b)
    if op == "concatenation":
        return str(a) + str(b)
    if op == "reverse concatenation":
        return str(b) + str(a)
    raise ValueError(op)


def _build_prompt(examples: list[Example], question: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to "
        "equations. Below are a few examples:"
    ]
    for ex in examples:
        lines.append(f"{ex.input_value} = {ex.output_value}")
    lines.append(f"Now, determine the result for: {question}")
    return "\n".join(lines)


def generate(n: int, seed: int = 28) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        sym = rng.choice(OPERATORS)
        op = rng.choice(OPS)
        n_ex = rng.choice(N_EXAMPLES_CHOICES)

        seen: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int]] = []
        while len(pairs) < n_ex + 1:
            a, b = rng.randint(LO, HI), rng.randint(LO, HI)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append((a, b))

        ex_pairs, (qa, qb) = pairs[:-1], pairs[-1]
        examples = [Example(f"{a}{sym}{b}", _apply(op, a, b)) for a, b in ex_pairs]
        q_input = f"{qa}{sym}{qb}"
        true_answer = _apply(op, qa, qb)

        pid = (
            "gen_" + hashlib.sha256(f"eqnum_{seed}_{len(out)}".encode()).hexdigest()[:8]
        )
        prob = Problem(
            id=pid,
            category="equation_numeric_deduce",
            examples=examples,
            question=q_input,
            answer="",
        )
        trace = reasoning_equation_numeric(prob)  # solver = source of truth
        if trace is None:
            continue
        answer = extract_answer(trace)
        if not answer or not compare_answer(true_answer, answer):
            continue
        prob.answer = answer
        prob.prompt = _build_prompt(examples, q_input)
        out.append(prob)
    return out
