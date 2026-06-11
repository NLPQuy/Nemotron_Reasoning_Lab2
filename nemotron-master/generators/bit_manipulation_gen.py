"""Procedural in-distribution generator for the `bit_manipulation` category (exp28).

Each output bit i is a pairwise boolean op between two input bit positions that rotate
together with i: out[i] = OP(in[(p0+i)%8], in[(s0+i)%8]). This is exactly the
stride-1 rotation rule family the deterministic solver searches, so it can rediscover
the rule. The solver `reasoning_bit_manipulation` is the source of truth: its boxed
output is verified (exact binary match) against the constructed answer; mismatches and
unsolved instances are skipped (zero label noise).

This module only builds Problem objects + their prompt strings. Appending them is done
by ../generate_instances.py.
"""

from __future__ import annotations

import hashlib
import random

from reasoners.bit_manipulation import _evaluate_binary, reasoning_bit_manipulation
from reasoners.store_types import Example, Problem
from reasoning import compare_answer, extract_answer

N_BITS = 8
FAMILIES = ("XOR", "OR", "AND", "NAND", "NOR")
# bit_manipulation traces are intrinsically long (the solver enumerates a per-bit rule
# search). Fewer examples keep most completions under corpus.py's GLOBAL_LENGTH_CAP
# (7600) so they survive the length gate; 6-7 still disambiguates the rule.
N_EXAMPLES_CHOICES = (6, 6, 7)


def _rand_bits(rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(N_BITS))


def _apply_rule(bits: str, family: str, p0: int, s0: int) -> str:
    out = []
    for i in range(N_BITS):
        a = bits[(p0 + i) % N_BITS]
        b = bits[(s0 + i) % N_BITS]
        out.append(_evaluate_binary(a, b, family))
    return "".join(out)


def _build_prompt(examples: list[Example], question: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary "
        "numbers. The transformation involves operations like bit shifts, rotations, "
        "XOR, AND, OR, NOT, and possibly majority or choice functions.",
        "",
        "Here are some examples of input -> output:",
    ]
    for ex in examples:
        lines.append(f"{ex.input_value} -> {ex.output_value}")
    lines.append("")
    lines.append(f"Now, determine the output for: {question}")
    return "\n".join(lines)


def generate(n: int, seed: int = 28) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        family = rng.choice(FAMILIES)
        p0 = rng.randrange(N_BITS)
        s0 = rng.randrange(N_BITS)
        if p0 == s0:
            continue  # degenerate (would collapse to unary)

        n_ex = rng.choice(N_EXAMPLES_CHOICES)
        seen: set[str] = set()
        inputs: list[str] = []
        while len(inputs) < n_ex + 1:
            b = _rand_bits(rng)
            if b in seen:
                continue
            seen.add(b)
            inputs.append(b)

        ex_inputs, q_input = inputs[:-1], inputs[-1]
        examples = [Example(b, _apply_rule(b, family, p0, s0)) for b in ex_inputs]
        true_answer = _apply_rule(q_input, family, p0, s0)

        pid = "gen_" + hashlib.sha256(f"bit_{seed}_{len(out)}".encode()).hexdigest()[:8]
        prob = Problem(
            id=pid,
            category="bit_manipulation",
            examples=examples,
            question=q_input,
            answer="",
        )
        trace = reasoning_bit_manipulation(prob)  # solver = source of truth
        if trace is None:
            continue
        answer = extract_answer(trace)
        if not answer or not compare_answer(true_answer, answer):
            continue
        prob.answer = answer
        prob.prompt = _build_prompt(examples, q_input)
        out.append(prob)
    return out
