"""Procedural in-distribution generator for the `cryptarithm_deduce` category.

Cryptarithm here is a 5-symbol puzzle: input `A0 A1 OP B0 B1`, where each symbol is a
hidden digit 0-9 and the middle operator symbol is one operation. The output encodes the
operation's result. Two families, both forward-constructed (we pick the rule, build the
examples, so the answer is known by construction — zero label noise, like exp20 gravity):

  - concat (exp28): output = A0A1B0B1 (fwd) or B0B1A0A1 (rev).
  - arithmetic (exp31, branch A): pick an injective symbol->digit map + an arithmetic op
    {add, abs_diff, mul}, compute on the two-digit operands, encode the result back to
    symbols. This covers the ~non-concat slice the production reasoner now solves.

The deterministic solver `reasoning_cryptarithm` is the source of truth: it is run on
each candidate and we keep the instance only when its boxed answer matches the
constructed answer (so the examples pin the rule unambiguously and the trace is exact).

This module only builds Problem objects + their prompt strings. Appending them is done
by ../generate_instances.py.
"""

from __future__ import annotations

import hashlib
import random

from reasoners.cryptarithm import reasoning_cryptarithm
from reasoners.store_types import Example, Problem
from reasoning import compare_answer, extract_answer

# Operand symbols measured from the real corpus (digits never appear).
SYMBOLS = list("!\"#$%&'()+/:<>?@[\\]^`{|}")
# Operator symbols sit at the middle position; keep them inside the same pool.
OPERATORS = list("*+-/?^&|")
N_EXAMPLES_CHOICES = (3, 4, 5, 5)

# exp31 branch A: arithmetic families (mirror reasoners.cryptarithm._ARITH_OPS 0/1/2).
ARITH_OPS = {
    "add": lambda a, b: a + b,
    "abs_diff": lambda a, b: abs(a - b),
    "mul": lambda a, b: a * b,
}
# Fraction of generated problems that use an arithmetic (non-concat) rule.
ARITH_FRACTION = 0.5


def _build_prompt(examples: list[Example], question: str) -> str:
    lines = [
        "In Alice's Wonderland, a secret set of transformation rules is applied to "
        "equations. Below are a few examples:"
    ]
    for ex in examples:
        lines.append(f"{ex.input_value} = {ex.output_value}")
    lines.append(f"Now, determine the result for: {question}")
    return "\n".join(lines)


def _make_input(rng: random.Random, op: str) -> str:
    chars = [rng.choice(SYMBOLS) for _ in range(4)]
    return chars[0] + chars[1] + op + chars[2] + chars[3]


def _concat(inp: str, kind: str) -> str:
    a0, a1, _op, b0, b1 = inp[0], inp[1], inp[2], inp[3], inp[4]
    return a0 + a1 + b0 + b1 if kind == "fwd" else b0 + b1 + a0 + a1


def _num_to_digits(n: int) -> tuple[int, ...]:
    if n == 0:
        return (0,)
    d = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


def _concat_candidate(rng: random.Random):
    """Forward-construct a concat problem -> (examples, q_input, true_answer)."""
    op = rng.choice(OPERATORS)
    kind = rng.choice(("fwd", "rev"))
    n_ex = rng.choice(N_EXAMPLES_CHOICES)
    seen: set[str] = set()
    inputs: list[str] = []
    while len(inputs) < n_ex + 1:
        inp = _make_input(rng, op)
        if inp in seen:
            continue
        seen.add(inp)
        inputs.append(inp)
    examples = [Example(inp, _concat(inp, kind)) for inp in inputs[:-1]]
    return examples, inputs[-1], _concat(inputs[-1], kind)


def _arith_candidate(rng: random.Random):
    """Forward-construct an arithmetic problem: pick an injective symbol->digit map
    + an op in {add, abs_diff, mul}, compute on the 2-digit operands, encode the
    result. Returns (examples, q_input, true_answer)."""
    op_name = rng.choice(list(ARITH_OPS))
    op_fn = ARITH_OPS[op_name]
    op_sym = rng.choice(OPERATORS)
    syms = rng.sample(SYMBOLS, 10)  # 10 distinct symbols -> every digit encodable
    d2s = {d: syms[d] for d in range(10)}
    n_ex = rng.choice(N_EXAMPLES_CHOICES)

    def encode(num: int) -> str:
        return "".join(d2s[x] for x in _num_to_digits(num))

    seen: set = set()
    items: list[tuple[str, str]] = []
    guard = 0
    while len(items) < n_ex + 1 and guard < 200:
        guard += 1
        a0, a1, b0, b1 = (rng.randrange(10) for _ in range(4))
        key = (a0, a1, b0, b1)
        if key in seen:
            continue
        seen.add(key)
        res = op_fn(a0 * 10 + a1, b0 * 10 + b1)
        inp = d2s[a0] + d2s[a1] + op_sym + d2s[b0] + d2s[b1]
        items.append((inp, encode(res)))
    if len(items) < n_ex + 1:
        return None
    examples = [Example(i, o) for i, o in items[:-1]]
    return examples, items[-1][0], items[-1][1]


def generate(n: int, seed: int = 28) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        use_arith = rng.random() < ARITH_FRACTION
        cand = _arith_candidate(rng) if use_arith else _concat_candidate(rng)
        if cand is None:
            continue
        examples, q_input, true_answer = cand

        pid = (
            "gen_" + hashlib.sha256(f"crypt_{seed}_{len(out)}".encode()).hexdigest()[:8]
        )
        prob = Problem(
            id=pid,
            category="cryptarithm_deduce",
            examples=examples,
            question=q_input,
            answer="",
        )
        trace = reasoning_cryptarithm(prob)  # solver = source of truth
        if trace is None:
            continue
        answer = extract_answer(trace)
        if not answer or not compare_answer(true_answer, answer):
            continue
        prob.answer = answer
        prob.prompt = _build_prompt(examples, q_input)
        out.append(prob)
    return out
