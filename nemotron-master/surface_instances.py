"""exp25 — surface randomization of solver-invariant features (Batch-3 Idea 6).

Domain-randomization idea: vary a surface feature the solver is provably invariant to,
so the model learns the rule rather than a fixed surface form. For gravity, the answer
is the MEDIAN of per-example k = d/t^2, which is invariant to example ORDER. We shuffle
the example lines (rebuilding the prompt + letting the solver re-derive the trace in the
new order) and keep the instance only if the solver re-verifies the SAME answer.

Restricted to gravity for now (confirmed order-invariant + we own its prompt template).
Extend per category only after confirming the chosen knob is truly solver-invariant.

Usage:
    uv run python3 surface_instances.py --category gravity --n 300 [--seed 25]
    uv run python3 surface_instances.py --clear
Then: uv run python3 reasoning.py && uv run python3 corpus.py
"""

from __future__ import annotations

import argparse
import hashlib
import random

from generators.gravity_gen import _build_prompt
from instance_io import append_problems, clear_prefix, load_real_problems
from reasoners.gravity import reasoning_gravity
from reasoners.store_types import Problem
from reasoning import compare_answer, extract_answer

PREFIX = "rand_"

INVARIANT_KNOB = {"gravity": "reorder example lines (median is order-invariant)"}


def generate(category: str, n: int, seed: int) -> list[Problem]:
    if category != "gravity":
        raise SystemExit(f"surface knob not defined/verified for category={category}")
    rng = random.Random(seed)
    src = load_real_problems(category, n * 2, rng)
    out: list[Problem] = []
    for prob in src:
        if len(out) >= n:
            break
        shuffled = prob.examples[:]
        rng.shuffle(shuffled)
        if [e.input_value for e in shuffled] == [e.input_value for e in prob.examples]:
            continue  # no actual reorder
        new = Problem(
            id="",
            category="gravity",
            examples=shuffled,
            question=prob.question,
            answer=prob.answer,
        )
        # Re-verify the solver still yields the SAME answer under the new order.
        trace = reasoning_gravity(new)
        if trace is None or not compare_answer(prob.answer, extract_answer(trace)):
            continue
        new.id = PREFIX + hashlib.sha256(f"{seed}_{prob.id}".encode()).hexdigest()[:8]
        new.prompt = _build_prompt(shuffled, prob.question)
        out.append(new)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="gravity")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    removed = clear_prefix(PREFIX)
    print(f"Cleared {removed} existing {PREFIX}* instances.")
    if args.clear:
        return
    if args.n <= 0:
        print("Nothing to generate (pass --n > 0).")
        return

    problems = generate(args.category, args.n, args.seed)
    append_problems(problems)
    print(
        f"Appended {len(problems)} surface-randomized {args.category} instances ({PREFIX}*)."
    )
    print("Next: uv run python3 reasoning.py && uv run python3 corpus.py")


if __name__ == "__main__":
    main()
