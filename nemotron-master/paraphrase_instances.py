"""exp24 — prompt-paraphrase augmentation (Batch-3 Idea 5).

For each sampled real problem, produce a para_* copy with the SAME structured fields
(examples / question / answer) but a reworded prompt string. Because the solver builds
the trace from the structured fields (not the prompt text) and the prompt is masked in
the corpus, the trace + answer are identical by construction (rule_found guaranteed);
only the phrasing the completion is conditioned on changes.

Paraphrase = deterministic, meaning-preserving phrase swaps (numbers/data untouched).
Problems where no swap applies are skipped (so every para_* truly differs from its source).

Usage:
    uv run python3 paraphrase_instances.py --category gravity --n 300 [--seed 24]
    uv run python3 paraphrase_instances.py --clear
Then: uv run python3 reasoning.py && uv run python3 corpus.py
"""

from __future__ import annotations

import argparse
import hashlib
import random

from instance_io import append_problems, clear_prefix, load_real_problems
from reasoners.store_types import Problem

PREFIX = "para_"

# Meaning-preserving swaps seen in the Wonderland prompts. Source must appear verbatim;
# none touch numbers or example lines.
SWAPS: list[tuple[str, str]] = [
    ("In Alice's Wonderland,", "In the land of Wonderland,"),
    ("Here are some example observations:", "Below are some example observations:"),
    (
        "Here are some examples of input -> output:",
        "Below are some examples of input -> output:",
    ),
    ("Now, determine the output for:", "Now, find the output for:"),
    ("Now, determine the falling distance", "Now, find the falling distance"),
    ("Now, determine", "Now, work out"),
    ("secretly changed", "secretly altered"),
    (
        "a secret bit manipulation rule transforms",
        "a hidden bit manipulation rule transforms",
    ),
    ("determine the", "find the"),
]


def paraphrase(prompt: str, rng: random.Random) -> str | None:
    """Apply a deterministic subset of meaning-preserving swaps; None if nothing applies."""
    applicable = [(a, b) for a, b in SWAPS if a in prompt]
    if not applicable:
        return None
    out = prompt
    changed = False
    for a, b in applicable:
        # apply each applicable swap with 0.7 probability, but guarantee >=1 change
        if rng.random() < 0.7:
            out = out.replace(a, b)
            changed = True
    if not changed:
        a, b = applicable[0]
        out = out.replace(a, b)
    return out


def generate(category: str, n: int, seed: int) -> list[Problem]:
    rng = random.Random(seed)
    src = load_real_problems(category, n * 2, rng)  # oversample; some may have no swap
    out: list[Problem] = []
    for prob in src:
        if len(out) >= n:
            break
        new_prompt = paraphrase(prob.prompt, rng)
        if new_prompt is None or new_prompt == prob.prompt:
            continue
        pid = (
            PREFIX
            + hashlib.sha256(f"{category}_{seed}_{prob.id}".encode()).hexdigest()[:8]
        )
        out.append(
            Problem(
                id=pid,
                category=prob.category,
                examples=prob.examples,
                question=prob.question,
                answer=prob.answer,
                prompt=new_prompt,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="gravity")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=24)
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
        f"Appended {len(problems)} paraphrased {args.category} instances ({PREFIX}*)."
    )
    print("Next: uv run python3 reasoning.py && uv run python3 corpus.py")


if __name__ == "__main__":
    main()
