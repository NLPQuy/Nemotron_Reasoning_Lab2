"""Char-index augmenter: return the character at a 0-based index of a word.

Input:  banana 2
Output: n

Low-level string primitive (indexing) used by cipher position mappings.
Masked auxiliary task, no \\boxed{}. Mirrors splitting.py structure.
"""

from __future__ import annotations

import hashlib
import random
import string

LINES_PER_PROBLEM = 100
N_PROBLEMS = 300
DEMO_LINES = 3


def _word(rng: random.Random) -> str:
    length = rng.randint(3, 8)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _pair(rng: random.Random) -> tuple[str, str]:
    word = _word(rng)
    idx = rng.randint(0, len(word) - 1)
    return f"{word} {idx}", word[idx]


def generate() -> list[dict[str, str]]:
    """Generate char-index problems. Returns list of dicts with id, prompt, completion, category."""
    rng = random.Random(222)
    problems = []

    for i in range(N_PROBLEMS):
        demo_pairs = [_pair(rng) for _ in range(DEMO_LINES)]
        sample_input_lines = [f"{j:02d} {inp}" for j, (inp, _) in enumerate(demo_pairs)]
        sample_output_lines = [
            f"{j:02d} {inp} -> {out}" for j, (inp, out) in enumerate(demo_pairs)
        ]

        test_inputs = []
        test_answers = []
        for row_num in range(LINES_PER_PROBLEM):
            inp, out = _pair(rng)
            test_inputs.append(f"{row_num:02d} {inp}")
            test_answers.append(f"{row_num:02d} {inp} -> {out}")

        prompt = (
            "In Alice's Wonderland, secret processing rules are used on text.\n\n"
            "This is a sample input.\n"
            + "\n".join(sample_input_lines)
            + "\n\n"
            + "This is a sample output.\n"
            + "\n".join(sample_output_lines)
            + "\n\n"
            + "This is your input.\n"
            + "\n".join(test_inputs)
        )

        answer = "\n".join(test_answers)
        pid = hashlib.sha256(f"char_index_{i}".encode()).hexdigest()[:8]
        problems.append(
            {
                "id": pid,
                "prompt": prompt,
                "completion": answer,
                "category": "char_index",
            }
        )

    print(f"[char_index] Generated {len(problems)} problems")
    return problems
