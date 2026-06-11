"""Digit-extract augmenter: keep only the digits of a mixed alphanumeric string.

Input:  a1b2c3
Output: 123

Low-level string primitive (digit filtering) used by numeral/equation parsing.
Masked auxiliary task, no \\boxed{}. Mirrors splitting.py structure.
"""

from __future__ import annotations

import hashlib
import random
import string

LINES_PER_PROBLEM = 100
N_PROBLEMS = 300
DEMO_LINES = 3


def _mixed(rng: random.Random) -> str:
    length = rng.randint(4, 9)
    alphabet = string.ascii_lowercase + string.digits
    s = "".join(rng.choice(alphabet) for _ in range(length))
    # guarantee at least one digit so the answer is non-empty
    if not any(c.isdigit() for c in s):
        pos = rng.randint(0, len(s) - 1)
        s = s[:pos] + rng.choice(string.digits) + s[pos + 1 :]
    return s


def _pair(rng: random.Random) -> tuple[str, str]:
    s = _mixed(rng)
    return s, "".join(c for c in s if c.isdigit())


def generate() -> list[dict[str, str]]:
    """Generate digit-extract problems. Returns list of dicts with id, prompt, completion, category."""
    rng = random.Random(223)
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
        pid = hashlib.sha256(f"digit_extract_{i}".encode()).hexdigest()[:8]
        problems.append(
            {
                "id": pid,
                "prompt": prompt,
                "completion": answer,
                "category": "digit_extract",
            }
        )

    print(f"[digit_extract] Generated {len(problems)} problems")
    return problems
