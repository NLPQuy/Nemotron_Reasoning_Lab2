"""Count-substring augmenter: count occurrences of a character in a word.

Input:  banana a
Output: 3

Low-level string primitive (counting) used by cryptarithm/cipher frequency steps.
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
    # bias the target char toward one present in the word so counts are non-trivial
    target = (
        rng.choice(word) if rng.random() < 0.7 else rng.choice(string.ascii_lowercase)
    )
    return f"{word} {target}", str(word.count(target))


def generate() -> list[dict[str, str]]:
    """Generate count-substring problems. Returns list of dicts with id, prompt, completion, category."""
    rng = random.Random(221)
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
        pid = hashlib.sha256(f"count_substring_{i}".encode()).hexdigest()[:8]
        problems.append(
            {
                "id": pid,
                "prompt": prompt,
                "completion": answer,
                "category": "count_substring",
            }
        )

    print(f"[count_substring] Generated {len(problems)} problems")
    return problems
