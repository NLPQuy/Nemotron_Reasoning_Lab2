"""Procedural in-distribution generator for the `cipher` category (exp28).

A random bijective substitution cipher (a-z permutation) is applied to plaintext
sentences drawn from the Wonderland vocabulary. Example sentences are chosen so their
letters cover every letter appearing in the question, so the deterministic solver
`reasoning_cipher` can decrypt directly from the learned mapping. The solver is the
source of truth: its boxed output is verified (case-insensitive) against the chosen
plaintext; uncovered/unsolved instances are skipped (zero label noise).

This module only builds Problem objects + their prompt strings. Appending them is done
by ../generate_instances.py.
"""

from __future__ import annotations

import hashlib
import random
import string

from reasoners.cipher import _load_wonderland, reasoning_cipher
from reasoners.store_types import Example, Problem
from reasoning import compare_answer, extract_answer

ALPHABET = string.ascii_lowercase
N_EXAMPLES_CHOICES = (4, 4, 5, 5)
Q_WORDS_CHOICES = (3, 4, 4, 5)
MAX_EXAMPLES = 8


def _make_cipher(rng: random.Random) -> dict[str, str]:
    shuffled = list(ALPHABET)
    rng.shuffle(shuffled)
    return dict(zip(ALPHABET, shuffled))


def _encrypt(text: str, plain_to_cipher: dict[str, str]) -> str:
    return "".join(plain_to_cipher.get(c, c) for c in text)


def _build_prompt(examples: list[Example], question: str) -> str:
    lines = [
        "In Alice's Wonderland, secret encryption rules are used on text. "
        "Here are some examples:"
    ]
    for ex in examples:
        lines.append(f"{ex.input_value} -> {ex.output_value}")
    lines.append(f"Now, decrypt the following text: {question}")
    return "\n".join(lines)


def generate(n: int, seed: int = 28) -> list[Problem]:
    rng = random.Random(seed)
    words = _load_wonderland()
    out: list[Problem] = []
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        plain_to_cipher = _make_cipher(rng)

        q_nwords = rng.choice(Q_WORDS_CHOICES)
        q_words = [rng.choice(words) for _ in range(q_nwords)]
        q_plain = " ".join(q_words)
        needed = set("".join(q_words))

        # Greedily pick example sentences covering every needed letter.
        n_ex = rng.choice(N_EXAMPLES_CHOICES)
        ex_plain: list[str] = []
        covered: set[str] = set()
        budget = MAX_EXAMPLES * 4
        while (covered < needed or len(ex_plain) < n_ex) and len(
            ex_plain
        ) < MAX_EXAMPLES:
            budget -= 1
            if budget < 0:
                break
            sent = " ".join(rng.choice(words) for _ in range(rng.choice((3, 4, 5))))
            ex_plain.append(sent)
            covered |= set(sent.replace(" ", ""))
        if not needed <= covered:
            continue

        examples = [Example(_encrypt(p, plain_to_cipher), p) for p in ex_plain]
        q_cipher = _encrypt(q_plain, plain_to_cipher)

        pid = (
            "gen_"
            + hashlib.sha256(f"cipher_{seed}_{len(out)}".encode()).hexdigest()[:8]
        )
        prob = Problem(
            id=pid,
            category="cipher",
            examples=examples,
            question=q_cipher,
            answer="",
        )
        trace = reasoning_cipher(prob)  # solver = source of truth
        if trace is None:
            continue
        answer = extract_answer(trace)
        if not answer or not compare_answer(q_plain, answer):
            continue
        prob.answer = answer
        prob.prompt = _build_prompt(examples, q_cipher)
        out.append(prob)
    return out
