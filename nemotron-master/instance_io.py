"""Shared helpers for appending/clearing synthetic instances (Batch-3 exp20/24/25).

Every synthetic instance carries a prefixed id (gen_ procedural, para_ paraphrase,
rand_ surface) and is written to the same three places the pipeline reads:
  problems/<pid>.jsonl, problems.jsonl, train.csv. Each helper is prefix-scoped so the
three augmentation modes stay independent (clearing para_ never touches gen_).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from reasoners.store_types import Problem

PROBLEMS_DIR = Path(__file__).parent / "problems"
PROBLEMS_INDEX = Path(__file__).parent / "problems.jsonl"
REASONING_DIR = Path(__file__).parent / "reasoning"
TRAIN_CSV = Path(__file__).parent / "train.csv"


def clear_prefix(prefix: str) -> int:
    """Remove all <prefix>* instances from problems/, problems.jsonl, train.csv, reasoning/."""
    removed = 0
    for path in PROBLEMS_DIR.glob(f"{prefix}*.jsonl"):
        path.unlink()
        removed += 1
    for path in REASONING_DIR.glob(f"{prefix}*.txt"):
        path.unlink()

    if PROBLEMS_INDEX.exists():
        kept = [
            line
            for line in PROBLEMS_INDEX.read_text().splitlines()
            if line.strip() and not json.loads(line)["id"].startswith(prefix)
        ]
        PROBLEMS_INDEX.write_text("\n".join(kept) + "\n")

    if TRAIN_CSV.exists():
        with open(TRAIN_CSV, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or ["id", "prompt", "answer"]
            rows = [r for r in reader if not r["id"].startswith(prefix)]
        with open(TRAIN_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return removed


def append_problems(problems: list[Problem]) -> None:
    """Write each Problem to problems/<pid>.jsonl + problems.jsonl + train.csv."""
    for prob in problems:
        (PROBLEMS_DIR / f"{prob.id}.jsonl").write_text(json.dumps(prob.to_payload()))

    with open(PROBLEMS_INDEX, "a") as f:
        for prob in problems:
            f.write(
                json.dumps(
                    {
                        "id": prob.id,
                        "category": prob.category,
                        "status": "rule_unknown",
                        "submission": "",
                    }
                )
                + "\n"
            )

    with open(TRAIN_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "prompt", "answer"])
        for prob in problems:
            writer.writerow(
                {"id": prob.id, "prompt": prob.prompt, "answer": prob.answer}
            )


def load_real_problems(category: str, n: int, rng: random.Random) -> list[Problem]:
    """Sample up to n real (non-prefixed) rule_found problems of a category as Problem objects."""
    ids = [
        e["id"]
        for line in PROBLEMS_INDEX.read_text().splitlines()
        if line.strip()
        for e in [json.loads(line)]
        if e["category"] == category
        and e.get("status") == "rule_found"
        and e["id"][:1].isalnum()
        and not e["id"].startswith(("gen_", "para_", "rand_"))
    ]
    rng.shuffle(ids)
    out: list[Problem] = []
    for pid in ids[:n]:
        payload = json.loads((PROBLEMS_DIR / f"{pid}.jsonl").read_text())
        out.append(Problem.from_payload(payload))
    return out
