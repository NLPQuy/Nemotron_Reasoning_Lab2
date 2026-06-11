"""Score model predictions on the held-out eval slice (Batch-3).

Reuses the grader's own extract_answer + compare_answer so local per-category numbers
match how the competition grades (binary exact / float rel-tol 1e-2 / case-insensitive).

Inputs:
  --slice eval_slice.jsonl   : {id, category, prompt, answer}  (from build_eval_slice.py)
  --preds preds.jsonl        : {id, output}  raw model generations (from infer_slice.py).
                               "output" may be the full greedy text incl. \\boxed{...};
                               extract_answer pulls the final boxed value. Alternatively a
                               record may carry "prediction" already-extracted.

Outputs per-category and macro exact-match, plus how many predictions are missing /
have no boxed answer (a format-zero proxy).

Usage:
    uv run python3 eval_slice.py --preds preds.jsonl [--slice eval_slice.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from reasoning import compare_answer, extract_answer

SLICE_JSONL = Path(__file__).parent / "eval_slice.jsonl"


def _prediction(rec: dict) -> str:
    if "prediction" in rec and rec["prediction"] is not None:
        return str(rec["prediction"])
    return extract_answer(str(rec.get("output", "")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True)
    parser.add_argument("--slice", default=str(SLICE_JSONL))
    args = parser.parse_args()

    slice_recs = {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(args.slice).read_text().splitlines()
        if line.strip()
    }
    preds = {
        json.loads(line)["id"]: json.loads(line)
        for line in Path(args.preds).read_text().splitlines()
        if line.strip()
    }

    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    missing = 0
    no_box = 0

    for pid, rec in slice_recs.items():
        cat = rec["category"]
        total[cat] += 1
        if pid not in preds:
            missing += 1
            continue
        pred = _prediction(preds[pid])
        if not pred:
            no_box += 1
            continue
        if compare_answer(rec["answer"], pred):
            correct[cat] += 1

    cats = sorted(total)
    w = 30
    print(f"{'Category':<{w}} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * (w + 24))
    accs = []
    for cat in cats:
        acc = correct[cat] / total[cat] if total[cat] else 0.0
        accs.append(acc)
        print(f"{cat:<{w}} {correct[cat]:>8} {total[cat]:>6} {acc * 100:>7.1f}%")
    print("-" * (w + 24))
    micro = sum(correct.values()) / sum(total.values()) if total else 0.0
    macro = sum(accs) / len(accs) if accs else 0.0
    print(
        f"{'MICRO (overall)':<{w}} {sum(correct.values()):>8} {sum(total.values()):>6} {micro * 100:>7.1f}%"
    )
    print(f"{'MACRO (mean of categories)':<{w}} {'':>8} {'':>6} {macro * 100:>7.1f}%")
    print(f"\nmissing predictions: {missing}   no-boxed (format-zero): {no_box}")


if __name__ == "__main__":
    main()
