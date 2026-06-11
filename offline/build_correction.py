#!/usr/bin/env python3
import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.common.corpus_io import read_rollouts, write_rows  # noqa: E402
from offline.common.tokenize_format import make_example_from_ids  # noqa: E402
from offline.common.verify import compare_answer, extract_answer  # noqa: E402

DEFAULT_BRIDGE = (
    "\n\nWait - let me recheck. The result above is inconsistent; redoing it:\n\n"
)
COMPLETION_TAIL = "\n</think>\n\\boxed{{{answer}}}<|im_end|>"
BOXED_RE = re.compile(r"\\boxed\{([^}]*)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build self-correction traces: wrong attempt -> bridge -> gold fix."
    )
    p.add_argument("--rollouts", default="rollouts.jsonl")
    p.add_argument("--gold_corpus", default="corpus/parts/S_solver.jsonl")
    p.add_argument("--out", default="corpus/parts/G_correct.jsonl")
    p.add_argument(
        "--categories",
        default="cipher,equation_numeric_deduce",
        help="Comma-separated categories to use. Keep short traces only.",
    )
    p.add_argument("--max_len", type=int, default=7600)
    p.add_argument(
        "--tokenizer_json_path",
        default=str(ROOT / "nemotron-master" / "tokenizer.json"),
    )
    p.add_argument("--bridge", default=DEFAULT_BRIDGE)
    return p.parse_args()


def parse_categories(raw: str) -> set[str]:
    return {c.strip() for c in raw.split(",") if c.strip()}


def split_prompt_completion(row: dict) -> tuple[list[int], list[int]]:
    mask = list(row["mask"])
    tokens = list(row["tokens"])
    try:
        first_unmasked = mask.index(1)
    except ValueError as exc:
        raise ValueError(
            f"{row.get('problem_id', '')}: no unmasked gold tokens"
        ) from exc
    return tokens[:first_unmasked], tokens[first_unmasked:]


def strip_after_think(text: str) -> str:
    return text.split("\n</think>", 1)[0].rstrip()


def cut_before_final_boxed(text: str) -> str:
    text = strip_after_think(text)
    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return text.rstrip()
    non_empty = [m for m in matches if m.group(1).strip()]
    target = non_empty[-1] if non_empty else matches[-1]
    return text[: target.start()].rstrip()


def decode_ids(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)


def load_gold(
    path: str | Path,
    categories: set[str],
    tokenizer: Tokenizer,
) -> dict[str, dict]:
    gold: dict[str, dict] = {}
    for row in read_rollouts(ROOT / path):
        category = str(row.get("category", ""))
        if category not in categories:
            continue
        prompt_ids, completion_ids = split_prompt_completion(row)
        completion_text = decode_ids(tokenizer, completion_ids)
        reasoning = strip_after_think(completion_text)
        answer = extract_answer(reasoning) or extract_answer(completion_text)
        if not answer:
            continue
        gold[str(row["problem_id"])] = {
            "category": category,
            "prompt_ids": prompt_ids,
            "reasoning": reasoning.rstrip(),
            "answer": answer,
        }
    return gold


def is_wrong_rollout(rec: dict, gold_answer: str) -> bool:
    if float(rec.get("reward", 0.0)) > 0.0:
        return False
    pred = str(rec.get("pred") or extract_answer(str(rec.get("text", ""))))
    return not compare_answer(gold_answer, pred)


def make_correction_row(
    rec: dict,
    gold: dict,
    tokenizer: Tokenizer,
    *,
    bridge: str,
    max_len: int,
) -> tuple[dict, int, int]:
    wrong_text = str(
        rec.get("text") or decode_ids(tokenizer, rec["completion_token_ids"])
    )
    attempt = cut_before_final_boxed(wrong_text)
    if not attempt:
        raise ValueError("empty wrong attempt after boxed cut")

    answer = str(gold["answer"])
    fix = gold["reasoning"].rstrip()
    if not compare_answer(answer, extract_answer(fix)):
        fix = f"{fix}\n\\boxed{{{answer}}}"

    if not compare_answer(answer, extract_answer(fix)):
        raise ValueError("gold fix does not verify")

    learning_text = f"{bridge}{fix}{COMPLETION_TAIL.format(answer=answer)}"
    attempt_ids = tokenizer.encode(attempt, add_special_tokens=False).ids
    learning_ids = tokenizer.encode(learning_text, add_special_tokens=False).ids
    total_len = len(rec["prompt_token_ids"]) + len(attempt_ids) + len(learning_ids)
    if total_len >= max_len:
        raise ValueError("over_max_len")

    row = make_example_from_ids(
        list(rec["prompt_token_ids"]),
        attempt_ids + learning_ids,
        problem_id=str(rec["problem_id"]),
        category=str(rec.get("category", "")),
        weight=1.0,
        sign=1.0,
    )
    prompt_len = len(rec["prompt_token_ids"])
    row["mask"] = [0] * prompt_len + [0] * len(attempt_ids) + [1] * len(learning_ids)
    if len(row["tokens"]) != len(row["mask"]):
        raise AssertionError(f"{row['problem_id']}: len(tokens) != len(mask)")

    combined_reasoning = f"{attempt}{bridge}{fix}"
    if not compare_answer(answer, extract_answer(combined_reasoning)):
        raise ValueError("combined correction does not verify")
    return row, len(attempt_ids), len(learning_ids)


def iter_examples(args: argparse.Namespace, stats: Counter) -> Iterator[dict]:
    categories = parse_categories(args.categories)
    tokenizer = Tokenizer.from_file(args.tokenizer_json_path)
    gold_by_id = load_gold(args.gold_corpus, categories, tokenizer)
    emitted_ids: set[str] = set()
    spots: list[tuple[str, str, int, int, int, str]] = []

    for rec in read_rollouts(ROOT / args.rollouts):
        category = str(rec.get("category", ""))
        if category not in categories:
            continue
        stats[f"seen:{category}"] += 1
        pid = str(rec["problem_id"])
        if pid in emitted_ids:
            continue
        gold = gold_by_id.get(pid)
        if gold is None:
            stats[f"missing_gold:{category}"] += 1
            continue
        if not is_wrong_rollout(rec, str(gold["answer"])):
            stats[f"skipped_not_wrong:{category}"] += 1
            continue

        try:
            row, attempt_len, learn_len = make_correction_row(
                rec,
                gold,
                tokenizer,
                bridge=args.bridge,
                max_len=args.max_len,
            )
        except ValueError as exc:
            stats[f"skipped_build:{category}:{exc}"] += 1
            continue

        emitted_ids.add(pid)
        stats["rows"] += 1
        stats[f"cat:{category}"] += 1
        stats[f"learn_tokens:{category}"] += learn_len
        if len(spots) < 3:
            spots.append(
                (
                    pid,
                    category,
                    len(row["tokens"]),
                    attempt_len,
                    learn_len,
                    str(gold["answer"]),
                )
            )
        yield row

    stats["_spots"] = spots  # type: ignore[assignment]


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    write_rows(ROOT / args.out, iter_examples(args, stats))

    print(f"Wrote {stats['rows']} correction rows to {args.out}")
    print(f"max_len={args.max_len} bridge={args.bridge.strip()!r}")
    print("Correction rows per category:")
    for key, value in sorted(stats.items()):
        if key.startswith("cat:"):
            category = key[4:]
            avg_learn = stats[f"learn_tokens:{category}"] / value if value else 0.0
            print(f"  {category:28s} rows={value:6d} avg_learn_tokens={avg_learn:8.1f}")

    spots = stats.get("_spots", [])
    if spots:
        print("Spot-check rows:")
        for pid, category, total_len, attempt_len, learn_len, answer in spots:
            print(
                f"  {pid} {category}: len={total_len} "
                f"attempt_mask0={attempt_len} bridge_fix_mask1={learn_len} "
                f"final=\\boxed{{{answer}}}"
            )

    skipped = {
        key: value
        for key, value in sorted(stats.items())
        if key.startswith("skipped_") or key.startswith("missing_gold:")
    }
    if skipped:
        print("Skipped:")
        for key, value in skipped.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
