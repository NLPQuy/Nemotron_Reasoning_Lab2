#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
import sys
from typing import Pattern

ROOT = Path(__file__).resolve().parents[1]
NEMOTRON_ROOT = ROOT / "nemotron-master"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.common.corpus_io import write_rows  # noqa: E402
from offline.common.tokenize_format import load_tokenizers, make_example  # noqa: E402
from offline.common.verify import compare_answer, extract_answer, format_ok  # noqa: E402

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_COMPRESSOR_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
PROTECT_PATTERNS = [
    re.compile(r"\\boxed"),
    re.compile(r"^Applying to "),
    re.compile(r"^Input$"),
    re.compile(r"^Output$"),
    re.compile(r"^\d+\s+[01]$"),
]

_COMPRESSOR = None
_COMPRESSOR_MODEL = DEFAULT_COMPRESSOR_MODEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compress bit_manipulation reasoning traces"
    )
    p.add_argument("--problems_jsonl", default=str(NEMOTRON_ROOT / "problems.jsonl"))
    p.add_argument("--train_csv", default=str(NEMOTRON_ROOT / "train.csv"))
    p.add_argument("--reasoning_dir", default=str(NEMOTRON_ROOT / "reasoning"))
    p.add_argument("--output", default=str(ROOT / "corpus_bit_compressed.jsonl"))
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument(
        "--tokenizer_json_path", default=str(NEMOTRON_ROOT / "tokenizer.json")
    )
    p.add_argument("--compressor_model", default=DEFAULT_COMPRESSOR_MODEL)
    p.add_argument("--ratios", default="0.7,0.6,0.5")
    p.add_argument("--max_problems", type=int, default=0)
    return p.parse_args()


def load_problems(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_train(path: str | Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            rows[str(row["id"])] = {"prompt": row["prompt"], "answer": row["answer"]}
    return rows


def get_compressor():
    global _COMPRESSOR
    if _COMPRESSOR is None:
        try:
            from llmlingua import PromptCompressor
        except ImportError as exc:
            raise SystemExit(
                "llmlingua is required for compress_traces.py; install it with uv add llmlingua."
            ) from exc
        _COMPRESSOR = PromptCompressor(
            model_name=_COMPRESSOR_MODEL,
            device_map="cpu",
            use_llmlingua2=True,
        )
    return _COMPRESSOR


def _line_protected(line: str, protect_patterns: list[Pattern[str]]) -> bool:
    return any(p.search(line) for p in protect_patterns)


def _restore_protected_lines(
    original: str,
    compressed: str,
    protect_patterns: list[Pattern[str]],
) -> str:
    compressed_lines = compressed.splitlines()
    compressed_set = set(compressed_lines)
    missing = [
        line
        for line in original.splitlines()
        if _line_protected(line, protect_patterns) and line not in compressed_set
    ]
    if not missing:
        return compressed
    return compressed.rstrip("\n") + "\n" + "\n".join(missing)


def compress_text(text: str, ratio: float, protect_patterns: list[Pattern[str]]) -> str:
    if ratio < 0.5:
        raise ValueError("compress_traces.py forbids ratios below 0.5")
    compressor = get_compressor()
    force_tokens = ["\\boxed", "Input", "Output", "Applying", "answer"]
    try:
        result = compressor.compress_prompt(
            text,
            rate=ratio,
            force_tokens=force_tokens,
            force_reserve_digit=True,
            drop_consecutive=True,
        )
    except TypeError:
        result = compressor.compress_prompt(text, rate=ratio)
    compressed = (
        result.get("compressed_prompt", result) if isinstance(result, dict) else result
    )
    return _restore_protected_lines(str(text), str(compressed), protect_patterns)


def main() -> None:
    args = parse_args()
    global _COMPRESSOR_MODEL
    _COMPRESSOR_MODEL = args.compressor_model

    ratios = [float(r.strip()) for r in args.ratios.split(",") if r.strip()]
    if any(r < 0.5 for r in ratios):
        raise SystemExit("All compression ratios must be >= 0.5")

    train_rows = load_train(args.train_csv)
    problems = [
        p
        for p in load_problems(args.problems_jsonl)
        if p.get("category") == "bit_manipulation"
    ]
    if args.max_problems:
        problems = problems[: args.max_problems]

    chat_tok, comp_tok = load_tokenizers(args.model_path, args.tokenizer_json_path)
    reasoning_dir = Path(args.reasoning_dir)

    out_rows = []
    kept = 0
    dropped = 0
    for problem in problems:
        pid = str(problem["id"])
        train_row = train_rows.get(pid)
        reasoning_path = reasoning_dir / f"{pid}.txt"
        if train_row is None or not reasoning_path.exists():
            dropped += 1
            continue

        answer = train_row["answer"]
        original = reasoning_path.read_text().rstrip("\n")
        compressed = ""
        for ratio in ratios:
            candidate = compress_text(original, ratio, PROTECT_PATTERNS)
            if format_ok(candidate) and compare_answer(
                answer, extract_answer(candidate)
            ):
                compressed = candidate
                break

        if not compressed:
            dropped += 1
            continue

        out_rows.append(
            make_example(
                train_row["prompt"],
                compressed,
                answer,
                chat_tok=chat_tok,
                comp_tok=comp_tok,
                problem_id=pid,
                category="bit_manipulation",
                weight=1.0,
                sign=1.0,
            )
        )
        kept += 1

    write_rows(args.output, out_rows)
    print(f"kept={kept} dropped={dropped} output={args.output}")


if __name__ == "__main__":
    main()
