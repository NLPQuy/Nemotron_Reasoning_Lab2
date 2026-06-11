#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.common.corpus_io import write_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Repack nemotron-master/corpus.jsonl segment files into trainer JSONL rows."
    )
    p.add_argument("--corpus_jsonl", default="nemotron-master/corpus.jsonl")
    p.add_argument("--corpus_dir", default="nemotron-master/corpus")
    p.add_argument("--out", default="corpus/parts/S_solver.jsonl")
    return p.parse_args()


def read_segments(path: Path) -> tuple[list[int], list[int]]:
    segments: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))

    tokens: list[int] = []
    mask: list[int] = []
    for seg in sorted(segments, key=lambda s: int(s["pos"])):
        seg_type = seg["type"]
        seg_tokens = list(seg["tokens"])
        if seg_type == "masked":
            seg_mask = [0] * len(seg_tokens)
        elif seg_type == "unmasked":
            seg_mask = [1] * len(seg_tokens)
        else:
            raise ValueError(f"Unknown segment type {seg_type!r} in {path}")
        tokens.extend(seg_tokens)
        mask.extend(seg_mask)

    return tokens, mask


def iter_rows(args: argparse.Namespace, stats: Counter) -> Iterator[dict]:
    corpus_jsonl = ROOT / args.corpus_jsonl
    corpus_dir = ROOT / args.corpus_dir
    with corpus_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get("included", False):
                continue

            problem_id = str(rec["problem_id"])
            segment_name = str(rec.get("segment") or "synthetic.jsonl")
            segment_path = corpus_dir / problem_id / segment_name
            tokens, mask = read_segments(segment_path)

            expected = int(rec["token_count"])
            if len(tokens) != len(mask):
                raise AssertionError(f"{problem_id}: len(tokens) != len(mask)")
            if len(tokens) != expected:
                raise AssertionError(
                    f"{problem_id}: segment tokens {len(tokens)} != token_count {expected}"
                )
            if sum(mask) != int(rec.get("unmasked_token_count", sum(mask))):
                raise AssertionError(f"{problem_id}: unmasked_token_count mismatch")

            stats["rows"] += 1
            stats[f"cat:{rec['category']}"] += 1
            yield {
                "problem_id": problem_id,
                "category": rec["category"],
                "tokens": tokens,
                "mask": mask,
                "weight": 1.0,
                "sign": 1.0,
            }


def main() -> None:
    args = parse_args()
    stats: Counter = Counter()
    write_rows(ROOT / args.out, iter_rows(args, stats))

    print(f"Wrote {stats['rows']} rows to {args.out}")
    print("Rows per category:")
    for key, value in sorted(stats.items()):
        if key.startswith("cat:"):
            print(f"  {key[4:]:28s} {value:6d}")


if __name__ == "__main__":
    main()
