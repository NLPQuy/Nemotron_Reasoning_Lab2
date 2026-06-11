"""Pack the regenerated corpus into the Kaggle snapshot layout the trainer reads.

The trainer (Continuer_Nemotron_Notebook.py / exp<N>.py, Kaggle branch) reads:
    CORPUS_PATH/<pid>/synthetic.json   -> a single JSON object {"tokens": [...], "mask": [...]}
    TRAIN_ORDER_PATH                   -> logprobs/index.jsonl, lines {"epoch": 0, "problem_id": pid}

But corpus.py writes corpus/<pid>/synthetic.jsonl as *segments* (interleaved
masked/unmasked) and corpus.jsonl as an index. This script converts that into the
snapshot layout so it can be uploaded as a Kaggle dataset.

Output layout (upload this whole directory as one Kaggle dataset):
    <out>/tokens/<pid>/synthetic.json     # {tokens, mask}
    <out>/logprobs/index.jsonl            # training order, epoch 0

Then in exp<N>.py point the override knob at the dataset root, e.g.
    EXP21_GATED_CORPUS = "/kaggle/input/<your-dataset>"
(the knob derives CORPUS_PATH=<root>/tokens and TRAIN_ORDER_PATH=<root>/logprobs/index.jsonl).

Usage:
    uv run python3 pack_kaggle_snapshot.py [--out kaggle_snapshot]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "corpus"
CORPUS_INDEX = Path(__file__).parent / "corpus.jsonl"
TOKEN_LIMIT = 8192


def reconstruct_tokens_mask(segment_path: Path) -> tuple[list[int], list[int]]:
    """Rebuild flat (tokens, mask) from a synthetic.jsonl segment file.

    mask = 1 for unmasked (completion, contributes to loss), 0 for masked (prompt).
    """
    segments = [
        json.loads(line) for line in segment_path.read_text().splitlines() if line
    ]
    segments.sort(key=lambda s: s["pos"])
    tokens: list[int] = []
    mask: list[int] = []
    for seg in segments:
        seg_tokens = seg["tokens"]
        tokens.extend(seg_tokens)
        mask.extend([1 if seg["type"] == "unmasked" else 0] * len(seg_tokens))
    return tokens, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="kaggle_snapshot",
        help="output snapshot root (upload this dir as a Kaggle dataset)",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="path to a file of problem ids (one per line) to hold OUT of the snapshot "
        "(e.g. eval_slice_ids.txt) so eval on them is leak-free",
    )
    args = parser.parse_args()

    if not CORPUS_INDEX.exists():
        print(f"No {CORPUS_INDEX}. Run reasoning.py then corpus.py first.")
        return

    exclude_ids: set[str] = set()
    if args.exclude:
        exclude_ids = {
            line.strip()
            for line in Path(args.exclude).read_text().splitlines()
            if line.strip()
        }

    out_root = Path(args.out)
    tokens_dir = out_root / "tokens"
    logprobs_dir = out_root / "logprobs"
    if out_root.exists():
        shutil.rmtree(out_root)
    tokens_dir.mkdir(parents=True)
    logprobs_dir.mkdir(parents=True)

    index_entries = [
        json.loads(line) for line in CORPUS_INDEX.read_text().splitlines() if line
    ]

    written = 0
    skipped_empty = 0
    excluded = 0
    order: list[str] = []
    max_seq = 0
    for entry in index_entries:
        if not entry.get("included", True):
            continue
        pid = entry["problem_id"]
        if pid in exclude_ids:
            excluded += 1
            continue
        seg_path = CORPUS_DIR / pid / "synthetic.jsonl"
        if not seg_path.exists():
            raise FileNotFoundError(f"missing segment file for {pid}: {seg_path}")

        tokens, mask = reconstruct_tokens_mask(seg_path)

        # Trainer-side invariants (mirror the load loop): drop empty / fully-masked.
        if not tokens or not any(mask):
            skipped_empty += 1
            continue
        assert len(tokens) == len(mask), f"{pid}: tokens/mask length mismatch"
        assert len(tokens) <= TOKEN_LIMIT, f"{pid}: {len(tokens)} > {TOKEN_LIMIT}"

        max_seq = max(max_seq, len(tokens))
        pid_dir = tokens_dir / pid
        pid_dir.mkdir(parents=True, exist_ok=True)
        with open(pid_dir / "synthetic.json", "w") as f:
            json.dump({"tokens": tokens, "mask": mask}, f)

        order.append(pid)
        written += 1

    # Training order, epoch 0 (trainer dedupes and replays epoch-0 entries).
    with open(logprobs_dir / "index.jsonl", "w") as f:
        for pid in order:
            f.write(json.dumps({"epoch": 0, "problem_id": pid}) + "\n")

    print(f"Packed snapshot -> {out_root}/")
    print(f"  tokens/<pid>/synthetic.json : {written} problems")
    print(f"  logprobs/index.jsonl        : {len(order)} ordered ids")
    if skipped_empty:
        print(f"  skipped (empty/fully-masked): {skipped_empty}")
    if exclude_ids:
        print(f"  held out (--exclude)        : {excluded}")
    print(f"  max seq length              : {max_seq} (<= {TOKEN_LIMIT})")
    print()
    print(
        "Next: upload the whole directory as a Kaggle dataset, then set in exp<N>.py:"
    )
    print('  EXP21_GATED_CORPUS = "/kaggle/input/<your-dataset-slug>"')


if __name__ == "__main__":
    main()
