# >>> EXP18 START
"""Skeleton for offline EXP18 preference-pair generation.

TODO:
- Load prompts/problems and a trained adapter.
- Sample K candidate traces per problem offline.
- Verify candidates with nemotron-master reasoners.
- Write JSONL rows shaped as:
  {"problem_id": "...", "chosen": {"tokens": [...], "mask": [...]},
   "rejected": {"tokens": [...], "mask": [...]}}
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--problems", required=True)
    parser.add_argument("--out", default="pairs.jsonl")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    raise NotImplementedError(
        "EXP18 pair generation is an offline TODO: sample, verify, dedupe, "
        f"and write pairs to {args.out}."
    )


if __name__ == "__main__":
    main()
# <<< EXP18 END
