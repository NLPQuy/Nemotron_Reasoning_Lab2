#!/usr/bin/env python3
"""
Generate base offline rollouts with vLLM.

Probe mode samples a cheap uniform G over train.csv. Targeted mode reads either
a JSON category schedule or yield_report.json, applies a token-budget cap, and
allocates more depth to bit/cipher/equation without changing prompts.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline.common.ppl import ppl_from_logp  # noqa: E402
from offline.common.verify import (  # noqa: E402
    PROMPT_SUFFIX,
    compare_answer,
    extract_answer,
    format_ok,
)
from offline.common.vllm_engine import (  # noqa: E402
    free as free_vllm,
    load_engine,
    sample as vllm_sample,
)

DEFAULT_TOKEN_BUDGET = 150_000_000
PREFERRED_CATEGORIES = {
    "bit_manipulation",
    "cipher",
    "equation_numeric_deduce",
    "equation_numeric_guess",
}
SATURATED_CATEGORIES = {"numeral", "gravity", "unit_conversion"}


def _sampled_token_logprob(logprob_row: object, token_id: int) -> float:
    if not logprob_row:
        return 0.0
    item = None
    if isinstance(logprob_row, dict):
        item = logprob_row.get(token_id) or logprob_row.get(str(token_id))
        if item is None and len(logprob_row) == 1:
            item = next(iter(logprob_row.values()))
    if item is None:
        return 0.0
    return float(getattr(item, "logprob", item))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate offline rollouts with vLLM")
    p.add_argument("--mode", choices=("probe", "targeted"), default="probe")
    p.add_argument(
        "--model_path",
        default="unsloth/Nemotron-3-Nano-30B-A3B",
        help="HF model id or local path",
    )
    p.add_argument(
        "--adapter_path",
        default=None,
        help="Path to LoRA adapter dir (must contain adapter_config.json)",
    )
    p.add_argument(
        "--train_csv",
        default="nemotron-master/train.csv",
        help="Path to train.csv (columns: id, prompt, answer)",
    )
    p.add_argument("--output", default="rollouts.jsonl", help="Output JSONL file")
    p.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Probe G, or fallback G when targeted has no schedule",
    )
    p.add_argument(
        "--g_schedule",
        default="",
        help="JSON category->G, or yield_report.json path for targeted mode",
    )
    p.add_argument("--token_budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--max_tokens", type=int, default=7680, help="Max new tokens per completion"
    )
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument(
        "--max_problems", type=int, default=0, help="Limit number of problems (0 = all)"
    )
    p.add_argument(
        "--only_category",
        default="",
        help="Filter to one category (needs problems.jsonl)",
    )
    p.add_argument(
        "--problems_jsonl",
        default=None,
        help="Optional problems.jsonl with category per id",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip problem_ids already present for the selected stage",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Submit problems in batches (0 = all at once)",
    )
    return p.parse_args()


def load_rows(args: argparse.Namespace) -> list[dict]:
    category_map: dict[str, str] = {}
    candidates = [
        c for c in [args.problems_jsonl, "nemotron-master/problems.jsonl"] if c
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                category_map[str(rec["id"])] = str(rec["category"])
        if category_map:
            print(f"Loaded {len(category_map)} categories from {path}")
            break

    rows = []
    with open(args.train_csv, newline="") as f:
        for row in csv.DictReader(f):
            row_id = str(row.get("id") or row.get("problem_id") or "")
            if not row_id:
                continue
            row["id"] = row_id
            row["category"] = row.get("category", "") or category_map.get(
                row_id, "unknown"
            )
            if args.only_category and row["category"] != args.only_category:
                continue
            rows.append(row)

    if args.max_problems:
        rows = rows[: args.max_problems]
    return rows


def get_done_ids(output_path: str, stage: str) -> set[str]:
    done: set[str] = set()
    if not os.path.isfile(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("stage") == stage:
                done.add(str(rec["problem_id"]))
    return done


def rollout_stats(path: str) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "tok": 0.0, "correct": 0.0}
    )
    if not os.path.isfile(path):
        return stats
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            category = str(rec.get("category") or "unknown")
            stats[category]["n"] += 1
            stats[category]["tok"] += len(rec.get("completion_token_ids") or [])
            stats[category]["correct"] += float(rec.get("reward", 0.0))
    return stats


def _schedule_from_pass_rate(pass_at_1: float, fallback_g: int) -> int:
    if pass_at_1 >= 0.85:
        return 2
    if pass_at_1 >= 0.15:
        return 32
    if pass_at_1 >= 0.02:
        return 64
    if pass_at_1 >= 0.0:
        return 4
    return fallback_g


def _default_category_g(category: str, fallback_g: int) -> int:
    if category in SATURATED_CATEGORIES:
        return 2
    if category == "bit_manipulation":
        return 64
    if category.startswith("equation_numeric") or category == "cipher":
        return 32
    if "cryptarithm" in category or category.endswith("_guess"):
        return 4
    return fallback_g


def _load_schedule_arg(path_or_json: str) -> dict[str, Any]:
    if not path_or_json:
        return {}
    if os.path.isfile(path_or_json):
        with open(path_or_json) as f:
            return json.load(f)
    return json.loads(path_or_json)


def build_g_schedule(
    rows: list[dict],
    args: argparse.Namespace,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("category") or "unknown")] += 1

    if args.mode == "probe":
        return {category: max(1, int(args.group_size)) for category in counts}

    raw_schedule = _load_schedule_arg(args.g_schedule)
    categories_obj = (
        raw_schedule.get("categories") if isinstance(raw_schedule, dict) else None
    )
    probe_stats = rollout_stats(args.output)
    schedule: dict[str, int] = {}

    for category in counts:
        value = raw_schedule.get(category) if isinstance(raw_schedule, dict) else None
        if isinstance(value, (int, float)):
            schedule[category] = max(1, int(value))
            continue
        if isinstance(value, dict):
            explicit = value.get("G") or value.get("g")
            if explicit is not None:
                schedule[category] = max(1, int(explicit))
                continue
            pass_at_1 = float(value.get("pass@1", -1.0))
            schedule[category] = _schedule_from_pass_rate(pass_at_1, args.group_size)
            continue
        if isinstance(categories_obj, dict) and category in categories_obj:
            pass_at_1 = float(categories_obj[category].get("pass@1", -1.0))
            schedule[category] = _schedule_from_pass_rate(pass_at_1, args.group_size)
            continue
        if probe_stats.get(category, {}).get("n", 0.0) > 0:
            pass_at_1 = probe_stats[category]["correct"] / probe_stats[category]["n"]
            schedule[category] = _schedule_from_pass_rate(pass_at_1, args.group_size)
            continue
        schedule[category] = _default_category_g(category, args.group_size)

    return cap_schedule_by_budget(schedule, counts, probe_stats, args)


def _avg_tokens(
    category: str,
    probe_stats: dict[str, dict[str, float]],
    max_tokens: int,
) -> float:
    stat = probe_stats.get(category)
    if stat and stat["n"] > 0:
        return max(1.0, stat["tok"] / stat["n"])
    return float(max_tokens)


def _schedule_cost(
    schedule: dict[str, int],
    counts: dict[str, int],
    probe_stats: dict[str, dict[str, float]],
    max_tokens: int,
) -> float:
    return sum(
        schedule[category]
        * counts[category]
        * _avg_tokens(category, probe_stats, max_tokens)
        for category in schedule
    )


def _min_g(category: str) -> int:
    if category in PREFERRED_CATEGORIES or category.startswith("equation_numeric"):
        return 8
    return 1


def _cut_rank(category: str) -> tuple[int, str]:
    if category in SATURATED_CATEGORIES:
        return (0, category)
    if "cryptarithm" in category or category.endswith("_guess"):
        return (1, category)
    if category in PREFERRED_CATEGORIES or category.startswith("equation_numeric"):
        return (3, category)
    return (2, category)


def cap_schedule_by_budget(
    schedule: dict[str, int],
    counts: dict[str, int],
    probe_stats: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> dict[str, int]:
    if args.token_budget <= 0:
        return schedule
    capped = dict(schedule)
    cost = _schedule_cost(capped, counts, probe_stats, args.max_tokens)
    if cost <= args.token_budget:
        return capped

    categories = sorted(capped, key=_cut_rank)
    changed = True
    while cost > args.token_budget and changed:
        changed = False
        for category in categories:
            if cost <= args.token_budget:
                break
            if capped[category] <= _min_g(category):
                continue
            capped[category] -= 1
            changed = True
            cost = _schedule_cost(capped, counts, probe_stats, args.max_tokens)
    return capped


def log_schedule(
    rows: list[dict],
    schedule: dict[str, int],
    args: argparse.Namespace,
) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("category") or "unknown")] += 1
    stats = rollout_stats(args.output)
    print("category                         n      G    avg_tok    est_tokens")
    total = 0.0
    for category in sorted(counts):
        avg_tok = _avg_tokens(category, stats, args.max_tokens)
        est = schedule[category] * counts[category] * avg_tok
        total += est
        print(
            f"{category:28s} {counts[category]:6d} {schedule[category]:6d} "
            f"{avg_tok:10.1f} {est:13.0f}"
        )
    print(f"estimated_completion_tokens={total:.0f} token_budget={args.token_budget}")


def _prompt_ids_for_rows(rows: list[dict], tokenizer: Any) -> list[list[int]]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"] + PROMPT_SUFFIX}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for row in rows
    ]


def _records_from_outputs(
    rows: list[dict], outputs: Any, stage: str
) -> tuple[list[dict], int]:
    records: list[dict] = []
    mixed_groups = 0
    for row, output in zip(rows, outputs):
        prompt_ids = list(output.prompt_token_ids)
        rewards: list[float] = []
        group_records: list[dict] = []

        for completion in output.outputs:
            completion_ids = list(completion.token_ids)
            text = completion.text
            pred = extract_answer(text)
            reward = (
                1.0 if format_ok(text) and compare_answer(row["answer"], pred) else 0.0
            )

            old_logp = [0.0] * len(prompt_ids)
            completion_logprobs = completion.logprobs or []
            for j, token_id in enumerate(completion_ids):
                logprob_row = (
                    completion_logprobs[j] if j < len(completion_logprobs) else None
                )
                old_logp.append(_sampled_token_logprob(logprob_row, token_id))

            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            group_records.append(
                {
                    "problem_id": row["id"],
                    "category": row.get("category", "rollout"),
                    "prompt_token_ids": prompt_ids,
                    "completion_token_ids": completion_ids,
                    "text": text,
                    "pred": pred,
                    "answer": row["answer"],
                    "reward": reward,
                    "old_logp": old_logp,
                    "ppl_approx": ppl_from_logp(old_logp, mask),
                    "stage": stage,
                }
            )
            rewards.append(reward)

        if len(set(rewards)) >= 2:
            mixed_groups += 1
        records.extend(group_records)
    return records, mixed_groups


def process_batch(
    rows: list[dict],
    tokenizer: Any,
    llm: Any,
    lora_request: Any | None,
    args: argparse.Namespace,
    fout: Any,
    g_schedule: dict[str, int],
) -> tuple[int, int]:
    rows_by_g: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        category = str(row.get("category") or "unknown")
        rows_by_g[max(1, int(g_schedule.get(category, args.group_size)))].append(row)

    n_written = 0
    n_mixed_groups = 0
    for g, group_rows in sorted(rows_by_g.items()):
        prompt_token_ids = _prompt_ids_for_rows(group_rows, tokenizer)
        outputs = vllm_sample(
            llm,
            prompt_token_ids,
            lora_request,
            n=g,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            logprobs=1,
        )
        records, mixed_groups = _records_from_outputs(group_rows, outputs, args.mode)
        for rec in records:
            fout.write(json.dumps(rec) + "\n")
        n_written += len(records)
        n_mixed_groups += mixed_groups

    fout.flush()
    return n_written, n_mixed_groups


def main() -> None:
    args = parse_args()

    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    rows = load_rows(args)
    print(f"Total problems loaded: {len(rows)}")

    if args.resume:
        done_ids = get_done_ids(args.output, args.mode)
        rows = [r for r in rows if r["id"] not in done_ids]
        print(
            f"Resume stage={args.mode}: {len(done_ids)} already done, {len(rows)} remaining"
        )

    if not rows:
        print("Nothing to generate.")
        return

    g_schedule = build_g_schedule(rows, args)
    log_schedule(rows, g_schedule, args)

    print(f"Loading tokenizer and vLLM engine from {args.model_path}...")
    llm, tokenizer, lora_request = load_engine(
        args.model_path,
        args.adapter_path,
        max_model_len=args.max_model_len,
    )
    if lora_request is None:
        print(f"No adapter at {args.adapter_path!r}, using base model")
    else:
        print(f"LoRA adapter loaded from {args.adapter_path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    mode = "a" if args.resume or args.mode == "targeted" else "w"
    batch_size = args.batch_size if args.batch_size > 0 else len(rows)

    total_written = 0
    total_mixed_groups = 0
    t0 = time.time()

    try:
        with open(args.output, mode) as fout:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                n_written, n_mixed = process_batch(
                    batch, tokenizer, llm, lora_request, args, fout, g_schedule
                )
                total_written += n_written
                total_mixed_groups += n_mixed

                done = start + len(batch)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(rows) - done) / rate if rate > 0 else 0.0
                mixed_rate = total_mixed_groups / done if done > 0 else 0.0
                print(
                    f"[{done}/{len(rows)}] stage={args.mode} "
                    f"written={total_written} mixed_groups={total_mixed_groups} "
                    f"mixed_rate={mixed_rate:.1%} "
                    f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
                )
    finally:
        free_vllm(llm)

    elapsed = time.time() - t0
    print(f"\nDone: {total_written} rollouts stage={args.mode}")
    print(f"Mixed-reward groups: {total_mixed_groups}")
    print(f"Total time: {elapsed / 3600:.2f}h")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
