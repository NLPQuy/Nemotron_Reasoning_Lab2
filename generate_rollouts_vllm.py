#!/usr/bin/env python3
"""
generate_rollouts_vllm.py — Sinh GSPO rollouts bằng vLLM trên RunPod/Modal.

Output rollouts.jsonl dùng trực tiếp với exp27.py:
    GSPO_ROLLOUTS=/path/to/rollouts.jsonl

Ước tính thời gian trên RTX PRO 6000 (95GB):
    9500 bài × G=8 × ~2000 token avg ≈ 152M tokens
    vLLM throughput ~8000-15000 tok/s → khoảng 3-5 giờ cho toàn bộ

Usage:
    python generate_rollouts_vllm.py \\
        --model_path unsloth/Nemotron-3-Nano-30B-A3B \\
        --adapter_path /path/to/adapter \\
        --train_csv train.csv \\
        --output rollouts.jsonl \\
        --group_size 8 \\
        --max_problems 0
"""

import argparse
import csv
import gc
import json
import os
import re
import time


# ── Helpers (copy từ exp27.py, không thay đổi) ──────────────────────────────

PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def extract_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def compare_answer(stored: str, pred: str) -> bool:
    stored = stored.strip()
    pred = pred.strip()
    if re.fullmatch(r"[01]+", stored):
        return pred.lower() == stored.lower()
    try:
        stored_num = float(stored)
        pred_num = float(pred)
        if stored_num == 0:
            return abs(pred_num) < 1e-2
        return abs(stored_num - pred_num) / abs(stored_num) <= 1e-2
    except ValueError:
        return pred.lower() == stored.lower()


def format_ok(text: str) -> bool:
    return len(re.findall(r"\\boxed\{", text)) >= 1


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


def _free_vllm(llm) -> None:
    import torch
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment
        destroy_distributed_environment()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate GSPO rollouts with vLLM")
    p.add_argument("--model_path", default="unsloth/Nemotron-3-Nano-30B-A3B",
                   help="HF model id or local path")
    p.add_argument("--adapter_path", default=None,
                   help="Path to LoRA adapter dir (must contain adapter_config.json)")
    p.add_argument("--train_csv", default="nemotron-master/train.csv",
                   help="Path to train.csv (columns: id, prompt, answer)")
    p.add_argument("--output", default="rollouts.jsonl",
                   help="Output JSONL file")
    p.add_argument("--group_size", type=int, default=8,
                   help="Number of completions per problem (G)")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=7680,
                   help="Max new tokens per completion")
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--max_problems", type=int, default=0,
                   help="Limit number of problems (0 = all)")
    p.add_argument("--only_category", default="",
                   help="Filter to one category (needs problems.jsonl)")
    p.add_argument("--problems_jsonl", default=None,
                   help="Optional problems.jsonl with category per id")
    p.add_argument("--resume", action="store_true",
                   help="Skip problem_ids already present in output file")
    p.add_argument("--batch_size", type=int, default=0,
                   help="Submit problems in batches (0 = all at once)")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def load_rows(args) -> list[dict]:
    # Optional: load category map
    category_map: dict[str, str] = {}
    candidates = [c for c in [args.problems_jsonl, "nemotron-master/problems.jsonl"] if c]
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
            row["category"] = row.get("category", "") or category_map.get(row_id, "")
            if args.only_category and row["category"] != args.only_category:
                continue
            rows.append(row)

    if args.max_problems:
        rows = rows[:args.max_problems]
    return rows


def get_done_ids(output_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.isfile(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            if line.strip():
                done.add(str(json.loads(line)["problem_id"]))
    return done


def process_batch(rows, tokenizer, llm, lora_request, args, fout) -> tuple[int, int]:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest  # noqa: F401

    sampling_params = SamplingParams(
        n=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=1,
    )

    prompt_token_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"] + PROMPT_SUFFIX}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for row in rows
    ]

    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    n_written = 0
    n_groups_kept = 0

    for row, output in zip(rows, outputs):
        prompt_ids = list(output.prompt_token_ids)
        records: list[dict] = []
        rewards: list[float] = []

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
                logprob_row = completion_logprobs[j] if j < len(completion_logprobs) else None
                old_logp.append(_sampled_token_logprob(logprob_row, token_id))

            tokens = prompt_ids + completion_ids
            mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
            records.append({
                "problem_id": row["id"],
                "category": row.get("category", "rollout"),
                "tokens": tokens,
                "mask": mask,
                "old_logp": old_logp,
                "reward": reward,
            })
            rewards.append(reward)

        if len(set(rewards)) < 2:
            continue  # pure correct or pure wrong — useless for GSPO

        n_groups_kept += 1
        for rec in records:
            fout.write(json.dumps(rec) + "\n")
            n_written += 1

    fout.flush()
    return n_written, n_groups_kept


def main():
    args = parse_args()

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load rows ────────────────────────────────────────────────────────────
    rows = load_rows(args)
    print(f"Total problems loaded: {len(rows)}")

    if args.resume:
        done_ids = get_done_ids(args.output)
        rows = [r for r in rows if r["id"] not in done_ids]
        print(f"Resume: {len(done_ids)} already done, {len(rows)} remaining")

    if not rows:
        print("Nothing to generate.")
        return

    # ── Load tokenizer + vLLM ───────────────────────────────────────────────
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading vLLM engine from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    lora_request = None
    if args.adapter_path and os.path.isfile(os.path.join(args.adapter_path, "adapter_config.json")):
        try:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest("gspo_adapter", 1, args.adapter_path)
            print(f"LoRA adapter loaded from {args.adapter_path}")
        except Exception as e:
            print(f"WARNING: Could not load LoRA adapter ({e}), using base model")
    else:
        print(f"No adapter at {args.adapter_path!r}, using base model")

    # ── Generate ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    mode = "a" if args.resume else "w"
    batch_size = args.batch_size if args.batch_size > 0 else len(rows)

    total_written = 0
    total_groups = 0
    t0 = time.time()

    try:
        with open(args.output, mode) as fout:
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                n_written, n_groups = process_batch(batch, tokenizer, llm, lora_request, args, fout)
                total_written += n_written
                total_groups += n_groups

                done = start + len(batch)
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(rows) - done) / rate if rate > 0 else 0
                mixed_rate = total_groups / done if done > 0 else 0
                print(
                    f"[{done}/{len(rows)}] groups_kept={total_groups} "
                    f"mixed_rate={mixed_rate:.1%} "
                    f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
                )
    finally:
        _free_vllm(llm)

    elapsed = time.time() - t0
    print(f"\nDone: {total_written} rollouts from {total_groups} mixed-reward groups")
    print(f"Total time: {elapsed/3600:.2f}h")
    print(f"Output: {args.output}")
    print(f"\nSet in exp27.py: GSPO_ROLLOUTS=\"{os.path.abspath(args.output)}\"")


if __name__ == "__main__":
    main()
