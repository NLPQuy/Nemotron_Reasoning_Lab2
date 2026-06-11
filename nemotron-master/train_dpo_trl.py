"""DPO fine-tuning of the rank-32 adapter via HuggingFace TRL (Batch-4 exp29).

Continues training the EXISTING 0.86 LoRA adapter (so the artifact stays a single
rank-32 adapter — submission-valid) with TRL's DPOTrainer on the preference pairs from
build_dpo_pairs.py. The reference model is the adapter-disabled base (PEFT default when
ref_model=None); drift away from 0.86 is controlled by a small LR + few steps + the
post-train holdout gate, NOT by a frozen-0.86 reference (loading a 2nd 30B copy would
exceed GPU memory). See research/ideation/plan-exp29.md §2.

RUN ON GPU (RTX PRO 6000 / Kaggle competition image with trl+peft+vllm for Nemotron-H).
Not runnable on a typical local box. Reference: enhance_cot/redi/experiments_trl/open_r1_dpo.py.

Usage:
    # smoke first (1 step, 2 pairs) to verify Nemotron-H × TRL compatibility:
    uv run python3 train_dpo_trl.py --base <base> --adapter <0.86-dir> \
        --pairs dpo_pairs_exp29.jsonl --out adapter_exp29_smoke --smoke
    # real run:
    uv run python3 train_dpo_trl.py --base <base> --adapter <0.86-dir> \
        --pairs dpo_pairs_exp29.jsonl --out adapter_exp29 \
        --beta 0.1 --lr 5e-7 --max_steps 50
"""

from __future__ import annotations

import argparse

# Must match infer_slice.py / corpus.py / the grader exactly (prompt construction).
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
MAX_MODEL_LEN = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="base model path or HF id")
    parser.add_argument(
        "--adapter",
        required=True,
        help="path to the 0.86 LoRA adapter dir (continued as the trainable policy)",
    )
    parser.add_argument("--pairs", default="dpo_pairs_exp29.jsonl")
    parser.add_argument("--out", default="adapter_exp29")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL coefficient")
    # exp35 (REDI): add an SFT/NLL term on the chosen completion (TRL rpo_alpha).
    # REDI = preference loss on (correct, incorrect) PLUS supervised loss on the
    # correct trace; rpo_alpha=0.0 leaves plain DPO (exp29) unchanged.
    parser.add_argument(
        "--rpo_alpha",
        type=float,
        default=0.0,
        help="REDI/RPO SFT-mix weight on chosen (0.0 = plain DPO)",
    )
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 step on 2 pairs — verifies model+TRL run before the full job",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_prompt(prompt_raw: str) -> str:
        # Same templated prefix the model saw at rollout time (ends at the opening <think>).
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_raw + PROMPT_SUFFIX}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    import json
    from pathlib import Path

    rows = [
        json.loads(line)
        for line in Path(args.pairs).read_text().splitlines()
        if line.strip()
    ]
    if args.smoke:
        rows = rows[:2]
    if not rows:
        raise SystemExit(f"No pairs in {args.pairs}")

    dataset = Dataset.from_list(
        [
            {
                "prompt": to_prompt(r["prompt_raw"]),
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            }
            for r in rows
        ]
    )
    print(f"Loaded {len(dataset)} DPO pairs from {args.pairs}")

    # Policy = base + 0.86 adapter, trainable (artifact stays one rank-32 adapter).
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=True)
    model.enable_input_require_grads()  # needed for grad checkpointing through PEFT

    cfg = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=1 if args.smoke else args.max_steps,
        logging_steps=1 if args.smoke else 5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=args.beta,
        loss_type="sigmoid",
        rpo_alpha=args.rpo_alpha if args.rpo_alpha > 0 else None,  # exp35 REDI SFT-mix
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        remove_unused_columns=False,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,  # already a PEFT model → do NOT pass peft_config
        ref_model=None,  # PEFT: reference = adapter-disabled base (see plan §2.2)
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    # PEFT saves lm_head LoRA as "base_model.model.lm_head.*", but Nemotron-H keeps
    # lm_head under backbone — rename to match (same fix as Continuer_Nemotron_Notebook.py),
    # so the adapter loads correctly under vLLM AND in submission.zip. No-op if absent.
    import os

    from safetensors.torch import load_file, save_file

    st_path = os.path.join(args.out, "adapter_model.safetensors")
    if os.path.isfile(st_path):
        tensors = load_file(st_path)
        renamed = {
            k.replace(
                "base_model.model.lm_head.", "base_model.model.backbone.lm_head."
            ): v
            for k, v in tensors.items()
        }
        if renamed != tensors:
            save_file(renamed, st_path)
            print(
                "Renamed lm_head LoRA keys -> backbone.lm_head (vLLM/submission compat)"
            )

    print(f"Saved DPO-refined adapter -> {args.out}")
    print(
        "Next (GPU): infer_slice.py --adapter",
        args.out,
        "(greedy) then eval_slice.py on eval_holdout_ids.txt",
    )


if __name__ == "__main__":
    main()
