# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Nemotron finetuning pipeline
# ============================================================
# EXP27 — GSPO sequence-level policy objective  (Batch-3 Idea 8)
# Base: Continuer_Nemotron_Notebook.py (unmodified except marked blocks)
# Ref: refs/trl/trl/trainer/grpo_trainer.py sequence importance weights | Knob: GSPO_ENABLE=True | Rollback: set GSPO_ENABLE=False
# ============================================================

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Shared config ─────────────────────────────────────────────────────
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

MAX_SEQ_LEN = 8192
NUM_STEPS = 1000
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
RESET_WEIGHTS = (
    True  # if True, skip loading pretrained adapter; train from fresh LoRA init
)
IN_PROJ_ONLY = False
MOE_TIE_WEIGHTS = True  # if True, tie one side of MoE expert LoRA across all 128 experts (Tinker-style)
ORIGINAL_PROBLEMS_ONLY = (
    False  # if True, filter examples to only problem_ids listed in train.csv
)
SHUFFLE_DATASET = False

# >>> EXP27 START
import os

GSPO_ENABLE = True
GSPO_ROLLOUTS = os.environ.get("GSPO_ROLLOUTS", "/kaggle/working/rollouts.jsonl")
GSPO_GROUP_SIZE = 8
GSPO_TEMPERATURE = 0.9
GSPO_TOP_P = 0.95
GSPO_ROLLOUT_TRAIN_CSV = os.environ.get(
    "GSPO_ROLLOUT_TRAIN_CSV",
    os.environ.get(
        "GSPO_TRAIN_CSV",
        "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv",
    ),
)
GSPO_ONLY_CATEGORY = ""
GSPO_MAX_PROBLEMS = 0
GSPO_EPS_LOW = 3e-4
GSPO_EPS_HIGH = 3e-4
GSPO_BETA_KL = 0.0
# <<< EXP27 END

KAGGLE_DATASET = "huikang/nemotron-data"
MINUTES = 60

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "in_proj",
    "out_proj",
    "lm_head",
]

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os

# >>> EXP27 START
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def extract_answer(text: str) -> str:
    """Extract the final boxed answer, matching nemotron-master/reasoning.py."""
    import re

    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def compare_answer(stored: str, pred: str) -> bool:
    """Return whether a predicted answer matches the stored competition label."""
    import re

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
    """Basic guard against reward hacking malformed completions."""
    import re

    return len(re.findall(r"\\boxed\{", text)) >= 1


def _sampled_token_logprob(logprob_row: object, token_id: int) -> float:
    """Extract vLLM's sampled-token logprob across minor API variations."""
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
    """Release vLLM distributed state and CUDA memory across vLLM versions."""
    import gc

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
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _generate_rollouts() -> None:
    import csv
    import gc
    import json

    category_map: dict[str, str] = {}
    category_candidates = [
        os.environ.get("GSPO_PROBLEMS_JSONL", "problems.jsonl"),
        "nemotron-master/problems.jsonl",
        "/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master/problems.jsonl",
    ]
    for path in category_candidates:
        if not path or not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                category_map[str(rec["id"])] = str(rec["category"])
        if category_map:
            print(f"EXP27: loaded {len(category_map)} categories from {path}")
            break

    rows: list[dict[str, str]] = []
    with open(GSPO_ROLLOUT_TRAIN_CSV, newline="") as f:
        for row in csv.DictReader(f):
            row_id = str(row.get("id") or row.get("problem_id") or "")
            if not row_id:
                continue
            category = row.get("category", "") or category_map.get(row_id, "")
            row["id"] = row_id
            row["category"] = category
            if GSPO_ONLY_CATEGORY and category != GSPO_ONLY_CATEGORY:
                continue
            rows.append(row)
    if GSPO_MAX_PROBLEMS:
        rows = rows[:GSPO_MAX_PROBLEMS]
    print(f"EXP27: generating rollouts for {len(rows)} problems, G={GSPO_GROUP_SIZE}")

    if IS_KAGGLE:
        import kagglehub

        model_path = os.environ.get("GSPO_MODEL_PATH")
        if not model_path:
            model_path = kagglehub.model_download(
                "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
            )
        adapter_src = os.environ.get("GSPO_ADAPTER", "/kaggle/tmp/pretrained_adapter")
    else:
        model_path = os.environ.get(
            "GSPO_MODEL_PATH", "unsloth/Nemotron-3-Nano-30B-A3B"
        )
        adapter_src = os.environ.get("GSPO_ADAPTER", "/merged/weights")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = None
    try:
        llm = LLM(
            model=model_path,
            enable_lora=True,
            max_lora_rank=32,
            max_model_len=8192,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        lora_request = None
        if os.path.isfile(os.path.join(adapter_src, "adapter_config.json")):
            try:
                from vllm.lora.request import LoRARequest

                lora_request = LoRARequest("gspo_adapter", 1, adapter_src)
                print(f"EXP27: sampling with LoRA adapter at {adapter_src}")
            except Exception as exc:
                print(f"EXP27: could not attach LoRA adapter ({exc}); sampling base model")
        else:
            print(f"EXP27: adapter not found at {adapter_src}; sampling base model")

        sampling_params = SamplingParams(
            n=GSPO_GROUP_SIZE,
            temperature=GSPO_TEMPERATURE,
            top_p=GSPO_TOP_P,
            max_tokens=int(os.environ.get("GSPO_MAX_TOKENS", "7680")),
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

        out_dir = os.path.dirname(GSPO_ROLLOUTS)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        n_written = 0
        n_groups_kept = 0
        with open(GSPO_ROLLOUTS, "w") as fout:
            for row, output in zip(rows, outputs):
                prompt_ids = list(output.prompt_token_ids)
                records: list[dict[str, object]] = []
                rewards: list[float] = []
                for completion in output.outputs:
                    completion_ids = list(completion.token_ids)
                    text = completion.text
                    pred = extract_answer(text)
                    reward = (
                        1.0
                        if format_ok(text) and compare_answer(row["answer"], pred)
                        else 0.0
                    )
                    old_logp = [0.0] * len(prompt_ids)
                    completion_logprobs = completion.logprobs or []
                    for j, token_id in enumerate(completion_ids):
                        logprob_row = (
                            completion_logprobs[j]
                            if j < len(completion_logprobs)
                            else None
                        )
                        old_logp.append(_sampled_token_logprob(logprob_row, token_id))
                    tokens = prompt_ids + completion_ids
                    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
                    records.append(
                        {
                            "problem_id": row["id"],
                            "category": row.get("category", "rollout"),
                            "tokens": tokens,
                            "mask": mask,
                            "old_logp": old_logp,
                            "reward": reward,
                        }
                    )
                    rewards.append(reward)

                if len(set(rewards)) < 2:
                    continue
                n_groups_kept += 1
                for record in records:
                    fout.write(json.dumps(record) + "\n")
                    n_written += 1

        print(
            f"EXP27: wrote {n_written} rollouts "
            f"({n_groups_kept} mixed-reward groups) -> {GSPO_ROLLOUTS}"
        )
    finally:
        if llm is not None:
            _free_vllm(llm)
        gc.collect()


# <<< EXP27 END

IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
IS_MODAL_WORKER = "MODAL_TASK_ID" in os.environ
IS_MODAL_LAUNCHER = not IS_KAGGLE and not IS_MODAL_WORKER

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Env-specific install (Kaggle only; Modal image has packages pre-installed) ──
if IS_KAGGLE:
    import subprocess

    subprocess.run(
        "pip install -q --no-index --find-links /kaggle/input/datasets/mayukh18/nemotron-packages/packages "
        "unsloth trl peft transformers datasets accelerate bitsandbytes",
        shell=True,
        check=True,
    )
    subprocess.run(
        "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        shell=True,
        check=True,
    )
    subprocess.run(
        "pip install -q /kaggle/input/datasets/mayukh18/nemotron-packages/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
        shell=True,
        check=True,
    )
    for _wd in ["/kaggle/input/datasets/llkh0a/rtx-wheels/wheels"]:
        if os.path.isdir(_wd):
            subprocess.run(
                [
                    "pip",
                    "install",
                    "-q",
                    "--no-index",
                    "--find-links",
                    _wd,
                    "protobuf==6.33.5",
                    "sentencepiece",
                    "safetensors",
                    "huggingface_hub",
                ],
                check=False,
            )
    subprocess.run("rm -rf /kaggle/tmp/*", shell=True, check=True)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def run_training() -> None:
    """Full training flow. Runs on Kaggle at module level or inside Modal container via train_remote()."""
    import gc
    import json
    import math
    import random
    import subprocess
    import sys
    import time

    import torch

    # ── Env-specific paths + adapter source ──────────────────────────
    if IS_KAGGLE:
        import kagglehub

        CORPUS_PATH = "/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master/training/sft/04-08-16-14/tokens"
        TRAIN_ORDER_PATH = "/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master/training/sft/04-08-16-14/logprobs/index.jsonl"
        TRAIN_CSV_PATH = "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"
        ADAPTER_SRC = "/kaggle/tmp/pretrained_adapter"
        if not RESET_WEIGHTS:
            import zipfile as _zipfile

            _adapter_zip = "/kaggle/input/notebooks/huikang/tinker-submission-notebook/submission.zip"
            os.makedirs(ADAPTER_SRC, exist_ok=True)
            with _zipfile.ZipFile(_adapter_zip, "r") as _zf:
                _zf.extractall(ADAPTER_SRC)
        MODEL_PATH = kagglehub.model_download(
            "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
        )
    else:  # IS_MODAL_WORKER
        MODEL_PATH = "unsloth/Nemotron-3-Nano-30B-A3B"
        CORPUS_PATH = "/data/corpus_preprocessed.jsonl"
        TRAIN_CSV_PATH = "/data/train.csv"
        ADAPTER_SRC = "/merged/weights"
        OUTPUT_DIR = "/output/weights"

    # ── GPU + kernel sanity check (runs on both Kaggle and Modal worker) ──
    import causal_conv1d
    import mamba_ssm

    cc = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, sm_{cc[0] * 10 + cc[1]}")
    print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
    print(
        f"mamba_ssm={mamba_ssm.__version__}, causal_conv1d={causal_conv1d.__version__}"
    )
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if IS_MODAL_WORKER:
        assert cc == (12, 0), (
            f"Expected sm_120 (RTX PRO 6000), got sm_{cc[0] * 10 + cc[1]}"
        )
    from causal_conv1d import causal_conv1d_fn

    _x = torch.randn(1, 512, 32, device="cuda", dtype=torch.bfloat16)+4e-3
    _w = torch.randn(512, 4, device="cuda", dtype=torch.bfloat16)
    causal_conv1d_fn(_x, _w, None, activation="silu")
    print("causal_conv1d CUDA kernel: OK")

    # Clear stale HF modules cache (Modal-only; bug: persists across runs)
    if IS_MODAL_WORKER:
        import shutil as _shutil

        hf_modules = os.path.join(
            os.environ.get("HF_HOME", "/root/.cache/huggingface"), "modules"
        )
        if os.path.exists(hf_modules):
            _shutil.rmtree(hf_modules)

    # ── Load corpus into `examples` list ─────────────────────────────
    examples: list[dict] = []

    if IS_KAGGLE:
        # Load problem_ids in training order from logprobs/index.jsonl (epoch 0).
        # Each entry has {epoch, step, problem_id, ...}; epoch-0 entries are the
        # original training order, which we replay here.
        ordered_ids: list[str] = []
        seen: set[str] = set()
        with open(TRAIN_ORDER_PATH) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("epoch", 0) != 0:
                    continue
                pid = rec["problem_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                ordered_ids.append(pid)
        print(
            f"Loaded {len(ordered_ids)} problem_ids in training order from "
            f"{TRAIN_ORDER_PATH}"
        )

        for sid in ordered_ids:
            seg_path = os.path.join(CORPUS_PATH, sid, "synthetic.json")
            assert os.path.isfile(seg_path), (
                f"problem_id {sid} from training order missing in corpus: {seg_path}"
            )
            with open(seg_path) as f:
                rec = json.load(f)
            tokens = rec["tokens"]
            mask = rec["mask"]
            if not tokens:
                continue
            if len(tokens) > MAX_SEQ_LEN:
                tokens = tokens[:MAX_SEQ_LEN]
                mask = mask[:MAX_SEQ_LEN]
            if not any(mask):
                continue
            examples.append(
                {
                    "problem_id": sid,
                    "tokens": tokens[:-1],
                    "targets": tokens[1:],
                    "weights": [float(m) for m in mask[1:]],
                }
            )
    else:  # IS_MODAL_WORKER
        with open(CORPUS_PATH) as f:
            for line in f:
                rec = json.loads(line.strip())
                tokens = rec["tokens"]
                mask = rec["mask"]
                if len(tokens) > MAX_SEQ_LEN:
                    tokens = tokens[:MAX_SEQ_LEN]
                    mask = mask[:MAX_SEQ_LEN]
                if not any(mask):
                    continue
                examples.append(
                    {
                        "problem_id": rec["problem_id"],
                        "tokens": tokens[:-1],
                        "targets": tokens[1:],
                        "weights": [float(m) for m in mask[1:]],
                    }
                )

    if ORIGINAL_PROBLEMS_ONLY:
        import csv

        with open(TRAIN_CSV_PATH) as f:
            original_ids = {row["id"] for row in csv.DictReader(f)}
        before = len(examples)
        examples = [e for e in examples if e["problem_id"] in original_ids]
        print(
            f"ORIGINAL_PROBLEMS_ONLY=True: filtered {before} → {len(examples)} examples "
            f"using {len(original_ids)} ids from {TRAIN_CSV_PATH}"
        )

    # >>> EXP27 START
    if GSPO_ENABLE:
        if not os.path.exists(GSPO_ROLLOUTS):
            print(f"EXP27: rollouts chưa có -> sinh mới vào {GSPO_ROLLOUTS}")
            _generate_rollouts()
        else:
            print(f"EXP27: tái dùng rollouts sẵn có tại {GSPO_ROLLOUTS}")

        rollout_records: list[dict] = []
        with open(GSPO_ROLLOUTS) as f:
            for line in f:
                if line.strip():
                    rollout_records.append(json.loads(line))

        rewards_by_problem: dict[str, list[float]] = {}
        for rec in rollout_records:
            rewards_by_problem.setdefault(str(rec["problem_id"]), []).append(
                float(rec["reward"])
            )

        gspo_examples: list[dict] = []
        for rec in rollout_records:
            rewards = rewards_by_problem[str(rec["problem_id"])]
            if len(rewards) < 2 or all(r == rewards[0] for r in rewards):
                continue
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            advantage = (float(rec["reward"]) - mean_r) / (var_r**0.5 + 1e-4)

            tokens = rec["tokens"]
            mask = rec["mask"]
            old_logp = rec["old_logp"]
            if len(tokens) > MAX_SEQ_LEN:
                tokens = tokens[:MAX_SEQ_LEN]
                mask = mask[:MAX_SEQ_LEN]
                old_logp = old_logp[:MAX_SEQ_LEN]
            if len(old_logp) == len(tokens):
                old_logp = old_logp[1:]
            else:
                old_logp = old_logp[: max(0, len(tokens) - 1)]
            if not any(mask):
                continue
            gspo_examples.append(
                {
                    "problem_id": rec["problem_id"],
                    "category": rec.get("category", "rollout"),
                    "tokens": tokens[:-1],
                    "targets": tokens[1:],
                    "weights": [float(m) for m in mask[1:]],
                    "old_logp": [float(v) for v in old_logp],
                    "advantage": float(advantage),
                }
            )
        if not gspo_examples:
            raise RuntimeError(
                "GSPO rollout file had no usable mixed-reward groups after filtering"
            )
        print(
            f"EXP27 GSPO loaded {len(gspo_examples)} rollout examples "
            f"from {GSPO_ROLLOUTS} (group target G={GSPO_GROUP_SIZE})"
        )
        examples = gspo_examples
    # <<< EXP27 END

    total_unmasked = sum(sum(e["weights"]) for e in examples)
    total_tokens = sum(len(e["tokens"]) for e in examples)
    print(
        f"Loaded {len(examples)} examples, {total_tokens:,} tokens "
        f"(unmasked={total_unmasked:,.0f})"
    )

    # ── Load base model ──────────────────────────────────────────────
    # >>> EXP27 START
    from unsloth import FastLanguageModel

    from cut_cross_entropy import linear_cross_entropy
    from peft import LoraConfig
    from peft.tuners.lora import Linear as LoraLinear

    # <<< EXP27 END

    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    if IS_MODAL_WORKER:
        hf_cache_vol.commit()  # noqa: F821 — defined at module level on non-Kaggle

    # ── Wrap in LoRA ─────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    FastLanguageModel.for_training(model)

    # ── Patch Mamba CUDA fast path ───────────────────────────────────
    nemotron_mod = None
    for _name, _m in sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
            nemotron_mod = _m
            break
    assert nemotron_mod is not None, "Could not find modeling_nemotron_h module"
    print(f"is_fast_path_available was: {nemotron_mod.is_fast_path_available}")
    nemotron_mod.is_fast_path_available = True  # type: ignore[attr-defined]
    print("Patched is_fast_path_available = True")

    # ── Manually add lm_head LoRA (Unsloth drops it for MoE) ─────────
    _causal_lm = model
    while hasattr(_causal_lm, "model"):
        _causal_lm = _causal_lm.model
    _lm_head = _causal_lm.lm_head
    if not isinstance(_lm_head, LoraLinear):
        _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
        model.base_model._create_and_replace(
            _cfg,
            "default",
            target=_lm_head,
            target_name="lm_head",
            parent=_causal_lm,
        )
        print("Manually added LoRA to lm_head")
    else:
        print("lm_head already has LoRA")

    # ── Cast LoRA params to fp32 (base model stays bf16 except MoE router) ──
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)

    for name, param in model.named_parameters():
        if ".lora_" in name:
            assert param.dtype == torch.float32, (
                f"LoRA param {name} expected fp32, got {param.dtype}"
            )
            continue

        is_router = (
            ".mixer.gate." in name
        )  # NemotronHTopkRouter.weight + e_score_correction_bias
        # Nemotron-H loads the MoE router (`mixer.gate`) in fp32 on purpose.
        # Ref: transformers/src/transformers/models/nemotron_h/modeling_nemotron_h.py
        #
        #   class NemotronHPreTrainedModel(PreTrainedModel):
        #       _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
        #
        #   class NemotronHTopkRouter(nn.Module):
        #       def __init__(self, config):
        #           self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        #           self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        #       def forward(self, hidden_states):
        #           router_logits = F.linear(
        #               hidden_states.type(torch.float32),
        #               self.weight.type(torch.float32),
        #           )
        #           return router_logits
        #
        # The per-forward fp32 cast on `self.weight` plus the strict list entry
        # mean the gate weight is promoted to fp32 at load time.
        if is_router:
            assert param.dtype == torch.float32, (
                f"param {name} expected fp32, got {param.dtype}"
            )
            continue

        assert param.dtype == torch.bfloat16, (
            f"param {name} expected bf16, got {param.dtype}"
        )
        continue

    print("Verified: LoRA params fp32, base params bf16 (MoE router fp32)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {trainable:,} trainable / {total:,} total parameters")

    # ── Patch forward with Cut Cross-Entropy ─────────────────────────
    _base = model
    while hasattr(_base, "model"):
        _base = _base.model

    def _patched_causal_forward(
        input_ids=None, attention_mask=None, labels=None, **kwargs
    ):
        backbone_out = _base.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("position_ids", "past_key_values", "use_cache")
            },
        )
        hidden_states = backbone_out[0]
        lm_head = _base.lm_head
        base_w = lm_head.base_layer.weight
        lora_A = lm_head.lora_A["default"].weight
        lora_B = lm_head.lora_B["default"].weight
        scaling = lm_head.scaling["default"]
        lm_weight = base_w + scaling * lora_B @ lora_A
        if labels is not None:
            per_token_ce = linear_cross_entropy(
                hidden_states, lm_weight, labels, reduction="none"
            )
            loss = per_token_ce.mean()
        else:
            per_token_ce = None
            loss = None
        model._cached_per_token_ce = per_token_ce  # type: ignore[attr-defined]
        return loss

    _base.forward = _patched_causal_forward
    print("Patched CausalLM.forward with CCE (no logits materialization)")

    # ── Load adapter weights (unless RESET_WEIGHTS) ──────────────────
    if RESET_WEIGHTS:
        print(
            "RESET_WEIGHTS=True — skipping pretrained adapter load; using fresh LoRA init"
        )
        loaded = 0
        adapter_weights: dict = {}
    else:
        print(f"Loading adapter from {ADAPTER_SRC}...")
        from peft import load_peft_weights

        adapter_weights = load_peft_weights(ADAPTER_SRC)

        model_sd = model.state_dict()
        new_sd: dict = {}
        loaded = 0
        for ak, av in adapter_weights.items():
            if ak in model_sd:
                new_sd[ak] = av
                loaded += 1
                continue
            ak_with_default = ak.replace(
                ".lora_A.weight", ".lora_A.default.weight"
            ).replace(".lora_B.weight", ".lora_B.default.weight")
            if ak_with_default in model_sd:
                new_sd[ak_with_default] = av
                loaded += 1
                continue
            ak_lm = ak.replace(".backbone.lm_head.", ".lm_head.")
            ak_lm_default = ak_lm.replace(
                ".lora_A.weight", ".lora_A.default.weight"
            ).replace(".lora_B.weight", ".lora_B.default.weight")
            if ak_lm_default in model_sd:
                new_sd[ak_lm_default] = av
                loaded += 1
                continue

        model.load_state_dict(new_sd, strict=False)
        assert loaded == len(adapter_weights), (
            f"Not all adapter weights loaded: {loaded}/{len(adapter_weights)}"
        )
        print(f"  Loaded {loaded}/{len(adapter_weights)} weights into model")

    # ── Freeze all LoRA params except in_proj (if IN_PROJ_ONLY) ──
    print(f"{IN_PROJ_ONLY=}")
    if IN_PROJ_ONLY:
        for name, param in model.named_parameters():
            if param.requires_grad and ".in_proj." not in name:
                param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  {trainable_params:,} trainable / {frozen_params:,} frozen")

    # ── MoE tied-weight params (Tinker convention) ───────────────────
    # Tinker ties whichever LoRA side touches the hidden dim:
    #   gate_up_proj / up_proj / w1 / gate_proj  -> tie A (input/hidden side)
    #   down_proj / w2                           -> tie B (output/hidden side)
    # We keep Unsloth's batched [num_experts, ...] tensor layout; "tying" means
    # all 128 expert slices are kept identical. Saving the adapter naturally
    # emits 128 per-expert copies, so submission.zip is untied downstream.
    moe_tied_params: list[torch.Tensor] = []
    if MOE_TIE_WEIGHTS:
        w1_proj_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
        w2_proj_names = ("down_proj", ".w2.")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if ".experts." not in name or ".lora_" not in name:
                continue
            is_w1 = any(p in name for p in w1_proj_names)
            is_w2 = any(p in name for p in w2_proj_names)
            is_A = ".lora_A." in name
            is_B = ".lora_B." in name
            should_tie = (is_w1 and is_A) or (is_w2 and is_B)
            if not should_tie:
                continue
            if param.dim() < 2 or param.shape[0] <= 1:
                continue
            moe_tied_params.append(param)

        def _tie_param_init() -> None:
            """Make all 128 expert slices identical (mean-and-broadcast)."""
            with torch.no_grad():
                for p in moe_tied_params:
                    mean = p.data.mean(dim=0, keepdim=True)
                    p.data.copy_(mean.expand_as(p.data))

        def _tie_grads() -> None:
            # Sum (not mean) across the expert dim: if W is the shared LoRA factor
            # and each expert uses a copy W_i = W, chain rule gives
            # dL/dW = sum_i dL/dW_i. Inactive experts contribute 0 and router
            # weights are already baked into active g_i, so there's no
            # double-counting. Summing keeps all 128 slices identical after each
            # AdamW step and reproduces the true shared-weight update; mean would
            # be off by a 1/128 lr rescale (and not exactly equivalent under
            # AdamW's eps/weight-decay).
            with torch.no_grad():
                for p in moe_tied_params:
                    if p.grad is None:
                        continue
                    grad_sum = p.grad.sum(dim=0, keepdim=True)
                    p.grad.copy_(grad_sum.expand_as(p.grad))

        print(f"MoE weight tying: {len(moe_tied_params)} params identified for tying")
        if moe_tied_params:
            print(f"  example shapes: {[tuple(p.shape) for p in moe_tied_params[:3]]}")
        _tie_param_init()  # start from a tied state
    else:

        def _tie_grads() -> None:
            pass

    # ── Training loop ────────────────────────────────────────────────
    gc.collect()
    torch.cuda.empty_cache()

    device = next(model.parameters()).device
    optimizer: torch.optim.AdamW | None = None

    indices = list(range(len(examples)))
    if SHUFFLE_DATASET:
        rng = random.Random(0)
        rng.shuffle(indices)
        print(f"SHUFFLE_DATASET=True: shuffled {len(indices)} examples (seed=0)")
    else:
        print(f"SHUFFLE_DATASET=False: keeping corpus order ({len(indices)} examples)")

    training_log: list[str] = []

    def _log(msg: str) -> None:
        print(msg, flush=True)
        training_log.append(msg)

    max_steps = len(examples) // BATCH_SIZE
    num_steps = NUM_STEPS
    if num_steps > max_steps:
        _log(
            f"WARNING: NUM_STEPS={NUM_STEPS} exceeds max_steps={max_steps} "
            f"({len(examples)} examples // {BATCH_SIZE} batch). Clamping to {max_steps}."
        )
        num_steps = max_steps

    _log(
        f"Training: {num_steps} steps, batch_size={BATCH_SIZE}, "
        f"micro_batch_size={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}"
    )

    step = 0
    for batch_start in range(0, len(indices), BATCH_SIZE):
        if step >= num_steps:
            break
        batch_indices = indices[batch_start : batch_start + BATCH_SIZE]
        batch = [examples[i] for i in batch_indices]
        batch_tokens = [e["tokens"] for e in batch]
        batch_targets = [e["targets"] for e in batch]
        batch_weights = [e["weights"] for e in batch]
        # >>> EXP27 START
        batch_old_logp = [e.get("old_logp") for e in batch]
        batch_advantages = [float(e.get("advantage", 0.0)) for e in batch]
        # <<< EXP27 END

        n = len(batch)
        n_accum = math.ceil(n / MICRO_BATCH_SIZE)
        total_loss_sum = 0.0
        total_weight_sum = 0.0

        for mb_start in range(0, n, MICRO_BATCH_SIZE):
            mb_end = min(mb_start + MICRO_BATCH_SIZE, n)
            mb_toks = batch_tokens[mb_start:mb_end]
            mb_tgts = batch_targets[mb_start:mb_end]
            mb_wts = batch_weights[mb_start:mb_end]
            # >>> EXP27 START
            mb_old_logp = batch_old_logp[mb_start:mb_end]
            mb_advantages = batch_advantages[mb_start:mb_end]
            # <<< EXP27 END

            n_micro = len(mb_toks)
            max_len = max(len(t) for t in mb_toks)
            total_len = sum(len(t) for t in mb_toks)

            padded_input = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            padded_targets = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            padded_weights = torch.zeros(
                n_micro, max_len, dtype=torch.float32, device=device
            )
            # >>> EXP27 START
            padded_old_logp = torch.zeros(
                n_micro, max_len, dtype=torch.float32, device=device
            )
            padded_adv = torch.zeros(
                n_micro, max_len, dtype=torch.float32, device=device
            )
            # <<< EXP27 END
            attention_mask = torch.zeros(
                n_micro, max_len, dtype=torch.long, device=device
            )
            for i in range(n_micro):
                seq_len = len(mb_toks[i])
                padded_input[i, :seq_len] = torch.tensor(mb_toks[i], dtype=torch.long)
                padded_targets[i, :seq_len] = torch.tensor(mb_tgts[i], dtype=torch.long)
                padded_weights[i, :seq_len] = torch.tensor(
                    mb_wts[i], dtype=torch.float32
                )
                # >>> EXP27 START
                if mb_old_logp[i] is not None:
                    old_len = min(seq_len, len(mb_old_logp[i]))
                    padded_old_logp[i, :old_len] = torch.tensor(
                        mb_old_logp[i][:old_len], dtype=torch.float32
                    )
                    padded_adv[i, :seq_len] = float(mb_advantages[i])
                # <<< EXP27 END
                attention_mask[i, :seq_len] = 1

            t0 = time.time()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(
                    input_ids=padded_input,
                    attention_mask=attention_mask,
                    labels=padded_targets,
                    use_cache=False,
                )
                per_token_ce = model._cached_per_token_ce  # type: ignore[attr-defined]
                # >>> EXP27 START
                if GSPO_ENABLE:
                    mask = padded_weights
                    cur_logp = -per_token_ce
                    log_ratio = cur_logp - padded_old_logp
                    seq_logw = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(
                        min=1.0
                    )
                    seq_logw = torch.clamp(seq_logw, -20.0, 20.0).unsqueeze(-1)
                    coef_1 = torch.exp(seq_logw)
                    coef_2 = torch.clamp(
                        coef_1, 1.0 - GSPO_EPS_LOW, 1.0 + GSPO_EPS_HIGH
                    )
                    per_tok = -torch.min(coef_1 * padded_adv, coef_2 * padded_adv)
                    loss = (
                        (per_tok * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
                    ).mean()
                    weighted_loss = per_tok * mask
                    weight_sum_t = mask.sum()
                    loss_sum_t = weighted_loss.sum()
                else:
                    weighted_loss = per_token_ce * padded_weights
                    weight_sum_t = padded_weights.sum()
                    loss_sum_t = weighted_loss.sum()
                    loss = (
                        loss_sum_t / weight_sum_t
                        if weight_sum_t > 0
                        else loss_sum_t * 0.0
                    )
                # <<< EXP27 END

            (loss / n_accum).backward()
            total_loss_sum += loss_sum_t.item()
            total_weight_sum += weight_sum_t.item()
            del loss, per_token_ce, weighted_loss

            t_end = time.time()
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            mem_gb = torch.cuda.memory_allocated() / 1e9
            mb_idx = mb_start // MICRO_BATCH_SIZE
            print(
                f"    micro-batch {mb_idx}: {n_micro} seqs, max_len={max_len}, "
                f"total_len={total_len}, wall={t_end - t0:.1f}s, "
                f"peak={peak_gb:.1f}GB, mem={mem_gb:.1f}GB"
            )

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=LEARNING_RATE,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.0,
            )
        lr = LEARNING_RATE * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        _tie_grads()  # average MoE expert grads before clip+step so Adam stays in sync
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1e9
        )
        optimizer.step()
        optimizer.zero_grad()
        loss_mean = total_loss_sum / total_weight_sum if total_weight_sum > 0 else 0
        step += 1
        _log(
            f"  step {step}/{num_steps}: "
            f"loss:mean={loss_mean:.6f}, grad_norm={grad_norm:.4f}, lr={lr:.2e}"
        )

    print(
        f"\nTraining complete. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    )

    # ── Save adapter + rename lm_head keys (identical on both sides) ──
    from safetensors.torch import load_file, save_file

    save_dir = "." if IS_KAGGLE else OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    for _f in os.listdir(save_dir):
        if _f.startswith("adapter"):
            os.remove(os.path.join(save_dir, _f))
    model.save_pretrained(save_dir)
    st_path = os.path.join(save_dir, "adapter_model.safetensors")
    tensors = load_file(st_path)
    renamed = {
        k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
        for k, v in tensors.items()
    }
    save_file(renamed, st_path)

    # ── Clean unsloth compiled cache (runs on both) ──────────────────
    _ucache = "unsloth_compiled_cache"
    if os.path.isdir(_ucache):
        import shutil as _sh

        _sh.rmtree(_ucache)

    # ── Package & ship (divergent) ───────────────────────────────────
    if IS_KAGGLE:
        import zipfile

        adapter_files = [f for f in os.listdir(save_dir) if f.startswith("adapter")]
        SUBMISSION_ZIP = "submission.zip"
        with zipfile.ZipFile(SUBMISSION_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in adapter_files:
                zf.write(os.path.join(save_dir, fname), fname)
        for fname in adapter_files:
            os.remove(os.path.join(save_dir, fname))
        print(f"Wrote {SUBMISSION_ZIP}")
    else:  # IS_MODAL_WORKER
        import shutil
        import tempfile

        with open(os.path.join(save_dir, "training_log.txt"), "w") as f:
            f.write("\n".join(training_log) + "\n")
        output_vol.commit()  # noqa: F821 — defined at module level on non-Kaggle

        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(os.path.join(kaggle_dir, "access_token"), "w") as f:
            f.write(os.environ["KAGGLE_API_TOKEN"])
        upload_dir = tempfile.mkdtemp()
        for fname in os.listdir(save_dir):
            shutil.copy(os.path.join(save_dir, fname), upload_dir)
        metadata = {"id": KAGGLE_DATASET, "title": KAGGLE_DATASET.split("/")[1]}
        with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
            json.dump(metadata, f)
        print(f"Uploading to Kaggle {KAGGLE_DATASET}...")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "version",
                "-p",
                upload_dir,
                "-m",
                "post-finetuned adapter + compiled wheels",
            ],
            check=True,
        )
        print("Kaggle upload complete.")
    print("Training complete.")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ── Modal glue: image, app, volumes, train_remote, main ──────────────
# Defined at module level on non-Kaggle so the worker's module import
# registers train_remote with the app. On Kaggle, skipped entirely
# (modal package is not installed there).
if not IS_KAGGLE:
    import modal

    train_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("git", "build-essential", "clang")
        .pip_install(
            "torch==2.10.0",
            extra_index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install(
            "safetensors>=0.5.0",
            "transformers>=4.56.2",
            "accelerate>=1.0.0",
            "peft>=0.15.0",
            "bitsandbytes>=0.45.0",
            "huggingface_hub>=0.36.2",
            "hf-transfer>=0.1.9",
            "numpy",
            "pillow",
            "torchvision",
            "datasets",
            "sentencepiece",
            "xformers",
            "cut-cross-entropy>=25.1.0",
            "wheel",
            "setuptools",
            "trl",
            "kaggle>=1.6.0",
        )
        .run_commands(
            'python -c "import torch.utils.cpp_extension as e; p=e.__file__; '
            "t=open(p).read().replace('raise RuntimeError(CUDA_MISMATCH_MESSAGE', 'pass  # '); "
            "open(p,'w').write(t)\"",
            "TORCH_CUDA_ARCH_LIST='12.0' pip wheel --no-build-isolation --wheel-dir /wheels mamba_ssm==2.3.1 causal_conv1d==1.6.1",
            "pip install --no-deps /wheels/mamba_ssm-*.whl /wheels/causal_conv1d-*.whl",
            "pip install --no-deps 'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo'",
            "pip install --no-deps 'unsloth[base] @ git+https://github.com/unslothai/unsloth'",
        )
        .pip_install("einops")
        .env({"HF_HOME": "/root/.cache/huggingface"})
    )

    hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
    merged_vol = modal.Volume.from_name("merged-adapter", create_if_missing=True)
    corpus_vol = modal.Volume.from_name("corpus-data", create_if_missing=True)
    output_vol = modal.Volume.from_name("post-finetune-output", create_if_missing=True)

    app = modal.App("post-finetune-pipeline")

    @app.function(
        image=train_image,
        gpu="RTX-PRO-6000",
        volumes={
            "/root/.cache/huggingface": hf_cache_vol,
            "/merged": merged_vol,
            "/data": corpus_vol,
            "/output": output_vol,
        },
        timeout=6 * 60 * MINUTES,
        secrets=[modal.Secret.from_local_environ(["KAGGLE_API_TOKEN"])],
    )
    def train_remote() -> None:
        run_training()

    if IS_MODAL_LAUNCHER:

        @app.local_entrypoint()
        def main() -> None:
            train_remote.remote()

# %% [code] {"jupyter":{"outputs_hidden":false}}
# On Kaggle, trigger training directly after cells load.
# On Modal worker, Modal's runtime calls train_remote() which calls run_training().
# On Modal launcher, neither fires (main() submits the remote call instead).
if IS_KAGGLE:
    run_training()
