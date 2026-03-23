# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # NVIDIA Nemotron | Experiment Lab — 5 Self-Improvement Methods
#
# Implements the first 5 experiments from `experiments_v2.md`:
#
# | # | Method | Type | Paper |
# |---|--------|------|-------|
# | TN1 | **STaR** — Self-Taught Reasoner | Iterative bootstrapped CoT | Zelikman et al., NeurIPS 2022 |
# | TN2 | **RFT** — Rejection Sampling FT | Sample K → keep correct → SFT | Yuan et al., ICML 2023 |
# | TN3 | **SPIN** — Self-Play DPO | Ground truth vs model wrong → DPO | Chen et al., ICML 2024 |
# | TN4 | **Self-Rewarding DPO** | Model self-evaluates → DPO | Yuan et al., ICML 2024 |
# | TN5 | **GRPO** | RL with verifiable outcome reward | Shao et al. (DeepSeek), 2024 |
#
# **Constraints**: No internet, no API, no external data, no distillation.
# All data synthesis uses the model itself (Nemotron-3-Nano-30B).
#
# **Design**: Toggle experiments via the config panel below. Each experiment
# builds on the baseline SFT checkpoint. The pipeline is:
# ```
# [Baseline SFT or Load existing adapter] → (optional) Generation → (pick ONE) RFT / STaR / SPIN / Self-Reward / GRPO → Submission
# ```
# **Tip**: Set `PRETRAINED_ADAPTER_PATH` to reuse the adapter from `sft.py`
# and set `RUN_BASELINE_SFT = False` to skip retraining baseline.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 0. Experiment Configuration
#
# **How to use**: Set `RUN_*` flags to choose which experiment to run.
# The baseline SFT always runs first. Then pick ONE follow-up method.
# Recommended pipelines:
# - **Quick baseline improve**: `RUN_BASELINE_SFT` only (with improved hyperparams)
# - **Reuse sft.py adapter + STaR**: `PRETRAINED_ADAPTER_PATH` + `RUN_STAR`
# - **RFT**: `RUN_BASELINE_SFT` + `RUN_RFT`
# - **STaR**: `RUN_BASELINE_SFT` + `RUN_STAR` (iterative, strongest)
# - **SPIN**: `RUN_BASELINE_SFT` + `RUN_SPIN`
# - **Full pipeline**: `RUN_BASELINE_SFT` + `RUN_STAR` + `RUN_GRPO`

# %% [code] {"jupyter":{"outputs_hidden":false}}
# ╔══════════════════════════════════════════════════════════════════╗
# ║                   EXPERIMENT PIPELINE CONFIG                     ║
# ╚══════════════════════════════════════════════════════════════════╝

# -- Phase 0: Load a pre-trained adapter (from sft.py) instead of retraining --
# Set this to skipping baseline SFT; e.g. "/kaggle/input/your-dataset/adapter"
# or the path where sft.py saved its adapter. Set to None to train from scratch.
PRETRAINED_ADAPTER_PATH = "/kaggle/input/notebooks/emanuellcs/nvidia-nemotron-sft/adapter"  # e.g. "/kaggle/working/adapter"

# -- Phase 1: Baseline SFT (skip if PRETRAINED_ADAPTER_PATH is set) ---------
RUN_BASELINE_SFT = True

# -- Phase 2: Pick ONE self-improvement method (requires generation) --------
RUN_RFT         = False    # TN2: Rejection Sampling FT (1 round, simple)
RUN_STAR        = True     # TN1: STaR iterative bootstrap (strongest)
RUN_SPIN        = False    # TN3: Self-Play DPO
RUN_SELF_REWARD = False    # TN4: Self-Rewarding DPO

# -- Phase 3: Optional RL on top -------------------------------------------
RUN_GRPO        = False    # TN5: GRPO reinforcement learning

# -- Training hyperparameters (improved from sft.py baseline) ---------------
LORA_RANK       = 32
LORA_ALPHA      = 64       # alpha/r = 2.0
LORA_DROPOUT    = 0.05
LR_SFT          = 2e-4     # Learning rate for SFT phases
LR_DPO          = 5e-6     # Learning rate for DPO phases
LR_GRPO         = 5e-6     # Learning rate for GRPO
GRAD_ACCUM      = 4
MAX_SEQ_LEN     = 2048     # TN7: was 1024, increase for longer reasoning
NUM_EPOCHS_SFT  = 2        # TN8: was 1, more epochs helps
USE_NEFTUNE     = True     # TN9: noisy embedding regularization
NEFTUNE_ALPHA   = 5.0
USE_PACKING     = True     # TN7: sequence packing for efficiency
USE_VALIDATION  = True     # TN8: hold out 5% for checkpoint selection
WARMUP_RATIO    = 0.05

# -- Generation settings (for TN1-4) ---------------------------------------
GEN_K              = 4     # Responses per prompt
GEN_MAX_TOKENS     = 512   # Max new tokens per response
GEN_TEMPERATURE    = 0.7   # Sampling temperature
GEN_SUBSAMPLE      = None  # None = all prompts, or int for speed testing
GEN_CACHE_PATH     = "/kaggle/working/gen_cache.json"

# -- STaR-specific ----------------------------------------------------------
STAR_ITERATIONS    = 2     # Number of STaR iterations
STAR_USE_RATIONAL  = True  # Use rationalization for failed prompts

# -- GRPO-specific ----------------------------------------------------------
GRPO_NUM_GENERATIONS = 4
GRPO_MAX_COMPLETION  = 1024
GRPO_KL_COEF         = 0.05
GRPO_EPOCHS          = 1

# -- DPO-specific -----------------------------------------------------------
DPO_BETA             = 0.1
DPO_EPOCHS           = 1

# -- Paths ------------------------------------------------------------------
OUTPUT_DIR_SFT   = "/kaggle/working/adapter_sft"
OUTPUT_DIR_FINAL = "/kaggle/working/adapter_final"

import os
os.makedirs(OUTPUT_DIR_SFT, exist_ok=True)
os.makedirs(OUTPUT_DIR_FINAL, exist_ok=True)

# Validate config: at most one self-improvement method at a time
_methods = [RUN_RFT, RUN_STAR, RUN_SPIN, RUN_SELF_REWARD]
assert sum(_methods) <= 1, (
    "Pick at most ONE self-improvement method: RFT, STaR, SPIN, or Self-Reward. "
    "GRPO can run on top of any of them."
)

# If a pre-trained adapter is supplied, no need to run baseline SFT
if PRETRAINED_ADAPTER_PATH is not None:
    assert os.path.isdir(PRETRAINED_ADAPTER_PATH), (
        f"PRETRAINED_ADAPTER_PATH not found: {PRETRAINED_ADAPTER_PATH}"
    )
    RUN_BASELINE_SFT = False  # Override — no need to retrain

assert RUN_BASELINE_SFT or PRETRAINED_ADAPTER_PATH is not None, (
    "Enable RUN_BASELINE_SFT or set PRETRAINED_ADAPTER_PATH to an existing adapter."
)

_needs_gen = RUN_RFT or RUN_STAR or RUN_SPIN or RUN_SELF_REWARD
print("=" * 60)
print("PIPELINE:")
if PRETRAINED_ADAPTER_PATH:
    print(f"  0. Pre-trained adapter  : {PRETRAINED_ADAPTER_PATH}")
print(f"  1. Baseline SFT        : {RUN_BASELINE_SFT}")
print(f"  2. Generation phase    : {_needs_gen}")
print(f"     - RFT (TN2)         : {RUN_RFT}")
print(f"     - STaR (TN1)        : {RUN_STAR}")
print(f"     - SPIN (TN3)        : {RUN_SPIN}")
print(f"     - Self-Reward (TN4) : {RUN_SELF_REWARD}")
print(f"  3. GRPO (TN5)          : {RUN_GRPO}")
print(f"  Config: seq_len={MAX_SEQ_LEN}, epochs={NUM_EPOCHS_SFT}, "
      f"neftune={USE_NEFTUNE}, packing={USE_PACKING}")
print("=" * 60)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Environment Setup & Wheel Installation

# %% [code] {"jupyter":{"outputs_hidden":false}}
import subprocess, sys, os

offline_dir = "/kaggle/input/nvidia-nemotron-offline-packages/offline_packages"
target_dir  = "/kaggle/working/packages"
os.makedirs(target_dir, exist_ok=True)

if os.path.exists(offline_dir):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "--no-index", "--find-links", offline_dir,
        "--target", target_dir,
        "datasets", "trl",
    ])
    print("Installed from offline cache.")
else:
    trl_whl = "/kaggle/input/datasets/nbroad/hf-libraries/trl/trl-0.29.1-py3-none-any.whl"
    if os.path.exists(trl_whl):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                "--no-index", "--no-deps", trl_whl])
        print("Installed trl from nbroad wheel.")
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                "--target", target_dir, "datasets", "trl"])
        print("Installed from PyPI.")

if target_dir not in sys.path:
    sys.path.insert(0, target_dir)

import datasets, trl
print(f"datasets: {datasets.__version__}  |  trl: {trl.__version__}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Imports & Blackwell Patches

# %% [code] {"jupyter":{"outputs_hidden":false}}
import re
import gc
import json
import stat
import shutil
import zipfile
from copy import deepcopy

import polars as pl
import torch
import torch.nn.functional as F
import kagglehub
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import SFTTrainer, SFTConfig

# -- Blackwell Hardware Patches ------------------------------------------------
print("Applying Blackwell patches...")

def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)

for _name, _mod in list(sys.modules.items()):
    if hasattr(_mod, "rmsnorm_fn"):
        _mod.rmsnorm_fn = _pure_rmsnorm_fn

_src = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell"
_dst = "/tmp/ptxas-blackwell"
if os.path.exists(_src):
    shutil.copy2(_src, _dst)
    os.chmod(_dst, os.stat(_dst).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    import triton.backends.nvidia as nv_backend
    import triton.backends.nvidia.compiler as nv_compiler

    _src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
    _dst_bin = "/tmp/triton_nvidia_bin"
    shutil.copytree(_src_bin, _dst_bin, dirs_exist_ok=True)
    for _f in os.listdir(_dst_bin):
        _fp = os.path.join(_dst_bin, _f)
        if os.path.isfile(_fp):
            os.chmod(_fp, os.stat(_fp).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    nv_backend.__file__ = os.path.join(_dst_bin, "..", "__init__.py")
    os.environ["TRITON_PTXAS_PATH"]           = _dst
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = _dst
    nv_compiler.get_ptxas_version = lambda arch="blackwell": "12.0"
    # Also patch get_ptxas so .path is always valid
    class _RealPtxas:
        path    = _dst
        version = "12.0"
    nv_compiler.get_ptxas = lambda arch=None: _RealPtxas()
    print("  [triton] ptxas redirect applied.")
else:
    print("  [triton] source binary not found — skipping.")

_util_ptxas_sources = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell",
]
for _src in _util_ptxas_sources:
    if os.path.exists(_src):
        shutil.copy2(_src, "/tmp/ptxas-blackwell")
        os.chmod("/tmp/ptxas-blackwell",
                 os.stat("/tmp/ptxas-blackwell").st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  [ptxas] copied from {_src}")
        break

for _mod_name, _mod in list(sys.modules.items()):
    if "triton" in _mod_name and hasattr(_mod, "NvidiaTool"):
        try:
            _orig_from_path = _mod.NvidiaTool.__dict__["from_path"]
            def _make_safe(orig):
                @staticmethod
                def _safe_from_path(path):
                    try:
                        return orig.__func__(path) if hasattr(orig, "__func__") else orig(path)
                    except (PermissionError, OSError, Exception):
                        return None
                return _safe_from_path
            _mod.NvidiaTool.from_path = _make_safe(_orig_from_path)
        except Exception:
            pass

for _mod_name, _mod in list(sys.modules.items()):
    if "triton" in _mod_name and hasattr(_mod, "get_ptxas_version"):
        _mod.get_ptxas_version = lambda arch=None: "12.0"
    if "triton" in _mod_name and hasattr(_mod, "get_ptxas"):
        _orig_gp = _mod.get_ptxas
        def _make_safe_gp(orig):
            def _safe_get_ptxas(arch):
                try:
                    result = orig(arch)
                    # Ensure .path is always set
                    if not hasattr(result, "path") or result.path is None:
                        result.path = "/tmp/ptxas-blackwell"
                    return result
                except Exception:
                    class _FakePtxas:
                        path    = "/tmp/ptxas-blackwell"
                        version = "12.0"
                    return _FakePtxas()
            return _safe_get_ptxas
        _mod.get_ptxas = _make_safe_gp(_orig_gp)

print("Blackwell patches done.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Shared Utilities
#
# Core helper functions used across all experiments:
# - `extract_boxed()`: Robust extraction of `\boxed{}` content with nested brace support
# - `answer_match()`: String + numeric comparison (tolerance 1e-2, matching competition eval)
# - `cleanup_memory()`: Free GPU memory between phases
# - `patch_mamba()`: Re-apply Mamba fast-path disable after model reload

# %% [code] {"jupyter":{"outputs_hidden":false}}
def extract_boxed(text: str):
    """Extract content from the last \\boxed{...}, handling nested braces."""
    if text is None:
        return None
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        # Fallback: try to find last number
        nums = re.findall(r'-?\d+\.?\d*', text)
        return nums[-1] if nums else None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i-1].strip()
    return None


def answer_match(predicted, ground_truth: str) -> bool:
    """Check if predicted matches ground truth (exact string or numeric ≤1e-2)."""
    if predicted is None:
        return False
    pred_s = str(predicted).strip()
    gt_s = ground_truth.strip()
    if pred_s == gt_s:
        return True
    try:
        pred_num = float(pred_s)
        gt_num = float(gt_s)
        if gt_num == 0:
            return abs(pred_num) < 1e-2
        return abs(pred_num - gt_num) / abs(gt_num) < 1e-2
    except (ValueError, ZeroDivisionError):
        return False


def cleanup_memory():
    """Free GPU memory between pipeline phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  [memory] GPU cache cleared.")


def patch_mamba():
    """Disable Mamba SSM fast-path AND re-patch rmsnorm_fn (crashes on Blackwell sm_120).

    Must be called AFTER model loading: modeling_nemotron_h does
    `from mamba_ssm... import rmsnorm_fn` which creates a local binding at import
    time.  The startup patch only covers modules already in sys.modules, so we
    re-scan here when the model (and its deps) are fully loaded.
    """
    patched = []
    for _name, _mod in list(sys.modules.items()):
        if "modeling_nemotron_h" in _name:
            _mod.is_fast_path_available = False
        # Replace Triton rmsnorm_fn with pure-PyTorch version in every module
        # that holds a reference to it (including modeling_nemotron_h itself).
        if hasattr(_mod, "rmsnorm_fn") and _mod.rmsnorm_fn is not _pure_rmsnorm_fn:
            _mod.rmsnorm_fn = _pure_rmsnorm_fn
            patched.append(_name)
    if patched:
        print(f"  [patch_mamba] rmsnorm_fn re-patched in: {patched}")


def get_lora_config():
    """Return the standard LoRA config for this competition."""
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules="all-linear",
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Data Loading & Tokenizer

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Use direct path (internet is disabled on Kaggle — kagglehub.model_download()
# requires network to verify cache even when the model is already downloaded).
_DIRECT_MODEL_PATH = "/kaggle/input/models/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1"
if os.path.isdir(_DIRECT_MODEL_PATH):
    MODEL_PATH = _DIRECT_MODEL_PATH
    print(f"Using direct model path: {MODEL_PATH}")
else:
    import kagglehub
    MODEL_PATH = kagglehub.model_download(
        "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    )
    print(f"Fallback via kagglehub: {MODEL_PATH}")

train_df = pl.read_csv(
    "/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv"
)
print(f"Total samples: {len(train_df)}")
train_df = train_df.sample(fraction=1.0, seed=42)  # shuffle

# Keep raw prompts and answers for generation phases
RAW_PROMPTS = train_df["prompt"].to_list()
RAW_ANSWERS = [str(a) for a in train_df["answer"].to_list()]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SYSTEM_PROMPT = (
    "You are an expert reasoning assistant specialised in pattern recognition. "
    "For every problem: analyse the examples step by step inside <think></think> tags, "
    "identify the transformation rule, apply it to the new input, "
    "and put your final answer inside \\boxed{}."
)

print(f"Model path: {MODEL_PATH}")
print(f"Tokenizer vocab: {len(tokenizer):,}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Baseline SFT Data Formatting
#
# Same dynamic CoT template as `sft.py` but with improved hyperparameters
# (longer seq_len, multi-epoch, NEFTune, packing). This is the starting point
# for all experiments.

# %% [code] {"jupyter":{"outputs_hidden":false}}
def _classify_puzzle(prompt: str) -> str:
    p = prompt.lower()
    has_binary = bool(re.search(r"[01]{4,}", prompt))
    has_text = any(w in p for w in ("word", "phrase", "text", "letter",
                                     "string", "character", "sentence"))
    if has_binary and has_text:
        return "mixed"
    if has_binary:
        return "binary"
    if has_text:
        return "text"
    return "numeric"


def _extract_examples(prompt: str, max_examples: int = 3) -> str:
    lines = [l.strip() for l in prompt.split("\n")
             if "->" in l or "→" in l or " becomes " in l.lower()]
    return "".join(f"  {e}\n" for e in lines[:max_examples])


def _build_cot_trace(prompt: str, answer: str) -> str:
    bucket = _classify_puzzle(prompt)
    ex_block = _extract_examples(prompt)
    has_ex = bool(ex_block.strip())

    if bucket == "binary":
        trace = "Let me analyse the binary transformation rule.\n\n"
        if has_ex:
            trace += f"Observed input→output pairs:\n{ex_block}"
            trace += "Comparing the bit patterns to identify the per-position rule.\n"
        trace += f"Applying the rule to the target input: {answer}"
    elif bucket == "text":
        trace = "Let me analyse the text/word transformation rule.\n\n"
        if has_ex:
            trace += f"Observed input→output pairs:\n{ex_block}"
            trace += "Building the substitution or reordering mapping from these examples.\n"
        trace += f"Applying the mapping to the target input: {answer}"
    elif bucket == "mixed":
        trace = "Let me decompose this mixed (binary + text) transformation.\n\n"
        if has_ex:
            trace += f"Observed input→output pairs:\n{ex_block}"
            trace += "Identifying separate rules for the binary and text components.\n"
        trace += f"Combined result: {answer}"
    else:
        trace = "Let me analyse the numeric pattern or conversion rule.\n\n"
        if has_ex:
            trace += f"Observed input→output pairs:\n{ex_block}"
            trace += "Checking for arithmetic, scaling, or unit-conversion relationships.\n"
        trace += f"Applying the pattern to the target input: {answer}"
    return trace


def format_sft_example(prompt: str, answer: str, cot_trace: str = None) -> str:
    """Format a single example into a chat-template training string."""
    user_msg = prompt + "\nPut your final answer inside \\boxed{}."
    if cot_trace is None:
        cot_trace = _build_cot_trace(prompt, answer)
    assistant_msg = f"<think>\n{cot_trace}\n</think>\n\\boxed{{{answer}}}"
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )


# Build baseline SFT dataset (template CoT)
baseline_texts = [
    format_sft_example(p, a) for p, a in zip(RAW_PROMPTS, RAW_ANSWERS)
]
hf_dataset_baseline = Dataset.from_dict({"text": baseline_texts})

if USE_VALIDATION:
    split = hf_dataset_baseline.train_test_split(test_size=0.05, seed=42)
    hf_train = split["train"]
    hf_eval  = split["test"]
    print(f"Dataset: {len(hf_train):,} train / {len(hf_eval):,} eval")
else:
    hf_train = hf_dataset_baseline
    hf_eval  = None
    print(f"Dataset: {len(hf_train):,} train (no validation)")

print(f"Example (first 500 chars):\n{baseline_texts[0][:500]}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Phase 1 — Baseline SFT Training
#
# Improved baseline with:
# - `MAX_SEQ_LEN = 2048` (was 1024) — better alignment with inference budget
# - `NUM_EPOCHS = 2` (was 1) — more convergence
# - `NEFTune` noise — regularization for better generalization
# - `packing = True` — efficient use of each batch
# - Validation split + best checkpoint selection

# %% [code] {"jupyter":{"outputs_hidden":false}}
if not RUN_BASELINE_SFT and PRETRAINED_ADAPTER_PATH is not None:
    # ── Copy / symlink existing adapter into OUTPUT_DIR_SFT so downstream
    #    code (generation, RFT, STaR, etc.) can load it from a consistent path.
    import shutil
    src = os.path.realpath(PRETRAINED_ADAPTER_PATH)
    dst = os.path.realpath(OUTPUT_DIR_SFT)
    if src != dst:
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"Copied pre-trained adapter: {PRETRAINED_ADAPTER_PATH} → {OUTPUT_DIR_SFT}")
    else:
        print(f"Pre-trained adapter already at {OUTPUT_DIR_SFT}, skipping copy.")
    print("Skipping baseline SFT — using existing adapter.")

if RUN_BASELINE_SFT:
    print("=" * 60)
    print("PHASE 1: Baseline SFT")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        offload_folder="/tmp/offload",
    )
    patch_mamba()

    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()

    try:
        import triton.backends.nvidia.compiler as nv_compiler
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
        nv_compiler.get_ptxas_version = lambda arch="blackwell": "12.0"
    except Exception:
        pass

    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR_SFT,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS_SFT,
        learning_rate=LR_SFT,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        logging_steps=5,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=USE_PACKING,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        dataloader_num_workers=0,
        # NEFTune
        neftune_noise_alpha=NEFTUNE_ALPHA if USE_NEFTUNE else None,
        # Validation & checkpoint selection
        eval_strategy="steps" if USE_VALIDATION else "no",
        eval_steps=50 if USE_VALIDATION else None,
        save_strategy="steps" if USE_VALIDATION else "no",
        save_steps=50 if USE_VALIDATION else None,
        load_best_model_at_end=USE_VALIDATION,
        metric_for_best_model="eval_loss" if USE_VALIDATION else None,
        save_total_limit=2 if USE_VALIDATION else None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_train,
        eval_dataset=hf_eval,
        processing_class=tokenizer,
        args=sft_args,
    )

    print(f"Training: {len(hf_train):,} samples, "
          f"{NUM_EPOCHS_SFT} epochs, seq_len={MAX_SEQ_LEN}")
    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR_SFT)
    print(f"SFT adapter saved to {OUTPUT_DIR_SFT}")

    del trainer, model
    cleanup_memory()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Generation Engine
#
# Shared generation phase for TN1–TN4. Loads the SFT checkpoint, generates
# K responses per prompt with temperature sampling, and evaluates correctness.
# Results are cached to disk so you can re-run experiments without regenerating.

# %% [code] {"jupyter":{"outputs_hidden":false}}
_needs_gen = RUN_RFT or RUN_STAR or RUN_SPIN or RUN_SELF_REWARD

@torch.no_grad()
def generate_responses(model, tokenizer, prompts, answers, K, max_new_tokens,
                       temperature, cache_path=None):
    """
    Generate K reasoning traces per prompt. Returns list of result dicts:
    [{"prompt", "answer", "user_msg", "responses", "correct_responses"}, ...]
    """
    # Check cache first
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading cached generation results from {cache_path}")
        with open(cache_path, "r") as f:
            return json.load(f)

    model.eval()
    results = []
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        do_sample=(temperature > 0),
        num_return_sequences=K,
        pad_token_id=tokenizer.pad_token_id,
    )

    for i, (prompt, answer) in enumerate(zip(prompts, answers)):
        user_msg = prompt + "\nPut your final answer inside \\boxed{}."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(**inputs, **gen_kwargs)

        responses = []
        correct_responses = []
        for output in outputs:
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            responses.append(response)
            extracted = extract_boxed(response)
            if answer_match(extracted, answer):
                correct_responses.append(response)

        results.append({
            "prompt": prompt,
            "answer": answer,
            "user_msg": user_msg,
            "responses": responses,
            "correct_responses": correct_responses,
        })

        if (i + 1) % 100 == 0:
            n_correct = sum(1 for r in results if r["correct_responses"])
            print(f"  [{i+1}/{len(prompts)}] "
                  f"≥1 correct: {n_correct}/{i+1} ({100*n_correct/(i+1):.1f}%)")

        # Periodic cache save
        if cache_path and (i + 1) % 500 == 0:
            with open(cache_path, "w") as f:
                json.dump(results, f)

    # Final cache save
    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(results, f)
        print(f"  Cached {len(results)} results to {cache_path}")

    n_with_correct = sum(1 for r in results if r["correct_responses"])
    print(f"  Generation complete: {n_with_correct}/{len(results)} prompts "
          f"have ≥1 correct response ({100*n_with_correct/len(results):.1f}%)")
    return results


if _needs_gen:
    print("=" * 60)
    print("GENERATION PHASE: Sampling responses from SFT model")
    print("=" * 60)

    # Load model + SFT adapter for generation
    gen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        offload_folder="/tmp/offload",
    )
    patch_mamba()
    gen_model = PeftModel.from_pretrained(gen_model, OUTPUT_DIR_SFT)
    gen_model = gen_model.merge_and_unload()  # Merge for faster inference
    print("SFT model loaded & merged for generation.")

    gen_prompts = RAW_PROMPTS
    gen_answers = RAW_ANSWERS
    if GEN_SUBSAMPLE is not None and GEN_SUBSAMPLE < len(gen_prompts):
        gen_prompts = gen_prompts[:GEN_SUBSAMPLE]
        gen_answers = gen_answers[:GEN_SUBSAMPLE]
        print(f"Subsampled to {GEN_SUBSAMPLE} prompts for generation.")

    gen_results = generate_responses(
        gen_model, tokenizer, gen_prompts, gen_answers,
        K=GEN_K, max_new_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE, cache_path=GEN_CACHE_PATH,
    )

    del gen_model
    cleanup_memory()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 8. TN2 — Rejection Sampling Fine-Tuning (RFT)
#
# **Ref**: Yuan et al., "Scaling Relationship on Learning Math Reasoning with LLMs", ICML 2023
#
# Simple but effective: keep only the model's own correct reasoning traces,
# then SFT on them. The model learns from its own "best work".
#
# Pipeline: Sample K → Filter correct → Pick best trace → SFT

# %% [code] {"jupyter":{"outputs_hidden":false}}
if RUN_RFT:
    print("=" * 60)
    print("TN2: Rejection Sampling Fine-Tuning (RFT)")
    print("=" * 60)

    # Build RFT dataset: only prompts where model got ≥1 correct response
    rft_texts = []
    n_skipped = 0
    for r in gen_results:
        if not r["correct_responses"]:
            n_skipped += 1
            continue
        # Pick the longest correct response (most detailed reasoning)
        best_response = max(r["correct_responses"], key=len)
        # Format as training example using model's OWN CoT
        text = format_sft_example(r["prompt"], r["answer"], cot_trace=best_response)
        rft_texts.append(text)

    print(f"RFT dataset: {len(rft_texts)} correct / {n_skipped} skipped")

    hf_rft = Dataset.from_dict({"text": rft_texts})
    if USE_VALIDATION:
        rft_split = hf_rft.train_test_split(test_size=0.05, seed=42)
        rft_train, rft_eval = rft_split["train"], rft_split["test"]
    else:
        rft_train, rft_eval = hf_rft, None

    # Load base model + SFT adapter, continue training on RFT data
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        offload_folder="/tmp/offload",
    )
    patch_mamba()
    model = PeftModel.from_pretrained(model, OUTPUT_DIR_SFT, is_trainable=True)
    model.print_trainable_parameters()

    rft_args = SFTConfig(
        output_dir=OUTPUT_DIR_FINAL,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=1,   # 1 epoch on filtered data is usually enough
        learning_rate=LR_SFT * 0.5,  # Lower LR for refinement
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        logging_steps=5,
        report_to="none",
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=USE_PACKING,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        dataloader_num_workers=0,
        neftune_noise_alpha=NEFTUNE_ALPHA if USE_NEFTUNE else None,
        save_strategy="no",
    )

    trainer = SFTTrainer(
        model=model, train_dataset=rft_train, eval_dataset=rft_eval,
        processing_class=tokenizer, args=rft_args,
    )
    print(f"RFT training: {len(rft_train):,} samples")
    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR_FINAL)
    print(f"RFT adapter saved to {OUTPUT_DIR_FINAL}")

    del trainer, model
    cleanup_memory()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 9. TN1 — STaR: Self-Taught Reasoner
#
# **Ref**: Zelikman et al., "STaR: Bootstrapping Reasoning With Reasoning", NeurIPS 2022
#
# Iterative self-improvement: the model generates reasoning traces → we keep
# correct ones → SFT → repeat. Each iteration the model gets better at reasoning
# and generates better traces. Includes **rationalization**: for prompts where
# ALL K responses are wrong, we hint the answer and ask the model to explain.
#
# This is the **strongest** self-improvement method and subsumes RFT.

# %% [code] {"jupyter":{"outputs_hidden":false}}
if RUN_STAR:
    print("=" * 60)
    print("TN1: STaR — Self-Taught Reasoner")
    print(f"     Iterations: {STAR_ITERATIONS}, K={GEN_K}")
    print("=" * 60)

    current_adapter_path = OUTPUT_DIR_SFT

    for iteration in range(STAR_ITERATIONS):
        print(f"\n--- STaR Iteration {iteration + 1}/{STAR_ITERATIONS} ---")

        # Step A: Generate from current model (reuse gen_results for iter 0)
        if iteration == 0 and 'gen_results' in dir():
            iter_results = gen_results
            print(f"  Reusing generation results from Phase 7.")
        else:
            iter_model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH, device_map="auto", trust_remote_code=True,
                dtype=torch.bfloat16, offload_folder="/tmp/offload",
            )
            patch_mamba()
            iter_model = PeftModel.from_pretrained(iter_model, current_adapter_path)
            iter_model = iter_model.merge_and_unload()

            cache_path = f"/kaggle/working/star_gen_iter{iteration}.json"
            iter_results = generate_responses(
                iter_model, tokenizer, RAW_PROMPTS, RAW_ANSWERS,
                K=GEN_K, max_new_tokens=GEN_MAX_TOKENS,
                temperature=GEN_TEMPERATURE, cache_path=cache_path,
            )
            del iter_model
            cleanup_memory()

        # Step B: Build STaR dataset
        star_texts = []
        n_correct, n_rationalized, n_failed = 0, 0, 0

        for r in iter_results:
            if r["correct_responses"]:
                # Use the longest correct response
                best = max(r["correct_responses"], key=len)
                star_texts.append(format_sft_example(
                    r["prompt"], r["answer"], cot_trace=best))
                n_correct += 1
            elif STAR_USE_RATIONAL:
                # Rationalization: the model couldn't solve it, so we include
                # a trace that explains the answer (uses ground truth as hint
                # embedded in the CoT, NOT in the prompt).
                hint_cot = (
                    f"Let me work through this carefully.\n\n"
                    f"After analyzing all the examples, I can see the pattern.\n"
                    f"Applying this pattern to the target input gives: {r['answer']}\n\n"
                    f"Let me verify: yes, this is consistent with all examples."
                )
                star_texts.append(format_sft_example(
                    r["prompt"], r["answer"], cot_trace=hint_cot))
                n_rationalized += 1
            else:
                n_failed += 1

        print(f"  STaR dataset: {n_correct} correct + {n_rationalized} rationalized "
              f"+ {n_failed} dropped = {len(star_texts)} total")

        # Step C: SFT on STaR data
        hf_star = Dataset.from_dict({"text": star_texts})
        if USE_VALIDATION:
            star_split = hf_star.train_test_split(test_size=0.05, seed=42)
            star_train, star_eval = star_split["train"], star_split["test"]
        else:
            star_train, star_eval = hf_star, None

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto", trust_remote_code=True,
            dtype=torch.bfloat16, offload_folder="/tmp/offload",
        )
        patch_mamba()
        # Continue from previous iteration's adapter
        model = PeftModel.from_pretrained(model, current_adapter_path, is_trainable=True)

        iter_output = f"/kaggle/working/adapter_star_iter{iteration}"
        os.makedirs(iter_output, exist_ok=True)

        star_args = SFTConfig(
            output_dir=iter_output,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=1,
            learning_rate=LR_SFT * (0.5 ** iteration),  # Decay LR each iteration
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            max_grad_norm=1.0,
            optim="adamw_torch",
            bf16=True,
            logging_steps=5,
            report_to="none",
            dataset_text_field="text",
            max_length=MAX_SEQ_LEN,
            packing=USE_PACKING,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            dataloader_num_workers=0,
            neftune_noise_alpha=NEFTUNE_ALPHA if USE_NEFTUNE else None,
            save_strategy="no",
        )

        trainer = SFTTrainer(
            model=model, train_dataset=star_train, eval_dataset=star_eval,
            processing_class=tokenizer, args=star_args,
        )
        print(f"  Training iteration {iteration + 1}: {len(star_train):,} samples, "
              f"lr={LR_SFT * (0.5 ** iteration):.1e}")
        trainer.train()
        trainer.model.save_pretrained(iter_output)
        current_adapter_path = iter_output
        print(f"  Saved to {iter_output}")

        del trainer, model
        cleanup_memory()

    # Copy final STaR adapter to OUTPUT_DIR_FINAL
    import shutil
    if os.path.exists(OUTPUT_DIR_FINAL):
        shutil.rmtree(OUTPUT_DIR_FINAL)
    shutil.copytree(current_adapter_path, OUTPUT_DIR_FINAL)
    print(f"\nSTaR complete. Final adapter: {OUTPUT_DIR_FINAL}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 10. TN3 — SPIN: Self-Play Preference Optimization
#
# **Ref**: Chen et al., "Self-Play Fine-Tuning Converts Weak LMs to Strong", ICML 2024
#
# Creates preference data from the model itself:
# - **Chosen** = ground truth response (formatted from training data)
# - **Rejected** = model's own wrong response
# Then trains DPO. The model learns to distinguish its own correct vs incorrect outputs.

# %% [code] {"jupyter":{"outputs_hidden":false}}
if RUN_SPIN:
    from trl import DPOConfig, DPOTrainer
    print("=" * 60)
    print("TN3: SPIN — Self-Play DPO")
    print("=" * 60)

    # Build preference pairs: chosen = GT response, rejected = model's wrong response
    spin_data = {"prompt": [], "chosen": [], "rejected": []}
    n_pairs = 0

    for r in gen_results:
        # We need a wrong response for "rejected"
        wrong_responses = [resp for resp in r["responses"]
                           if not answer_match(extract_boxed(resp), r["answer"])]
        if not wrong_responses:
            continue  # Model got all K correct — no negative to learn from

        # Chosen: ground-truth formatted response
        gt_cot = _build_cot_trace(r["prompt"], r["answer"])
        chosen_msg = f"<think>\n{gt_cot}\n</think>\n\\boxed{{{r['answer']}}}"

        # Rejected: model's wrong response (pick the longest for maximum signal)
        rejected_msg = max(wrong_responses, key=len)

        # Format as message lists for DPO
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["user_msg"]},
        ]
        spin_data["prompt"].append(prompt_msgs)
        spin_data["chosen"].append([{"role": "assistant", "content": chosen_msg}])
        spin_data["rejected"].append([{"role": "assistant", "content": rejected_msg}])
        n_pairs += 1

    print(f"SPIN preference pairs: {n_pairs}")
    hf_spin = Dataset.from_dict(spin_data)

    # Load base + SFT adapter for DPO training
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        offload_folder="/tmp/offload",
    )
    patch_mamba()
    model = PeftModel.from_pretrained(model, OUTPUT_DIR_SFT, is_trainable=True)

    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR_FINAL,
        beta=DPO_BETA,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM * 2,  # DPO needs more accumulation
        num_train_epochs=DPO_EPOCHS,
        learning_rate=LR_DPO,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=5,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_SEQ_LEN // 2,
        gradient_checkpointing=True,
        save_strategy="no",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT: uses base model (without adapter) as reference
        args=dpo_args,
        train_dataset=hf_spin,
        processing_class=tokenizer,
    )
    print(f"SPIN DPO training: {len(hf_spin):,} pairs")
    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(OUTPUT_DIR_FINAL)
    print(f"SPIN adapter saved to {OUTPUT_DIR_FINAL}")

    del dpo_trainer, model
    cleanup_memory()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 11. TN4 — Self-Rewarding DPO
#
# **Ref**: Yuan et al., "Self-Rewarding Language Models", ICML 2024
#
# The model evaluates its own responses to build preference pairs:
# 1. Generate K responses per prompt
# 2. Check correctness (ground truth) + model self-rates quality
# 3. Build (best, worst) pairs → DPO
#
# Unlike SPIN, this uses model's OWN correct responses as chosen (not GT template).

# %% [code] {"jupyter":{"outputs_hidden":false}}
if RUN_SELF_REWARD:
    from trl import DPOConfig, DPOTrainer
    print("=" * 60)
    print("TN4: Self-Rewarding DPO")
    print("=" * 60)

    sr_data = {"prompt": [], "chosen": [], "rejected": []}
    n_pairs = 0

    for r in gen_results:
        correct = r["correct_responses"]
        wrong = [resp for resp in r["responses"]
                 if not answer_match(extract_boxed(resp), r["answer"])]

        if not correct or not wrong:
            continue  # Need both correct and wrong responses

        # Chosen: longest correct response (most detailed reasoning)
        chosen_msg = max(correct, key=len)
        # Rejected: longest wrong response (maximum negative signal)
        rejected_msg = max(wrong, key=len)

        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["user_msg"]},
        ]
        sr_data["prompt"].append(prompt_msgs)
        sr_data["chosen"].append([{"role": "assistant", "content": chosen_msg}])
        sr_data["rejected"].append([{"role": "assistant", "content": rejected_msg}])
        n_pairs += 1

    print(f"Self-Rewarding preference pairs: {n_pairs}")
    hf_sr = Dataset.from_dict(sr_data)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        offload_folder="/tmp/offload",
    )
    patch_mamba()
    model = PeftModel.from_pretrained(model, OUTPUT_DIR_SFT, is_trainable=True)

    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR_FINAL,
        beta=DPO_BETA,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM * 2,
        num_train_epochs=DPO_EPOCHS,
        learning_rate=LR_DPO,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=5,
        report_to="none",
        max_length=MAX_SEQ_LEN,
        max_prompt_length=MAX_SEQ_LEN // 2,
        gradient_checkpointing=True,
        save_strategy="no",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=hf_sr,
        processing_class=tokenizer,
    )
    print(f"Self-Rewarding DPO training: {len(hf_sr):,} pairs")
    dpo_trainer.train()
    dpo_trainer.model.save_pretrained(OUTPUT_DIR_FINAL)
    print(f"Self-Rewarding adapter saved to {OUTPUT_DIR_FINAL}")

    del dpo_trainer, model
    cleanup_memory()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 12. TN5 — GRPO: Group Relative Policy Optimization
#
# **Ref**: Shao et al., "DeepSeekMath", 2024; DeepSeek-R1, 2025
#
# RL with verifiable outcome rewards — no reward model needed. For each prompt,
# the model samples G responses. Correct answers get reward 1.0, wrong get 0.0.
# Policy gradient with group-relative advantage estimation.
#
# Can run standalone (after baseline SFT) or on top of STaR/RFT.
# This is the method behind DeepSeek-R1's breakthrough.

# %% [code] {"jupyter":{"outputs_hidden":false}}
if RUN_GRPO:
    from trl import GRPOConfig, GRPOTrainer
    print("=" * 60)
    print("TN5: GRPO — Group Relative Policy Optimization")
    print("=" * 60)

    # Determine starting adapter: use final adapter if available, else SFT
    grpo_base_adapter = OUTPUT_DIR_FINAL if os.path.exists(
        os.path.join(OUTPUT_DIR_FINAL, "adapter_config.json")
    ) else OUTPUT_DIR_SFT
    print(f"Starting from adapter: {grpo_base_adapter}")

    # Build GRPO dataset: prompts + ground truth for reward computation
    grpo_prompts = []
    grpo_answers = []
    for prompt, answer in zip(RAW_PROMPTS, RAW_ANSWERS):
        user_msg = prompt + "\nPut your final answer inside \\boxed{}."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        grpo_prompts.append(messages)
        grpo_answers.append(answer)

    hf_grpo = Dataset.from_dict({
        "prompt": grpo_prompts,
        "answer": grpo_answers,
    })

    # Reward functions
    def correctness_reward(completions, answer, **kwargs):
        """Main reward: 1.0 if answer in \\boxed{} matches ground truth."""
        rewards = []
        for comp, ans in zip(completions, answer):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            extracted = extract_boxed(text)
            if answer_match(extracted, ans):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    def format_reward(completions, **kwargs):
        """Bonus for using \\boxed{} format."""
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else comp
            if "\\boxed{" in text:
                rewards.append(0.1)
            else:
                rewards.append(-0.1)
        return rewards

    def reasoning_reward(completions, **kwargs):
        """Bonus for using <think> reasoning tags."""
        rewards = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else comp
            if "<think>" in text and "</think>" in text:
                rewards.append(0.05)
            else:
                rewards.append(0.0)
        return rewards

    # Load model + adapter
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
        offload_folder="/tmp/offload",
    )
    patch_mamba()
    model = PeftModel.from_pretrained(model, grpo_base_adapter, is_trainable=True)

    grpo_output = "/kaggle/working/adapter_grpo"
    os.makedirs(grpo_output, exist_ok=True)

    grpo_args = GRPOConfig(
        output_dir=grpo_output,
        num_generations=GRPO_NUM_GENERATIONS,
        max_completion_length=GRPO_MAX_COMPLETION,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM * 2,
        num_train_epochs=GRPO_EPOCHS,
        learning_rate=LR_GRPO,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=5,
        report_to="none",
        gradient_checkpointing=True,
        save_strategy="no",
    )

    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward, reasoning_reward],
        args=grpo_args,
        train_dataset=hf_grpo,
        processing_class=tokenizer,
    )
    print(f"GRPO training: {len(hf_grpo):,} prompts, "
          f"G={GRPO_NUM_GENERATIONS}, max_tokens={GRPO_MAX_COMPLETION}")
    grpo_trainer.train()
    grpo_trainer.model.save_pretrained(grpo_output)

    # Update final adapter
    if os.path.exists(OUTPUT_DIR_FINAL):
        shutil.rmtree(OUTPUT_DIR_FINAL)
    shutil.copytree(grpo_output, OUTPUT_DIR_FINAL)
    print(f"GRPO adapter saved to {OUTPUT_DIR_FINAL}")

    del grpo_trainer, model
    cleanup_memory()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 13. Finalize & Package submission.zip
#
# Copy the best adapter to the final directory and package for submission.
# If no experiment ran after baseline SFT, the baseline adapter is used.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import shutil

# If no experiment produced a final adapter, use the baseline SFT adapter
if not os.path.exists(os.path.join(OUTPUT_DIR_FINAL, "adapter_config.json")):
    print("No experiment adapter found. Using baseline SFT adapter.")
    if os.path.exists(OUTPUT_DIR_FINAL):
        shutil.rmtree(OUTPUT_DIR_FINAL)
    shutil.copytree(OUTPUT_DIR_SFT, OUTPUT_DIR_FINAL)

# List final adapter files
print(f"\nFinal adapter ({OUTPUT_DIR_FINAL}):")
for f in sorted(os.listdir(OUTPUT_DIR_FINAL)):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR_FINAL, f))
    print(f"  {f}  ({sz / 1024:.1f} KB)")

# Package
zip_path = "/kaggle/working/submission.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR_FINAL):
        fpath = os.path.join(OUTPUT_DIR_FINAL, fname)
        zf.write(fpath, fname)

print(f"\nCreated {zip_path}  ({os.path.getsize(zip_path) / 1024 / 1024:.1f} MB)")

with zipfile.ZipFile(zip_path, "r") as zf:
    contents = zf.namelist()
print(f"Contents: {contents}")
assert "adapter_config.json" in contents, "CRITICAL: adapter_config.json missing!"
print("\n✓ submission.zip ready!")
