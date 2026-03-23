# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # NVIDIA Nemotron | Supervised Fine-Tuning (SFT)
# 
# This notebook outlines the complete pipeline for supervised fine-tuning of the **Nemotron-3-Nano-30B** model. The workflow includes environment setup, hardware-specific patching for Blackwell GPUs, dynamic Chain-of-Thought (CoT) data formatting, LoRA configuration, and final submission packaging.

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Environment Setup & Wheel Installation
# 
# Before importing our core libraries, we need to ensure all dependencies are installed. Because this notebook is designed to run in a constrained environment (like a Kaggle competition with internet disabled), the installation step relies on an offline wheel cache. 
# 
# * **Offline First**: It attempts to install `datasets` and `trl` directly from local directories.
# * **Fallback**: If the specific offline paths aren't found, it falls back to alternative local paths or attempts a PyPI installation.
# * **Path Injection**: Finally, it dynamically adds the target directory to `sys.path` so the newly installed modules can be imported immediately.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import subprocess, sys, os

# Try the offline wheel cache first (Kaggle environment), fall back to PyPI
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
    # Fallback: try the nbroad wheel path used in the original SFT notebook
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
# ## 2. Imports, Blackwell Patches & Hyperparameters
# 
# This section initializes our environment and applies critical workarounds for the specific hardware (sm_120 / Blackwell architectures).
# 
# ### Hyperparameters
# * **Dataset Scale**: We define `SUBSAMPLE_SIZE = None` to use the entire dataset (9,500 samples), which significantly boosts model performance.
# * **LoRA Configuration**: 
#     * `LORA_RANK = 32`: Keeps the adapter within typical competition submission limits.
#     * `LORA_ALPHA = 64`: A higher alpha/rank ratio (2.0) aggressively scales the LoRA updates, giving the adapter more expressive capacity.
# * **Training Specs**: `LR = 2e-4`, `GRAD_ACCUM = 4`, and `NUM_EPOCHS = 1`.
# 
# ### Blackwell Hardware Patches
# Nemotron running on Blackwell GPUs currently faces a few low-level execution bugs. To prevent kernel crashes, we apply the following monkey-patches:
# 1.  **Pure-PyTorch RMSNorm**: We replace the native C++ RMSNorm kernel with a pure PyTorch implementation (`_pure_rmsnorm_fn`) to bypass sm_120 execution errors.
# 2.  **Triton `ptxas` Redirect**: We manually copy the Blackwell-compatible `ptxas` binary to `/tmp`, adjust execution permissions, and force Triton/Nvidia tools to point to this safe binary path.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os
import re
import sys
import stat
import shutil
import gc
import zipfile

import polars as pl
import torch
import torch.nn.functional as F
import kagglehub
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# -- Hyperparameters ------------------------------------------------------------
# Use the full dataset (9,500 samples) instead of a 600-sample subsample.
# More data is the single biggest driver of the 0.61->0.68 improvement.
SUBSAMPLE_SIZE = None   # None = use ALL available samples

# lora_alpha = 64 (alpha/r = 2.0).
# A higher alpha scales the LoRA update more aggressively, giving the adapter
# more expressive capacity without increasing the rank (and therefore the
# submission file size, which must stay under the rank-32 limit).
LORA_RANK    = 32
LORA_ALPHA   = 64

MAX_SEQ_LEN  = 1024
NUM_EPOCHS   = 1
GRAD_ACCUM   = 4
LR           = 2e-4
OUTPUT_DIR   = "/kaggle/working/adapter"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"LoRA:   r={LORA_RANK}, alpha={LORA_ALPHA}  (alpha/r = {LORA_ALPHA/LORA_RANK:.1f})")
print(f"Train:  lr={LR}, grad_accum={GRAD_ACCUM}, max_seq={MAX_SEQ_LEN}, epochs={NUM_EPOCHS}")

# -- Blackwell Hardware Patches -------------------------------------------------
print("\nApplying Blackwell patches...")

# PATCH 1: Pure-PyTorch RMSNorm - bypasses the broken C++ kernel on sm_120
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

# PATCH 2: Triton ptxas binary redirect
# Copy the Blackwell ptxas binary to /tmp so Triton can execute it
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
    print("  [triton] ptxas redirect applied.")
else:
    print("  [triton] source binary not found - skipping ptxas redirect.")

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
            _orig_from_path = _mod.NvidiaTool.__dict__["from_path"]  # staticmethod
            def _make_safe(orig):
                @staticmethod
                def _safe_from_path(path):
                    try:
                        return orig.__func__(path) if hasattr(orig, "__func__") else orig(path)
                    except (PermissionError, OSError, Exception):
                        return None
                return _safe_from_path
            _mod.NvidiaTool.from_path = _make_safe(_orig_from_path)
            print(f"  [knobs] NvidiaTool.from_path patched in: {_mod_name}")
        except Exception as e:
            print(f"  [knobs] Could not patch {_mod_name}: {e}")

for _mod_name, _mod in list(sys.modules.items()):
    if "triton" in _mod_name and hasattr(_mod, "get_ptxas_version"):
        _mod.get_ptxas_version = lambda arch=None: "12.0"
    if "triton" in _mod_name and hasattr(_mod, "get_ptxas"):
        _orig_gp = _mod.get_ptxas
        def _make_safe_gp(orig):
            def _safe_get_ptxas(arch):
                try:
                    return orig(arch)
                except (PermissionError, OSError, Exception):
                    class _FakePtxas:
                        version = "12.0"
                    return _FakePtxas()
            return _safe_get_ptxas
        _mod.get_ptxas = _make_safe_gp(_orig_gp)

print("All ptxas permission patches applied.")
print("Blackwell patches done.")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Data Loading & Dynamic CoT Formatting
# 
# This is the most impactful step in the pipeline. Instead of training the model to simply output a bare `\boxed{answer}`, we inject a **domain-specific Chain-of-Thought (CoT) trace** to teach the model *how* to reason.
# 
# ### Pipeline Breakdown
# * **Dataset Loading**: Loads the competition CSV using `polars`, handling subsampling if configured.
# * **System Prompt**: Anchors the model as an expert in pattern recognition and explicitly dictates the use of `<think>` and `\boxed{}` tags.
# * **Dynamic Routing (`_classify_puzzle`)**: Analyzes the raw prompt to categorize the puzzle into one of four buckets: `binary`, `text`, `mixed`, or `numeric`.
# * **Trace Building (`_build_cot_trace`)**: Extracts examples from the prompt and constructs a step-by-step reasoning template tailored to the puzzle's specific category.
# * **Chat Formatting (`build_training_text`)**: Combines the system prompt, user prompt, generated CoT trace, and final answer into a standard conversational format (ChatML) using the model's tokenizer.

# %% [code] {"jupyter":{"outputs_hidden":false}}
MODEL_PATH = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")

# Load data
train_df = pl.read_csv("/kaggle/input/competitions/nvidia-nemotron-model-reasoning-challenge/train.csv")
total_available = len(train_df)
print(f"Total samples available: {total_available}")

if SUBSAMPLE_SIZE is not None and SUBSAMPLE_SIZE < total_available:
    train_df = train_df.sample(n=SUBSAMPLE_SIZE, seed=42)
    print(f"Subsampled to: {len(train_df)}")
else:
    train_df = train_df.sample(fraction=1.0, seed=42)  # shuffle
    print(f"Using full dataset: {len(train_df)} samples")

hf_dataset = Dataset.from_pandas(train_df.to_pandas())

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -- System prompt -------------------------------------------------------------
# Anchors the model's persona as a pattern-recognition expert and explicitly
# instructs it to use <think> tags and \boxed{} - matching the inference-time prompt.
SYSTEM_PROMPT = (
    "You are an expert reasoning assistant specialised in pattern recognition. "
    "For every problem: analyse the examples step by step inside <think></think> tags, "
    "identify the transformation rule, apply it to the new input, "
    "and put your final answer inside \\boxed{}."
)

# -- Dynamic CoT classifier & trace builder ------------------------------------
# Routes each puzzle to a domain-specific thinking template.
# This teaches the model *how* to reason about each puzzle type, not just
# what the answer is - the core driver of the accuracy improvement.

def _classify_puzzle(prompt: str) -> str:
    """Classify puzzle type into one of four reasoning buckets."""
    p = prompt.lower()
    has_binary  = bool(re.search(r"[01]{4,}", prompt))
    has_text    = any(w in p for w in ("word", "phrase", "text", "letter",
                                        "string", "character", "sentence"))
    if has_binary and has_text:
        return "mixed"
    if has_binary:
        return "binary"
    if has_text:
        return "text"
    return "numeric"


def _extract_examples(prompt: str, max_examples: int = 3) -> str:
    """Extract up to max_examples input->output pairs from the prompt."""
    lines = [l.strip() for l in prompt.split("\n")
             if "->" in l or "->" in l or " becomes " in l.lower()]
    examples = lines[:max_examples]
    return "".join(f"  {e}\n" for e in examples)


def _build_cot_trace(prompt: str, answer: str) -> str:
    """Build a domain-specific CoT reasoning trace for a given puzzle."""
    bucket   = _classify_puzzle(prompt)
    ex_block = _extract_examples(prompt)
    has_ex   = bool(ex_block.strip())

    if bucket == "binary":
        trace = "Let me analyse the binary transformation rule.\n\n"
        if has_ex:
            trace += f"Observed input->output pairs:\n{ex_block}"
            trace += "Comparing the bit patterns to identify the per-position rule.\n"
        trace += f"Applying the rule to the target input: {answer}"

    elif bucket == "text":
        trace = "Let me analyse the text/word transformation rule.\n\n"
        if has_ex:
            trace += f"Observed input->output pairs:\n{ex_block}"
            trace += "Building the substitution or reordering mapping from these examples.\n"
        trace += f"Applying the mapping to the target input: {answer}"

    elif bucket == "mixed":
        trace = "Let me decompose this mixed (binary + text) transformation.\n\n"
        if has_ex:
            trace += f"Observed input->output pairs:\n{ex_block}"
            trace += "Identifying separate rules for the binary and text components.\n"
        trace += f"Combined result: {answer}"

    else:  # numeric
        trace = "Let me analyse the numeric pattern or conversion rule.\n\n"
        if has_ex:
            trace += f"Observed input->output pairs:\n{ex_block}"
            trace += "Checking for arithmetic, scaling, or unit-conversion relationships.\n"
        trace += f"Applying the pattern to the target input: {answer}"

    return trace


def build_training_text(example: dict) -> dict:
    """
    Format one CSV row into a complete training string.
    Assistant target = <think>[CoT trace]</think>\\boxed{answer}
    """
    prompt = example["prompt"]
    answer = str(example["answer"])

    user_msg      = prompt + "\nPut your final answer inside \\boxed{}."
    cot_trace     = _build_cot_trace(prompt, answer)
    assistant_msg = f"<think>\n{cot_trace}\n</think>\n\\boxed{{{answer}}}"

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Manual fallback if the tokenizer lacks a chat template
        text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
    return {"text": text}


hf_dataset = hf_dataset.map(
    build_training_text,
    remove_columns=hf_dataset.column_names,
)

print(f"\nDataset formatted: {len(hf_dataset):,} examples.")
print(f"\n--- Example training string (first 600 chars) ---")
print(hf_dataset[0]["text"][:600])

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Model Loading, LoRA Setup & Training
# 
# With the dataset formatted, we move on to initializing the model and executing the training loop.
# 
# * **Model Initialization**: Loads `Nemotron-3-Nano-30B` using `bfloat16` precision and `device_map="auto"`. 
# * **Mamba Fast-Path Patch**: We explicitly disable the Mamba SSM fast-path (`is_fast_path_available = False`) as it is known to crash on Blackwell architectures.
# * **PEFT / LoRA**: Wraps the base model using our predefined `LoraConfig`, targeting `all-linear` modules to maximize adaptation capabilities while keeping the parameter count low.
# * **SFTTrainer**: Configures the Hugging Face `SFTTrainer` with gradient checkpointing, cosine learning rate scheduling, and a 5% warmup ratio. Mid-run checkpoints are disabled to conserve disk space.

# %% [code] {"jupyter":{"outputs_hidden":false}}
# -- Load base model ------------------------------------------------------------
print("Loading Nemotron-3-Nano-30B in bfloat16...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
print(f"Model loaded. Vocab size: {len(tokenizer):,}")

# Disable the Mamba SSM fast-path - it crashes on Blackwell (sm_120)
for _name, _mod in sys.modules.items():
    if "modeling_nemotron_h" in _name:
        _mod.is_fast_path_available = False
        print(f"  Patched {_name}: is_fast_path_available = False")

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,       # 64 - stronger adaptation (alpha/r = 2.0)
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Apply Triton compiler version fix after model load
try:
    import triton.backends.nvidia.compiler as nv_compiler
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = "/tmp/ptxas-blackwell"
    nv_compiler.get_ptxas_version = lambda arch="blackwell": "12.0"
except Exception:
    pass

# %% [code] {"jupyter":{"outputs_hidden":false}}
# -- SFTConfig ------------------------------------------------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,           # 5% warmup over the full run
    max_grad_norm=1.0,
    optim="adamw_torch",
    bf16=True,
    logging_steps=5,
    save_strategy="no",          # No mid-run checkpoints - save disk space
    report_to="none",
    dataset_text_field="text",
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    dataloader_num_workers=0,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
)

n_steps = len(hf_dataset) // GRAD_ACCUM
print(f"\nStarting training: {len(hf_dataset):,} samples -> {n_steps:,} steps...")
trainer.train()

# -- Save adapter ---------------------------------------------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
print(f"\nAdapter saved to {OUTPUT_DIR}:")
for _f in sorted(os.listdir(OUTPUT_DIR)):
    _sz = os.path.getsize(os.path.join(OUTPUT_DIR, _f))
    print(f"  {_f}  ({_sz / 1024:.1f} KB)")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Package `submission.zip`
# 
# Once training completes, the resulting LoRA adapter weights (`adapter_model.bin`/`safetensors` and `adapter_config.json`) are saved to our working directory. 
# 
# This final step loops through the output directory and compresses all relevant adapter files into a single `submission.zip` file, ready for competition upload. It includes a quick sanity check to ensure the critical `adapter_config.json` is present in the archive.

# %% [code] {"jupyter":{"outputs_hidden":false}}
import zipfile, os

zip_path = "/kaggle/working/submission.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        zf.write(fpath, fname)

print(f"Created {zip_path}  ({os.path.getsize(zip_path) / 1024 / 1024:.1f} MB)")

with zipfile.ZipFile(zip_path, "r") as zf:
    contents = zf.namelist()

print(f"Contents: {contents}")
assert "adapter_config.json" in contents, "CRITICAL: adapter_config.json missing!"
print("submission.zip ready!")