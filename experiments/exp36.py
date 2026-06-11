# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # EXP36 — LogicPuzzleRL offline DPO pairs for cryptarithm  (Batch-4 Idea 3, P2, T3)
# ============================================================
# Paper: arXiv:2506.04821 — check official repo trước khi chạy:
#   curl -s https://api.github.com/search/repositories?q=LogicPuzzleRL | python3 -m json.tool | grep html_url
# HIGH RISK — XL effort. Cần vLLM temperature sampling infrastructure.
# Prerequisite CỨNG:
#   1. exp29 (DPO infrastructure) đã hoạt động
#   2. Pass@20 cryptarithm ≥ 2% (falsification test — verify trước khi build DPO pairs)
# Upstream:
#   1. vLLM n=20 temperature=0.8 trên cryptarithm/equation_numeric_guess (~873 problems)
#   2. build_dpo_pairs.py (extend từ exp29) — threshold ≥1 correct + ≥1 incorrect
#   3. DPO training 150 steps với SFT anchor 20%, beta=0.1, LR=1e-5
# ============================================================

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os

NEMOTRON_MASTER = os.path.join(os.path.dirname(__file__), "..", "nemotron-master")

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
ADAPTER_PATH = None   # adapter 0.86 hoặc kết quả exp27/28

# DPO hyperparams (từ Idea 3 description)
N_ROLLOUTS = 20
TEMPERATURE = 0.8
DPO_STEPS = 150
DPO_LR = 1e-5
DPO_BETA = 0.1
SFT_ANCHOR_RATIO = 0.2

TARGET_CATEGORIES = ["cryptarithm_deduce", "cryptarithm_guess", "equation_numeric_guess"]

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Pre-check bắt buộc — Pass@20 cryptarithm

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Chạy bước này TRƯỚC khi bất kỳ implementation nào:
PASS_CHECK_CMD = f"""
cd {NEMOTRON_MASTER} && \\
uv run python3 infer_slice.py \\
  --base    {BASE_MODEL} \\
  --adapter {ADAPTER_PATH} \\
  --slice   eval_slice.jsonl \\   # chỉ dùng cryptarithm subset
  --temperature {TEMPERATURE} \\
  --n_samples {N_ROLLOUTS} \\
  --out     cryptarithm_rollouts_check.jsonl

# Count problems với ít nhất 1 correct trong 20 rollouts:
python3 -c "
import json
from pathlib import Path
# Load eval_slice để biết correct answers
# Load rollout preds
# Filter cryptarithm category
# Count pass@20
print('TODO: implement sau khi biết schema của infer_slice output')
"
"""
print("Pre-check — Pass@20 cryptarithm:")
print(PASS_CHECK_CMD)
print()
print("ABORT THRESHOLD: < 20 problems với ≥1 correct trong 20 rollouts (~2% của 873)")
print("Nếu không đạt → hypothesis sai → ABORT toàn bộ exp36")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Bước 1 — Full rollout collection (chỉ chạy sau pre-check passed)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Tạo cryptarithm/equation_guess slice từ eval_slice.jsonl (hoặc full test problems)
FULL_ROLLOUT_CMD = f"""
cd {NEMOTRON_MASTER} && \\
uv run python3 infer_slice.py \\
  --base    {BASE_MODEL} \\
  --adapter {ADAPTER_PATH} \\
  --slice   cryptarithm_eq_guess_problems.jsonl \\
  --temperature {TEMPERATURE} \\
  --n_samples {N_ROLLOUTS} \\
  --out     rollouts_exp36.jsonl
# Runtime ước tính: ~873 problems × 20 rollouts × ~2 min/rollout = ~29h (vLLM batching giảm ~10x)
"""
print("Full rollout (sau pre-check):")
print(FULL_ROLLOUT_CMD)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Bước 2–3 — Build pairs và DPO training

# %% [code] {"jupyter":{"outputs_hidden":false}}
BUILD_AND_TRAIN_CMD = f"""
# Build pairs (extend build_dpo_pairs.py từ exp29):
cd {NEMOTRON_MASTER} && \\
uv run python3 build_dpo_pairs.py \\
  --slice  cryptarithm_eq_guess_problems.jsonl \\
  --preds  rollouts_exp36.jsonl \\
  --out    dpo_pairs_exp36.jsonl

# Verify: >= 20 valid pairs (nếu không → abort)
python3 -c "
import json
pairs = [json.loads(l) for l in open('dpo_pairs_exp36.jsonl')]
print(f'Valid pairs: {{len(pairs)}}')
assert len(pairs) >= 20, 'Insufficient pairs — ABORT'
"

# DPO training ({DPO_STEPS} steps, SFT anchor {SFT_ANCHOR_RATIO*100:.0f}%):
# Dùng DPO infrastructure từ exp29
"""
print(BUILD_AND_TRAIN_CMD)
print()
print(f"Falsification: accuracy trên 30 held-out eval_slice problems (không dùng trong pairs)")
print(f"Nếu ≤ baseline 0.86 → DPO chỉ memorize pairs, không generalize → abort")
