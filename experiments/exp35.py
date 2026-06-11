# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # EXP35 — REDI negative-trace training  (Batch-4 Idea 7, P1, T1)
# ============================================================
# Paper: arXiv:2505.24850
# Official source: enhance_cot/redi/ (github.com/Tim-Siu/reinforcement-distillation)
# Key files để đọc:
#   - enhance_cot/redi/rllm/          — core REDI algorithm (framework-agnostic logic)
#   - enhance_cot/redi/experiments_trl/open_r1_dpo.py — TRL implementation (reference)
# Upstream:
#   1. Collect negative traces cho 1,167 rule_unknown problems (greedy infer)
#   2. loss_config.py — thêm REDILossConfig (REINFORCE-style negative loss)
#   3. train_sft.py Stage-2: 50–100 steps với REDI objective, LR=5e-6, SFT-mix 30%
# Prerequisite CỨNG: exp18 (batch-2) đã chạy → confirm DPO/preference viable trên Nemotron-H
# Rollback: dùng SFT-only adapter, bỏ Stage-2
# ============================================================

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os

NEMOTRON_MASTER = os.path.join(os.path.dirname(__file__), "..", "nemotron-master")
ENHANCE_COT = os.path.join(os.path.dirname(__file__), "..", "enhance_cot")
REDI_DIR = os.path.join(ENHANCE_COT, "redi")

# REDI Stage-2 hyperparams
REDI_STEPS = 100
REDI_LR = 5e-6
SFT_MIX_RATIO = 0.3   # 30% SFT positives mixed để tránh catastrophic forgetting

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Bước 0 — Đọc REDI algorithm

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(f"REDI core algorithm: {os.path.join(REDI_DIR, 'rllm/')}")
print(f"REDI TRL reference: {os.path.join(REDI_DIR, 'experiments_trl/open_r1_dpo.py')}")
print()
print("Key questions khi đọc rllm/:")
print("1. Loss formula chính xác: REINFORCE hay DPO hay custom?")
print("2. Negative trace được weight như thế nào so với positive?")
print("3. Có reference model cần không? (reference-free vs KL-anchored)")
print()
print("⚠️ KHÔNG port TRL code trực tiếp — adapt algorithm sang Tinker LossConfig interface")
print("⚠️ Verify LossFnType.__args__ có 'redi' hay không → nếu không → implement từ đầu")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Bước 1 — Collect negative traces (GPU)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Identify rule_unknown problems từ corpus:
COLLECT_CMD = f"""
cd {NEMOTRON_MASTER}

# Tạo rule_unknown_problems.jsonl từ corpus entries với status != rule_found
# Đọc corpus_preprocessed.jsonl và lọc theo category + status
python3 -c "
import json
unknown = []
with open('problems.jsonl') as f:
    for line in f:
        p = json.loads(line)
        # TODO: xác nhận field name chứa rule status trong problems.jsonl
        # Đọc một vài dòng đầu để biết schema
        break
print('Schema check — đọc problems.jsonl trước khi filter')
"

# Sau khi biết schema:
uv run python3 infer_slice.py \\
  --base    nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
  --adapter <path-to-SFT-adapter> \\
  --slice   rule_unknown_problems.jsonl \\
  --out     negatives_exp35.jsonl
"""
print("Step 1 — Collect negatives:")
print(COLLECT_CMD)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Bước 2 — REDILossConfig trong loss_config.py

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Sau khi đọc rllm/ và hiểu formula, implement REDILossConfig.
# Pattern: extend LossConfig, override name với Tinker-supported type
# (hoặc nếu Tinker không support custom type, wrap qua CrossEntropyLossConfig với negative weights)

# Skeleton structure (KHÔNG implement — chỉ show interface):
# class REDILossConfig(LossConfig):
#     negative_weight: float = 1.0  # weight của negative loss term
#     # ... theo exact formula từ rllm/

# Trong train_sft.py main():
# Stage-1: SFT trên positives (như hiện tại)
# Stage-2: REDI trên negatives (100 steps, LR=5e-6, SFT-mix 30%)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Falsification

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("Falsification sau Stage-2 REDI:")
print("  uv run python3 infer_slice.py → eval_slice.py → score per-category")
print()
print(f"Hard threshold: gravity/cipher accuracy KHÔNG giảm > 1 pp vs Stage-1 baseline")
print(f"Nếu giảm → Stage-2 overcorrecting → reduce REDI_LR hoặc increase SFT_MIX_RATIO")
print(f"Nếu cryptarithm không improve > 0 pp → REDI signal quá yếu → abandon")
