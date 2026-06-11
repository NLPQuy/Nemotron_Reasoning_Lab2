# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # EXP38 — RL→SFT ordering: DPO trước, SFT sau  (Batch-4 Idea 12, P1, T2)
# ============================================================
# Paper: arXiv:2509.21128v2 — NO official repo.
# HIGH RISK. Feasibility 2/5. Chỉ chạy sau exp29 và exp18 (batch-2) đã validated.
# Hypothesis: DPO 50 steps (squeeze correct reasoning mode) →
#             SFT full corpus 1000 steps (expand coverage) = tốt hơn SFT-only.
# Warning: Paper study *online* RL; chúng ta dùng offline DPO → squeeze effect yếu hơn.
# Prerequisite CỨNG:
#   1. exp18 (batch-2) confirm DPO viable trên Nemotron-H
#   2. exp29 DPO pairs (dpo_pairs_exp29.jsonl) available
# Rollback: nếu Stage-A eval_slice < 0.85 → dùng SFT-only (baseline)
# ============================================================

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os

NEMOTRON_MASTER = os.path.join(os.path.dirname(__file__), "..", "nemotron-master")

# Stage-A (DPO) config — squeeze phase
DPO_PAIRS = os.path.join(NEMOTRON_MASTER, "dpo_pairs_exp29.jsonl")
STAGE_A_STEPS = 50
STAGE_A_LR = 1e-5
STAGE_A_BETA = 0.1
STAGE_A_SFT_MIX = 0.10    # 10% SFT mix (thấp để maximize squeeze signal)
STAGE_A_ABORT_THRESHOLD = 0.85   # eval_slice accuracy sau Stage-A

# Stage-B (SFT) config — expand phase
STAGE_B_STEPS = 1000
STAGE_B_LR = 2e-4           # standard SFT LR

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Pre-flight check

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("=== Pre-flight checks ===")
print()
print("1. exp18 (batch-2) đã chạy và kết quả DPO viable?")
print("   → Xem tracker/leaderboard.md hoặc tracker/rounds/ cho exp18 result")
print("   → Nếu exp18 không improve → DPO không hoạt động trên Nemotron-H → ABORT exp38")
print()
print("2. dpo_pairs_exp29.jsonl tồn tại?")
dpo_pairs_exists = os.path.exists(DPO_PAIRS)
print(f"   → {DPO_PAIRS}: {'EXISTS ✓' if dpo_pairs_exists else 'MISSING ✗ — chạy exp29 trước'}")
print()
print("3. Output entropy check — verify DPO thực sự 'squeeze' distribution:")
print("   Sau Stage-A, compute token-level entropy của adapter trên eval_slice")
print("   Nếu entropy KHÔNG giảm > 5% so với SFT baseline → squeeze effect không đủ → abort")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Stage-A — DPO squeeze (50 steps)

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("=== Stage-A: DPO squeeze ===")
STAGE_A_CMD = f"""
cd {NEMOTRON_MASTER}

# Chạy DPO với SFT mix thấp (10%) để maximize squeeze effect
# Dùng DPO infrastructure từ exp29
# NOTE: start từ adapter 0.86 (KHÔNG từ SFT-trained adapter — đây là RL→SFT bắt đầu từ base)
# Training command phụ thuộc vào DPO path đã được setup trong exp29

# Sau Stage-A: eval_slice check bắt buộc
uv run python3 infer_slice.py --base <base> --adapter <stage-a-output> --slice eval_slice.jsonl --out preds_stage_a.jsonl
uv run python3 eval_slice.py --preds preds_stage_a.jsonl
"""
print(STAGE_A_CMD)
print(f"ABORT threshold: overall accuracy < {STAGE_A_ABORT_THRESHOLD}")
print("→ Nếu < threshold: Stage-A too aggressive, catastrophic forgetting")
print("   Fix: tăng STAGE_A_SFT_MIX (0.10 → 0.20) rồi retry Stage-A")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Stage-B — SFT expand (1000 steps, standard)

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("=== Stage-B: SFT expand (chỉ chạy nếu Stage-A passed threshold) ===")
STAGE_B_CMD = f"""
cd {NEMOTRON_MASTER}
# Chạy standard SFT từ Stage-A adapter (không phải từ 0.86)
uv run python3 train_sft.py
# Cfg(): CrossEntropyLossConfig() (standard), lr={STAGE_B_LR}, steps={STAGE_B_STEPS}
# Adapter init: Stage-A output (RESET_WEIGHTS=False equivalent)
"""
print(STAGE_B_CMD)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Falsification cuối

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("Falsification sau Stage-B:")
print("Submit Stage-B adapter; nếu leaderboard score ≤ 0.86 (baseline) →")
print("  RL→SFT ordering KHÔNG có lợi trong offline DPO setting này")
print("  → Submit best of {exp27, exp28, exp29, exp30} thay thế")
print()
print("Comparison expected:")
print("  Baseline: 0.86")
print("  SFT-only (exp27 best): ~0.87 (mục tiêu)")
print("  DPO→SFT (exp38): lý thuyết > SFT-only nếu paper's finding transfer")
