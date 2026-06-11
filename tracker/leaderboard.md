# Leaderboard Tracker

Target: **0.88+** | Current best: **0.86** (baseline — chưa idea nào vượt)

| Round | Score | Δ | Config summary | Notes |
|-------|-------|---|----------------|-------|
| baseline | 0.86 | — | `Continuer_Nemotron_Notebook.py` default | Starting point |
| exp12 (NEFTune) | 0.83 | −0.03 | `NEFTUNE_ALPHA=10` | Noise embedding hại exact-match (đúng cảnh báo devil's-advocate) |
| exp13 (self-verify traces) | 0.68 | −0.18 | corpus verify-then-box | **Hồi quy nặng nhất** — verify tail lật đáp án / dài trace |
| exp19 (stream-of-search) | 0.79 | −0.07 | corpus SoS backtracking | Trace dài + meander → truncate/sai |
| exp11 (DoRA) | ERROR | — | `USE_DORA=True` | Adapter chỉ ~500MB (thiếu tensor expert) → hỏng |
| exp14 (PiSSA) | ERROR | — | `init_lora_weights="pissa"` | Unsloth chỉ nhận True/False/gaussian/loftq/corda |
| exp15 (scratchpad) | pending | — | corpus arithmetic scratchpad | Chưa có điểm |
| exp16/17/18 | chưa chạy | — | — | anchored-KL / soup / preference |
| **exp20 (gravity scale)** | **0.85** | **−0.01** | corpus 21+22+400 gravity gen (`kaggle_snapshot_exp20`) | ⚠️ trộn 3 thay đổi (gate+augmenter+gravity); chưa isolate. Nghi phạm: verify gate cắt data category khó; gravity bão hòa. Xem [round_3.md](rounds/round_3.md) |
| **exp32 (curriculum)** | **0.86** | **0** | AdaSTaR-inspired curriculum, continue 0.86 (RESET=False, 270 step, LR 2e-5), corpus huikang gốc | Best of batch-4. Flat = continue-train ≈ NO-OP (270 step LR thấp từ adapter hội tụ → cơ chế động-học vô hình). Cần fresh-train + control mới test được. |
| exp30 (OXA reweight) | 0.85 | −0.01 | OXA per-example reweight, continue 0.86, corpus huikang | Suppress bài dễ bỏ reinforcement → hơi hại. Continue ≈ no-op. |
| exp31 (cryptarithm corpus) | 0.65 | −0.21 | corpus REGENERATE (verify gate + arith solver + 1000 gen_ crypt), `mlbang/reasoning-data-exp31` | **Corpus HỎNG, không phải lỗi cryptarithm:** GLOBAL_LENGTH_CAP=7600 cắt 63% bit_manipulation (358/1370 vào corpus) → mất 2/3 category lớn. Confound nặng. Fix length-cap trước khi dùng corpus regen. |

**Insight 2026-06-11:** continue-train (RESET=False, LR 2e-5, ~270 step) từ 0.86 ≈ no-op → che mọi cơ chế training-dynamics (curriculum/VCORE). Muốn test exp32/exp33 thật phải **fresh-train (RESET=True, 1000 step, LR 2e-4) + control off**, baseline fresh ≈0.85. Corpus regen có bug length-cap (bit_manip dài bị cắt 63%).

---

## Score history (chronological)

- **2026-06-01** — baseline `0.86` (pretrained adapter, default config)
- **2026-06-01** — batch-2 sweep (chi tiết [round_2.md](rounds/round_2.md)):
  - exp12 NEFTune `0.83`; exp13 self-verify `0.68`; exp19 stream-of-search `0.79`
  - exp11 DoRA `ERROR` (adapter 500MB); exp14 PiSSA `ERROR` (Unsloth init không hỗ trợ)
  - exp15 pending; exp16/17/18 chưa chạy
- **Kết luận tạm:** mọi exp đã hoàn tất đều **hồi quy** dưới 0.86. Baseline ở đỉnh nhọn — thay đổi *data-time đổi phân phối trace* (exp13/19) và *regularizer thêm nhiễu* (exp12) đều hại; idea *đổi parameterization/init* (exp11/14) vướng tường tương thích Unsloth/MoE.
- **2026-06-10** — batch-4: exp33 (VCORE online), exp30 (OXA), exp32 (curriculum) port sang trainer Unsloth (offline, độc lập DPO) — sẵn sàng chạy, chưa có điểm.
  - **Nhánh DPO (exp29/35/36/38) BỎ ƯU TIÊN:** exp29 OOM ở rollout inference trên RTX PRO 6000 96GB (naive Mamba path do thiếu wheel causal_conv1d + n_samples=10 song song). DPO-train còn ngốn hơn → memory-bound trên 1×96GB. Focus = exp33/32/30.
- **2026-06-04** — batch-3 data-aug, submission đầu (chi tiết [round_3.md](rounds/round_3.md)):
  - exp20 (21+22+gravity) `0.85` (−0.01). **Confounded** — bundle gate+augmenter+gravity vs corpus gốc; chưa tách được.
  - Phát hiện chính: corpus gốc 0.86 **vốn chứa trace boxed-sai** (rule_unknown); verify gate exp21 xóa ~600 cryptarithm + ~115 eq_guess + ~238 bit_manip → **nghi phạm chính** của −0.01, có thể hơn cả gravity. Gravity solver-100% đã bão hòa → scale phẳng (đúng cảnh báo batch-3).
  - **Next:** submit isolate 21+22 / exp24 / exp25; dựng eval-slice per-category (chờ gỡ inference offline); ngừng scale category bão hòa; exp26 DROP.
