# Chạy các exp NON-RL (batch-4) — guide hợp nhất

Sau khi BỎ nhánh DPO (exp29/35/36/38 — memory-bound trên 1×96GB), các exp non-RL chia 2 nhóm:

| exp | Cơ chế | Trạng thái | Memory |
|-----|--------|-----------|--------|
| **exp33** | VCORE per-token online reweight | ✅ SẴN SÀNG (trainer + runbook) | An toàn (~2-3× compute/step) |
| **exp32** | AdaSTaR-inspired curriculum sampling | ✅ SẴN SÀNG | An toàn (không pre-pass) |
| **exp30** | OXA per-example reweight | ✅ SẴN SÀNG (đã vá OOM) | Có pre-pass — chạy cuối, subsample |
| exp31 | Procedural cryptarithm + RFT | ⚠️ Trainer xong, THIẾU data-gen (rollout) | Rollout OOM-risk (như exp29) |
| exp34 | Paraphrase traces (Claude API) | ⚠️ Chưa port; cần API local + rebuild | Train an toàn; off-dist RISK |
| exp37 | GeoRA LoRA init | ⛔ ABORTED (feasibility 0/5) | =PiSSA(exp14 ERROR)+base-mutation bất khả |

**Thứ tự khuyến nghị: exp33 → exp32 → exp30.** exp31/34/37 chưa plug-and-play (xem §3).

---

## 1. Setup Kaggle CHUNG cho cả 3 (exp33/32/30)

Mỗi `expN.py` **chính là notebook Kaggle** (chạy `run_training()` ở module level khi `IS_KAGGLE`).
Dán cả file vào kernel GPU (RTX PRO 6000), chỉnh hằng số config đầu file, Run.

**Dataset attach (giống baseline Continuer — KHÔNG dataset mới):**
1. Wheels: `mayukh18/nemotron-packages` + `llkh0a/rtx-wheels`.
2. Corpus tokens: `huikang/huikang-nemotron-repository-snapshot`.
3. Model: `metric/nemotron-3-nano-30b-a3b-bf16`.
4. Adapter 0.86 (khi continue-train): `bngtbnh04/adapter-0-86` (hoặc submission.zip huikang).

**Bài học memory (từ 2 lần OOM thực tế — áp dụng CHO MỌI run):**
- ⚙️ Đặt env kernel: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (giảm fragment; đúng gợi ý message OOM).
- GPU thật = RTX PRO 6000 Blackwell **96GB** (sm_120). Model 30B bf16 ≈ 60GB → còn ~35GB cho activations/optimizer.
- **Generation (rollout) là thủ phạm OOM** — exp33/32/30 KHÔNG generate giữa train nên an toàn. exp31 thì có (nên risky).
- **Continue-train (RESET_WEIGHTS=False) khuyến nghị**: né confound fresh-init (đã gây 0.85), A/B sạch vs 0.86.

**Quy trình mỗi exp:** smoke gate (NUM_STEPS nhỏ) → kiểm log → full → A/B vs 0.86 → **DỪNG trước submit, hỏi.**

---

## 2. Quick card 3 exp sẵn sàng (chi tiết trong runbook riêng)

### exp33 — VCORE  → `exp33_kaggle.md`  (ĐẶT CƯỢC CHÍNH)
```python
RESET_WEIGHTS=False; SHUFFLE_DATASET=True; NUM_STEPS=300; LEARNING_RATE=2e-5
EXP33_USE_VCORE=True; EXP33_VCORE_TEMPERATURE=1.0; EXP33_VCORE_EPSILON=2e-5; EXP33_VCORE_ANCHOR_STEPS=1
```
Gate smoke 50 bước: G1 không OOM/NaN (probe ×2 forward → nếu OOM, ANCHOR_STEPS=2 hoặc MICRO_BATCH_SIZE=2);
G2 variance-control `c<1` thật kích hoạt; G3 adapter r=32.

### exp32 — Curriculum  → `exp32_kaggle.md`
```python
RESET_WEIGHTS=False; NUM_STEPS=300; LEARNING_RATE=2e-5
EXP32_USE_CURRICULUM=True; EXP32_EMA_ALPHA=0.3; EXP32_PRIORITY_TEMP=1.0
```
Gate smoke 30 bước: chống starvation (đếm id distinct; nếu < ~5×BATCH_SIZE → tăng PRIORITY_TEMP→2.0).

### exp30 — OXA  → `exp30_kaggle.md`  (CHẠY CUỐI — có pre-pass nặng)
```python
RESET_WEIGHTS=False; NUM_STEPS=300; LEARNING_RATE=2e-5
EXP30_USE_OXA=True; EXP30_OXA_MODE="percentile"; EXP30_OXA_MAX_EXAMPLES=2000; EXP30_OXA_MICRO_BATCH=2
```
⚠️ Đã vá OOM (CPU-RAM kill ở pre-pass): `model.eval()` + dọn cache + batch 2. Vẫn nên `MAX_EXAMPLES=2000`
(score 2000, còn lại weight=1.0) cho lần đầu + env `expandable_segments`. Theo dõi log progress mỗi 50 batch.

---

## 3. exp31 / exp34 / exp37 — CHƯA chạy được (triage trung thực)

### exp31 — Procedural cryptarithm + RFT
- **Trainer XONG** (`exp31.py`, knob `EXP31_CORPUS` trỏ snapshot). Thiếu phần DATA.
- **Blocker:** viết `generators/cryptarithm_procedural.py` → gen 1500 bài → **rollout `infer_slice` temp=0.5 n=10** (filter giữ correct = RFT) → pack snapshot. Rollout generation = **đúng chỗ OOM exp29** (đã vá `--gen_chunk` nhưng vẫn nặng + chậm).
- **Gate falsify TRƯỚC train:** pass@10 ≥ 2% trên 200 bài; < 2% → ABORT.
- → Cần port data-gen + chấp nhận rollout chậm/memory-risk. Hỏi nếu muốn làm.

### exp34 — Paraphrase traces
- **Chưa port.** Pipeline: extend `paraphrase_instances.py` cho `reasoning/*.txt` → **Claude API chạy LOCAL** (offline Kaggle không gọi API được) paraphrase ~8600 trace (~$6-10) → verify gate (`compare_answer` + ≤7600 token) → rebuild corpus → pack → trainer pointer (như exp31).
- **Pre-check bắt buộc:** paraphrase 100 trace, logprob adapter 0.86 không tăng > 0.1 → ABORT.
- **RỦI RO:** off-distribution — [[feedback_experiment_lessons]] ghi exp26 (model-written) DROP, exp19 (SoS) −0.07. Paraphrase nhẹ hơn nhưng cùng họ rủi ro. Cân nhắc ROI thấp.

### exp37 — GeoRA LoRA init — ⛔ ABORTED (feasibility 0/5, không port)
Verdict đầy đủ: [exp37_feasibility.md](exp37_feasibility.md). Tóm tắt:
- **Paper thật (arXiv:2601.09361) = "GeoRA for RLVR" = SVD pretrained weights (họ PiSSA) + freeze
  residual anchor, cho RL.** Mô tả "gradient FIM sweep" trong stub là **hallucinated**, không phải method.
- **(c) Unsloth seal = NO đã chứng minh:** exp14 (PiSSA, cùng họ) ERROR — Unsloth chỉ nhận
  True/False/gaussian/loftq/corda cho `init_lora_weights`.
- **Hard-constraint conflict:** GeoRA/PiSSA serve base đã mutate `W_res=W−B@A`; grader serve base GỐC
  + adapter vLLM → bất khả. Bỏ subtraction → double top-r directions → disturb 0.86 ngay. Continue từ
  0.86 (B≠0) + overwrite A → `B_trained@A_geora` vô nghĩa → phá adapter.
- Config grounded: hidden=2688, 52 layers, Mamba/attn hybrid, MoE 128 expert. Memory KHÔNG phải blocker.
- **Salvage duy nhất (ROI thấp):** EVA-style activation-aware init (forward-only, B=0, RESET_WEIGHTS=True,
  train fresh) — không continue được từ 0.86. Hỏi user nếu muốn.

---

## 4. Sau khi có điểm
Cập nhật `tracker/rounds/round_4.md` (copy từ `round_template.md`) + append `tracker/leaderboard.md`
(Δ vs 0.86, ghi rõ continue vs fresh + knob). Mỗi exp = 1 cơ chế, A/B sạch.
```
