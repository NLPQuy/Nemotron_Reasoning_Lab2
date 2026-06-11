# Kickoff prompt — phát triển exp37 (GeoRA LoRA init). Dán vào session mới.

---

Tôi đang dự thi **NVIDIA Nemotron Model Reasoning Challenge** (Kaggle). Deliverable = 1 LoRA adapter
rank-32 cho base `Nemotron-3-Nano-30B-A3B-BF16` (Mamba/MoE hybrid), nộp `submission.zip` (phải có
`adapter_config.json`). Mục tiêu 0.86 → 0.88+. **Task session này: phát triển exp37 (GeoRA
geometry-aware LoRA init) — bản offline, chạy được trên Kaggle GPU.**

## Đọc TRƯỚC khi làm gì (theo thứ tự), đừng tin trí nhớ:
1. `CLAUDE.md` (root) + `nemotron-master/CLAUDE.md` — kiến trúc, 2 codebase, ràng buộc, `uv` only.
2. Memory: `project_exp33_vcore_port.md` (state batch-4 + DPO branch BỎ + exp30 OXA OOM đã vá),
   `project_exp31_cryptarithm_AB.md`, `project_architecture.md`, `feedback_experiment_lessons.md`.
3. `experiments/exp37.py` — stub thiết kế GeoRA hiện tại (feasibility checks + pseudocode). ĐỌC KỸ.
4. `experiments/RUN_NON_RL.md` — §3 triage exp37 (HIGH RISK, feasibility 2/5).
5. `experiments/exp33.py` + `experiments/exp30.py` — **mẫu port chuẩn** (copy Continuer + `# >>> EXPN`
   markers + knob, base-path giữ nguyên khi off). exp30 có **pre-pass forward/backward sweep memory-safe**
   (model.eval cho no-grad / del + empty_cache) — GeoRA FIM cũng là sweep, học pattern này.
6. `tracker/leaderboard.md` — điểm thật (baseline 0.86, chưa idea nào vượt).

## Ràng buộc cứng (vi phạm = vô hiệu)
- Inference lúc chấm: **greedy** temp 0, `max_tokens 7680`, `max_model_len 8192`, **vLLM**.
- LoRA **rank ≤ 32**, 1 adapter, vLLM-loadable. Adapter phải có đủ tensor cho 128 expert (DoRA từng
  hỏng vì thiếu — xem feedback). lm_head LoRA keys rename `backbone.lm_head` lúc save.
- Kaggle competition kernel **OFFLINE** (không pip/HF-download). GPU = **RTX PRO 6000 Blackwell 96GB
  (sm_120)**. Base + adapter phải mount path LOCAL.
- `uv` only trong `nemotron-master/`. exp files ở `experiments/` không có uv project → `py_compile`/AST.
  (Lưu ý: đĩa hay đầy → `uv sync` đứt làm thiếu .so CUDA; nếu `import torch` lỗi libcusparseLt →
  `uv sync --reinstall-package nvidia-cusparselt-cu13`. Dùng `uv run --no-sync` để tránh full-sync.)

## State đã chốt (đừng dựng lại)
- **Trainer Kaggle thật = `Continuer_Nemotron_Notebook.py`** (Unsloth). Forward monkey-patch
  cut_cross_entropy → `model._cached_per_token_ce`; loop tự áp per-token weight; MOE_TIE_WEIGHTS
  sum grad qua 128 expert; lm_head LoRA thêm tay; LoRA cast fp32, base bf16 (MoE router fp32).
- **Đã port offline (DONE):** exp33 (VCORE online), exp32 (curriculum), exp30 (OXA). exp31 (cryptarithm
  A+B) đã build corpus. **Nhánh DPO (exp29/35/36/38) BỎ** vì OOM memory-bound trên 1×96GB.
- **Bài học OOM (áp cho exp37 FIM sweep):** generation/rollout OOM; OXA pre-pass forward sweep gây
  **CPU-RAM OOM-kill** (Unsloth offload hook tích pinned buffer qua sweep no-backward) → fix bằng
  `model.eval()` (khi no-grad) + xóa cached tensor + `del` + `empty_cache()` định kỳ + batch nhỏ.
  GeoRA FIM CẦN backward (không eval được) → bound N nhỏ + del + empty_cache + chỉ vài layer.

## exp37 — GeoRA (từ stub, arXiv:2601.09361, KHÔNG có official repo)
**Ý tưởng:** thay vì init LoRA A ngẫu nhiên, init theo hình học gradient của task (Fisher Info) để
LoRA bắt đúng subspace quan trọng. Continue từ adapter 0.86 (RESET_WEIGHTS=False).
**Thuật toán (đọc paper Section 3 trước khi code, đừng đoán):**
1. Forward+backward N (10–50) bài **cryptarithm** (category yếu) → thu per-layer gradient `G_l`.
2. Outer product `G_l G_lᵀ` (xấp xỉ FIM) — CHỈ cho `N_TOP_LAYERS` (4–8) để giới hạn memory.
3. `SVD(FIM_l)` → top-`LORA_RANK` right singular vectors = `A_geora`.
4. `A_new = GEORA_BLEND·A_geora + (1−GEORA_BLEND)·A_current`. **B giữ = 0** (nếu adapter 0.86 có
   B≠0, overwrite A + B≠0 = disturb ngay → check/xử lý B trước).
5. Continue train 200–500 step, LR 1e-5.

**Feasibility checks BẮT BUỘC trước khi implement (stub §"Feasibility"):**
- (a) Memory budget FIM: d_model² per layer; Nemotron-H d_model? → chỉ N_TOP_LAYERS.
- (b) Layer names Nemotron-H: GeoRA chỉ hợp **FFN (up_proj/down_proj) + lm_head**; SSM mixer
  (`in_proj/out_proj`, `mixer.gate`) hình học KHÔNG interpretable → KHÔNG apply GeoRA ở đó.
- (c) Unsloth có cho overwrite LoRA A sau `get_peft_model()` không (requires_grad, gán lại được)?

**Falsification (mini-run 50 step):**
- gravity/cipher accuracy KHÔNG giảm > 2pp vs 0.86 (nếu giảm → GeoRA init disturb capacity → abort).
- cryptarithm loss PHẢI giảm vs random-LoRA-init baseline (nếu không → FIM không bắt geometry → abandon).
- OOM khi tính FIM → giảm N_CRYPTARITHM_SAMPLES (50→10–20), N_TOP_LAYERS (8→4), dùng grad accum.

## Quy tắc làm việc
- Copy `Continuer_Nemotron_Notebook.py` → `experiments/exp37.py`; mọi sửa bọc `# >>> EXP37 START/END`;
  knob trong block config; **off (EXP37_USE_GEORA=False) ⇒ base-path nguyên vẹn** (diff chỉ trong markers).
- Gọi lại hàm/cấu trúc gốc, đừng tự suy. Sau sửa: `py_compile experiments/exp37.py` + diff non-EXP37
  lines == base. Viết runbook Kaggle `experiments/exp37_kaggle.md` (mount, knob, gate, memory-safe).
- DỪNG trước full-train/submit/upload-dataset; hỏi user. Có điểm → `tracker/rounds/round_<N>.md` +
  `tracker/leaderboard.md`.

**Khuyến cáo trung thực ngay từ đầu:** exp37 feasibility 2/5 — GeoRA thiết kế cho Transformer thuần,
Nemotron-H là Mamba/MoE; FIM sweep có rủi ro OOM (như OXA). Làm **feasibility checks (a)(b)(c) +
mini-run gate TRƯỚC** khi đầu tư full. Nếu (b) cho thấy chỉ FFN+lm_head apply được (phần nhỏ tham số)
hoặc (c) Unsloth seal weights → khả năng cao bất khả → báo user cân nhắc bỏ, dồn sang việc khác.

Bắt đầu.
