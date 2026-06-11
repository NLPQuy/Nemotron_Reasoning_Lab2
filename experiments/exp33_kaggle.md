# exp33 (VCORE online) trên Kaggle — runbook

VCORE = reweight per-token **online** trong vòng train (variance-controlled), port thật từ
`enhance_cot/vcore`. KHÔNG phải static mask — không có snapshot mới, không cần upload data.
`experiments/exp33.py` **chính là notebook Kaggle** (chạy `run_training()` ở module level khi
`IS_KAGGLE`). Bạn chỉ dán cả file vào 1 cell Kaggle GPU (RTX PRO 6000), chỉnh vài hằng số ở
block config đầu file, rồi Run.

> Cơ chế / chi phí: mỗi anchor step làm thêm 1 lần probe (forward+backward trên batch ngẫu nhiên
> B') + 1 lần forward tính `pre_loss` của batch thật. → mỗi step ≈ **2–3× compute** so với base.
> Hạ `EXP33_VCORE_ANCHOR_STEPS` (probe thưa hơn) để khấu hao nếu chậm/OOM.

---

## 0. Dataset cần attach (y hệt baseline Continuer — KHÔNG có dataset mới)
1. **Wheels:** `mayukh18/nemotron-packages` + `llkh0a/rtx-wheels` (causal_conv1d, mamba_ssm, unsloth…).
2. **Corpus tokens:** `huikang/huikang-nemotron-repository-snapshot`
   (path `…/training/sft/04-08-16-14/tokens` đã hard-code trong file).
3. **Model:** `metric/nemotron-3-nano-30b-a3b-bf16` (kagglehub, đã add làm input).
4. **Adapter 0.86** (CHỈ khi continue-train): notebook `huikang/tinker-submission-notebook`
   → `submission.zip` (path đã hard-code ở dòng 142). Nếu mount chỗ khác, sửa `_adapter_zip`.

Bật **GPU**. Offline OK (mọi thứ từ dataset local; không pip internet).

---

## 1. Chọn chế độ chạy — sửa block config đầu `exp33.py`

**Khuyến nghị: continue-train từ 0.86** (insight #4: né confound fresh-init đã gây 0.85; VCORE là
"SFT giám sát tốt hơn" → tinh chỉnh trên nền 0.86 là A/B sạch nhất). Đặt:
```python
RESET_WEIGHTS = False          # nạp adapter 0.86, refine thay vì train fresh
SHUFFLE_DATASET = True
NUM_STEPS = 300                 # continue ngắn
LEARNING_RATE = 2e-5           # LR nhỏ cho continue
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
# VCORE (đã có sẵn trong file):
EXP33_USE_VCORE = True
EXP33_VCORE_TEMPERATURE = 1.0
EXP33_VCORE_EPSILON = 2e-5
EXP33_VCORE_ANCHOR_STEPS = 1   # 1 = probe mỗi step; tăng (2–4) nếu cần khấu hao compute
```
Nếu muốn lặp lại đúng recipe huikang (fresh): giữ `RESET_WEIGHTS=True`, `NUM_STEPS=1000`,
`LEARNING_RATE=2e-4` — nhưng dính confound fresh-init, A/B kém sạch.

---

## 2. GATE bắt buộc — smoke 50 bước TRƯỚC khi đốt full

VCORE đụng vòng train (backup/probe/restore LoRA + reweight). Phải xác nhận **chạy được trên
Nemotron-H (Mamba/MoE) + cut_cross_entropy** trước. Đặt tạm `NUM_STEPS = 50`, Run, và kiểm log:

- **G1 — không crash/OOM/NaN:** 50 step chạy hết; `loss:mean` hữu hạn, KHÔNG nan/inf.
  (probe nhân đôi forward → nếu OOM: giảm `MICRO_BATCH_SIZE` xuống 2, hoặc `EXP33_VCORE_ANCHOR_STEPS=2`.)
- **G2 — VCORE thật sự kích hoạt:** `loss:mean` (báo cáo = CE mean thường) vẫn giảm hợp lý.
  Variance-control `c=loss_uniform/loss_weighted` chỉ co (≤1) → loss backward ≤ uniform; nếu loss
  bằng đúng base từng step → reweight không tác dụng (kiểm `EXP33_USE_VCORE`, `mb_pre` có rỗng?).
- **G3 — adapter hợp lệ:** chạy xong tạo `submission.zip`; unzip phải có `adapter_config.json` với
  `r=32`.

Muốn quan sát reweight rõ hơn: tạm thêm 1 dòng print trong `_vcore_weighted_loss` log
`c.item()` + `(loss_weighted/loss_uniform)` → kỳ vọng c<1 ở phần lớn step (variance control đang co
các token high-variance). G2 FAIL (c≡1.0 mọi step) → eps/tau lệch scale, thử `EXP33_VCORE_EPSILON`
khớp đúng probing-LR.

G1/G2/G3 PASS → mới đặt lại `NUM_STEPS=300` (hoặc 1000) chạy full.

---

## 3. Full run → đánh giá

1. Đặt `NUM_STEPS` về giá trị thật (300 continue / 1000 fresh), Run. Output: `submission.zip` ở
   `/kaggle/working/` (file đã tự đóng gói, rank=32, lm_head keys đã rename).
2. **A/B = leaderboard thật** (offline không có vLLM; eval-slice hf-cache đã vá nếu muốn đo local —
   xem `infer_slice.py`). So với 0.86:
   - Continue-control (VCORE off, cùng RESET_WEIGHTS=False/LR/steps) phải ≈0.86 → xác nhận continue
     không phá. Nếu chưa có, chạy 1 lần `EXP33_USE_VCORE=False` làm mốc.
   - VCORE on > control → VCORE cộng giá trị. ≤ control → drop (giữ 0.86).
3. **DỪNG trước submit** — hỏi user trước khi nộp `submission.zip` lên Kaggle.

---

## 4. Rollback / ghi chú
- Tắt cơ chế: `EXP33_USE_VCORE = False` → đường chạy = base Continuer (đã verify base-path
  giữ nguyên hành vi).
- Chỉ sửa loss/train; KHÔNG đụng inference, `compare_answer`, `extract_answer`, format corpus.
- Có điểm → cập nhật `tracker/rounds/round_4.md` + `tracker/leaderboard.md` (Δ vs 0.86, ghi rõ
  continue vs fresh + knob VCORE).
- Hằng số VCORE mặc định lấy từ upstream (`vcore_epsilon=2e-5`, `temperature=1.0`,
  `anchor_steps=1`). Sweep `temperature∈{0.5,1,2}` nếu G2 ok nhưng LB phẳng.
```
