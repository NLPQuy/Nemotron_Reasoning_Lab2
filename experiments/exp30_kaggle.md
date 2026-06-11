# exp30 (OXA) trên Kaggle — runbook

OXA = exploration-aware SFT: reweight **mỗi example** theo độ tự tin của chính adapter 0.86 —
**promote** trace khó-mà-đúng (avg CE cao = logprob thấp), **suppress** trace đã thuộc (avg CE thấp).
Weight là **scalar per-example** → nhân vào completion mask; trainer áp `float(m)` sẵn có →
**KHÔNG sửa vòng train, KHÔNG snapshot mới**. `experiments/exp30.py` chính là notebook Kaggle.

> CAVEAT: corpus đã verify-gate (mọi trace đúng) → set-B "high-conf INCORRECT" của paper = rỗng.
> Ở đây set-B = "high-conf correct" (đã thuộc) → down-weight như curriculum bỏ bớt bài dễ. Nếu
> muốn set-B thật (trace sai) phải dùng corpus chưa-gate + is_correct (xem batch-4 hướng khác).

---

## 0. Dataset attach — y hệt baseline Continuer (KHÔNG có dataset mới)
1. Wheels: `mayukh18/nemotron-packages` + `llkh0a/rtx-wheels`.
2. Corpus tokens: `huikang/huikang-nemotron-repository-snapshot`.
3. Model: `metric/nemotron-3-nano-30b-a3b-bf16`.
4. **Adapter 0.86 (BẮT BUỘC):** `huikang/tinker-submission-notebook` → `submission.zip`
   (dòng ~142). OXA **phải** đo adapter 0.86 nên cần nó.

Bật GPU. Offline OK.

---

## 1. Config bắt buộc đầu `exp30.py`
```python
RESET_WEIGHTS = False          # BẮT BUỘC — OXA đo độ tự tin của adapter 0.86
SHUFFLE_DATASET = True
NUM_STEPS = 300                 # continue-train từ 0.86
LEARNING_RATE = 2e-5
EXP30_USE_OXA = True
EXP30_OXA_MODE = "percentile"  # robust; calibrate theo corpus (khuyến nghị)
EXP30_OXA_PROMOTE_PCT = 0.30   # promote 30% khó nhất
EXP30_OXA_SUPPRESS_PCT = 0.30  # suppress 30% dễ nhất
EXP30_OXA_PROMOTE_WEIGHT = 2.0
EXP30_OXA_SUPPRESS_WEIGHT = 0.3
EXP30_OXA_MAX_EXAMPLES = 2000     # 0 = toàn corpus; đặt 200 cho smoke
EXP30_OXA_MICRO_BATCH=2
```
`exp30.py` assert `RESET_WEIGHTS=False` khi OXA on → nếu quên sẽ báo lỗi ngay.

---

## 2. Chi phí & GATE — smoke trước

**Pre-pass OXA = thêm ~1 epoch forward-only** (no_grad, ~18k example) trước khi train. Tốn thời gian
+ VRAM (forward seq tới 7680 token). Smoke trước:

- Đặt `EXP30_OXA_MAX_EXAMPLES = 200`, `NUM_STEPS = 30`, Run.
- **G1 — pre-pass chạy & phân phối hợp lý:** log in `avg_ce dist: min/p25/median/p75/max`. KHÔNG
  nan/inf; spread đủ rộng (p75 > p25) để promote/suppress có nghĩa. Nếu phân phối phẳng (mọi avg_ce
  ~bằng nhau) → OXA vô tác dụng.
- **G2 — reweight đúng đếm:** log `OXA reweighted: N promoted, M suppressed`. Với percentile 0.30/0.30
  trên 200 mẫu → ~60 promote, ~60 suppress. Nếu lệch nặng (vd 190 promote) → threshold sai.
- **G3 — train chạy + adapter hợp lệ:** 30 step không OOM/NaN; `submission.zip` có `adapter_config.json`
  r=32.

VRAM OOM ở pre-pass → giảm `MICRO_BATCH_SIZE` (2). G1/G2/G3 PASS → bỏ cap
(`EXP30_OXA_MAX_EXAMPLES=0`), `NUM_STEPS=300`, full run.

---

## 3. Full + đánh giá
1. Full run → `submission.zip` ở `/kaggle/working/`.
2. A/B vs 0.86 (leaderboard thật):
   - Continue-control (`EXP30_USE_OXA=False`, cùng RESET/LR/steps) phải ≈0.86.
   - OXA on > control → OXA cộng giá trị; ≤ → drop (giữ 0.86).
3. Nếu suppress (set-B) hại (bỏ quá nhiều bài) → đặt `EXP30_OXA_SUPPRESS_PCT=0.0` (chỉ promote,
   khớp đúng caveat corpus-gated) và chạy lại.
4. **DỪNG trước submit** — hỏi user.

---

## 4. Rollback / ghi chú
- Tắt: `EXP30_USE_OXA=False` → đường base Continuer (đã verify base-path nguyên vẹn).
- Chỉ đổi trọng số loss per-example; KHÔNG đụng inference/`compare_answer`/`extract_answer`/token format.
- avg CE = −logprob proxy: avg_ce CAO = model ÍT tự tin = bài khó → promote. avg_ce THẤP = thuộc rồi → suppress.
- Có điểm → cập nhật `tracker/rounds/round_4.md` + `tracker/leaderboard.md`.
- Khác exp33: exp30 là **per-example static reweight** (rẻ về code, 1 pre-pass forward); exp33 là
  **per-token online** (sửa vòng train, ~2-3× compute). Hai cơ chế độc lập — chạy/đo riêng.
```
