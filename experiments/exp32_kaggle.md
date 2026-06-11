# exp32 (AdaSTaR-inspired curriculum) trên Kaggle — runbook

Curriculum thích nghi: mỗi step train **BATCH_SIZE example ÍT THUỘC nhất** (EMA cross-entropy cao),
chọn bằng Gumbel top-k. Spirit của AdaSTaR ("train cái chưa học được") nhưng dùng CE làm proxy độ
khó → **không cần generation/verify giữa train** (bản faithful win/lose-heap thì cần, bất khả với
forward đã monkey-patch CCE). `experiments/exp32.py` chính là notebook Kaggle (như Continuer).

> ⚠️ **Đây là bản tractable, KHÔNG phải AdaSTaR thật** (win/lose min-heap dựa correctness). CE thấp
> ≠ chắc chắn đúng (model có thể tự tin mà sai), nhưng trong corpus verify-gated mọi trace đều đúng
> nên CE là proxy "đã thuộc" hợp lý. Ghi rõ điều này khi báo cáo điểm.

---

## 0. Dataset attach — y hệt baseline Continuer (KHÔNG dataset mới)
Wheels + corpus `huikang/...snapshot` + model `metric/nemotron-...` (+ adapter 0.86 nếu continue).
Bật GPU. Offline OK.

---

## 1. Config đầu `exp32.py`
```python
RESET_WEIGHTS = False        # khuyến nghị continue từ 0.86 (A/B sạch); True nếu muốn fresh
SHUFFLE_DATASET = True       # bị BỎ QUA khi curriculum bật (sampler tự chọn), giữ True cho an toàn
NUM_STEPS = 300              # continue; 1000 nếu fresh
LEARNING_RATE = 2e-5
EXP32_USE_CURRICULUM = True
EXP32_EMA_ALPHA = 0.3        # trọng số CE mới trong EMA
EXP32_PRIORITY_TEMP = 1.0    # <1 = tham (bám bài khó); >1 = phẳng (gần uniform)
EXP32_UNSEEN_PRIORITY = 1e6  # bài chưa train được ưu tiên thấy trước
```

---

## 2. GATE — smoke + chống suy biến

Cơ chế lấy mẫu thay vòng lặp tuyến tính → phải xác nhận không suy biến (chỉ lặp lại vài bài khó).
Đặt `NUM_STEPS = 30`, Run, kiểm log:

- **G1 — chạy được:** 30 step không OOM/NaN; `loss:mean` hữu hạn, có xu hướng giảm.
- **G2 — đa dạng mẫu (chống starvation):** trong 30 step không lặp đi lặp lại y một nhúm bài. Cách
  kiểm nhanh: tạm in `len(set(batch_indices))` / số id distinct cộng dồn. Nếu < ~5×BATCH_SIZE id
  distinct qua 30 step → temp quá thấp → tăng `EXP32_PRIORITY_TEMP` (1.0 → 2.0).
- **G3 — adapter hợp lệ:** `submission.zip` có `adapter_config.json` r=32.

G1/G2/G3 PASS → `NUM_STEPS=300`, full run.

---

## 3. Full + A/B
1. Full run → `submission.zip`.
2. A/B vs 0.86:
   - Control = `EXP32_USE_CURRICULUM=False` (cùng RESET/LR/steps) → đường base linear, phải ≈ baseline.
   - Curriculum on > control → curriculum cộng giá trị; ≤ → drop.
3. Kỳ vọng: category yếu (cryptarithm, equation_guess — CE cao) được train nhiều hơn → cải thiện
   ở đó; category bão hòa (gravity/numeral — CE thấp) bị giảm tần suất. Nếu category yếu KHÔNG nhích
   → CE-proxy không tương quan với accuracy ở đây → bản curriculum này vô hiệu.
4. **DỪNG trước submit**, hỏi user.

---

## 4. Rollback / ghi chú
- Tắt: `EXP32_USE_CURRICULUM=False` → vòng walk tuyến tính (base-path, đã verify giữ nguyên hành vi).
- Tham số: `EMA_ALPHA` cao = phản ứng nhanh với CE mới (nhiễu hơn); `PRIORITY_TEMP` điều tiết
  greedy↔uniform. Bắt đầu 0.3 / 1.0.
- Chỉ đổi thứ tự/lấy mẫu dữ liệu; KHÔNG đụng loss/inference/grader/format.
- Khác exp30 (per-example weight tĩnh từ 1 pre-pass) và exp33 (per-token online): exp32 đổi
  **example nào được train + tần suất**, không đổi trọng số token. 3 cơ chế độc lập.
- Có điểm → `tracker/rounds/round_4.md` + `tracker/leaderboard.md`.
```
