# Round 3 — Batch-3 data-augmentation, first submission (exp20)

**Date**: 2026-06-04
**Score**: exp20 = **0.85**
**Δ vs previous best**: **−0.01** (vs baseline 0.86)

---

## Hypothesis

Batch-3 ([batch-3.md](../../research/ideation/batch-3.md)) chỉ làm **data-time augmentation đúng-phân-phối/đúng-format** để tránh hồi quy như round-2. exp20 = "procedural in-distribution instance scaling": dùng solver tất định sinh thêm **400 bài gravity** (solver 100%, đáp án solver tự gán → 0 nhãn sai), kỳ vọng thêm gradient đúng mà không dịch phân phối.

## Config changes

exp20 KHÔNG đổi trainer; nó train trên **corpus regenerate** (`kaggle_snapshot_exp20`) khác corpus gốc 0.86 ở **3 điểm cùng lúc**:

```text
1. exp21 verify gate  : corpus.py drop trace có boxed != train.csv answer
                        → bỏ RẤT NHIỀU bài category khó:
                          cryptarithm_deduce 91.8%, cryptarithm_guess 93.3%,
                          equation_numeric_guess 84.6%, bit_manipulation 14.9%
2. exp22 augmenters   : + 4 augmenter chuỗi masked (reverse/count/char_index/digit_extract), +1200 problems
3. exp20 gravity scale: + 400 gravity gen_ (1597 → 1997)
```

## Training run

| Field | Value |
|-------|-------|
| Platform | Kaggle |
| Corpus | `kaggle_snapshot_exp20` (18392 entries, regenerated) |
| Steps / LR | mặc định (NUM_STEPS=1000, LR=2e-4) |
| Knob | `EXP20_SCALED_CORPUS = "/kaggle/input/<slug>"` |

## Result

| Split | Score |
|-------|-------|
| Public LB | **0.85** |

## Insights

### 1. ⚠️ Điểm 0.85 **không quy được cho gravity** — đang trộn 3 thay đổi
exp20 snapshot bundle **gate (21) + augmenters (22) + gravity (20)** so với corpus gốc 0.86. −0.01 có thể đến từ bất kỳ cái nào. **Bắt buộc isolate** trước khi kết luận: submit `kaggle_snapshot` (chỉ 21+22) để tách phần gate+augmenter khỏi gravity.

### 2. Nghi phạm số 1: **verify gate (exp21) cắt mất data category khó**
Điểm mấu chốt phát hiện được: corpus gốc 0.86 (huikang) **vốn ĐÃ chứa các trace boxed-sai** — `corpus.py` gốc include mọi pid có reasoning file, kể cả `rule_unknown`. Verify gate của exp21 **xóa ~600 cryptarithm + ~115 equation_guess + ~238 bit_manipulation** trace đó. Giả thuyết: những trace này — dù đáp án cuối sai — vẫn dạy **format + văn phong reasoning** cho category khó, và bỏ chúng làm **mất coverage** → hại. Tức "bộ lọc chất lượng" mà mình tưởng là pure-positive có thể **net-âm**. Đây là khả năng lớn hơn cả gravity.

### 3. Gravity bão hòa → scale phẳng (đúng cảnh báo batch-3.md)
gravity solver 100% (model gần như đã thuộc). batch-3.md §exp20 đã flag: *"gravity là solver-100% → scaling có thể gain phẳng"*. Kết quả khớp: thêm 400 gravity (đẩy gravity lên ~25% corpus) **không cộng điểm**, thậm chí có thể **làm lệch mixture** (category dễ chiếm tỉ trọng lớn hơn → hại macro). → **Ngừng scale category đã bão hòa**; nếu scale, phải nhắm category còn headroom.

### 4. Đỉnh nhọn round-2 vẫn đúng
Kể cả augmentation "an toàn nhất" (đúng-phân-phối, 0 nhãn sai) cũng **chưa vượt 0.86**. Củng cố: baseline cực nhạy, mọi thay đổi corpus phải đo trên slice held-out trước khi tin.

### 5. Bài học quy trình
Mình dựng "baseline regenerate" (gate + augmenter luôn bật) **đồng thời** với việc thêm gravity → **không bao giờ có so sánh sạch "corpus 0.86 + chỉ gravity"**. Lẽ ra phải submit từng lớp một. Từ giờ: **mỗi snapshot = đúng 1 thay đổi**, và **eval-slice per-category** để biết regress ở ĐÂU (đặc biệt cryptarithm/equation sau gate).

## Next actions (ưu tiên)

1. **Submit `kaggle_snapshot` (21+22)** để tách gate+augmenter vs gravity. Nếu cũng ≤0.86 → **gate là thủ phạm** → nới gate (giữ `rule_unknown`, chỉ drop sai-nghiêm-trọng) hoặc tách riêng exp21 vs exp22.
2. **Submit `kaggle_snapshot_exp24` / `kaggle_snapshot_exp25`** (paraphrase / surface, đã build) — cùng nền 21+22, delta thuần.
3. **Gấp: gỡ inference offline** (vLLM/hybrid-cache) để chạy **eval-slice per-category** → thấy gate cắt hại category nào → mới làm exp23 (mixture) có cơ sở.
4. **Ngừng scale gravity**; nếu làm exp20 tiếp, nhắm category có headroom (bit_manipulation 85%, equation_deduce 90%) — nhưng solver sinh được instance verify-đúng cho chúng khó hơn.
5. exp26 đã **DROP** (rủi ro cao).

## Status

- [x] Submitted (exp20 = 0.85)
- [x] Result recorded in leaderboard.md
- [ ] Isolate: submit 21+22 baseline, exp24, exp25
- [ ] eval-slice per-category (blocked on offline inference)
