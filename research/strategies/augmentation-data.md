# Chiến lược Data Augmentation — Phân tích tính khả thi

**Mục tiêu**: cải tiến baseline **0.86 → 0.88+** bằng *data augmentation* (mở rộng/biến đổi dữ liệu train), KHÔNG đổi inference (vẫn 1 lần greedy, vLLM, rank ≤ 32).
**Ngày**: 2026-06-02
**Nguồn ràng buộc**: [competition_info.md](../../competition_info.md), [CLAUDE.md](../../CLAUDE.md), kết quả [tracker/rounds/round_2.md](../../tracker/rounds/round_2.md).

---

## 0. TL;DR — kết luận khả thi

| Chiến lược | Khả thi | Rủi ro hồi quy | Kỳ vọng | Ưu tiên |
|-----------|---------|----------------|---------|---------|
| A. Mở rộng augmenter chuỗi phụ trợ (spelling/split/…) | ✅ Cao | 🟢 Thấp (masked, no-boxed) | +0.0–0.3 | Sớm, an toàn |
| B. Scale thêm *instance* trong-phân-phối qua solver tất định | ✅ Cao | 🟢 Thấp (format y hệt) | +0.2–0.6 | **#1** |
| C. Cân bằng coverage giữa 7 category | ✅ Vừa | 🟡 Vừa (mix LB ẩn) | +0.1–0.5 | Sau khi đo |
| D. Paraphrase *đề bài* (giữ nguyên đáp án) | ⚠️ Vừa | 🟡 Vừa | +0.1–0.4 | Thử có kiểm soát |
| E. Augment *nhiều dạng boxed tương đương* | ❌ Thấp | 🔴 Cao (exact-match) | ±0 | Tránh |
| F. LLM-paraphrase / viết lại *nội dung reasoning* | ❌ Thấp | 🔴 Cao (đã chứng minh ở round-2) | âm | Tránh |
| G. Self-generated verified instances (STaR-offline) | ⚠️ Vừa | 🟡 Vừa | +0.2–0.8 | Big-bet, sau |

**Một câu**: augmentation *cộng thêm dữ liệu đúng-phân-phối, đúng-format* (A, B, C) là khả thi và an toàn; augmentation *viết lại nội dung trace hoặc đổi dạng đáp án* (E, F) gần như chắc chắn hồi quy trên grader exact-match — **bằng chứng thực nghiệm round-2** đã cho thấy điều này.

---

## 1. Ràng buộc quyết định mọi thứ

- **Grader exact-match**: chấm chuỗi trong `\boxed{...}` (string match HOẶC sai số tương đối ≤ 1e-2; binary match **chính xác**). → mọi augmentation đụng tới *giá trị/định dạng đáp án* cực kỳ rủi ro.
- **Greedy, budget 7680 token**: trace dài → truncate trước `\boxed` → 0 điểm. → augmentation làm *dài* output (thêm bước, backtrack, verify) đã regress ở round-2.
- **vLLM, rank 32, 1 adapter**: augmentation là thay đổi *data-time* thuần; không đụng được decoding.
- **Format token là single source of truth** ([corpus.py](../../nemotron-master/corpus.py)): completion = `"{reasoning}\n</think>\n\\boxed{{answer}}<|im_end|>"`, prompt mask 0, completion mask 1. Mọi dữ liệu mới phải đi qua đúng pipeline này.

## 2. Bài học thực nghiệm (round-2) — trọng số lớn nhất

Baseline 0.86 nằm trên **đỉnh nhọn**. Các thay đổi *data-time đổi phân phối trace* đều hồi quy nặng:
- exp13 (self-verify traces): **0.68** (−0.18) — verify tail lật đáp án + dài trace.
- exp19 (stream-of-search backtracking): **0.79** (−0.07) — trace lan man, truncate.
- exp12 (noise embedding): **0.83** (−0.03) — phá tái tạo token chính xác.

→ **Nguyên tắc thiết kế augmentation cho task này**: chỉ *thêm* dữ liệu mà **phân phối trace/đáp án/độ dài gần như không đổi**. Thêm "kỹ năng" ở dạng task phụ masked (không boxed) là an toàn; viết lại nội dung CoT là nguy hiểm.

## 3. Hiện trạng pipeline augmentation

- [augmentation.py](../../nemotron-master/augmentation.py) ráp 5 augmenter chuỗi: `spelling, concatenation, splitting, matching, lstrip` → `augmentations/<id>.txt`.
- Các task phụ này **không có reasoning, không có `\boxed{}`** (corpus.py: "no reasoning, no boxed"); chúng chỉ dạy kỹ năng chuỗi cấp thấp, đưa vào corpus dưới dạng prompt-masked + completion-unmasked.
- 7 category lõi có **solver tất định** trong [reasoners/](../../nemotron-master/reasoners/): `bit_manipulation, cipher, cryptarithm, equation_numeric, gravity, numeral, unit_conversion`. Mỗi solver tự verify đáp án trước khi ghi trace (`status=rule_found`).

**Hệ quả khả thi quan trọng**: vì solver tất định + tự verify, việc **sinh thêm instance đúng-phân-phối là gần như free và không thể tạo nhãn sai** — đây là nền tảng cho chiến lược B (an toàn nhất, đòn bẩy mạnh nhất).

## 4. Phân tích từng chiến lược

### A. Mở rộng augmenter chuỗi phụ trợ — ✅ khả thi, 🟢 an toàn
Thêm augmenter mới (vd: reverse-string, char-index, count-substring, digit-extraction) hoặc tăng số lượng các augmenter hiện có. Vì chúng masked + no-boxed, **không đụng phân phối boxed-reasoning** → rủi ro hồi quy thấp.
- *Cơ chế*: thêm module trong `augmenters/`, đăng ký ở `augmentation.py`, chạy lại `corpus.py`.
- *Kỳ vọng*: nhỏ nhưng thực — củng cố kỹ năng nền (tách/ghép/đếm ký tự) mà cipher/cryptarithm dùng tới.
- *Rủi ro*: nếu tỷ trọng task phụ quá lớn, loãng tín hiệu reasoning → cap tỷ lệ (vd ≤ 15–20% corpus).
- *Falsification*: A/B trên slice; nếu macro exact-match không +≥0.3pp và không category nào rớt → bỏ.

### B. Scale instance trong-phân-phối qua solver — ✅ khả thi, 🟢 an toàn — **ĐÒN BẨY #1**
Dùng chính solver tất định để **sinh thêm bài mới cùng category, cùng format**, đáp án tự verify. Đây là "augmentation" đúng nghĩa nhất: nhiều dữ liệu đúng, 0 nhãn sai, phân phối giữ nguyên.
- *Cơ chế*: tăng số instance/seed trong từng `reasoners/<cat>.py`; chạy `reasoning.py → corpus.py`.
- *Kỳ vọng*: vừa–khá (nhiều gradient signal đúng-phân-phối); rủi ro thấp nhất trong các đòn bẩy "tăng điểm".
- *Rủi ro*: chỉ giúp nếu model còn under-fit category đó; nếu đã bão hòa → phẳng. Cần đo bucket lỗi trước.
- *Falsification*: tăng 1.5–2× instance cho 1–2 category đang yếu, A/B slice; cần +≥0.3pp ở category đó.

### C. Cân bằng coverage giữa category — ✅ khả thi, 🟡 vừa
Up-sample category hiếm / down-sample category dày để macro-accuracy đều hơn.
- *Rủi ro chính*: **mix leaderboard ẩn** — cân bằng sai có thể hại category mà LB chấm nhiều. (Cùng cảnh báo với batch-1 idea-6.)
- *Điều kiện khả thi*: phải có **slice eval theo category** trước, nếu không là đoán mò.

### D. Paraphrase đề bài (giữ nguyên đáp án) — ⚠️ khả thi có kiểm soát, 🟡 vừa
Biến đổi *cách diễn đạt prompt* (đồng nghĩa, đổi thứ tự mệnh đề) trong khi đáp án + format completion giữ nguyên → tăng robustness với cách hỏi lạ trên LB.
- *Vì prompt bị mask (loss 0)*, thay đổi prompt **không** trực tiếp dịch phân phối *loss/trace* → an toàn hơn paraphrase reasoning.
- *Rủi ro*: paraphrase tự động (LLM) có thể đổi nghĩa bài → solver verify lại để loại bài lệch.
- *Falsification*: thêm 1 biến thể paraphrase/bài cho 1 category, A/B slice.

### E. Nhiều dạng `\boxed{}` tương đương — ❌ ít khả thi, 🔴 rủi ro cao
Dạy model rằng `0.5`, `1/2`, `.5` đều đúng. Nhưng grader binary match chính xác và chuẩn hóa riêng → dạy đa-dạng-form dễ khiến model chọn form **không** khớp normalizer của grader → 0 điểm. **Tránh** trừ khi nắm chắc `compare_answer` chuẩn hóa thế nào (xem [reasoning.py](../../nemotron-master/reasoning.py)).

### F. LLM-paraphrase / viết lại nội dung reasoning — ❌ không nên, 🔴 rủi ro cao
Đây chính là lớp thay đổi đã làm exp13/exp19 hồi quy: đổi nội dung/độ dài/phong cách trace làm lệch phân phối khỏi đỉnh nhọn. Không tự verify được "trace mới vẫn dẫn tới đáp án đúng ở greedy". **Tránh.**

### G. Self-generated verified instances (STaR-offline) — ⚠️ big-bet, 🟡 vừa
Sinh trace mới bằng adapter hiện tại cho bài đang sai, **chỉ giữ trace verify đúng**, thêm vào corpus. Khác B ở chỗ trace do *model* viết (đa dạng path) thay vì solver.
- *Rủi ro*: drift/độ dài (compose với cap độ dài); trùng batch-1 idea-7 → coi như mở rộng, không lặp.
- *Điều kiện*: cần verifier round-trip + cap độ dài; làm sau khi B/A đã bank.

## 5. Lộ trình khả thi đề xuất

0. **[Bắt buộc] Dựng eval slice held-out theo category** (vLLM greedy, ~200 bài) + **đo bucket lỗi** `{format, truncation, arithmetic-slip, method-wrong}`. Không có cái này thì mọi augmentation là đoán mò (round-2 đã trả giá).
1. **B trước** (scale instance trong-phân-phối cho 1–2 category yếu nhất theo bucket) — an toàn, đòn bẩy mạnh nhất.
2. **A** (thêm augmenter chuỗi, cap ≤ 20%) — củng cố kỹ năng nền, rủi ro thấp.
3. **C/D** chỉ sau khi có slice theo category; kiểm soát tỷ lệ, verify lại đáp án.
4. **G** big-bet cuối; **E/F tránh**.

**Kỷ luật**: mọi augmentation A/B trên *cùng slice, cùng seed, cùng NUM_STEPS*; chỉ giữ nếu macro exact-match +≥0.3pp **và** không category nào rớt >1pp. Augmentation nào *làm dài trace trung bình* phải kèm đo cap-hit-rate 7680.

## 6. Câu hỏi mở (đưa vào research prompt)
- Tỷ lệ task-phụ:reasoning tối ưu là bao nhiêu trước khi loãng tín hiệu?
- `compare_answer` chuẩn hóa đáp án thế nào (để biết E có cứu được hay không)?
- Mix category của leaderboard — có proxy nào suy ra từ điểm public không?
- Augmentation đúng-phân-phối (B) bão hòa ở mức bao nhiêu instance/category?

---

> Xem prompt research kèm theo (do người dùng chạy) để khảo sát literature cho từng chiến lược, ưu tiên A/B/C/D, và buộc mỗi đề xuất phải nêu *rủi ro dịch phân phối* + *test falsification trên slice*.
