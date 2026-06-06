# Hiện trạng Data — NVIDIA Nemotron Reasoning Challenge

> Cập nhật: 2026-06-02  
> Nguồn: `nemotron-master/problems.jsonl`, `corpus.jsonl`, `generation.jsonl`

---

## 1. Tổng quan

| Metric | Giá trị |
|---|---|
| Tổng bài trong `train.csv` | 9,500 |
| Reasoning files có nội dung | 9,500 (100%) |
| Entries trong `corpus.jsonl` | 17,963 |
| Unmasked tokens (training signal) | 40,871,870 |
| Masked tokens (prompt, không train) | 9,672,519 |
| Tổng tokens | 50,544,389 |

**Lưu ý quan trọng:** `corpus.py` hiện tại đưa **tất cả** bài vào corpus kể cả `rule_unknown` và `hypothesis_formed` — tức là model đang train trên cả CoT **chưa được verify**.

---

## 2. Phân bố token theo category

| Category | Entries | Unmasked tokens | % corpus | Ghi chú |
|---|---|---|---|---|
| bit_manipulation | 1,602 | 10,778,410 | 26.4% | Deterministic solver |
| gravity | 1,597 | 5,694,239 | 13.9% | Deterministic solver |
| cipher | 1,576 | 4,916,239 | 12.0% | Deterministic solver |
| unit_conversion | 1,594 | 3,882,764 | 9.5% | Deterministic solver |
| concatenation | 1,500 | 3,549,069 | 8.7% | Augmentation |
| splitting | 1,500 | 3,543,247 | 8.7% | Augmentation |
| equation_numeric_deduce | 596 | 3,455,149 | 8.5% | Deterministic solver (partial) |
| spelling | 648 | 2,575,296 | 6.3% | Augmentation |
| equation_numeric_guess | 136 | 822,948 | 2.0% | **Phần lớn unsolved** |
| matching | 4,515 | 516,676 | 1.3% | Augmentation (downsampled) |
| lstrip | 300 | 441,726 | 1.1% | Augmentation |
| cryptarithm_deduce | 659 | 415,042 | 1.0% | **Phần lớn unsolved** |
| numeral | 1,576 | 180,202 | 0.4% | Deterministic solver |
| cryptarithm_guess | 164 | 100,863 | 0.2% | **Phần lớn unsolved** |
| **TOTAL** | **17,963** | **40,871,870** | **100%** | |

---

## 3. Trạng thái giải (solve status) theo category

> `rule_found` = solver xác minh đúng, tạo CoT deterministically  
> `hypothesis_formed` = có investigation file nhưng chưa có solver  
> `rule_unknown` = không có solver, CoT là guessed/speculative  

| Category | rule_found | hypothesis_formed | rule_unknown | Tổng | % đúng | Tình trạng |
|---|---|---|---|---|---|---|
| cipher | 1,576 | 0 | 0 | 1,576 | **100%** | ✅ Hoàn chỉnh |
| gravity | 1,597 | 0 | 0 | 1,597 | **100%** | ✅ Hoàn chỉnh |
| numeral | 1,576 | 0 | 0 | 1,576 | **100%** | ✅ Hoàn chỉnh |
| unit_conversion | 1,594 | 0 | 0 | 1,594 | **100%** | ✅ Hoàn chỉnh |
| equation_numeric_deduce | 540 | 22 | 34 | 596 | 91% | ⚠️ Gần hoàn chỉnh |
| bit_manipulation | 1,364 | 121 | 117 | 1,602 | 85% | ⚠️ Còn 238 lỗ hổng |
| equation_numeric_guess | 21 | 35 | 80 | 136 | 15% | 🔴 Phần lớn sai |
| cryptarithm_guess | 11 | 25 | 128 | 164 | 7% | 🔴 Gần như toàn sai |
| cryptarithm_deduce | 54 | 46 | 559 | 659 | **8%** | 🔴 Gần như toàn sai |

**Tổng unsolved (rule_unknown + hypothesis_formed): 1,167 bài**

---

## 4. Vấn đề: Contaminated training data

Corpus hiện tại bao gồm tất cả entries kể cả `rule_unknown`. Điều này có nghĩa:

- **cryptarithm_deduce**: 605/659 entries (92%) là CoT **guessed/sai** — nhưng byte-size tương tự hệt rule_found (avg 1,817 vs 1,771 bytes). Model không có tín hiệu nào để phân biệt đúng/sai.
- **cryptarithm_guess**: 153/164 entries (93%) là CoT sai.
- **bit_manipulation**: 238 entries CoT sai, nhưng tỷ lệ nhỏ hơn (15%).

**Hệ quả:** Model học pattern sai cho cryptarithm, có thể làm giảm điểm trên các bài đó thay vì tăng.

### Số lượng CoT bị nhiễm trong corpus

| Category | Entries bị nhiễm (rule_unknown + hypothesis) | % trong category |
|---|---|---|
| cryptarithm_deduce | 605 | 92% |
| cryptarithm_guess | 153 | 93% |
| equation_numeric_guess | 115 | 85% |
| bit_manipulation | 238 | 15% |
| equation_numeric_deduce | 56 | 9% |
| **TỔNG** | **1,167** | **6.5% của toàn corpus** |

---

## 5. Model pass rates (generation.jsonl)

Pass rate của model hiện tại trên toàn bộ training set (bất kỳ run nào đúng):

| Category | Đúng / Tổng | Pass rate | Nhận xét |
|---|---|---|---|
| numeral | 1,531 / 1,576 | **97.1%** | ✅ Gần saturated |
| unit_conversion | 1,259 / 1,594 | **79.0%** | ✅ Tốt |
| gravity | 1,055 / 1,597 | **66.1%** | ✅ Khá tốt |
| cipher | 573 / 1,576 | **36.4%** | ⚠️ Còn cải thiện được |
| equation_numeric_deduce | 183 / 596 | **30.7%** | ⚠️ Còn nhiều headroom |
| bit_manipulation | 153 / 1,602 | **9.6%** | 🔴 Yếu, và là category lớn nhất (26% corpus) |
| equation_numeric_guess | 6 / 136 | **4.4%** | 🔴 Yếu |
| cryptarithm_deduce | 2 / 659 | **0.3%** | 🔴 Gần như 0 |
| cryptarithm_guess | 0 / 164 | **0.0%** | 🔴 Hoàn toàn không giải được |

**Observation:** `bit_manipulation` có pass rate chỉ 9.6% nhưng chiếm 26.4% token corpus — đây là gap lớn nhất về ROI.

---

## 6. Phân tích cơ hội cải thiện

### Tier 1: Tác động cao, khả thi ngay

| Cơ hội | Category | Bài giải thêm được | Cơ chế |
|---|---|---|---|
| Fix/improve `reasoners/cryptarithm.py` | cryptarithm_deduce | +605 bài | Solver hiện chỉ giải được 54/659 (8%) |
| Fix bit_manipulation edge cases | bit_manipulation | +238 bài | Investigate `investigators/bit_manipulation.py` |
| Giải equation_numeric_guess | equation_numeric_guess | +115 bài | Không có solver hiện tại |

### Tier 2: Scale augmentation (zero-logic-change)

| Augmenter | N hiện tại | N đề xuất | Token tăng thêm (ước tính) |
|---|---|---|---|
| concatenation | 1,500 | 5,000 | +~7M |
| splitting | 1,500 | 5,000 | +~7M |
| lstrip | 300 | 2,000 | +~3M |
| matching (bỏ downsampling) | 4,515 | ~10,000–20,000 | +~1M |

### Tier 3: Làm sạch contaminated data

Lọc `rule_unknown` entries ra khỏi corpus (đặc biệt cho cryptarithm) có thể **tăng điểm** bằng cách ngừng train model trên CoT sai.

**Cách làm:** Trong `corpus.py`, thêm điều kiện `if problem.status == "rule_found"` trước khi include. Rủi ro thấp, có thể rollback ngay.

---

## 7. Tệp nguồn liên quan

| File | Mục đích |
|---|---|
| `nemotron-master/problems.jsonl` | Trạng thái giải từng bài (status, category) |
| `nemotron-master/corpus.jsonl` | Training data cuối cùng (17,963 entries) |
| `nemotron-master/generation.jsonl` | Model inference results (pass rate) |
| `nemotron-master/reasoning/<id>.txt` | CoT trace cho từng bài |
| `nemotron-master/reasoning.py` | Chạy deterministic solvers → tạo reasoning files |
| `nemotron-master/corpus.py` | Tokenize + mask → corpus.jsonl |
| `nemotron-master/reasoners/cryptarithm.py` | Solver cryptarithm (hiện chỉ cover 8%) |
| `nemotron-master/investigators/cryptarithm_deduce.py` | Analysis scripts cho cryptarithm |

---

## 8. Ưu tiên hành động (theo ROI)

1. **Xem xét lọc contaminated data** — bỏ `rule_unknown` entries khỏi corpus, đặc biệt cryptarithm. Chi phí: thấp, rủi ro: thấp.
2. **Scale augmentation** — tăng concatenation/splitting/lstrip lên 5,000. Chi phí: rất thấp (không cần viết logic mới).
3. **Fix bit_manipulation reasoner** — 26% corpus, 9% pass rate. Cải thiện reasoner là leverage cao nhất.
4. **Crack cryptarithm_deduce** — 92% entries hiện bị nhiễm. Nếu solver được cải thiện, cộng thêm 605 verified traces.
5. **Giải equation_numeric_guess** — 136 bài hoàn toàn không có solver.
