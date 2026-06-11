# Phân tích Phân phối Dữ liệu — NVIDIA Nemotron Reasoning Challenge

> Mục đích: Cung cấp nền tảng thực tế để ra quyết định chiến lược reasoning trajectory.  
> Được tạo: 2026-06-04. Dữ liệu: `nemotron-master/problems.jsonl` (9,800 problems) + corpus thực tế.

---

## 1. Cấu trúc bài toán

Mỗi bài toán là một **few-shot rule inference task**: cho 4–8 ví dụ `input → output`, suy ra quy tắc ẩn rồi áp dụng lên input mới. Câu trả lời cần nằm trong `\boxed{...}`. Không có free-form explanation nào được chấp nhận ngoài boxed value.

Format inference: `temperature=0.0`, `max_tokens=7680`, `max_model_len=8192` — model phải giải đúng theo greedy decode một lần duy nhất, không có self-consistency hay majority vote.

---

## 2. Phân phối theo Category

| Category | Số bài | % tổng | Solve rate (solver thủ công) | Trạng thái |
|---|---|---|---|---|
| gravity | 1,897 | 19.4% | **100%** | Fully solved |
| bit_manipulation | 1,602 | 16.3% | 85.1% | **238 unknown** |
| unit_conversion | 1,594 | 16.3% | **100%** | Fully solved |
| cipher | 1,576 | 16.1% | **100%** | Fully solved |
| numeral | 1,576 | 16.1% | **100%** | Fully solved |
| cryptarithm_deduce | 659 | 6.7% | 8.2% | **605 unknown** |
| equation_numeric_deduce | 596 | 6.1% | 90.6% | **56 unknown** |
| cryptarithm_guess | 164 | 1.7% | 6.7% | **153 unknown** |
| equation_numeric_guess | 136 | 1.4% | 15.4% | **115 unknown** |
| **TOTAL** | **9,800** | 100% | **88.1%** | **1,167 unknown** |

**Nhận xét chính:**
- 5 categories (gravity, unit_conversion, cipher, numeral, và phần lớn bit_manipulation) có solver hoàn hảo — reasoning trajectory cho những loại này đã ổn.
- `cryptarithm_deduce` + `cryptarithm_guess` có tổng **758 bài chưa giải** (~77%), là gap lớn nhất.
- `equation_numeric_guess` có 115 bài unknown — solver không handle được operator lạ (ví dụ `@`, `?`).
- `bit_manipulation` unknown 238 bài — likely các phép biến đổi tổ hợp nhiều hơn 1 op.

---

## 3. Phân phối Eval Slice (proxy cho test set)

Từ `eval_slice.jsonl` (170 bài):

| Category | n | Proportion |
|---|---|---|
| bit_manipulation | 25 | 14.7% |
| cipher | 25 | 14.7% |
| equation_numeric_deduce | 25 | 14.7% |
| gravity | 25 | 14.7% |
| numeral | 25 | 14.7% |
| unit_conversion | 25 | 14.7% |
| cryptarithm_deduce | 13 | 7.6% |
| equation_numeric_guess | 5 | 2.9% |
| cryptarithm_guess | 2 | 1.2% |

**Đây là tỷ lệ gần đúng của private leaderboard.** Với 9,800 test bài, ước tính:
- ~1,400–1,900 bài cho mỗi category lớn (6 loại đầu)
- ~700+ bài cryptarithm_deduce
- ~140 equation_numeric_guess
- ~47 cryptarithm_guess

---

## 4. Corpus Training thực tế (sau augmentation)

| Category | n trong corpus | Ghi chú |
|---|---|---|
| **matching** (augment) | 4,515 | Bài toán phụ trợ — không có trong test |
| gravity | 1,897 | 1:1 với problems |
| unit_conversion | 1,594 | 1:1 |
| cipher | 1,576 | 1:1 |
| numeral | 1,576 | 1:1 |
| **splitting** (augment) | 1,500 | Không có trong test |
| **concatenation** (augment) | 1,500 | Không có trong test |
| bit_manipulation | 1,360 | Chỉ bài có solver |
| **spelling** (augment) | 648 | Không có trong test |
| equation_numeric_deduce | 540 | Chỉ bài solved |
| **reverse** (augment) | 300 | Không có trong test |
| **count_substring** (augment) | 300 | Không có trong test |
| **char_index** (augment) | 300 | Không có trong test |
| **lstrip** (augment) | 300 | Không có trong test |
| **digit_extract** (augment) | 300 | Không có trong test |
| cryptarithm_deduce | **54** | Rất thiếu — chỉ 54/659 |
| equation_numeric_guess | **21** | Cực thiếu |
| cryptarithm_guess | **11** | Gần như không có |

**Tổng token: ~50.7M tokens.**

**Vấn đề nghiêm trọng:** Augmentation tasks (`matching`, `splitting`, etc.) chiếm **9,363/18,292 = 51%** corpus nhưng không xuất hiện trong test set — đây là off-distribution training signal rất lớn.

---

## 5. Đặc điểm từng Category — Reasoning Requirements

### 5.1 gravity
- **Bài toán**: Suy ra hằng số g ẩn từ `distance = 0.5 * g * t²`, rồi tính distance mới.
- **Reasoning**: Linear regression / curve fitting đơn giản. Trajectory ngắn-vừa (~1,076 words).
- **Solve rate**: 100%. Model cần học: fit parabola → compute g → apply.

### 5.2 numeral
- **Bài toán**: Chuyển đổi hệ đếm ẩn (thực ra luôn là Roman numeral).
- **Reasoning**: Cực ngắn (~59 words). Nhận diện Roman, viết lại.
- **Solve rate**: 100%. Simplest category.

### 5.3 unit_conversion
- **Bài toán**: Suy ra conversion factor từ examples `X unit → Y`, rồi convert input.
- **Reasoning**: Vừa (~735 words). Tính ratio từ examples, áp dụng.
- **Solve rate**: 100%.

### 5.4 cipher
- **Bài toán**: Suy ra cipher key từ ví dụ encode/decode text, áp dụng lên input mới.
- **Reasoning**: Vừa (~752 words). Cần detect shift/substitution pattern per character.
- **Solve rate**: 100%.

### 5.5 bit_manipulation
- **Bài toán**: Suy ra phép biến đổi bit 8-bit từ 8 ví dụ. Phép biến đổi có thể là: Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT áp dụng theo cột.
- **Reasoning**: **Rất dài** (~7,000 tokens). Solver duyệt exhaustive all per-column operations, match kết quả.
- **Solve rate**: 85.1% — 238 bài unknown là những trường hợp solver hiện tại không cover được (có thể là multi-op hoặc rotation/shift).
- **Gap**: 238 bài chưa có reasoning trajectory.

### 5.6 equation_numeric_deduce
- **Bài toán**: Bảng ký hiệu toán tử ẩn (`/`, `\`, `|`) áp dụng lên số. Suy ra mỗi ký hiệu là phép gì (div, mod, concat, etc.).
- **Reasoning**: Rất dài (~5,868 tokens). Duyệt từng operator symbol, thử tất cả numeric operations.
- **Solve rate**: 90.6% — 56 bài chưa giải (operator lạ hoặc nhiều symbols).
- **Gap**: Nhỏ, 56 bài.

### 5.7 equation_numeric_guess
- **Bài toán**: Tương tự deduce nhưng không phân biệt `guess` vs `deduce` — operator lạ hơn (ví dụ `@`, `?`).
- **Reasoning**: Dài (~6,110 tokens).
- **Solve rate**: 15.4% — 115/136 bài chưa giải. **Rất nghiêm trọng.** Model cần tự suy luận cho 115 bài này.

### 5.8 cryptarithm_deduce
- **Bài toán**: Ký hiệu đặc biệt (`@`, `!`, `&`, `%`, ...) áp dụng lên chuỗi ký tự. Suy ra quy tắc biến đổi từ examples.
- **Reasoning**: Ngắn (~143 words) — solver chỉ solve được 54/659 bài.
- **Gap cực lớn**: 605 bài không có trajectory. Với ~700 bài test ước tính, đây là category ảnh hưởng lớn nhất tới score.

### 5.9 cryptarithm_guess
- Tương tự cryptarithm_deduce nhưng harder.
- Solve rate: 6.7%, 153 unknown.

---

## 6. Phân tích Gap: Nơi Score Bị Mất

### Ước tính điểm mất (giả sử 9,800 test problems):

| Category | Ước tính test | Solver coverage | Bài "unknown" trong test | Score risk |
|---|---|---|---|---|
| gravity | ~1,900 | 100% | 0 | Thấp |
| unit_conversion | ~1,600 | 100% | 0 | Thấp |
| cipher | ~1,600 | 100% | 0 | Thấp |
| numeral | ~1,600 | 100% | 0 | Thấp |
| bit_manipulation | ~1,600 | 85% | ~240 | **Trung bình** |
| equation_numeric_deduce | ~600 | 91% | ~54 | Thấp |
| cryptarithm_deduce | ~700 | 8% | **~643** | **Rất cao** |
| equation_numeric_guess | ~140 | 15% | ~119 | **Cao** |
| cryptarithm_guess | ~60 | 7% | **~56** | **Cao** |

**Tổng bài không có reasoning trajectory ước tính: ~1,112/9,800 = 11.3%**

Đây là ceiling của solver-based approach: nếu model không thể tự suy luận cho 11.3% bài còn lại, điểm tối đa từ approach hiện tại là ~88.7%.

---

## 7. Đặc điểm Training Signal Hiện Tại

### 7.1 Imbalance nghiêm trọng
- Augmentation tasks (51% corpus): không xuất hiện trong test. Model học patterns không liên quan.
- `bit_manipulation` có trace dài nhất (~7K tokens), chiếm ~9.5M tokens / 50.7M tổng (19% corpus tokens).
- `cryptarithm` chỉ có 65 examples trong corpus nhưng có ~700 test bài.

### 7.2 Reasoning depth mismatch
- Solver-generated traces cực kỳ dài và exhaustive (duyệt all combinations).
- Model thực tế cần học **shortcut heuristics** chứ không phải brute-force.
- Traces dài có thể dạy model viết dài thay vì suy luận đúng.

### 7.3 Off-distribution augmentation
- Categories `matching`, `splitting`, `concatenation`, `reverse`, `char_index`, `digit_extract`, `lstrip`, `count_substring`, `spelling` = **auxiliary string tasks**.
- Chúng giúp model learn general pattern-matching, nhưng không trực tiếp giải test problems.
- Hypothesis: model học "viết trajectory theo format" hơn là "suy luận đúng rule".

### 7.4 Distribution của categories trong corpus vs test
```
Category          Corpus%   Test% (est)   Ratio
gravity            10.4%     19.4%        0.54x  ← under-represented
numeral             8.6%     16.1%        0.53x  ← under-represented  
bit_manipulation    7.4%     16.3%        0.45x  ← severely under
cryptarithm_deduce  0.3%      6.7%        0.04x  ← critically under
```

---

## 8. Key Bottlenecks (Tóm tắt)

1. **Cryptarithm gap (highest impact)**: 758 bài (cryptarithm_deduce + guess) thiếu trajectory. Solver không decode được symbolic operators. ~7% test score at stake.

2. **bit_manipulation 238 unknowns**: Solver không handle multi-operator compositions (e.g., `SHIFT` + `XOR`). ~2.4% test score.

3. **equation_numeric_guess 115 unknowns**: Operator không nằm trong vocabulary solver. ~1.2% test score.

4. **Augmentation off-distribution**: 51% training corpus không liên quan trực tiếp đến test. Có thể gây confusion và waste capacity.

5. **No self-correction**: Vì inference là greedy one-shot, model không có cơ hội verify/retry. Trajectory cần chứa explicit verification step.

---

## 9. Implications cho Reasoning Trajectory Strategy

Để phân tích được các chiến lược tiếp theo, cần trả lời:

1. **Cryptarithm**: Liệu có thể dùng LLM mạnh hơn (GPT-4o, Claude Opus) để generate reasoning trajectory cho 758 bài chưa giải? Quality vs. cost?

2. **Bit_manipulation**: Mở rộng solver để handle shift/rotation operations? Hay dùng exhaustive enumeration với LLM verification?

3. **Equation_numeric_guess**: Augment solver với thêm operator types hoặc dùng neural approach?

4. **Augmentation reduction**: Có nên giảm corpus từ 18K xuống ~9K bài thực tế (bỏ augmentation categories không liên quan)?

5. **Reasoning format**: Traces hiện tại rất dài (7K tokens). Có nên distill thành shorter, more accurate traces?
