# Phân tích approach của Tong Hui Kang — Progress Prize submission

> Nguồn: `nemotron-master/` (writeup: kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915, site: nemotron.huikang.dev)  
> Mục đích: Hiểu rõ kiến trúc ý tưởng để tìm điểm đột phá vượt 0.86.

---

## 1. Tổng quan ý tưởng

Tong Hui Kang (THK) xây dựng một pipeline **data-centric, fully deterministic** với triết lý cốt lõi:

> *"Nếu bạn có thể viết solver giải đúng bài toán, bạn có thể sinh ra reasoning trajectory hoàn hảo về mặt logic. Train model trên trajectory đó — model học cách suy luận, không phải học thuộc đáp án."*

Pipeline gồm 4 giai đoạn tuần tự:

```
reasoning.py  →  augmentation.py  →  corpus.py  →  train_sft.py
   (solver)        (aux tasks)       (tokenize)      (SFT/Tinker)
```

---

## 2. Giai đoạn 1: Deterministic Solver per Category

Mỗi category có một solver Python viết tay (`reasoners/<category>.py`):

| Category | Solver logic | Trace length |
|---|---|---|
| `gravity` | Fit `d = 0.5 * g * t²` bằng least squares → tính g → predict | ~1,076 words |
| `unit_conversion` | Tính conversion ratio từ examples → áp dụng | ~735 words |
| `cipher` | Detect substitution cipher per character → decode | ~752 words |
| `numeral` | Nhận diện hệ Roman numeral → chuyển đổi | ~59 words (cực ngắn) |
| `bit_manipulation` | **Exhaustive per-column search**: thử Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT cho từng bit column; match pattern → apply | **~7,000 tokens (dài nhất)** |
| `equation_numeric` | Thử tất cả arithmetic operators (+, -, *, /, mod, concat, etc.) cho từng symbol ẩn → suy ra mapping | ~5,800 tokens |
| `cryptarithm` | Thử decode symbolic operations (concatenation, reverse-concat, etc.) → match | ~143 words |

**Nguyên tắc sinh trace:** Solver KHÔNG chỉ trả về đáp án mà emit một **natural-language CoT** mirror lại từng bước của thuật toán. Chỉ khi `compare_answer(solver_answer, true_answer)` khớp thì trace mới được ghi ra `reasoning/<id>.txt` với status `rule_found`. Bài không giải được → `rule_unknown`.

**Kết quả solver coverage:**
- Gravity, cipher, numeral, unit_conversion: **100%**
- Bit_manipulation: **85.1%** (238 bài unknown)
- Equation_numeric_deduce: **90.6%** (56 bài unknown)
- Cryptarithm_deduce: **8.2%** (605 bài unknown)
- Cryptarithm_guess: **6.7%** (153 bài unknown)
- Equation_numeric_guess: **15.4%** (115 bài unknown)

---

## 3. Giai đoạn 2: Augmentation Tasks

THK thêm **9 loại bài phụ trợ** (không có trong test) để dạy model general pattern-matching:

| Augmenter | Mô tả | n training examples |
|---|---|---|
| `matching` | Tìm permutation khớp 2 chuỗi binary | 4,515 |
| `splitting` | Tách chuỗi theo delimiter | 1,500 |
| `concatenation` | Ghép chuỗi theo pattern | 1,500 |
| `spelling` | Nhận diện từ viết hoa/thường | 648 |
| `lstrip` | Strip prefix pattern | 300 |
| `reverse` | Đảo ngược chuỗi | 300 |
| `count_substring` | Đếm substring | 300 |
| `char_index` | Tìm index ký tự | 300 |
| `digit_extract` | Trích số từ chuỗi | 300 |

**Tổng augmentation: 9,663 examples — chiếm 51% corpus.**

**Lý do THK thêm augmentation:** Giả thuyết là các tasks này dạy model:
1. Kỹ năng đọc ví dụ theo cột/hàng (critical cho bit_manipulation)
2. Pattern matching tổng quát trước khi apply vào task cụ thể
3. Tăng data diversity chống overfitting SFT

---

## 4. Giai đoạn 3: Corpus Building

`corpus.py` tokenize và mask toàn bộ data:

```
Prompt (masked, weight=0) + Completion (unmasked, weight=1) → training tokens
```

**Format completion cố định:**
```
{reasoning_text}\n</think>\n\boxed{answer}<|im_end|>
```

**Tokenization:** Dùng chat template với `enable_thinking=True` (Nemotron-H specific).

**Quality gates (từ exp21):**
- `VERIFY_GATE=True`: Xóa bài nào mà solver trả lời sai (trace `rule_unknown`)
- `LENGTH_GATE=True`: Drop trace nào > 7,600 tokens (vượt inference budget 7,680)

**Kết quả corpus:** ~18,292 examples, ~50.7M tokens.

---

## 5. Giai đoạn 4: SFT Training (Tinker)

THK dùng **Tinker** — một custom SFT trainer (không phải Unsloth) chạy trên Modal cloud:

**Config cơ bản:**
- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (Mamba/MoE hybrid)
- LoRA rank: 32 (constraint cứng của competition)
- Stratified batching: phân phối đều tất cả category trong mỗi batch

**Loss functions THK explore:**
- `cross_entropy` (default)
- `importance_sampling` (off-policy IS)
- `ppo` (clipped policy gradient)
- `cispo` (confidence-based IS)
- `dro` (distributional robust, thêm β·KL penalty)

**Đặc điểm kỹ thuật:**
- Per-epoch logprob tracking để monitor overfitting
- Metrics: logprob_decreased/increased, KL per token, clip fractions
- Không có online RL loop — toàn bộ là **offline SFT** trên corpus tĩnh

**Score đạt được: 0.86 (Progress Prize winner)**

---

## 6. Phân tích: Tại sao bị stuck ở 0.86?

### 6.1 Trần cứng từ solver coverage (lý do chính)

**Ceiling tính toán được:**

Với 9,800 test problems, ước tính distribution giống train:
- 1,167 bài không có reasoning trajectory (11.9%)
- Model được train trên corpus **không chứa** reasoning đúng cho những bài này
- Với những bài `rule_unknown`, SFT corpus gốc **thực ra chứa traces có boxed answer sai** (submission ≠ true answer) — model học format đúng nhưng logic sai

```
Ceiling nếu không giải thêm: 100% - 11.9% ≈ 88.1%
Thực tế: 0.86 (thấp hơn ceiling vì model không perfectly recall tất cả)
```

**Cryptarithm là gap lớn nhất:**
- ~700 bài test ước tính, chỉ 65 training examples có trace đúng
- Model thấy 10× nhiều bài không có trace hơn có trace
- Dự đoán accuracy gần random (~8%) → mất ~6.5% tổng score từ đây

### 6.2 Corpus distribution mismatch

| Loại data | % corpus | % test questions |
|---|---|---|
| Augmentation (off-distribution) | **51%** | **0%** |
| bit_manipulation | 7.4% | 16.3% |
| gravity | 10.4% | 19.4% |
| cryptarithm | 0.3% | 6.7% |

Model bỏ 51% "capacity" training vào tasks không xuất hiện trong test. Nếu model capacity hữu hạn (LoRA rank 32), đây là sự lãng phí đáng kể.

**Lưu ý quan trọng:** Điều này không đơn giản là "augmentation = waste". THK thêm augmentation chính xác vì model cần các skills tổng quát. Câu hỏi là: liệu 51% có phải là **quá nhiều** không?

### 6.3 SFT-only, không có self-correction

**Inference protocol:** Greedy decode, một lần, không retry. Model phải đúng ngay từ đầu.

SFT thuần dạy model **reproduce** reasoning trace — không dạy model **verify** hay **backtrack** khi sai. Khi model gặp bài khó (cryptarithm, bit_manip unknown), nó:
1. Generate trace theo format đã học
2. Không có mechanism nào để detect "tôi không chắc"
3. Output boxed answer sai với confidence cao

**So sánh với RL:** Một model trained với RL/preference có thể học "khi nào nên thử hướng khác". SFT model không có tín hiệu này.

### 6.4 Trace quá dài cho bit_manipulation

Bit_manipulation traces trung bình **7,000 tokens** — exhaustive search tất cả 9 operators × 8 bit-columns × 8 pairs of inputs. 

**Vấn đề:** Model học được format và logic, nhưng trace dài chiếm phần lớn context window tại inference (7,680 token budget). Nếu bài phức tạp hơn (multi-op), model cần trace dài hơn nhưng budget không cho phép → truncation → sai.

Hơn nữa, **238 bài unknown** là những bài solver không cover được, likely vì chúng dùng operators phức tạp hơn (e.g., bit rotation, shift). Model chưa thấy trace đúng cho loại này.

### 6.5 Không có "hard negative" tường minh

Corpus của THK chỉ chứa traces **đúng** (cross-entropy trên correct trajectories). Không có cơ chế nào:
- Dạy model biết trajectory **sai** trông như thế nào
- Preference giữa đúng và sai cho cùng một input
- Self-consistency check giữa intermediate steps và boxed answer

Kết quả: model học **"viết CoT đúng format"** nhưng không học **"đừng viết CoT dẫn đến đáp án sai"**.

### 6.6 Augmenters "matching" có thể interfere với bit_manipulation

`matching` (4,515 examples) dùng cùng binary string format như `bit_manipulation` nhưng với task khác (find permutation). Với 4,515 vs 1,360 examples, model có thể confuse hai task này khi gặp binary string input.

---

## 7. Cấu trúc score hiện tại (ước tính)

```
Category              Test est.   Model accuracy est.   Score contribution
gravity               1,900       ~95%                  +18.5%
unit_conversion       1,600       ~92%                  +15.0%
cipher                1,600       ~90%                  +14.7%
numeral               1,600       ~97%                  +15.8%
bit_manipulation      1,600       ~78%                  +12.7%  [solver miss + recall]
equation_numeric      730         ~78%                  +5.8%
cryptarithm_deduce    700         ~7%                   +0.5%
cryptarithm_guess     60          ~5%                   +0.03%
equation_num_guess    140         ~12%                  +0.17%
                                           TOTAL ≈       0.83–0.86
```

*Ước tính thô — khớp với score thực 0.86.*

---

## 8. Tóm tắt: Điểm mạnh và giới hạn của approach THK

### Điểm mạnh
- **Fully verifiable training data**: mỗi trace được xác nhận đúng trước khi train → zero noise
- **Deterministic reasoning format**: model học một cách suy luận nhất quán, có thể kiểm tra
- **Domain-specific augmentation**: dạy skills prerequisite (pattern matching) trước
- **Sản phẩm sạch**: 0.86 chỉ bằng SFT thuần là remarkably good cho model 30B MoE với LoRA rank 32

### Giới hạn cơ bản (không thể fix bằng tweaking)
1. **Solver coverage ceiling** ≈ 88%: 1,167 bài không có correct trajectory → model không thể học cái không có
2. **SFT-only, no self-improvement**: không có loop để bootstrap từ những bài khó
3. **No negative learning**: model không biết tránh sai lầm, chỉ biết reproduce đúng
4. **Category imbalance**: 51% corpus off-distribution, cryptarithm underrepresented 50×

### Để vượt 0.86, cần ít nhất 1 trong 3 hướng
1. **Giải thêm cryptarithm** (impact lớn nhất: ~6-7% score at stake)
2. **Negative-aware training** (REDI/DPO) để giảm error rate trên bài đã có trace
3. **Reduce augmentation bias** và rebalance corpus theo test distribution
