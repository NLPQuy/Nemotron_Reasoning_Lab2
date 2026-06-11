# Implementation Plan — Batch 4 (exp27…exp38)

Mục tiêu: hiện thực hóa 12 ý tưởng trong [batch-4.md](batch-4.md) để đẩy **0.86 → 0.87+**.

> **Nguyên tắc bất di bất dịch (kế thừa từ batch-3):**
> - KHÔNG tự suy ra math / schema — đọc code gốc trong `nemotron-master/` trước khi viết.
> - KHÔNG viết lại nội dung hay độ dài trace (bài học exp13 0.68, exp19 0.79).
> - Mọi trace mới phải qua `compare_answer()` trước khi vào corpus.
> - Mọi loss mới phải verify trên eval_slice 50-step trước khi full run.

---

## 0. Mapping exp → idea

| exp | Batch-4 Idea | Cơ chế | Base |
|-----|-------------|--------|------|
| exp27 | Idea 1 — Extend bit_manipulation solver | Upstream solver, mới corpus | Continuer copy |
| exp28 | Idea 4 — Drop off-distribution augmentation | Upstream data edit, mới corpus | Continuer copy |
| exp29 | Idea 9 — Iterative DPO on eval_slice | infer_slice + DPO loss | nemotron-master runner |
| exp30 | Idea 5 — OXA offline exploration-aware SFT | OXA loss trong loss_config.py | nemotron-master runner |
| exp31 | Idea 2 — Procedural cryptarithm + RFT | generators/ + solver verify | Continuer copy |
| exp32 | Idea 8 — AdaSTaR adaptive sampling | Adaptive _stratified_batches | nemotron-master runner |
| exp33 | Idea 10 — VCORE key-token weighting | Float mask trong corpus.py | nemotron-master runner |
| exp34 | Idea 6 — Paraphrase traces (Shape-of-Thought) | Claude API paraphrase + corpus rebuild | nemotron-master runner |
| exp35 | Idea 7 — REDI negative-trace training | Negative loss từ enhance_cot/redi/ | nemotron-master runner |
| exp36 | Idea 3 — LogicPuzzleRL offline DPO | vLLM rollout + DPO pairs | nemotron-master runner |
| exp37 | Idea 11 — GeoRA LoRA init | SVD init trên 0.86 adapter | Continuer copy |
| exp38 | Idea 12 — RL→SFT ordering | DPO trước rồi SFT | nemotron-master runner |

**Run order (theo score + dependency):**
```
exp27 → exp28 → exp29 → exp30 → exp31 → exp32 → exp33 → exp34 → exp35 → exp36 → exp37 → exp38
```
exp37, exp38 chạy cuối vì phụ thuộc kết quả exp29 (DPO) và feasibility thấp.

---

## 1. Setup (làm 1 lần)

### 1a. Official source repos (đã clone)

```bash
# Đã clone vào enhance_cot/ (depth=1):
#   enhance_cot/redi/   — github.com/Tim-Siu/reinforcement-distillation  (REDI)
#   enhance_cot/adastar/ — github.com/reiss-koh/AdaSTaR                  (AdaSTaR)
#   enhance_cot/vcore/  — github.com/coder-gx/VCORE                      (VCORE, LLaMA Factory base)

# KHÔNG pip install từ các repo này — chỉ đọc algorithm để adapt sang Tinker/loss_config.py
```

### 1b. Tạo exp files Continuer-based (đã cp)

```bash
cd /path/to/repo
# Đã tạo: experiments/exp27.py, exp28.py, exp31.py
# exp file còn lại (nemotron-master based) nằm trong experiments/ nhưng là scripts độc lập
```

### 1c. Data-flow chuẩn (kế thừa batch-3)

```
edit upstream (reasoners/ | augmenters/ | corpus.py)
  → uv run python3 reasoning.py        # verify + ghi reasoning/<pid>.txt
  → uv run python3 corpus.py           # tokenize + mask → corpus/<pid>/synthetic.jsonl
  → pack_kaggle_snapshot.py            # layout tokens/<pid>/synthetic.json {tokens,mask}
  → upload Kaggle dataset / ghi Modal volume
  → exp<N>.py: trỏ EXP<N>_CORPUS sang snapshot mới
```

⚠️ **Gap format (verify trước mỗi lần upload):** `corpus.py` xuất `synthetic.jsonl` (segments),
trainer đọc `synthetic.json` `{tokens,mask}` (Kaggle) / `corpus_preprocessed.jsonl` (Modal).
Bước pack là bắt buộc — dùng lại `pack_kaggle_snapshot.py` trong `nemotron-master/`.

---

## 2. exp27 — Extend bit_manipulation solver *(Idea 1, P3, T1)* — 🏆 ƯU TIÊN 1

**Hypothesis:** Thêm operators LEFT_ROTATE(k), RIGHT_ROTATE(k), LEFT_SHIFT(k), RIGHT_SHIFT(k),
MAJORITY vào solver → giải thêm ~120–170 trong 238 unknown → corpus mới → retrain.

**Thời gian ước tính:** 2h engineering + 1h corpus rebuild + 4h training = **~7h total**

### Upstream (nemotron-master/)

**File:** `reasoners/bit_manipulation.py`

Đọc file gốc để hiểu `OPERATORS` dict hiện tại (9 operators). Thêm vào cuối dict:

```python
# >>> EXP27 START — đọc bit_manipulation.py dòng OPERATORS trước khi edit
# Thêm vào sau block OPERATORS hiện có (KHÔNG đổi các operator đã có):
# - LEFT_ROTATE_k / RIGHT_ROTATE_k với k ∈ {1..7}: dịch vòng bit theo hàng
# - LEFT_SHIFT_k / RIGHT_SHIFT_k với k ∈ {1..7}: dịch có fill-0
# - MAJORITY_abc: majority vote của 3 cột input (cần thêm tham số cột)
# Cách implement: đọc hàm test_operator() gốc để biết signature,
#   rồi extend — KHÔNG tự đoán input format.
# <<< EXP27 END
```

**Kiểm tra trước khi corpus rebuild:**
```bash
cd nemotron-master
uv run python3 reasoning.py 2>&1 | grep "bit_manipulation" | grep "rule_found" | wc -l
# Baseline: ~1122 rule_found. Target sau EXP27: >= 1150 (tức >= 28 newly solved)
# Nếu < 30 mới solved → abandon trước khi rebuild corpus
```

**Sau khi xác nhận >= 30 mới solved:**
```bash
uv run python3 corpus.py
uv run python3 pack_kaggle_snapshot.py --tag exp27
# Upload snapshot lên Kaggle dataset / Modal volume với tag EXP27
```

### exp27.py config

Sau khi upload snapshot, set trong exp27.py:
```python
EXP27_CORPUS = "<kaggle-dataset-root>"   # None → baseline
```

### Chạy

```bash
modal run experiments/exp27.py   # ~4h on RTX PRO 6000
```

### Falsification

Sau training, chạy `infer_slice.py` trên eval_slice.jsonl, dùng `eval_slice.py` để score.
Nếu bit_manipulation accuracy trên eval_slice không tăng → solver additions không cover test distribution.

---

## 3. exp28 — Drop off-distribution augmentation *(Idea 4, P3, T1)* — ƯU TIÊN 2

**Hypothesis:** Xóa 9,663 augmentation examples (51% corpus), thay bằng 2–3K in-distribution
mini-tasks từ existing `reasoners/` → giảm task-confusion, tập trung gradient vào 9 test categories.

**Thời gian ước tính:** 1.5h data edit + 1h corpus rebuild + 4h training = **~6.5h total**

### Upstream (nemotron-master/)

**File:** `augmentation.py`

Đọc file gốc để thấy các `problems.extend(...)` calls. Comment-out/remove tất cả augmenter calls:
```python
# >>> EXP28 START — xóa 9 augmenter calls (matching/splitting/concat/spelling/lstrip/reverse/count/char/digit)
# Comment-out (đừng xóa) để dễ rollback:
# for augmenter in AUGMENTERS: ...
# Thay bằng: viết augmenters/mini_rule.py (xem bên dưới)
# <<< EXP28 END
```

**File mới:** `augmenters/mini_rule.py`

Tạo `mini_rule.py` generate 200–300 simplified instances per category bằng cách gọi
`reasoning_<cat>(problem)` với narrow parameter ranges — KHÔNG tự tính đáp án.

Skeleton (đọc `augmenters/spelling.py` để hiểu interface `generate()` trả `list[dict]`):
```python
# augmenters/mini_rule.py
# KHÔNG implement — skeleton chỉ show interface cần match:
# def generate(n_per_cat=250) -> list[dict]:
#     results = []
#     for category in CATEGORIES:
#         solver = SOLVERS[category]
#         for _ in range(n_per_cat):
#             problem = generate_narrow_params(category)  # TODO: đọc generators/ để biết params
#             trace = solver(problem)
#             if trace and compare_answer(problem.answer, extract_answer(trace)):
#                 results.append({"id": ..., "prompt": ..., "completion": trace, "category": category})
#     return results
```

⚠️ **Trước khi implement:** đọc `augmenters/spelling.py` (dòng 50–115) để hiểu exact dict schema
`{id, prompt, completion, category}` và `augmentation.py` để biết cách register.

**Sau corpus rebuild:**
```bash
# Verify: corpus size giảm từ ~50.7M tokens xuống ~28M
uv run python3 corpus.py
# Check: không có augmentation entry nào còn category matching/splitting/...
uv run python3 pack_kaggle_snapshot.py --tag exp28
```

### Falsification sau 50-step mini-run

```bash
# Chạy 50-step với corpus exp28, evaluate eval_slice
# Nếu bit_manipulation hoặc cipher accuracy drops > 2pp vs baseline → restore augmentation
```

---

## 4. exp29 — Iterative DPO on eval_slice *(Idea 9, P6, T1)* — ƯU TIÊN 3

**Hypothesis:** 10 rollouts / problem từ eval_slice → DPO pairs (correct vs incorrect) → DPO 50 steps →
model learns to prefer own correct reasoning on test distribution.

**Thời gian ước tính:** 1h inference + 1h pair build + 2h DPO training = **~4h total** (cheapest)

**Dependency:** Cần GPU + trained adapter (baseline 0.86 hoặc kết quả exp27/exp28).

### Bước 1 — Sinh rollouts (GPU, nemotron-master/)

```bash
cd nemotron-master
uv run python3 infer_slice.py \
  --base   nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --adapter <path-to-0.86-adapter> \
  --slice  eval_slice.jsonl \
  --out    preds_exp29_temp05.jsonl \
  --temperature 0.5 \
  --n_samples 10
# Đọc infer_slice.py để xem args thực tế — KHÔNG đoán flag names
```

⚠️ `infer_slice.py` hiện hỗ trợ greedy (temperature=0). Trước khi chạy: kiểm tra xem file có
`--temperature` / `--n_samples` flags chưa; nếu chưa thì cần thêm.

### Bước 2 — Build DPO pairs (local)

Đọc output từ bước 1 cùng với `eval_slice.jsonl`. Với mỗi problem:
- Tìm ít nhất 1 correct và 1 incorrect rollout (verify bằng `compare_answer()` từ `reasoning.py`)
- Tạo pair `{prompt, chosen, rejected}`

Script: `nemotron-master/build_eval_slice.py` hiện đã có — kiểm tra xem có hỗ trợ pair output chưa.
Nếu chưa: tạo `nemotron-master/build_dpo_pairs.py` (đọc `build_eval_slice.py` làm template).

```bash
uv run python3 build_dpo_pairs.py \
  --slice eval_slice.jsonl \
  --preds preds_exp29_temp05.jsonl \
  --out   dpo_pairs_exp29.jsonl
# Expected: ~80–120 valid pairs (problem có ≥1 correct + ≥1 incorrect)
# Nếu < 20 pairs → rollout pass rate quá thấp → ABORT
```

### Bước 3 — DPO training (nemotron-master/train_sft.py)

DPO loss chưa có trong `loss_config.py`. Cần thêm `DPOLossConfig`. Nhưng **Tinker** (trainer engine)
có hỗ trợ DPO hay không phụ thuộc vào `tinker.types.LossFnType`.

**Trước khi implement:** chạy:
```bash
python3 -c "from tinker.types import LossFnType; print(LossFnType.__args__)"
```
Nếu `"dpo"` không có trong LossFnType → Tinker không native support DPO → cần dùng TRL.

**Fallback nếu Tinker không support DPO:** dùng enhance_cot/redi/experiments_trl/open_r1_dpo.py
làm reference — adapt sang Hugging Face TRL với adapter 0.86 như reference model.

### exp29.py role

Script orchestration — không phải Continuer copy. Xem `experiments/exp29.py`.

---

## 5. exp30 — OXA offline exploration-aware SFT *(Idea 5, P1, T1)*

**Paper:** arXiv:2603.16206 — không có official repo. Implement từ paper.

**Hypothesis:** Upweight corpus entries với low logprob correct traces (hard, under-explored),
downweight high-confidence incorrect examples → +6 Pass@1 (paper's Qwen2.5-1.5B result).

**Thời gian ước tính:** 2h logprob precompute + 2h loss_config edit + 4h training = **~8h total**

### Bước 1 — Logprob precompute (GPU)

```bash
cd nemotron-master
# Chạy forward pass (không sample) trên toàn corpus để lấy per-example avg logprob
# infer_slice.py đọc eval_slice.jsonl — cần script tương tự cho full corpus
# Tạo: nemotron-master/compute_corpus_logprobs.py (đọc train_sft.py load_corpus_entries() để biết format)
uv run python3 compute_corpus_logprobs.py \
  --base <base-model> --adapter <0.86-adapter> \
  --corpus corpus_preprocessed.jsonl \
  --out corpus_logprobs.jsonl
```

### Bước 2 — OXA loss trong loss_config.py

OXA hai objective theo paper: promote hard-correct (low logprob correct), suppress high-confident-incorrect.

Implement bằng cách extend `CrossEntropyWithWeightingLossConfig`. Đọc `loss_config.py` dòng 97–140
để hiểu `apply_weights()` signature trước khi viết:

```python
# Skeleton (đọc CrossEntropyWithWeightingLossConfig để hiểu cách override):
# class OXALossConfig(CrossEntropyWithWeightingLossConfig):
#     promote_threshold: float = -0.5   # per-token logprob, below = hard-correct
#     suppress_threshold: float = -0.1  # per-token logprob, above = high-conf-incorrect
#     promote_weight: float = 2.0
#     suppress_weight: float = 0.3
#     # Override apply_weights() để map entry-level logprob → per-token weight
#     # Cần đọc build_datum() trong train_sft.py để hiểu how weights are passed per-token
```

⚠️ **Critical check:** `apply_weights()` hiện apply per-token (list[float]). OXA cần per-example
thresholding. Verify rằng `corpus_logprobs.jsonl` có thể được loaded trong `main()` của `train_sft.py`
và mapped sang `float_advantages` trước khi implement.

### exp30.py role

Script tổ chức 2 bước trên. Xem `experiments/exp30.py`.

---

## 6. exp31 — Procedural cryptarithm + RFT *(Idea 2, P12, T2)*

**Hypothesis:** Sinh 1,000+ procedural cryptarithm problems → adapter rollout temperature=0.5 →
giữ correct completions (RFT) → thêm vào corpus → retrain.

**Thời gian ước tính:** 3h generator + 2h inference + 1h filter + 4h training = **~10h total**

### Upstream (nemotron-master/)

**File mới:** `generators/cryptarithm_procedural.py`

Đọc `reasoners/store_types.py` (Problem schema, dòng 20–86) và `problems.jsonl` để hiểu format.
Đọc `reasoners/bit_manipulation.py` (hoặc một reasoner đơn giản) để hiểu cấu trúc problem object.

Generator skeleton (KHÔNG tự implement — đọc schema trước):
```python
# Xem experiments/exp31.py dành phần TODO cho từng step sau khi đọc schema
```

**Pre-check (chạy trước khi build generator):**
```bash
cd nemotron-master
# Đếm cryptarithm examples hiện tại trong corpus:
python3 -c "
import json
n = sum(1 for l in open('corpus_preprocessed.jsonl')
        if json.loads(l).get('category','').startswith('cryptarithm'))
print(f'cryptarithm in corpus: {n}')
"
# Expected: ~65 entries. Target sau exp31: >= 200
```

**Bước quan trọng nhất — falsify trước khi build corpus:**
```bash
# Rollout 200 procedural problems với temperature=0.5, n=10
# Nếu pass@10 < 2% → không đủ signal → ABORT
uv run python3 infer_slice.py \
  --base <base> --adapter <0.86> \
  --slice procedural_cryptarithm_200.jsonl \
  --temperature 0.5 --n_samples 10 \
  --out procedural_preds.jsonl
python3 -c "
import json
preds = [json.loads(l) for l in open('procedural_preds.jsonl')]
# count problems với ít nhất 1 correct sample
..."
```

---

## 7. exp32 — AdaSTaR adaptive sampling *(Idea 8, P8, T2)*

**Official source:** `enhance_cot/adastar/` (github.com/reiss-koh/AdaSTaR)

**Key files từ AdaSTaR repo để đọc:**
- `utils_adastar.py` — hàm tính accuracy và update sampling weights
- `iteration_train.py` — training loop với adaptive sampling

**Hypothesis:** Thay static `_stratified_batches()` bằng adaptive sampling cập nhật mỗi 200 steps
dựa trên per-category eval_slice accuracy → upweight cryptarithm, downweight gravity/numeral.

**Thời gian ước tính:** 3h implementation + 5h training = **~8h total**

### Implementation trong nemotron-master/train_sft.py

Đọc AdaSTaR's `utils_adastar.py` để hiểu weight update formula, sau đó adapt vào `_stratified_batches()`:

```python
# Sau khi đọc utils_adastar.py:
# 1. Thêm compute_category_accuracy(eval_slice_path, model) function
# 2. Modify _stratified_batches() để nhận per-category weights
# 3. Trong main() loop: every 200 steps → compute accuracy → update weights
```

⚠️ AdaSTaR dùng JAX/TPU (`device_train.py`). Logic sampling trong `utils_adastar.py` là framework-agnostic
và có thể adapt sang Python/Tinker. Đọc kỹ trước khi port.

### exp32.py role

Script orchestration với knob `ADASTAR_EVAL_INTERVAL = 200` (steps giữa các lần update weights).
Xem `experiments/exp32.py`.

---

## 8. exp33 — VCORE key-token loss weighting *(Idea 10, P4, T2)*

**Official source:** `enhance_cot/vcore/` (github.com/coder-gx/VCORE — LLaMA Factory base)

**Key files từ VCORE repo để đọc:**
- `enhance_cot/vcore/llama_factory/` — tìm file chứa variance-controlled weighting logic
- README.md — algorithm description

**Hypothesis:** 3-tier float weight mask: `\boxed{}` tokens → 3.0, operator-decision tokens → 2.0,
boilerplate → 0.5. Sử dụng `CrossEntropyWithWeightingLossConfig` đã có trong loss_config.py.

**Thời gian ước tính:** 2h corpus.py edit + 1h verify + 4h training = **~7h total**

### Bước 1 — Đọc VCORE algorithm

```bash
ls enhance_cot/vcore/llama_factory/
# Tìm file liên quan đến loss weighting / token weighting
# README: https://github.com/coder-gx/VCORE — đọc để hiểu exact weight formula
```

### Bước 2 — Modify corpus.py

**File:** `nemotron-master/corpus.py`

Hiện tại mask là int 0/1. Cần convert sang float weights.

**Trước khi edit:** kiểm tra `CrossEntropyWithWeightingLossConfig.apply_weights()` xem có nhận
float list hay không (đọc loss_config.py dòng 117–139). Nếu có → chỉ cần change corpus output.

```python
# Trong corpus.py reasoning branch (dòng 176–227):
# Thay mask=[0,0,...,1,1,...] (int) bằng mask=[0.0, ..., 0.5, ..., 2.0, ..., 3.0]
# Logic:
#   - prompt tokens: 0.0 (không đổi)
#   - completion boilerplate: 0.5
#   - operator-decision tokens (regex per category): 2.0
#   - \boxed{} + </think> tokens: 3.0
# KHÔNG implement regex trước khi đọc actual trace format cho từng category
```

### Falsification

```bash
# Trước full training: 200-step mini-run
# Check: gradient norm cho decision tokens >= 2x boilerplate
# Nếu weights bị normalize away → CrossEntropyWithWeightingLossConfig không apply đúng
```

---

## 9. exp34 — Paraphrase traces (Shape-of-Thought) *(Idea 6, P12, T1)*

**Paper:** arXiv:2512.22255 — không có official repo. Dùng Claude API.

**Hypothesis:** Paraphrase solver traces bằng Claude API → vocabulary diversity → break gradient
saturation từ template-rigid traces → +1 pp theo Shape-of-Thought.

**Thời gian ước tính:** 3h API calls (~2K traces @ Claude Haiku) + 1h verify + 4h training = **~8h total**

⚠️ **Rủi ro chính:** LLM có thể thay đổi boxed answer (hallucination). Verification gate là bắt buộc.

### Quy trình (nemotron-master/)

`nemotron-master/paraphrase_instances.py` đã tồn tại. Đọc file này để hiểu interface,
sau đó extend cho `reasoning/*.txt` files (không phải chỉ problem instances).

```bash
cd nemotron-master
# Đọc paraphrase_instances.py trước:
head -50 paraphrase_instances.py

# Sau khi hiểu interface, chạy:
uv run python3 paraphrase_instances.py \
  --mode traces \              # thêm mode này nếu chưa có
  --input reasoning/ \
  --out reasoning_paraphrased/ \
  --verify                     # gọi compare_answer() sau mỗi paraphrase
```

**Verification gate (bắt buộc):**
- Extract `\boxed{}` từ paraphrased trace
- `compare_answer(original_answer, extracted_answer)` phải pass
- Token count ≤ 7,600 (LENGTH_GATE)
- Nếu fail → giữ original

**Pre-check trước khi full run:**
```bash
# Paraphrase 100 traces, tính average per-token logprob của adapter 0.86
# trên paraphrased vs original. Nếu logprob paraphrased KHÔNG cao hơn > 0.1 → abandon
```

---

## 10. exp35 — REDI negative-trace training *(Idea 7, P1, T1)*

**Official source:** `enhance_cot/redi/` (github.com/Tim-Siu/reinforcement-distillation)

**Key files từ REDI repo để đọc:**
```bash
# REDI sử dụng TRL — không thể import trực tiếp sang Tinker
# Đọc để hiểu algorithm:
cat enhance_cot/redi/experiments_trl/open_r1_dpo.py    # DPO với negative traces
cat enhance_cot/redi/experiments_trl/open_r1_sft.py    # SFT reference
# rllm/ — core REDI algorithm
ls enhance_cot/redi/rllm/
```

**Hypothesis:** Collect greedy outputs cho 1,167 rule_unknown problems (wrong answers) → REDI objective:
push logprob DOWN cho wrong-answer tokens → negative training signal.

**Thời gian ước tính:** 1h collect negatives + 2h REDI loss implement + 4h training = **~7h total**

**Dependency quan trọng:** Chạy exp18 (batch-2, offline preference) trước để verify DPO/preference
learning hoạt động trên Nemotron-H. Nếu exp18 fail → exp35 likely fail too.

### Bước 1 — Collect negative traces

```bash
cd nemotron-master
# Chạy greedy inference trên rule_unknown problems
uv run python3 infer_slice.py \
  --base <base> --adapter <0.86> \
  --slice <rule_unknown_problems.jsonl> \   # tạo từ corpus entries với status != rule_found
  --out negatives_exp35.jsonl
```

### Bước 2 — REDI loss trong loss_config.py

Từ REDI paper (đọc `enhance_cot/redi/rllm/` để hiểu): REINFORCE-style loss đẩy logprob âm
của negative tokens. Adapt vào `loss_config.py` như một `REDILossConfig(LossConfig)`.

⚠️ **KHÔNG port TRL code trực tiếp.** Đọc algorithm từ `rllm/` rồi implement theo Tinker's
`LossFnType` interface. Verify `LossFnType.__args__` để xem Tinker hỗ trợ loss type nào.

### Falsification

```bash
# Sau Stage-2 REDI 50 steps, check eval_slice:
# Nếu gravity/cipher accuracy drops > 1 pp → overcorrection → abort
```

---

## 11. exp36 — LogicPuzzleRL offline DPO *(Idea 3, P2, T3)* — HIGH RISK

**Paper:** arXiv:2506.04821 — verify xem có official repo chưa:
```bash
# Tìm official repo:
# site:github.com LogicPuzzleRL cryptarithm reinforcement learning 2025
```

**Hypothesis:** vLLM temperature=0.8, n=20 trên cryptarithm/equation_guess → DPO pairs →
DPO 150 steps → offline analog of LogicPuzzleRL.

**Thời gian ước tính:** 2h vLLM sampling + 1h pair construction + 3h DPO = **~6h total**
⚠️ High engineering effort — cần vLLM sampling infrastructure chưa có.

**Prerequisite:**
- exp29 DPO infrastructure đã hoạt động (verify Tinker hoặc TRL path)
- Pass@20 cryptarithm ≥ 2% (falsification test từ batch-4 idea 3)

### Quy trình skeleton

```bash
# 1. vLLM sampling n=20 trên cryptarithm/equation_guess (~873 problems)
# 2. build_dpo_pairs.py (extend từ exp29) với threshold ≥1 correct + ≥1 incorrect
# 3. DPO training với SFT anchor 20%
# Abort nếu < 20 problems có valid pairs
```

---

## 12. exp37 — GeoRA geometry-aware LoRA init *(Idea 11, P3, T2)* — HIGH RISK

**Paper:** arXiv:2601.09361 — NO official repo. Implement từ paper.

**Feasibility check bắt buộc trước khi code:**
```bash
# 1. Check memory budget cho FIM computation:
python3 -c "
import torch
# Nemotron-3-Nano-30B-A3B: ~30B params, rank=32
# FIM per layer = (d_model)^2 outer product → estimate memory
# If d_model=4096: 4096^2 * 4 bytes = 67MB per layer * N_layers
print('Estimated FIM memory (rough):', 67 * 80 / 1024, 'GB')  # ~5.2 GB if 80 layers
# Manageable nếu compute per-layer và không giữ toàn bộ
"
# 2. Verify Nemotron-H layer names để biết có thể apply GeoRA hay không:
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', trust_remote_code=True)
print(type(cfg))
"
```

⚠️ **Architecture risk:** GeoRA được design cho Transformer. Nemotron-H có SSM layers — FIM cho
selective scan matrices không có precedent. Chỉ apply GeoRA cho FFN + lm_head layers nếu SSM
geometry không interpretable.

**Thời gian ước tính:** 2h feasibility check + 3h implement (nếu feasible) + 4h training = **~9h total**
Cao nhất trong batch-4. Chỉ chạy sau khi tất cả ideas khác đã được thử.

---

## 13. exp38 — RL→SFT ordering (DPO first) *(Idea 12, P1, T2)* — HIGH RISK

**Paper:** arXiv:2509.21128v2 — NO official repo.

**Prerequisite cứng:**
1. exp18 (batch-2) đã chạy và confirm offline DPO hoạt động trên Nemotron-H
2. exp29 DPO pairs đã available

**Hypothesis:** DPO 50 steps (squeeze) → SFT full corpus (expand) = tốt hơn SFT-only.

**Thời gian ước tính:** 2h DPO stage + 4h SFT stage = **~6h total** (2× GPU time so với baseline)

**Falsification cứng:**
```bash
# Sau DPO 50 steps: eval_slice accuracy ≥ 0.85 (không catastrophic forget)
# Nếu < 0.85 → increase SFT-mix trong DPO stage (20–30%) rồi retry
# Nếu sau full SFT không exceed 0.86 → ordering hypothesis fails
```

---

## 14. Verify code — bắt buộc trước khi chốt mỗi exp

```bash
cd nemotron-master
uv run --frozen ruff format *.py
uv run --frozen ruff check *.py
uv run --frozen mypy *.py
uv run pytest .claude/          # nếu có test cho augmenter/solver mới
```

---

## 15. Thời gian tổng ước tính

| exp | Idea | Upstream | Training | Total | Priority |
|-----|------|----------|----------|-------|----------|
| exp27 | bit_manipulation solver | 2h | 4h | **6h** | 🔴 NOW |
| exp28 | drop augmentation | 1.5h | 4h | **5.5h** | 🔴 NOW |
| exp29 | DPO eval_slice | 2h | 2h | **4h** | 🟡 Week 1 |
| exp30 | OXA | 3h | 4h | **7h** | 🟡 Week 1 |
| exp31 | procedural crypto | 4h | 4h | **8h** | 🟡 Week 1-2 |
| exp32 | AdaSTaR | 3h | 5h | **8h** | 🟢 Week 2 |
| exp33 | VCORE | 2h | 4h | **6h** | 🟢 Week 2 |
| exp34 | paraphrase traces | 3h | 4h | **7h** | 🟢 Week 2 |
| exp35 | REDI | 3h | 4h | **7h** | 🟢 Week 2-3 |
| exp36 | LogicPuzzleRL DPO | 4h | 3h | **7h** | ⚪ Later |
| exp37 | GeoRA | 3h | 4h | **7h** | ⚪ Later |
| exp38 | RL→SFT ordering | 1h | 6h | **7h** | ⚪ Later |

**Total nếu chạy tuần tự tất cả:** ~83h GPU (~3.5 ngày). Khuyến nghị: chạy exp27+28 trước,
evaluate, rồi quyết định tiếp.

---

## 16. Dependency graph

```
exp27 (solver ext)  ──────────────────────────────────────────────────┐
exp28 (drop aug)    ──────────────────────────────────────────────────┤
exp18 (batch-2 DPO, chưa run) ──────► exp29 (DPO pairs) ────────────►│ submit
                                           │                          │
                                           └──► exp38 (RL→SFT order) │
exp30 (OXA)  ─────────────────────────────────────────────────────────┤
exp31 (procedural crypto)  ────────────────────────────────────────────┤
exp32 (AdaSTaR)  ──────────────────────────────────────────────────────┤
exp33 (VCORE) ─────────────────────────────────────────────────────────┤
exp34 (paraphrase) ────────────────────────────────────────────────────┤
exp35 (REDI) ──── requires exp18 first ────────────────────────────────┤
exp36 (DPO pairs, XL) ─── requires exp29 + pass@20 ≥ 2% ──────────────┤
exp37 (GeoRA, after all) ──────────────────────────────────────────────┘
```

---

## Provenance

Plan written: 2026-06-05. Official repos cloned into `enhance_cot/`:
- `redi/`: github.com/Tim-Siu/reinforcement-distillation (TRL-based, read for algorithm)
- `adastar/`: github.com/reiss-koh/AdaSTaR (JAX/TPU, read `utils_adastar.py` for sampling logic)
- `vcore/`: github.com/coder-gx/VCORE (LLaMA Factory-based, read for weighting formula)
- GeoRA, OXA, RL-SFT: no official repo — implement từ paper.
