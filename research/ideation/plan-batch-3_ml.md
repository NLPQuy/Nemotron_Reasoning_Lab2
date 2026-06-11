# Implementation Plan — Batch 3 (exp20…exp26) — DATA AUGMENTATION only

Mục tiêu: hiện thực hóa 7 ý tưởng trong [batch-3.md](batch-3.md) để đẩy **0.86 → 0.88+** **chỉ bằng data-time augmentation** (KHÔNG đụng inference/decoding). Khác batch-1/2: **phần chính của hầu hết idea nằm UPSTREAM** trong `nemotron-master/` (`reasoners/`, `augmenters/`, `corpus.py`, `reasoning.py`) — sinh/lọc data rồi **regenerate corpus**; file `exp<N>.py` ở repo gốc chỉ là **copy nguyên xi** của [Continuer_Nemotron_Notebook.py](../../Continuer_Nemotron_Notebook.py) và **trỏ `CORPUS_PATH`/`KAGGLE_DATASET` sang snapshot mới** (đúng cách exp13/15/19 đã làm).

> exp20 ↔ Idea 1, …, exp26 ↔ Idea 7 theo đúng thứ tự ranked trong batch-3.md.
>
> ⚠️ **Nguyên tắc viết code (theo yêu cầu):** với mọi đoạn đụng tới schema/solver, **đọc và copy code chính chủ** trong `nemotron-master/` — KHÔNG tự suy ra. Các chỗ chưa chắc (constructor `Problem`, math từng solver) thì **gọi lại hàm gốc** (`reasoning_<cat>`, `extract_answer`, `compare_answer`, `Problem.from_payload`) thay vì viết lại. Mục [§Verify](#12-verify-code--bắt-buộc-trước-khi-chốt-mỗi-exp) là cổng cứng trước khi coi exp là xong.
>
> 🧭 **Bài học round-2 (đỉnh nhọn):** mọi thứ *viết lại nội dung/độ dài trace* (exp13 0.68, exp19 0.79) hoặc *thêm nhiễu* (exp12 0.83) đều hồi quy. Batch-3 chỉ **THÊM dữ liệu đúng-phân-phối/đúng-format** và bọc bằng **cổng verify+length-cap (exp21)**. exp21 là **dependency**, làm trước. exp26 (self-trace) là rủi-ro-cao, làm cuối.

---

## 0. Setup chung (làm 1 lần)

```bash
cd /media/mlinh/DATA/projects/ML/Nemotron_Reasoning_Lab2
for i in $(seq 20 26); do cp Continuer_Nemotron_Notebook.py exp${i}.py; done
```

**Quy ước cho mọi exp file (giống batch-1/2):**
1. Sau dòng 2 (`# # Nemotron finetuning pipeline`) thêm banner:
   ```python
   # ============================================================
   # EXP<N> — <title>  (Batch-3 Idea <K>, DATA-AUG)
   # Base: Continuer_Nemotron_Notebook.py (unmodified except marked blocks)
   # Upstream: <reasoners/augmenters/corpus.py changes>  |  Corpus: <new snapshot>
   # Change: <1 dòng>  |  Knob: <KNOB>=<value>  |  Rollback: <KNOB> mặc định + corpus gốc
   # ============================================================
   ```
2. Mỗi vùng sửa bọc `# >>> EXP<N> START` … `# <<< EXP<N> END`.
3. Knob mới đặt trong block config **dòng 6–38**.
4. Chỉ đổi **một** cơ chế / file.

**Codepath đã verify (bản hiện tại):**

Trainer gốc `Continuer_Nemotron_Notebook.py`:
| Vùng | Dòng | Vai trò |
|------|------|---------|
| Config knobs | 6–38 | `LORA_RANK`, `NUM_STEPS`, `KAGGLE_DATASET` (25), `TARGET_MODULES`, `ORIGINAL_PROBLEMS_ONLY` (20) |
| `CORPUS_PATH` Kaggle | 109 | dir `tokens/<pid>/synthetic.json` `{tokens,mask}` |
| `TRAIN_ORDER_PATH` Kaggle | 110 | thứ tự train từ `logprobs/index.jsonl` (epoch 0) |
| `CORPUS_PATH` Modal | 125 | `/data/corpus_preprocessed.jsonl` (1 record/line `{problem_id,tokens,mask}`) |
| Load corpus Kaggle | 186–209 | đọc `synthetic.json`, build `examples=[{problem_id,tokens,targets,weights}]` |
| Load corpus Modal | 211–228 | tương tự từ jsonl |
| `ORIGINAL_PROBLEMS_ONLY` filter | 230–240 | mẫu lọc theo `train.csv` |

Upstream `nemotron-master/`:
| File | Hàm/Dòng | Vai trò |
|------|----------|---------|
| `reasoning.py` | `GENERATORS` 45–55 | category → solver `reasoning_<cat>(problem)` |
| `reasoning.py` | `extract_answer` 58–66 | lấy `\boxed{...}` cuối |
| `reasoning.py` | `compare_answer` 69–103 | **verifier**: binary exact / float rel-tol 1e-2 / else string ci |
| `reasoning.py` | loop 164–195 | `reasoning_text=generator(problem)`; `result=compare_answer(problem.answer, submission)`; **ghi file kể cả khi result=False** (status mới đổi) |
| `corpus.py` | reasoning branch 176–227 | include **mọi pid có `reasoning/<pid>.txt`**; boxed lấy từ **trace** (183–184) — KHÔNG check lại `answers[pid]` |
| `corpus.py` | aug branch 229–277 | `completion_text=f"{completion}\n</think><|im_end|>"` (240, **no boxed**), prompt `suffix=""` (245) |
| `corpus.py` | `TOKEN_LIMIT` | 43 = 8192 |
| `augmenters/spelling.py` | `generate()` 50–115 | mẫu augmenter: trả `list[{id,prompt,completion,category}]` |
| `augmentation.py` | `main` | gọi từng `augmenter.generate()`, ghi `[category]/[prompt]/[completion]` |
| `reasoners/store_types.py` | `Problem`/`Example` 20–86 | schema; `Problem.from_payload` 59–76, `to_payload` 78–86 |

**Data-flow chuẩn (mọi idea data-aug):**
```
edit upstream (reasoners/ | augmenters/ | corpus.py)
  → uv run python3 reasoning.py        # solver verify, ghi reasoning/<pid>.txt
  → uv run python3 corpus.py           # tokenize+mask → corpus/<pid>/synthetic.jsonl
  → BUILD SNAPSHOT: chuyển sang layout trainer đọc (tokens/<pid>/synthetic.json {tokens,mask})
  → upload Kaggle dataset / ghi Modal volume
  → exp<N>.py: trỏ KAGGLE_DATASET / CORPUS_PATH sang snapshot mới (chỉ đổi path)
```
⚠️ **Gap format đã verify:** `corpus.py` xuất `synthetic.jsonl` (segments), trong khi trainer đọc `synthetic.json` `{tokens,mask}` (Kaggle) / `corpus_preprocessed.jsonl` (Modal). **Bước BUILD SNAPSHOT là bắt buộc** (dùng lại pipeline đóng-gói của snapshot `04-08-16-14` hiện có; KHÔNG bỏ qua). Đây là điểm dễ vỡ nhất — xem [§Verify](#12-verify-code--bắt-buộc-trước-khi-chốt-mỗi-exp).

---

## exp20 — Procedural in-distribution instance scaling  *(Idea 1, P4, T1)* — 🏆 Big bet, UPSTREAM

**Hypothesis:** solver trong `reasoners/` **tự verify** (reasoning.py dòng 178), nên sinh thêm bài cùng category/format rồi để solver gán nhãn là **free + 0 nhãn sai + phân phối giữ nguyên** → thêm gradient đúng cho category under-fit. Chỉ scale 1–2 category yếu nhất (đo bằng slice).

**Cơ chế an-toàn-tuyệt-đối (quan trọng):** KHÔNG tự tính đáp án. Sinh `Problem` ứng viên (examples + question), **chạy chính `reasoning_<cat>(problem)`** để lấy trace, rồi lấy `extract_answer(trace)` làm `answer` ground-truth. Như vậy `compare_answer(problem.answer, submission)` **luôn `rule_found`** by-construction → zero label noise. Bài nào solver trả `None` thì bỏ.

**Edit chính (upstream, `nemotron-master/`):** thêm `generators/<cat>.py` (tách khỏi `reasoners/` để không nhầm với solver) — ví dụ khung cho `gravity` (đọc `reasoners/gravity.py` 14–87 + `store_types.py` để khớp field; KHÔNG đoán math):
```python
# nemotron-master/generators/gravity_gen.py   (NEW)
import json, random, hashlib
from pathlib import Path
from reasoners.store_types import Problem, Example
from reasoners.gravity import reasoning_gravity
from reasoning import extract_answer            # tái dùng, không viết lại

def gen_gravity(n: int, seed: int = 20) -> list[Problem]:
    rng = random.Random(seed)
    out: list[Problem] = []
    while len(out) < n:
        # sample tham số TRONG support quan sát từ problems.jsonl/train.csv (đo trước!)
        k = round(rng.uniform(K_LO, K_HI), 3)            # ranges = đo từ data thật
        ts = sorted(rng.sample(T_POOL, k=N_EXAMPLES))
        examples = [Example(str(t), str(round(k * t * t, 3))) for t in ts]
        q = str(rng.choice([t for t in T_POOL if str(t) not in {e.input_value for e in examples}]))
        pid = "gen_" + hashlib.sha256(f"gravity_{seed}_{len(out)}".encode()).hexdigest()[:8]
        prob = Problem(id=pid, category="gravity", examples=examples, question=q, answer="")
        trace = reasoning_gravity(prob)               # SOLVER = nguồn chân lý
        if trace is None:
            continue
        prob.answer = extract_answer(trace)           # gán nhãn theo solver
        out.append(prob)
    return out
```
Rồi **append** các `Problem` này vào `problems.jsonl` + `train.csv` (prompt = chuỗi đề bài đúng template category; nếu chưa chắc cách build prompt, **đọc cách `train.csv` hiện có render prompt cho category đó** và sao y) → chạy `reasoning.py` + `corpus.py` + BUILD SNAPSHOT.

> Đo support trước (bắt buộc): histogram tham số/độ dài mỗi category từ `problems.jsonl`+`train.csv`. Generator phải nằm trong support này (tránh OOD — xem exp25).

**Edit trong `exp20.py`:** chỉ banner + trỏ `KAGGLE_DATASET`/`CORPUS_PATH` sang snapshot đã scale. Không sửa logic train. (Tùy chọn chỉnh `NUM_STEPS` cho hợp số ví dụ mới.)

**Validate rẻ:** thêm 1.5–2× instance cho 1–2 category yếu, train cùng step/seed, chấm slice 200 bài/category. **Rollback:** corpus gốc; xóa các pid `gen_*`.

**Falsification:** macro exact-match **+≥0.3pp** và **không category nào −>1pp** và cap-hit-rate 7680 không tăng. Δ<0.3pp ⇒ category đã bão hòa, ngừng scale.

---

## exp21 — Solver-as-free-verifier gate + hard length cap  *(Idea 2, P6, T1)* — ⚡ Quick win, **DEPENDENCY**

**Hypothesis:** `corpus.py` hiện include **mọi** pid có `reasoning/<pid>.txt` và lấy boxed **từ trace** (dòng 183–184) mà **không** so lại với `answers[pid]` (train.csv). Trace `rule_unknown` (boxed ≠ stored) vẫn lọt vào corpus = nhãn sai. Thêm cổng: (a) chỉ nhận khi `compare_answer(stored, trace_boxed)` đúng; (b) drop trace dài quá `mean+σ` của category (chống truncate 7680). Cổng này làm exp20/22/24/25/26 an toàn.

**Edit chính (upstream, `nemotron-master/corpus.py`)** — chèn vào reasoning branch, ngay sau dòng 184 (`reasoning_answer = ...`), copy-khớp biến hiện có:
```python
# >>> EXP21 START  (corpus.py, trong vòng for problem_id, sau dòng 184)
from reasoning import compare_answer            # tái dùng verifier gốc
if VERIFY_GATE and not compare_answer(answer, reasoning_answer):
    dropped_wrong += 1
    continue                                    # bỏ trace boxed != stored answer
# <<< EXP21 END
```
Length-cap (sau khi có `completion_ids`, trước `entry = CorpusEntry(...)` dòng 204) — cap theo **category** đo từ baseline:
```python
# >>> EXP21 START
if LENGTH_GATE and len(completion_ids) > CAP_BY_CATEGORY.get(category, TOKEN_LIMIT):
    dropped_long += 1
    continue
# <<< EXP21 END
```
`CAP_BY_CATEGORY` = `{cat: int(mean+1σ)}` tính 1 lần từ corpus baseline (script đo riêng; in ra để review). Khởi tạo `dropped_wrong = dropped_long = 0` đầu `main()`, in cuối.

**Edit trong `exp21.py`:** trỏ corpus đã-gate; không sửa train.

**Validate rẻ:** gate trên corpus mặc định, train cùng step/seed. **Rollback:** `VERIFY_GATE=LENGTH_GATE=False` + corpus gốc.

**Falsification:** cap-hit-rate@7680 **giảm** và macro **≥ baseline**; **fail** nếu >5% ví dụ của bất kỳ category bị drop (cap quá chặt → nới σ).

---

## exp22 — Expand masked auxiliary string-skill augmenters  *(Idea 3, P5, T2)* — 🛡️ Safe bet, UPSTREAM

**Hypothesis:** cipher/cryptarithm/numeral cần primitive char-level (reverse, count, index, extract). Augmenter masked + **no `\boxed{}`** (corpus.py 240) **không thể** dịch phân phối boxed-reasoning — chính tính chất giữ baseline an toàn ở 0.86. Cap tổng aux ≤ 15–20% token.

**Edit chính (upstream, `nemotron-master/augmenters/`)** — mỗi module **mirror `spelling.py` `generate()`** (đọc 50–115; trả `list[{id,prompt,completion,category}]`, `id = sha256(...)[:8]`):
```python
# nemotron-master/augmenters/reverse.py   (NEW — khung theo spelling.py)
import hashlib, random
def generate() -> list[dict[str, str]]:
    rng = random.Random(42); problems = []
    for i in range(N_PROBLEMS):
        # mẫu prompt few-shot giống spelling.py (sample input → sample output → your input)
        ...
        pid = hashlib.sha256(f"reverse_{i}".encode()).hexdigest()[:8]
        problems.append({"id": pid, "prompt": prompt, "completion": answer, "category": "reverse"})
    return problems
```
Đăng ký trong `augmentation.py` (`from augmenters import ... reverse, count_substring, char_index, digit_extract` + `problems.extend(reverse.generate())` …). Chạy `augmentation.py` → `corpus.py` → SNAPSHOT.

**Edit trong `exp22.py`:** trỏ corpus mới; không sửa train.

**Validate rẻ:** đo aux token share ≤ 20% (in từ corpus.py stats 303–304); A/B slice. **Rollback:** bỏ đăng ký augmenter mới; corpus gốc.

**Falsification:** macro **+≥0.2pp** và cipher/cryptarithm/numeral **cùng** tăng; không category nào −>1pp.

---

## exp23 — Category-coverage + difficulty-stratified mixture weighting  *(Idea 4, P8, T1)* — cần slice trước

**Hypothesis:** corpus weights hiện chưa audit; reweight nghịch theo accuracy/category (đo trên slice) → tăng macro **nếu** mix slice ~ mix leaderboard. Rủi ro hidden-mix (medium).

**Edit (2 lựa chọn — ưu tiên upstream để giữ thứ tự train):**
1. **Upstream (khuyến nghị):** repeat/subsample pid theo `weight[cat]` khi build SNAPSHOT (nhân bản pid trong train-order). Giữ format y hệt.
2. **In-file (`exp23.py`)** nếu không build lại snapshot được — chèn **sau dòng 240** (sau khối `ORIGINAL_PROBLEMS_ONLY`), dùng `category` có sẵn trong corpus index (corpus.py ghi `category` ở `to_index_dict` 87–97 — cần load kèm, hoặc suy từ `problem_id` prefix):
   ```python
   # >>> EXP23 START
   if MIXTURE_WEIGHTS:                       # dict {category: float}
       import collections, random as _r
       _r.seed(0)
       rebalanced = []
       for e in examples:
           w = MIXTURE_WEIGHTS.get(_category_of(e["problem_id"]), 1.0)
           reps = int(w) + (1 if _r.random() < (w - int(w)) else 0)
           rebalanced += [e] * max(0, reps)
       examples = rebalanced
       print(f"EXP23 mixture: {len(examples)} examples after reweight")
   # <<< EXP23 END
   ```
   ⚠️ `_category_of()` cần map pid→category đáng tin; nếu corpus index không kèm category trong `examples`, **bổ sung `category` vào record khi load** (đọc corpus.jsonl) thay vì đoán prefix.

**Validate rẻ:** 2 vòng reweight→train→đo. **Rollback:** `MIXTURE_WEIGHTS=None`.

**Falsification:** macro **+≥0.3pp** và **min per-category accuracy không giảm** (cổng chống hidden-mix) và không category −>1pp. Thiếu slice theo category ⇒ **không chạy** (đoán mò).

---

## exp24 — Prompt paraphrase augmentation (masked prompt, fixed answer)  *(Idea 5, P11, T1)* — UPSTREAM

**Hypothesis:** prompt bị **mask=0** (corpus.py 194), nên paraphrase *đề bài* không dịch phân phối loss/trace → an toàn hơn nhiều so với paraphrase reasoning (đã hồi quy). Tăng robustness với cách hỏi lạ trên LB.

**Cơ chế:** với mỗi problem, sinh 1 paraphrase của **prompt** (template đổi từ đồng nghĩa/đảo mệnh đề trước; LLM offline tùy chọn). **Bắt buộc re-verify**: chạy lại solver trên problem paraphrased (hoặc giữ nguyên examples/answer, chỉ đổi văn phong phần mô tả) và xác nhận `answer` không đổi → loại paraphrase làm lệch. Completion = **trace gốc** (không đổi). Thêm như pid mới `para_*` với cùng `answer`, `prompt` mới.

**Edit chính (upstream):** script `paraphrase_prompts.py` (NEW) đọc `train.csv`, sinh `prompt'`, ghi pid mới vào `train.csv`+`problems.jsonl` (cùng `answer`, cùng category), **tái dùng `reasoning_<cat>`** để emit trace cho pid mới (đảm bảo trace khớp đề mới). Đi qua `reasoning.py`(verify) → `corpus.py` → SNAPSHOT. Bọc bằng cổng **exp21**.

**Edit trong `exp24.py`:** trỏ corpus mới; không sửa train.

**Validate rẻ:** thêm ≤1 paraphrase/bài cho 1–2 category; chấm slice **đã perturb cách hỏi**. **Rollback:** corpus gốc; xóa pid `para_*`.

**Falsification:** macro trên slice-perturbed **+≥0.3pp** và slice chuẩn không category −>1pp. (Không có solver re-verify ⇒ thành label-noise — cấm bỏ bước này.)

---

## exp25 — Domain randomization of solver-invariant surface features  *(Idea 6, P2, T3)* — UPSTREAM

**Hypothesis:** randomize **chỉ feature solver bất biến** (đổi tên biến/từ cryptarithm từ `wonderland.txt`/`dictionary.txt`, hoán vị mệnh đề độc lập, đổi cách trình bày số TRONG rel-tol 1e-2 — **không** cho binary) → ép học **luật** thay vì bề mặt; phân phối đáp án/cấu trúc trace giữ nguyên. Nguyên lý sim-to-real domain randomization.

**Cơ chế:** trong generator (exp20) hoặc trên problems gốc, jitter surface knob, **chạy lại solver `reasoning_<cat>`** và yêu cầu `compare_answer(old_answer, new_boxed)` đúng (đáp án bất biến). Knob nào làm đổi đáp án ⇒ knob đó KHÔNG bất biến ⇒ loại. Đi qua cổng exp21.

**Edit chính (upstream):** thêm `surface_randomize.py` (NEW) áp per-category knob; với cryptarithm rename token đọc từ `reasoners/wonderland.txt`/`dictionary.txt` (đọc cách solver dùng 2 file này trước khi đổi). `reasoning.py`(verify) → `corpus.py` → SNAPSHOT.

**Edit trong `exp25.py`:** trỏ corpus mới.

**Validate rẻ:** thêm biến thể cho 1–2 category; chấm slice **surface-perturbed**. **Rollback:** corpus gốc.

**Falsification:** macro trên slice-perturbed **+≥0.3pp**, slice chuẩn không regress, không category −>1pp. (Contrast: AbstRaL cảnh báo surface-var dưới SFT có thể yếu hơn abstraction → giữ liều vừa.)

---

## exp26 — STaR-offline self-traces, solver-filtered + length-capped  *(Idea 7, P12, T1)* — ⚠️ RỦI RO CAO, làm cuối

**Hypothesis:** với bài đang sai, để **adapter hiện tại** sinh K trace (sampling **chỉ offline**), **solver lọc** giữ trace đúng + ngắn (< cap), thêm vào corpus. Verifier là solver tất định ⇒ trace nhận luôn đúng. **NHƯNG** round-2: trace do model viết dịch phân phối khỏi đỉnh nhọn (exp13/19 âm) → đây là idea rủi ro nhất, **gate cứng bằng exp21 + category-gate + liều ≤10%**.

**Cơ chế (2-phase, như exp7 batch-1):**
1. **Phase A (offline):** `star_generate.py` (NEW) dùng vLLM/`model.generate` (temp>0 **chỉ khi sinh**) trên tập bài-sai (từ slice). Mỗi output → `extract_answer` → `compare_answer(stored, boxed)`; giữ trace đúng **và** `len(completion_ids) ≤ CAP_BY_CATEGORY` (exp21); dedupe; cân bằng category. Ghi `reasoning/<pid>.txt` (hoặc star jsonl).
2. **Phase B:** `corpus.py` (qua cổng exp21) → SNAPSHOT; `exp26.py` trỏ corpus mới. Liều self-trace ≤ 10% corpus.

**Edit trong `exp26.py`:** trỏ corpus mới; không sửa train.

**Validate rẻ:** thêm ≤10% self-trace cho 1 category yếu; A/B slice với guard nghiêm. **Rollback:** corpus gốc; xóa self-trace. **Bỏ ngay khi có bất kỳ regression.**

**Falsification:** macro **+≥0.3pp** **và** mean completion length **không tăng** **và** cap-hit-rate@7680 không tăng **và** không category −>1pp.

---

## 11. Thứ tự chạy đề xuất

> Theo "Next steps" của batch-3.md — dựng slice trước, bank an-toàn trước, big-bet/rủi-ro sau.

0. **Prerequisite (bắt buộc, không phải idea):** dựng **slice held-out theo category (~200 bài, vLLM greedy)** + bucket lỗi `{format, truncation, arithmetic-slip, method-wrong}`. Round-2 âm 5 lần vì thiếu nó. Slice này là input của *mọi* falsification dưới đây.

| Đợt | Chạy | Lý do |
|-----|------|-------|
| 1 (nền) | **exp21** (verify+length gate) | dependency: làm exp20/22/24/25/26 an toàn; rẻ nhất |
| 2 (an toàn, đòn bẩy) | **exp20** (scale instance) → **exp22** (aux augmenters ≤20%) | đúng-phân-phối/masked; rủi ro thấp nhất |
| 3 (có kiểm soát) | exp24 (paraphrase đề) → exp25 (surface DR) → exp23 (mixture, cần slice category) | đều qua cổng exp21 + solver re-verify |
| 4 (rủi ro cao) | exp26 (STaR self-trace) | chỉ sau khi đã bank wave 1–3; liều ≤10%, bỏ khi regress |

**Combo cuối:** gộp các thay đổi *độc lập* dương tính vào `exp_combo3.py`. Compose an toàn: exp20 (thêm instance) ⊕ exp22 (aux masked) ⊕ exp24/25 (paraphrase/surface) — đều qua **một** snapshot đã gate exp21. exp23 (mixture) áp **sau** khi đã chốt tập data. exp26 **không** trộn vào combo cho tới khi tự nó dương.

**Kỷ luật đo:** mọi A/B trên **cùng slice cố định, cùng seed, cùng `NUM_STEPS`**; chỉ giữ nếu macro **+≥0.3pp** (exp22: +≥0.2pp) **và** không category −>1pp **và** cap-hit-rate@7680 không tăng. Strategy E (đa-form boxed) **loại** — `compare_answer` (69–103) cho thấy `"1/2"`≠`"0.5"` và binary zero-tolerance.

---

## 12. Verify code — BẮT BUỘC trước khi chốt mỗi exp

> Yêu cầu người dùng: *thêm phần verify code*. Đây là cổng cứng; không exp nào "xong" nếu chưa qua hết mục áp dụng. Codex phải chạy và dán output, không tự khẳng định pass.

### 12.1 Static — mọi file upstream đã sửa (`nemotron-master/`)
```bash
cd nemotron-master
uv run --frozen ruff format reasoners/*.py augmenters/*.py corpus.py reasoning.py generators/*.py
uv run --frozen ruff check  reasoners/*.py augmenters/*.py corpus.py reasoning.py generators/*.py --fix
uv run --frozen mypy        corpus.py reasoning.py reasoners/*.py augmenters/*.py
uv run pytest                                   # tests nằm dưới .claude/
```
(exp file ở repo gốc **không** có uv project — chỉ kiểm tra `python -c "import ast; ast.parse(open('exp<N>.py').read())"` + `python -m py_compile exp<N>.py`.)

### 12.2 Verifier không đổi (chống tự-phá grader)
```bash
uv run python -m doctest reasoning.py -v       # compare_answer docstrings 75–85 phải pass
```
Khẳng định: KHÔNG sửa `compare_answer`/`extract_answer`. Nếu một idea cần đổi → DỪNG, đó là đổi grader-path (cấm trong batch này).

### 12.3 Dry-run pipeline trên N nhỏ trước khi full
```bash
# sinh ít (vd N=20/category), chạy thật pipeline, đọc stats
uv run python3 reasoning.py        # xem accuracy/category 226–245: pid gen_/para_ phải rule_found ~100%
uv run python3 corpus.py           # xem stats 298–304: #entries, max-seq, unmasked/category
```
Kiểm tra bằng mắt 2–3 file `reasoning/<gen_*>.txt`: kết thúc `\boxed{...}`, đáp án khớp `train.csv`.

### 12.4 Invariant theo từng idea
- **exp20:** với mọi pid `gen_*`: `compare_answer(train.csv.answer, extract_answer(trace)) == True` (in % rule_found cho `gen_*`, phải 100%). Tham số generator nằm trong support đã đo (in min/max vs baseline).
- **exp21:** in `dropped_wrong`, `dropped_long`/category; assert ≤5% mỗi category; in `CAP_BY_CATEGORY` để review. Trên corpus mặc định `dropped_wrong` nhỏ (đo độ "rò" hiện tại).
- **exp22:** in aug token share (từ corpus.py stats) — assert ≤20%; xác nhận entry aug **không** chứa `\boxed` (`assert "\\boxed" not in completion_text`).
- **exp23:** in phân phối category trước/sau reweight; assert mọi category vẫn > 0 ví dụ.
- **exp24/exp25:** với mọi pid `para_*`/`rand_*`: solver re-verify đáp án **bất biến** (in #loại do đổi đáp án); cho exp25 **assert không đụng** answer binary (`re.fullmatch(r"[01]+", answer)` → skip number-format jitter).
- **exp26:** in #self-trace giữ lại, mean/median completion length vs baseline category; assert ≤10% corpus; assert mọi trace ≤ `CAP_BY_CATEGORY`.

### 12.5 Snapshot ↔ trainer khớp format (điểm dễ vỡ nhất)
Sau BUILD SNAPSHOT, trước full run:
```bash
# Kaggle layout: phải có tokens/<pid>/synthetic.json với keys {tokens, mask}
python - <<'PY'
import json, glob, random
fs = glob.glob("<SNAPSHOT>/tokens/*/synthetic.json")
assert fs, "no synthetic.json — BUILD SNAPSHOT chưa chạy / sai layout"
r = json.load(open(random.choice(fs)))
assert set(r) >= {"tokens","mask"} and len(r["tokens"])==len(r["mask"])
assert any(r["mask"]) and len(r["tokens"])<=8192
print("snapshot OK:", len(fs), "problems")
PY
```
Đối chiếu `TRAIN_ORDER_PATH` (index.jsonl epoch-0) chứa đúng các pid mới; pid trong order mà thiếu file → trainer assert chết (dòng 188–190).

### 12.6 Smoke train + deliverable
1. **Smoke:** `NUM_STEPS=5` trên snapshot mới — chạy hết, loss hữu hạn, không OOM, adapter lưu ra.
2. **vLLM load-test** adapter (cổng cứng như batch-2): load + vài greedy gen, output có `\boxed{...}`, `adapter_config.json` tồn tại, rank ≤ 32.
3. **A/B slice** đúng kỷ luật §11; chỉ chốt nếu đạt Falsification của idea.

### 12.7 Checklist cuối mỗi exp
- [ ] §12.1 static sạch (ruff/mypy/pytest)  ·  [ ] §12.2 doctest compare_answer pass, verifier KHÔNG sửa
- [ ] §12.3 dry-run stats hợp lý  ·  [ ] §12.4 invariant idea pass  ·  [ ] §12.5 snapshot khớp format
- [ ] §12.6 smoke train + vLLM load OK  ·  [ ] A/B đạt ngưỡng falsification  ·  [ ] banner + `# >>> EXP<N>` markers + rollback ghi rõ
