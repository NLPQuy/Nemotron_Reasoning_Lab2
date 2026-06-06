# Implementation guide — `offline/` generators (batch-4)

> **Hướng dẫn code** (không phải code sẵn) để build stack sinh-data offline trong [plan-batch-4.md §13](plan-batch-4.md) / [batch-4.md](batch-4.md).
> Viết cho 1 agent code theo: mỗi file có **mục tiêu / nguồn port (line ref) / hàm phải tạo (signature) / thuật toán từng bước / schema I/O / lỗi-hay-mắc / validate / điều CẤM**.
> **Contract tổng:** RunPod/Modal GPU → `corpus_preprocessed.jsonl` (+ `rollouts.jsonl`, `pairs.jsonl`) → Continuer SFT **continue-từ-0.86**.
> Ràng buộc bất biến: off-policy, LoRA r32, verifier luật `compare_answer`, 7680 tok, greedy 1-pass.

---

## 0. Sự thật repo phải nắm trước khi code (đã đọc, có line ref)

| Cần biết | Ở đâu | Nội dung |
|---|---|---|
| **Token format (1 nguồn duy nhất)** | [corpus.py:185-194](../../nemotron-master/corpus.py#L185-L194) | completion = `f"{reasoning}\n</think>\n\\boxed{{{answer}}}<\|im_end\|>"`; prompt qua chat-template `enable_thinking=True` → mask `0`; completion → mask `1`. **Answer trong `\boxed{}` lấy từ regex của reasoning, KHÔNG phải answer gốc** ([corpus.py:183-184](../../nemotron-master/corpus.py#L183-L184)). |
| **2 tokenizer khác nhau** | [corpus.py:142-145](../../nemotron-master/corpus.py#L142-L145) | completion dùng `Tokenizer.from_file("tokenizer.json")` + `.encode(text, add_special_tokens=False).ids`; prompt dùng `AutoTokenizer.apply_chat_template(...)`. **Không trộn 2 cái.** |
| **PROMPT_SUFFIX** | [corpus.py:38-41](../../nemotron-master/corpus.py#L38-L41) = [generate_rollouts_vllm.py:33-36](../../generate_rollouts_vllm.py#L33-L36) | phải khớp grader, copy nguyên văn. |
| **verify helpers** | [generate_rollouts_vllm.py:39-65](../../generate_rollouts_vllm.py#L39-L65) | `extract_answer`/`compare_answer`/`format_ok` — copy nguyên, KHÔNG sửa logic. |
| **vLLM G=8 sampler** | [generate_rollouts_vllm.py:180-251](../../generate_rollouts_vllm.py#L180-L251) | đã có reward + `old_logp`. **Drop pure-group ở [L242-243](../../generate_rollouts_vllm.py#L242-L243) — phải XOÁ.** |
| **Loss loop ăn `weights`/token** | [Continuer:599-605](../../Continuer_Nemotron_Notebook.py#L599-L605) | `weighted_loss = per_token_ce * padded_weights; loss = loss_sum/weight_sum`. Điểm chèn `weight`/`sign`. |
| **Modal-worker đọc corpus** | [Continuer:210-228](../../Continuer_Nemotron_Notebook.py#L210-L228) | đọc jsonl, mỗi dòng `{problem_id, tokens, mask}` → `weights=[float(m) for m in mask[1:]]`. **Generator output khớp đây.** |
| **Kaggle đọc corpus (KHÁC)** | [Continuer:165-209](../../Continuer_Nemotron_Notebook.py#L165-L209) | per-id `<sid>/synthetic.json` theo `index.jsonl`. Cần `upload_kaggle.py` repack (§6). |
| **Continue-0.86 knobs** | [Continuer:11-23](../../Continuer_Nemotron_Notebook.py#L11-L23), [113-128](../../Continuer_Nemotron_Notebook.py#L113-L128) | `RESET_WEIGHTS=False`, `NUM_STEPS`/`LEARNING_RATE` đầu file. |
| **reasoners API** | [reasoning.py:42-56](../../nemotron-master/reasoning.py#L42-L56), [163-166](../../nemotron-master/reasoning.py#L163-L166) | `GENERATORS[category](problem) -> str \| None`; `Problem.load_from_json(id)` đọc `problems/<id>.jsonl`. |
| **Problem datatype** | [store_types.py:37-101](../../nemotron-master/reasoners/store_types.py#L37-L101) | `{id, category, examples:[{input_value,output_value}], question, answer, prompt}`. |

**Hệ quả thiết kế (quan trọng nhất):** generator chỉ cần emit JSONL `{problem_id, category, tokens, mask, weight, sign}` — **đúng** schema Modal-worker đang đọc + 2 field mới. Trainer chỉ sửa **4 dòng** (§2). Mọi sinh-token đi qua **1 hàm** `tokenize_format.make_example` để khỏi lệch format.

---

## 1. Milestones (code theo thứ tự rủi ro/chi phí tăng dần)

| M | File | GPU | Blocker? |
|---|---|:--:|:--:|
| **A** | `offline/common/*` + patch Continuer `weight`/`sign` | ✗ | **CÓ** (chặn tất cả) |
| **B** | `sample_rollouts.py` (1 lần, G=8 + ppl) | ✓ | nguồn cho C2 |
| **C1** | `measure_yield.py`, `compress_traces.py` (exp33), `evolve_solved.py` (exp37) | ✗ | làm song song M-B |
| **C2** | `build_rft.py`, `build_negatives.py`, `build_reward_weighted.py` | ✗ | sau M-B |
| **D** | `build_dpo_pairs`, `build_correction_traces`, `coldstart_inject`, `synth_weakness` | ✓/✗ | sau cùng |

**First PR = A + B + C1.** C2 + D là PR sau (dùng lại `rollouts.jsonl`).

---

## 2. Patch Continuer — corpus `weight`/`sign` (Milestone A, BLOCKER)

**Mục tiêu:** cho phép mỗi dòng corpus mang `weight` (scalar nhân loss) + `sign` (±1, học/đẩy-xa). Default phải **no-op bit-khớp** baseline.

**Schema dòng corpus (chuẩn mới):**
```jsonc
{"problem_id":"...", "category":"...", "tokens":[int...], "mask":[0/1...],
 "weight": 1.0,   // optional, default 1.0
 "sign":  1.0}    // optional, default +1.0; -1.0 = unlikelihood
```

**Sửa CHÍNH XÁC 2 chỗ, bọc marker `# >>> EXP_WEIGHT_SIGN START/END`:**

1. [Continuer:221-227](../../Continuer_Nemotron_Notebook.py#L221-L227) (Modal loader) — đổi cách build `weights`:
   - đọc `w = float(rec.get("weight",1.0)) * float(rec.get("sign",1.0))`
   - đổi `"weights": [float(m) for m in mask[1:]]` → `"weights": [w*float(m) for m in mask[1:]]`
   - **Cũng sửa nhánh Kaggle [Continuer:207](../../Continuer_Nemotron_Notebook.py#L207)** y hệt (đọc từ `rec` của synthetic.json) — nếu không, 2 path lệch nhau.

2. [Continuer:600-602](../../Continuer_Nemotron_Notebook.py#L600-L602) (loss) — mẫu số phải dùng `abs`:
   - `weight_sum_t = padded_weights.sum()` → `weight_sum_t = padded_weights.abs().sum()`
   - `weighted_loss` và `loss_sum_t` GIỮ NGUYÊN (đã mang dấu).

**Cơ chế:** `sign=-1` → `padded_weights` âm → `per_token_ce*âm` = gradient-ascent (unlikelihood) trên token sai. `.abs()` ở mẫu giữ scale loss khi trộn ±.

**Lỗi hay mắc:**
- ❌ Quên sửa nhánh Kaggle → train trên 2 máy ra khác nhau.
- ❌ Dùng `.sum()` (không abs) ở mẫu → 1 batch toàn negative ra mẫu âm → loss lật dấu loạn.
- ❌ Đổi forward CE — KHÔNG cần, CE đã `reduction="none"` ở [Continuer:386](../../Continuer_Nemotron_Notebook.py#L386).

**Validate (điều kiện merge M-A):** chạy loader trên corpus CŨ (không có 2 field) → `rec.get` trả default → `weights` ra **y hệt** trước patch. So 1 batch: `loss` bit-identical. Nếu lệch → sai.

**CẤM:** không thêm field nào khác ngoài `weight`/`sign`; không đổi thứ tự example; không đổi `MAX_SEQ_LEN` truncation.

---

## 3. `offline/common/` — module dùng chung (Milestone A)

> Mọi file dưới là **thư viện**, không có `__main__`. Đặt `offline/common/__init__.py` rỗng.

### 3.1 `verify.py`
- **Mục tiêu:** verify luật, dùng chung mọi generator.
- **Port:** copy **nguyên văn** [generate_rollouts_vllm.py:33-65](../../generate_rollouts_vllm.py#L33-L65) (`PROMPT_SUFFIX`, `extract_answer`, `compare_answer`, `format_ok`).
- **Hàm:** `extract_answer(text)->str`, `compare_answer(stored,pred)->bool`, `format_ok(text)->bool`, hằng `PROMPT_SUFFIX`.
- **Validate:** import `compare_answer` từ [reasoning.py:69](../../nemotron-master/reasoning.py#L69) làm oracle, assert khớp trên 20 cặp (gồm binary-string exact, numeric 1e-2, case-insensitive).
- **CẤM:** sửa regex/tolerance — phải khớp grader.

### 3.2 `tokenize_format.py` — **HÀM SINH TOKEN DUY NHẤT**
- **Mục tiêu:** mọi (prompt, reasoning, answer) → `{tokens, mask}` đúng format grader. Tất cả generator gọi đây, không ai tự ghép token.
- **Port:** logic [corpus.py:56-194](../../nemotron-master/corpus.py#L56-L194).
- **Hàm phải tạo:**
  - `load_tokenizers(model_path, tokenizer_json_path) -> (chat_tok, comp_tok)` — `chat_tok=AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)`, `comp_tok=Tokenizer.from_file(tokenizer_json_path)`.
  - `make_example(prompt_text, reasoning_text, answer, *, chat_tok, comp_tok, problem_id="", category="rollout", weight=1.0, sign=1.0) -> dict` (path TEXT → token).
  - `make_example_from_ids(prompt_ids, completion_ids, *, problem_id, category, weight=1.0, sign=1.0) -> dict` (path ROLLOUT — đã có token_ids, KHÔNG tokenize lại).
- **Thuật toán `make_example`:**
  1. `boxed = re.findall(r"\\boxed\{([^}]*)\}", reasoning_text)`; `ans = boxed[-1] if boxed else answer` (khớp [corpus.py:183-184](../../nemotron-master/corpus.py#L183-L184)).
  2. `completion = f"{reasoning_text.rstrip(chr(10))}\n</think>\n\\boxed{{{ans}}}<|im_end|>"`.
  3. `comp_ids = comp_tok.encode(completion, add_special_tokens=False).ids`.
  4. `prompt_ids = chat_tok.apply_chat_template([{"role":"user","content":prompt_text+PROMPT_SUFFIX}], tokenize=True, add_generation_prompt=True, enable_thinking=True)`.
  5. `tokens = prompt_ids + comp_ids`; `mask = [0]*len(prompt_ids) + [1]*len(comp_ids)`.
  6. truncate cả hai `[:8192]` (`TOKEN_LIMIT`, khớp [corpus.py:43](../../nemotron-master/corpus.py#L43)).
  7. return dict `{problem_id, category, tokens, mask, weight, sign}`.
- **`make_example_from_ids`:** bỏ bước 1-4, dùng `tokens=prompt_ids+completion_ids`, `mask=[0]*len+[1]*len`, truncate `[:8192]`.
- **Lỗi hay mắc:**
  - ❌ Dùng `chat_tok` để encode completion → thêm special tokens sai. Phải dùng `comp_tok` + `add_special_tokens=False`.
  - ❌ Quên `enable_thinking=True` → thiếu `<think>` mở đầu, lệch corpus.py.
  - ❌ Lấy answer gốc thay vì `\boxed` trong reasoning → mismatch khi reasoning tự kết luận.
- **Validate:** chạy `make_example` trên 3 problem có sẵn reasoning/*.txt → so **bit-khớp** dòng tương ứng trong `corpus.jsonl` / `corpus/<id>/synthetic.jsonl`.

### 3.3 `vllm_engine.py`
- **Mục tiêu:** load 0.86 adapter + batched sample, tách khỏi CLI.
- **Port:** [generate_rollouts_vllm.py:254-301](../../generate_rollouts_vllm.py#L254-L301) (load) + [180-206](../../generate_rollouts_vllm.py#L180-L206) (sample) + `_free_vllm` [81-98](../../generate_rollouts_vllm.py#L81-L98).
- **Hàm:** `load_engine(model_path, adapter_path, max_model_len=8192) -> (llm, tokenizer, lora_request)`; `sample(llm, prompt_token_ids, lora_request, *, n, temperature=0.9, top_p=0.95, max_tokens=7680, logprobs=1) -> outputs`; `free(llm)`.
- **Gotcha:** giữ `enable_lora=True, max_lora_rank=32, dtype="bfloat16", trust_remote_code=True` ([L282-289](../../generate_rollouts_vllm.py#L282-L289)). Adapter chỉ load nếu có `adapter_config.json` ([L292](../../generate_rollouts_vllm.py#L292)).
- **CẤM:** không đổi `max_lora_rank` (≠32 sẽ vỡ submission).

### 3.4 `ppl.py`
- **Mục tiêu:** điểm "khó-với-student" mỗi trace, để OXA-LP (đúng-PPL-cao→weight↑) và OXA-UL (sai-PPL-thấp→sign=−λ).
- **2 chế độ, ưu tiên (a):**
  - **(a) approx từ logprob vLLM (KHÔNG thêm GPU-pass):** `ppl_from_logp(old_logp, mask) -> float = exp(-mean(old_logp[i] for i where mask[i]==1))`. Dùng `old_logp` đã có ở [generate_rollouts_vllm.py:224-228](../../generate_rollouts_vllm.py#L224-L228).
  - **(b) HF chính xác (chỉ khi cần):** port [oxa cal_ppl.py `ppl_forward`](../../refs/oxa/1-prepare_sft_data/cal_ppl/cal_ppl.py), chunk `step_tokens=128`, NLL CHỈ trên token completion (mask==1) → `exp(mean_nll)`. Bỏ phần OOM-bisect cho gọn.
- **Hàm:** `ppl_from_logp(old_logp, mask)->float`; (tuỳ chọn) `trace_ppl_hf(model, tokenizer, text, comp_mask, step_tokens=128)->float`.
- **Khuyến nghị:** dùng (a) trong `sample_rollouts` luôn → mọi downstream có `ppl_approx` miễn phí. (b) để dành nếu (a) noise.

### 3.5 `dedup.py`
- **Mục tiêu:** Jaccard-dedup giữ trace đa dạng (RFT/DPO), gold-anchor.
- **Port:** [dpo-st make_rft_data.py `compare_similarity`](../../refs/dpo-st/utils/make_rft_data.py) (token-set IoU).
- **Hàm:** `jaccard(a_ids, b_ids)->float` (`len(set(a)&set(b))/len(set(a)|set(b))`); `dedup_keep_diverse(items, k, key, thr=0.7, anchor=None)->list` — `anchor` (nếu có) luôn đứng đầu & giữ; duyệt phần còn lại, giữ nếu `jaccard < thr` với mọi cái đã giữ; cap `k`.
- **Gotcha:** `key` trỏ tới token-ids của trace (dùng completion_ids để dedup, không phải toàn tokens kèm prompt).

### 3.6 `corpus_io.py`
- **Mục tiêu:** I/O chuẩn, mọi generator ghi giống nhau.
- **Hàm:** `write_rows(path, rows, append=False)` (mỗi row 1 dòng `json.dumps`); `read_rollouts(path) -> Iterator[dict]`; `merge_corpus(paths, out_path)` (gộp nhiều `corpus_*.jsonl`).
- **Gotcha:** dedup `problem_id` trùng KHÔNG bắt buộc cho Modal jsonl (cho phép nhiều trace/id); chỉ Kaggle repack mới cần suffix id (§6).

---

## 4. Lớp sampling (Milestone B) — **1 base-pass adaptive-G + các pass có điều kiện riêng**

> ⚠️ **Sửa quan niệm cũ:** KHÔNG phải "1 lượt G=8 nuôi mọi generator". Đúng là: **1 base-pass adaptive-G** (`sample_rollouts.py`) nuôi nhóm *hậu-xử-lý* (G1/G2/G3/G4); còn G6/G9/exp35 **mỗi cái có routine sampling RIÊNG** (prompt khác → không tái dùng được `rollouts.jsonl`). Hai lý do: thiếu **TYPE** (trajectory có điều kiện) và thiếu **DEPTH** (G=8 uniform quá nông cho category khó/mixed).

### 4.0 Routing — generator nào tái dùng rollouts, generator nào cần pass riêng

| Generator | Nguồn sampling | Vì sao |
|---|---|---|
| G1 `build_rft`, G2 `build_negatives`, G3 `build_reward_weighted`, G4 `build_dpo_pairs` | **tái dùng `rollouts.jsonl`** (hậu-xử-lý) | chỉ lọc/đánh-trọng-số trace base, không đổi prompt |
| G6 `build_correction_traces` (exp34) | **pass riêng, 2 vòng** | turn2 = sample CÓ ĐIỀU KIỆN trên `[prompt + attempt-sai + "sửa đi"]` → prompt khác |
| G9 `coldstart_inject` (exp35-assist) | **pass riêng** | prompt bị chèn k% lời-giải/skeleton → input khác hẳn base |
| exp35 `self_improve` | **pass riêng, lặp** | sample trên bài (n+k)-bit **tổng-hợp** (không có trong train.csv) + adapter cập nhật mỗi vòng |

→ Cả 3 pass riêng **gọi chung** core `common.vllm_engine.sample()` (đừng load engine nhiều lần — engine load ~vài phút; reuse cùng `llm` object trong 1 process nếu chạy nối tiếp). Mỗi cái tự build prompt rồi verify bằng `common.verify`.

### 4.1 `sample_rollouts.py` — base-pass, **G thích ứng theo category** (HS-STaR / b-star budget)
- **Mục tiêu:** harvest đủ DEPTH cho category cần nhất, không phí budget cho category đã bão hoà.
- **Vấn đề của G=8 uniform** (số từ leaderboard/yield thực):

  | Category | pass@1 | G=8 cho ra | Hệ quả |
  |---|--:|---|---|
  | numeral / gravity / unit_conv | 90–97% | ~8 đúng/bài | **thừa đúng, 0 cặp mixed** cho DPO |
  | cipher / equation | 31–36% | vài đúng + vài sai | OK cho mixed-pair |
  | bit_manipulation | 9.6% | **~0.8 đúng/bài** | **RFT đói** đúng category cần nhất |
  | cryptarithm / *_guess | ~0% | 0 đúng | info-ceiling (chấp nhận, G nhỏ) |

- **Thiết kế 2-stage:**
  1. **Probe (rẻ):** G_probe=8 uniform toàn `train.csv` → đây CHÍNH là input cho `measure_yield` (§5.1) → ước `pass@1` mỗi **category** (và mỗi **bài** nếu muốn per-problem budget).
  2. **Targeted:** đặt `G_cat` theo `p=pass@1`:
     - `p ≥ 0.85` (numeral/gravity/unit): `G=2` — chỉ cần vài trace đúng, chủ yếu để có baseline.
     - `0.15 ≤ p < 0.85` (cipher/equation): `G=24–32` — nguồn mixed-pair (DPO) + RFT giàu nhất.
     - `0.02 ≤ p < 0.15` (bit_manip): `G=48–64` — bơm để hứng trace đúng hiếm cho RFT + tạo group mixed.
     - `p < 0.02` (crypto/guess): `G=4` — info-ceiling, chỉ để xác nhận, không đổ budget; nhường cho G9 coldstart.
  - **Per-problem boundary (tuỳ chọn, b-star/adastar):** trong category, dồn thêm sample cho bài probe ra **mixed** (vừa đúng vừa sai) — nơi tín hiệu học cao nhất; bớt cho bài pure-correct/pure-wrong.
- **Budget — cap token-budget TỔNG (chốt):** đừng để `G_cat` cố định làm nổ giờ. Cơ chế:
  1. từ probe, ước `avg_tok_cat` (token/trace trung bình mỗi category, từ `completion_token_ids`).
  2. `--token_budget` (mặc định ~150M ≈ 3-5h RunPod). Tính `cost = Σ_cat (G_cat × N_cat × avg_tok_cat)`.
  3. nếu `cost > token_budget`: scale `G_cat` xuống theo **thứ tự ưu tiên** (giữ category đói = bit/cipher/equation, cắt category đã-bão-hoà trước — chúng vốn `G` nhỏ nên cắt ít ảnh hưởng). KHÔNG scale đều tay (sẽ lại làm bit_manip đói).
  4. log bảng `{category: G_final, est_tokens}` trước khi chạy để soi.
  - **`G_cat` ở §4.1 là TRẦN ưu tiên, không phải số cứng** — budget cap có quyền hạ.
- **Port + thay đổi trên [generate_rollouts_vllm.py](../../generate_rollouts_vllm.py)** (giữ load/engine/resume; refactor `process_batch` [180-251](../../generate_rollouts_vllm.py#L180-L251)):
  1. **XOÁ** drop pure-group [L242-243](../../generate_rollouts_vllm.py#L242-L243). Giữ `mixed_rate` để LOG.
  2. **THÊM `text`**: `"text": completion.text`.
  3. **THÊM `ppl_approx`** = `ppl_from_logp(old_logp, mask)`.
  4. **THÊM adaptive-G:** đọc `--g_schedule` (JSON `{category: G}` hoặc đường dẫn `yield_report.json` để tự suy) thay vì `n=args.group_size` cố định ([L184-190](../../generate_rollouts_vllm.py#L184-L190) tạo `SamplingParams` per-category).
- **CLI:** `--mode probe|targeted`; `--g_schedule <json>`; `--token_budget 150000000`; giữ `--temperature 0.9 --resume --max_problems`; adapter mặc định = 0.86.
- **Output `rollouts.jsonl` (1 dòng/completion):**
  ```jsonc
  {"problem_id","category","prompt_token_ids":[...],"completion_token_ids":[...],
   "text":"...reasoning...","pred":"...","answer":"...","reward":0|1,
   "old_logp":[...],"ppl_approx":1.34,"stage":"probe|targeted"}
  ```
- **Gotcha:**
  - Tách `prompt_token_ids`/`completion_token_ids` riêng → downstream ghép qua `make_example_from_ids`.
  - `old_logp` gồm phần prompt (=0.0, [L224](../../generate_rollouts_vllm.py#L224)); `ppl_from_logp` chỉ lấy completion theo mask.
  - Probe + targeted ghi CÙNG file (append, field `stage` phân biệt) → resume an toàn.
- **Volume:** probe ~76k + targeted dồn vào bit/cipher/equation → tổng ~150–250k tuỳ schedule (bound bằng token budget RunPod ~3-5h).
- **CẤM:** dùng G cố định cho mọi category (làm `build_rft` đói bit_manip); load engine lại cho mỗi pass có điều kiện (reuse `sample()` core).

---

## 5. Generators

> Mỗi file có `__main__` + CLI. CPU-only chạy máy thường; file dùng `reasoners/*` phải chạy trong `nemotron-master/` env (`uv run`).

### 5.1 `measure_yield.py` — P0 gate (CPU, đọc rollouts) — **làm TRƯỚC**
- **Mục tiêu:** quyết nhánh nào đáng đẩy; chặn đổ compute vào info-ceiling.
- **Đầu vào:** `rollouts.jsonl` + `problems.jsonl` (category).
- **Tính:** per-category `pass@1 = mean(reward)`, `pass@k = any(reward) trong group`; phân phối `ppl_approx`; **error-bucket** mỗi trace sai: `truncation` (len completion ≥ max_tokens-ε hoặc thiếu `\boxed`), `format` (`format_ok=False`), `arithmetic_slip` (có `\boxed` nhưng `compare_answer=False`). Đếm theo category.
- **Output:** `yield_report.json` + in bảng category × {pass@1, pass@k, %trunc, %slip, %format, ppl median}.
- **Quyết định ghi vào report:** category yield≈0 (crypto/guess) → **KHÔNG** đầu tư G7/G8/G9 vào đó (info-ceiling, [memory cryptarithm-unsolved-levers]).
- **CẤM:** không tin số leaderboard trước khi chạy decontaminate (§6) — nhưng yield-report thì chạy ngay được.

### 5.2 `compress_traces.py` — exp33 / G5 (CPU, đọc reasoning/*.txt) — **làm SỚM**
- **Mục tiêu:** gỡ nút 7680 cho `bit_manipulation` (trace tới 816 dòng).
- **Port:** [tokenskip LLMLingua.py](../../refs/tokenskip/LLMLingua.py) `PromptCompressor` ratio-prune. ASAP surprisal = fallback.
- **Hàm:** `compress_text(text, ratio, protect_patterns)->str`; `main()`.
- **Thuật toán:**
  1. lọc `category==bit_manipulation` từ `problems.jsonl`, đọc `reasoning/<id>.txt`.
  2. `PromptCompressor("microsoft/llmlingua-2-...", use_llmlingua2=True)`; `compress_prompt(text, rate=ratio)` với `ratio∈{0.7,0.6,0.5}` (thử 0.7 trước).
  3. **protect-list** (KHÔNG prune): mọi token số `\d`, dòng chứa `\boxed`, header cột-bit (regex theo `_emit_apply`/Input/Output trong [bit_manipulation.py:330+](../../nemotron-master/reasoners/bit_manipulation.py#L330)). LLMLingua hỗ trợ `force_tokens`/`force_reserve_digit` — bật.
  4. trace nén → `make_example(prompt, compressed, answer, weight=1, sign=1)`.
  5. **round-trip verify:** `compare_answer(answer, extract_answer(compressed_completion))` phải True, nếu không → DROP (đừng bơm trace mất đáp án).
- **Output:** `corpus_bit_compressed.jsonl` (thay phần bit_manip khi merge).
- **Distinct:** controllable (engine đo importance) — khác exp3 terse-tay (−0.28). Đừng prune tay.
- **Dep:** `uv add llmlingua`.
- **Falsify:** bit_manip pass↑ HOẶC %trunc↓ (đo bằng measure_yield).
- **CẤM:** ratio < 0.5; prune token số/`\boxed`; bỏ round-trip verify.

### 5.3 `evolve_solved.py` — exp37 / G8 (CPU, chạy trong `nemotron-master/`) — **làm SỚM**
- **Mục tiêu:** mutate seed `rule_found` → bài KHÓ HƠN, verify bằng solver của repo (free, đúng 100%).
- **Port taxonomy:** [wizardlm depth.py](../../refs/wizardlm/Evol_Instruct/depth.py) 5 operator (add-constraint/deepen/concretize/add-steps/breadth) + [metamath run_backward.sh](../../refs/metamath/code_for_generating_data/code/run_backward.sh) FOBAR.
- **Cơ chế AN TOÀN (round-trip verify — đọc kỹ):** bất kể mutate kiểu gì, một bài chỉ được giữ nếu:
  1. dựng `Problem` mới (id mới, examples/question/answer tự sinh, self-consistent),
  2. gọi `GENERATORS[category](problem)` ([reasoning.py:42-56](../../nemotron-master/reasoning.py#L42-L56)) → `reasoning_text` (hoặc `None`→drop),
  3. `compare_answer(problem.answer, extract_answer(reasoning_text))` == True → giữ; sai/None → DROP.
  → **solver-verify gate đảm bảo không bao giờ emit data sai**, kể cả khi mutate ra rule ngoài hypothesis-space (solver fail → drop).
- **Hàm:** `evolve_bit_manipulation(seed: Problem, ops, rng) -> list[Problem]`; `evolve_equation(seed, ops, rng) -> list[Problem]`; `roundtrip_keep(problems) -> list[(Problem, reasoning_text)]`; `main()`.
- **Thuật toán bit_manipulation (ưu tiên — domain đúng nhất):** seed có 8 ví dụ `input→output` theo per-bit rule.
  1. **suy rule từ seed:** với mỗi bit-vị-trí, brute-force family∈{I,NOT,0,1,XOR,OR,AND,*-NOT} + operand indices sao cho khớp toàn bộ 8 ví dụ (tái dùng `_evaluate_rule` [bit_manipulation.py:300-330](../../nemotron-master/reasoners/bit_manipulation.py#L300)).
  2. **harden (operator menu):** `add_operand` (unary→pair family), `longer_chain` (tăng số bit phụ thuộc / dùng stride), `nest_op` (đổi family phức tạp hơn). Mỗi op = 1 variant.
  3. **regen data:** sinh `K=8` input ngẫu nhiên mới + 1 question; `output=apply_rule(input)` cho mọi ví dụ + answer. Build `Problem(id=f"{seed.id}__e{n}", ...)`.
  4. round-trip verify (trên).
- **Thuật toán equation FOBAR:** ẩn 1 biến đã biết, hỏi tìm nó cho trước đáp án (backward); answer deterministic; verify qua `reasoning_equation_numeric`.
- **Output:** `corpus_evolved.jsonl` (qua `make_example`, weight=1, sign=1).
- **⚠️ OOD-risk:** train-acc↑ nhưng leaderboard có thể ↓. → **gate held-out category-level** (đừng tin train-acc).
- **Volume:** seed ~1,900 × 3–5 = +6k–9.5k.
- **Lỗi hay mắc:** ❌ tạo question/answer không self-consistent (answer phải = rule áp lên question); ❌ id trùng seed → ghi đè; ❌ chạy ngoài `nemotron-master/` → ImportError `reasoners`.
- **CẤM:** giữ bài mà solver không reproduce; bỏ round-trip verify; train rồi đo bằng train-acc.

### 5.4 `build_rft.py` — exp31 / G1 (CPU, đọc rollouts) [Milestone C2]
- **Mục tiêu:** thêm trace ĐÚNG đa dạng, gold-anchor.
- **Port:** [dpo-st make_rft_data.py `process_rft_train`](../../refs/dpo-st/utils/make_rft_data.py) (gold-anchor + Jaccard dedup `sim_threshold=0.7`, cap k).
- **Thuật toán:** group rollouts theo `problem_id`; lấy `reward==1`; nếu `status==rule_found` → anchor = `reasoning/<id>.txt` (gold); `dedup_keep_diverse(correct, k=4, key=completion_token_ids, thr=0.7, anchor=gold)`; mỗi giữ → `make_example_from_ids(weight=1, sign=1)`.
- **Output:** `corpus_rft.jsonl`. **Volume:** +10k–14k.
- **Distinct:** khác exp7 (thêm Jaccard + gold-anchor); khác corpus gốc (thêm đúng cho bài solver-chưa-giải).

### 5.5 `build_negatives.py` — exp28-UL + exp29 / G2 (CPU, đọc rollouts) [C2]
- **Mục tiêu:** học-đẩy-xa trace SAI (`sign=-1`).
- **Port:** [oxa make_ULloss_data.py](../../refs/oxa/1-prepare_sft_data/make_ULloss_data.py) (`MIN_PPL_THRESHOLD=1.2`) + [redi clean_base_data.py](../../refs/redi/experiments_trl/data_preprocess/clean_base_data.py) (all-wrong).
- **Thuật toán:** lấy `reward==0`; mode `oxa`: giữ `ppl_approx < ppl_thr` (sai high-confidence); mode `redi`: giữ tất; `make_example_from_ids(weight=λ, sign=-1)` (λ default 0.5). Nguồn thêm: 1,167 contaminated (`status≠rule_found`).
- **Output:** `corpus_neg.jsonl`. **Volume:** ~free.
- **Falsify:** format-box còn nguyên + cipher/eq không tụt → vỡ thì giảm λ.
- **CẤM:** sign=-1 trên trace `format_ok=False` (đẩy-xa rác vô nghĩa) — lọc trước.

### 5.6 `build_reward_weighted.py` — exp30 / G3 (CPU, đọc rollouts) [C2]
- **Mục tiêu:** bản nguyên-lý-đúng của exp27 (KHÔNG importance-clip).
- **Port:** [a-po model_generate.py](../../refs/a-po/preprocess/data_generation/model_generate.py) + README §Offline (V baseline β1=0.5, advantage-regress β2=1e-3).
- **Thuật toán:** group theo `problem_id`; `V = mean(rewards)` (hoặc soft-max-β1); `adv = reward − V`; chỉ giữ `reward==1` (adv>0); `weight = clip(1 + adv/β2_scale, w_min=0.5, w_max=3.0)`; `make_example_from_ids(weight, sign=1)`.
- **Output:** `corpus_rw.jsonl`.
- **Falsify:** vs CE thuần +0.5pp macro; reward thưa (measure_yield) → bỏ.
- **CẤM:** importance-weight clipping kiểu GSPO (chỗ exp27 degenerate).

### 5.7 Milestone D — spec ngắn (code sau)
> Phân loại theo §4.0: `build_dpo_pairs` **tái dùng** targeted-rollouts; còn `build_correction_traces`/`coldstart_inject`/`synth_weakness`+exp35 **cần pass sampling RIÊNG** (gọi `common.vllm_engine.sample`), KHÔNG hậu-xử-lý `rollouts.jsonl`.
- `build_dpo_pairs.py` (exp31/32) — **tái dùng rollouts**: mixed-group → (chosen=gold/correct, rejected=wrong); **step-dpo** ([refs/step-dpo](../../refs/step-dpo)) cho bit_manip: solver biết **bước lệch đầu** ([plan §16.1](plan-batch-4.md)) → cặp share-prefix, không cần critic-LLM. Output `pairs.jsonl`; **cần thêm DPO objective vào Continuer** (milestone riêng). ⚠️ Cặp mixed chỉ dồi dào ở cipher/equation/gravity — phụ thuộc targeted-G (§4.1) để có cặp ở bit_manip.
- `build_correction_traces.py` (exp34) — **PASS RIÊNG 2 vòng**: vòng1 = lấy trace SAI từ rollouts; vòng2 = `sample()` lại với prompt `[gốc + attempt-sai + "sửa đi"]` ([self-rewarding infer_math](../../refs/self-rewarding-correction/infer_math/) `process_prompt_turn2`); label bằng `compare_answer`; merge template `attempt→[VERIFY] wrong.→<sửa>→[VERIFY] correct.→\boxed{}`; **mask phần attempt sai** (weight=0 cho token attempt).
- `coldstart_inject.py` (exp35-assist) — **PASS RIÊNG**: `sample()` với prompt chèn k% gold-token/skeleton ([questa process.py](../../refs/questa/AReaL/datasets/process.py)) → verify-keep completion đầy đủ. Chỉ lúc gen (submission vẫn greedy thuần). Nhắm bit_manip-gap + equation hard.
- `self_improve.py` (exp35) — **PASS RIÊNG, LẶP**: sinh bài (n+k)-bit tổng-hợp → `sample()` → verify-keep → train → lặp với adapter mới. Length-generalization; stop-rule hội tụ (auto-cei [plan §17.2](plan-batch-4.md)).
- `synth_weakness.py` (exp36) — **CPU**: concept co-occurrence từ `failure_cases.jsonl` (P0) → generator repo sinh tổ-hợp operator hay-trượt ([sws data_synthesis.py](../../refs/sws/src/data_synthesis.py)).

---

## 6. `upload_kaggle.py` + decontaminate

- **Mục tiêu:** đóng gói corpus + chống rò rỉ train↔test.
- **2 format (vì 2 loader khác nhau, §0):**
  - **Modal (ưu tiên):** `merge_corpus([corpus_*.jsonl]) → corpus_preprocessed.jsonl` (jsonl thẳng, cho phép nhiều trace/id).
  - **Kaggle:** explode thành `<sid>/synthetic.json` + `index.jsonl`; nhiều trace/id → suffix `<id>__v2` để không đè.
- **Decontaminate:** port [pcl-reasoner decontaminate.py](../../refs/pcl-reasoner/data_preprocess/decontaminate.py) — lọc trace có nội dung trùng test trước khi tin số leaderboard.
- **Upload:** `kaggle datasets version -p <dir> -m "<msg>"`.

---

## 7. `setup_runpod.sh` mở rộng
- Base [setup_runpod.sh](../../setup_runpod.sh) (torch 2.6 cu124 + vLLM). Thêm: `llmlingua` (compress, CPU), `kaggle` CLI (upload). `tokenizers/transformers` đã có. **Không** thêm verl/ray/OpenRLHF/AReaL (đã loại).

---

## 8. Run order + budget (1 vòng)

| Bậc | Lệnh | GPU | Time |
|---|---|:--:|---|
| 0 | `compress_traces` + `evolve_solved` (CPU, song song) | ✗ | máy thường |
| 1a | `sample_rollouts.py --mode probe` (G=8 uniform) | ✓ | ~1h |
| 1b | `measure_yield` → suy `g_schedule` per-category | ✗ | giây |
| 1c | `sample_rollouts.py --mode targeted --g_schedule yield_report.json` (G lớn cho bit/cipher/eq) | ✓ | ~3-5h |
| 2 | `build_rft`/`build_negatives`/`build_reward_weighted` (tái dùng rollouts) | ✗ | phút |
| 3 | `upload_kaggle` → Continuer continue-0.86 (`RESET_WEIGHTS=False`, LR 1e-5–5e-5, NUM_STEPS giảm) | ✓ | train |
| (D) | `build_correction_traces` / `coldstart_inject` / `self_improve` | ✓ | **mỗi cái pass sampling riêng** (§4.0) |

**Base-pass adaptive-G nuôi G1–G4 (hậu-xử-lý);** G6/G9/exp35 = pass có điều kiện riêng. Tiết kiệm = reuse `vllm_engine` core + ghi chung `rollouts.jsonl`, KHÔNG phải "1 lượt nuôi tất cả".

---

## 9. Validation chung (gate trước khi train)

1. **Format khớp grader:** decode 5 row/generator → đúng `</think>\n\boxed{...}<|im_end|>`, prompt mask=0/completion mask=1. So bit-khớp 1 mẫu corpus.py.
2. **weight/sign no-op (M-A):** corpus cũ → loss bit-identical.
3. **negatives:** mọi row `sign=-1`, `reward==0`, `format_ok=True`.
4. **evolve/compress:** 100% trace giữ phải re-verify `compare_answer(answer, extract_answer(completion))==True`.
5. **measure_yield gate:** category yield≈0 → không đổ compute.
6. **decontaminate** train↔test trước khi tin số leaderboard.

---

## 10. Risks / rollback

| Risk | Mitig |
|---|---|
| `sign=-1` vỡ format/regress | λ nhỏ (0.5), `.abs()` mẫu; rollback = bỏ `corpus_neg.jsonl` |
| `evolve_solved` OOD | gate held-out category-level (§9.4) |
| `compress` cắt mất bước | protect-list + round-trip verify; ratio 0.7 trước |
| reward thưa → reward_weighted vô nghĩa | measure_yield P0 quyết trước |
| Continuer patch lệch baseline | marker + no-op test (§9.2) |
| crypto/guess kỳ vọng ảo | info-ceiling — không đầu tư |

---

## 11. Đã loại — KHÔNG code lại
- **G11/G12:** beyond-80-20→exp20, doremi→exp22, ESFT→exp23, group_DRO→exp25 (regressed).
- **on-policy:** GSPO/exp27 + mọi verl/ray/OpenRLHF/AReaL (vLLM-in-loop dead-end).
- **python-constraint CSP crypto** → exp24 regressed (info-ceiling, [plan §17.3](plan-batch-4.md)).

---

## 12. First-PR checklist (A + B + C1)

- [ ] `offline/common/__init__.py` (rỗng) + `verify.py` / `tokenize_format.py` / `vllm_engine.py` / `ppl.py` / `dedup.py` / `corpus_io.py` — §3
- [ ] Patch Continuer `weight`/`sign` (marker + no-op test §9.2) — §2
- [ ] `offline/sample_rollouts.py` (bỏ drop-pure-group, +text +ppl_approx) — §4
- [ ] `offline/measure_yield.py` — §5.1
- [ ] `offline/compress_traces.py` (exp33) — §5.2
- [ ] `offline/evolve_solved.py` (exp37, chạy trong nemotron-master/) — §5.3
- [ ] `setup_runpod.sh` += llmlingua, kaggle — §7
- [ ] Validate §9.1/§9.2/§9.4 pass
- [ ] 1 vòng end-to-end → leaderboard → `tracker/rounds/round_13.md` + `tracker/leaderboard.md`

> C2 (build_rft/negatives/reward_weighted) + D (dpo/correction/coldstart/synth) = PR sau, dùng lại `rollouts.jsonl`.
