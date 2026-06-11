# Plan chi tiết — exp29: Iterative DPO on eval_slice rollouts

Batch-4 Idea 9. Mục tiêu: tinh chỉnh adapter 0.86 bằng DPO (offline) để **ưu tiên cách suy
luận đúng của chính mô hình** trên phân phối bài thi → kỳ vọng +0.3–1.0 pp.

> Tài liệu này là **plan trước khi sửa code** (theo yêu cầu). Mọi quyết định kiến trúc đã neo
> vào trạng thái file thật (verified 2026-06-05). Sau khi bạn duyệt plan, tôi mới sửa code.
>
> **TRẠNG THÁI 2026-06-05: ĐÃ IMPLEMENT** (sau khi bạn duyệt: ref=base + LR nhỏ; env=image Kaggle).
> Đã viết: `infer_slice.py` (vá sampling), `build_dpo_pairs.py` (test OK 49 pair từ preds giả),
> `train_dpo_trl.py` (smoke mode), `experiments/exp29.py` (orchestration RUN=False). ruff sạch.
> CÒN LẠI = chạy thật trên RTX PRO 6000: điền ADAPTER_PATH, đặt RUN=True, theo §4.

Quyết định đã chốt theo yêu cầu của bạn:
- (a) ADAPTER_PATH đã có → coi như biến đầu vào, không phải lo.
- (b) Thêm `--temperature` + `--n_samples` (vá `infer_slice.py`, exp29.py đã truyền sẵn 2 flag này).
- (c) Viết mới `build_dpo_pairs.py` (plan ở §3).
- (d) Engine = **HuggingFace TRL `DPOTrainer`**, mẫu `enhance_cot/redi/experiments_trl/open_r1_dpo.py`.
- (e) Bỏ qua exp18 — giả định preference learning chạy tốt.
- GPU = **RTX PRO 6000** (Kaggle competition cấp). Mọi bước GPU chạy ở đó.

---

## 1. Bức tranh tổng thể — 5 bước

```
[1] infer_slice.py  (GPU)  → preds_exp29.jsonl    : mỗi bài làm 10 lần (temp=0.5)
[2] build_dpo_pairs.py (CPU)→ dpo_pairs_exp29.jsonl: ghép {prompt, chosen, rejected}
                            → eval_holdout_ids.txt : 30 bài để chấm, KHÔNG dùng build pair
[3] train_dpo_trl.py (GPU) → adapter_exp29/        : DPO 50 bước, TRL DPOTrainer
[4] infer_slice.py  (GPU)  → preds_holdout.jsonl   : greedy trên 30 bài holdout, adapter mới
[5] eval_slice.py   (CPU)  → so sánh holdout acc vs baseline → GIỮ hoặc BỎ
```

Mỗi bước có cổng dừng (falsification) ở §6. Nếu cổng nào fail → dừng, không tốn GPU bước sau.

---

## 2. Quyết định kiến trúc then chốt (đọc kỹ — ảnh hưởng tính hợp lệ submission)

### 2.1. Vì sao phải continue-train CHÍNH adapter 0.86 (không merge)
Luật thi: submission là **một LoRA adapter rank-32 trên base gốc**, vLLM-loadable. Do đó:
- KHÔNG được merge 0.86 vào base rồi thêm LoRA mới (tổng 2 LoRA rank-32 = rank-64, không ship được).
- => DPO phải **huấn luyện tiếp đúng bộ trọng số LoRA 0.86** (policy LoRA = adapter 0.86, để `is_trainable=True`). Kết quả cuối vẫn là **một adapter rank-32** → hợp lệ.

### 2.2. Hệ quả: reference model = base (LoRA tắt), KHÔNG phải 0.86
Trong TRL, khi `model` là PEFT và `ref_model=None`, DPOTrainer tính logprob tham chiếu bằng cách
**tắt adapter** → reference = base thuần. Nghĩa là KL được neo về **base**, không phải về 0.86.
- Cách "đúng lý thuyết" (ref = base+0.86 đông cứng) cần nạp model thứ 2 (~+60GB) → **vượt 96GB**
  của RTX PRO 6000 với model 30B. Nên **không khả thi về bộ nhớ**.
- Giải pháp kiểm soát trôi (drift) thay cho KL-anchor: **LR rất nhỏ + ít bước + cổng holdout**.
  Lấy `learning_rate=5e-7` (như ví dụ TRL của REDI), `max_steps=50`, `beta=0.1`. Nếu holdout
  tụt → giảm LR/steps hoặc bỏ. Đây là rủi ro R4 ở §7.

### 2.3. Định dạng prompt/completion của cặp DPO (phải khớp lúc sinh)
- `prompt` (lúc train) = `tokenizer.apply_chat_template([{"role":"user","content": prompt_raw + PROMPT_SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=True)`.
  → đúng tiền tố mô hình đã thấy lúc generate (kết thúc ngay ở `<think>` mở đầu).
- `chosen` / `rejected` = **văn bản output thô** của mẫu (gồm reasoning + `</think>` + `\boxed{...}`).
- `PROMPT_SUFFIX`, `enable_thinking`, max_tokens phải **giống hệt** `infer_slice.py`/`corpus.py`
  (đã verify chúng dùng chung chuỗi `PROMPT_SUFFIX`).
- `build_dpo_pairs.py` chỉ lưu **prompt_raw thô** (chưa template) để giữ bước CPU-only; việc
  apply chat template làm trong `train_dpo_trl.py` (map dataset). Tách bạch, dễ test.

---

## 3. Thay đổi/việc viết mới từng file

### 3.1. (b) Vá `nemotron-master/infer_slice.py`  — nhỏ, không phá hành vi cũ
Thêm 3 tham số, **default = hành vi greedy cũ** (exp27/28 không bị ảnh hưởng):
```
--temperature  float  default 0.0
--n_samples    int    default 1
--seed         int    default 0
```
- `SamplingParams(temperature=temperature, max_tokens=MAX_TOKENS, n=n_samples, seed=seed)`.
- Ghi output:
  - `n_samples == 1` → **giữ schema cũ** `{id, output, n_output_tokens, hit_cap}` (eval_slice.py vẫn chạy).
  - `n_samples > 1`  → `{id, outputs:[...], n_output_tokens:[...], hit_cap:[...]}` (list n phần tử).
- Lý do giữ 2 schema: bước [4] (chấm holdout) dùng greedy n=1 → cần schema cũ cho `eval_slice.py`.

### 3.2. (c) Viết mới `nemotron-master/build_dpo_pairs.py`  — CPU-only, không GPU
**Mẫu cấu trúc:** copy phong cách `build_eval_slice.py` (argparse + đọc jsonl + ghi jsonl).
**Input:**
- `--slice eval_slice.jsonl`  → `{id, category, prompt, answer}` (answer = ground truth).
- `--preds preds_exp29.jsonl` → `{id, outputs:[...]}` (từ bước [1]).
- `--holdout_n 30`, `--seed 7`, `--max_pairs_per_problem 1`, `--out dpo_pairs_exp29.jsonl`.

**Thuật toán:**
1. Tách holdout: sort id → seeded shuffle → lấy `holdout_n` id đầu làm **holdout**, ghi
   `eval_holdout_ids.txt`. Các id còn lại dùng để build pair.
2. Với mỗi bài build-pair: với mỗi sample trong `outputs`:
   - `pred = extract_answer(sample)` (dùng `reasoning.extract_answer`).
   - `ok = compare_answer(rec["answer"], pred)` (dùng `reasoning.compare_answer` — **source of truth**).
   - phân loại sample vào `correct[]` / `incorrect[]` (kèm văn bản thô + cờ `hit_cap`).
3. Chỉ tạo pair khi **có ≥1 correct và ≥1 incorrect**.
   - `chosen` = correct **ngắn nhất, hit_cap=False** (sạch, không bị cắt cụt).
   - `rejected` = incorrect ưu tiên loại **có boxed nhưng sai** (dạy "đúng vs sai tinh vi"); nếu
     không có thì lấy incorrect bất kỳ (kể cả format hỏng) ngắn nhất, hit_cap=False.
   - lặp tối đa `max_pairs_per_problem` cặp/bài (mặc định 1 để khỏi lệch về bài dễ).
4. Ghi mỗi dòng: `{id, category, prompt_raw, chosen, rejected}` (prompt_raw = `rec["prompt"]` thô).
5. In thống kê: tổng pair, pair/category, số bài bị bỏ. **ABORT nếu tổng < 20 pair.**

**Lưu ý phân phối:** cryptarithm hầu như không có correct → ít/không có pair; pair sẽ tập trung
ở bit_manipulation/equation_numeric (mức khó vừa). Chấp nhận cho v1, ghi rõ ở §7-R3.

### 3.3. (d) Viết mới `nemotron-master/train_dpo_trl.py`  — engine TRL DPO
**Deps mới** (xem §5 môi trường): `trl`, `peft`, `accelerate`, `datasets` (transformers đã pin 4.57.6).
**Luồng:**
1. `tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)`; nếu thiếu pad → `pad_token=eos`.
2. `model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, torch_dtype=bf16, use_cache=False)`.
3. `model = PeftModel.from_pretrained(model, adapter_086_path, is_trainable=True)`  ← policy = base+0.86, train tiếp.
4. `DPOConfig(beta=0.1, loss_type="sigmoid", learning_rate=5e-7, per_device_train_batch_size=1,
   gradient_accumulation_steps=8, gradient_checkpointing=True, bf16=True, max_steps=50,
   max_prompt_length=1024, max_length=8192, max_completion_length≈7000, logging_steps=5,
   save_strategy="no", report_to="none")`.
5. Dataset: đọc `dpo_pairs_exp29.jsonl` → `datasets.Dataset` → `.map` để đổi `prompt_raw`
   thành prompt-đã-template (§2.3); giữ `chosen/rejected` thô.
6. `DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, processing_class=tokenizer)`
   — **KHÔNG truyền peft_config** (model đã là PEFT). ref=None → reference = base (§2.2).
7. `trainer.train()` → `trainer.model.save_pretrained("adapter_exp29")` (vẫn rank-32, hợp lệ submission).
8. **Chế độ smoke**: cờ `--smoke` → `max_steps=1`, chỉ lấy 2 pair → kiểm Nemotron-H + TRL chạy được
   trước khi đốt GPU full (rủi ro R1).

### 3.4. Wiring `experiments/exp29.py` (orchestration)
Hiện exp29.py chỉ **in lệnh**. Cập nhật để gọi thật 3 script trên qua `subprocess` (hoặc giữ dạng
notebook cell cho Kaggle). Điền `ADAPTER_PATH`, giữ nguyên hằng số (DPO_STEPS=50, BETA=0.1...).
**Không** đổi logic train_sft.py / corpus.py / token format.

---

## 4. Quy trình chạy trên RTX PRO 6000 (Kaggle hoặc server của bạn)

> Tất cả lệnh chạy trong `nemotron-master/` (uv env). Bước [1][3][4] cần GPU; [2][5] CPU.

```bash
cd nemotron-master
ADAPTER=<đường-dẫn-adapter-0.86>          # (a) bạn đã có
BASE=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

# [1] Sinh rollouts: 170 bài × 10 mẫu, temperature 0.5  (~30–60 phút trên RTX PRO 6000)
uv run python3 infer_slice.py --base $BASE --adapter $ADAPTER \
  --slice eval_slice.jsonl --out preds_exp29.jsonl --temperature 0.5 --n_samples 10 --seed 0

# [2] Ghép cặp (CPU, vài giây)
uv run python3 build_dpo_pairs.py --slice eval_slice.jsonl --preds preds_exp29.jsonl \
  --out dpo_pairs_exp29.jsonl --holdout_n 30 --seed 7
#   → nếu in "pairs < 20" thì DỪNG (không đủ tín hiệu).

# [3a] SMOKE trước (bắt buộc): 1 bước, 2 pair — kiểm TRL + Nemotron-H không OOM/không lỗi
uv run python3 train_dpo_trl.py --base $BASE --adapter $ADAPTER \
  --pairs dpo_pairs_exp29.jsonl --out adapter_exp29_smoke --smoke
#   → chạy lọt mới làm bước [3b].

# [3b] DPO thật: 50 bước
uv run python3 train_dpo_trl.py --base $BASE --adapter $ADAPTER \
  --pairs dpo_pairs_exp29.jsonl --out adapter_exp29 \
  --beta 0.1 --lr 5e-7 --max_steps 50

# [4] Chấm holdout (greedy, adapter MỚI)
uv run python3 infer_slice.py --base $BASE --adapter adapter_exp29 \
  --slice eval_slice.jsonl --out preds_holdout_exp29.jsonl     # n_samples mặc định = 1 (greedy)

# [5] So điểm — chỉ tính trên 30 id holdout
uv run python3 eval_slice.py --preds preds_holdout_exp29.jsonl --slice eval_slice.jsonl
#   (so sánh thủ công các id trong eval_holdout_ids.txt với baseline đã đo ở Prerequisite #0)
```

> Lưu ý baseline holdout: trước [3], chạy [4]+[5] một lần với `--adapter $ADAPTER` (0.86) để có
> **mốc holdout của 0.86**. Đây là con số để so ở bước [5].

---

## 5. Môi trường / dependencies

- `nemotron-master/pyproject.toml` hiện **không có** trl/peft/accelerate/datasets/vllm trong list
  chính (chỉ tinker, torch>=2.11, transformers==4.57.6).
- Cần thêm (đường uv, KHÔNG pip): `uv add trl peft accelerate datasets`
  - ⚠️ `transformers` đang **pin 4.57.6** → phải chọn phiên bản `trl` tương thích (TRL gần đây yêu cầu
    transformers mới). Nếu xung đột: tạo **deps group riêng** cho DPO hoặc chạy trong **image Kaggle**
    của cuộc thi (đã có sẵn peft/vllm/transformers tương thích Nemotron-H).
- `vllm` cần cho bước [1][4] — verify đã có trong image GPU (infer_slice.py vốn yêu cầu vllm).
- Khuyến nghị thực dụng: chạy **toàn bộ exp29 trong một Kaggle GPU notebook** (RTX PRO 6000) nơi
  vllm + peft + transformers đã được cài cho Nemotron-H; chỉ `uv add trl` nếu thiếu.

---

## 6. Các cổng falsification (dừng sớm để khỏi tốn GPU)

| Cổng | Khi nào | Điều kiện PASS | Nếu FAIL |
|---|---|---|---|
| G1 — đủ tín hiệu | sau [2] | tổng pair ≥ 20 | ABORT exp29 (rollout pass-rate không hợp) |
| G2 — tương thích | sau [3a] smoke | 1 bước DPO chạy, không OOM/crash | sửa max_length/bs; nếu vẫn lỗi → chuyển engine (R1) |
| G3 — không thoái hoá | sau [5] | holdout micro-acc(adapter_exp29) **≥** holdout micro-acc(0.86) | BỎ adapter_exp29, ship 0.86 |
| G4 — có cải thiện | sau [5] | holdout micro-acc **>** 0.86 (dù chỉ +1 bài) | giữ 0.86 làm submission; ghi exp29 = neutral |

Chỉ khi G1→G4 đều xanh mới coi exp29 thành công và cập nhật `tracker/`.

---

## 7. Rủi ro & cách giảm

- **R1 (cao nhất) — Nemotron-H × TRL không tương thích.** Model là Mamba/MoE custom
  (`modeling_nemotron_h`, trust_remote_code); DPOTrainer ghép forward chosen+rejected + gradient
  checkpointing trên SSM layer chưa có tiền lệ. → **Giảm bằng G2 (smoke 1 bước)**. Nếu hỏng:
  fallback (i) Tinker `dro`/`importance_sampling` (đổi loss, không TRL), (ii) Unsloth DPO trong
  `Continuer_Nemotron_Notebook.py` (đã có Unsloth path cho model này).
- **R2 — Bộ nhớ.** 30B bf16 + 2 completion dài (tới ~7000 token). 96GB có thể chặt. → bs=1,
  grad_ckpt, `gradient_accumulation_steps`, lọc bỏ pair có completion quá dài; nếu vẫn OOM →
  giảm `max_completion_length` (cắt pair dài).
- **R3 — Pair lệch category.** Tập trung ở bit/equation, thiếu cryptarithm. → chấp nhận v1; nếu
  muốn phủ cryptarithm phải tăng `n_samples` hoặc giảm temperature có chọn lọc (việc của exp36).
- **R4 — Trôi khỏi 0.86 (ref=base).** → LR=5e-7 + 50 bước + G3. Nếu G3 fail thử LR=1e-7/steps=30.
- **R5 — Lệch định dạng prompt.** template/`PROMPT_SUFFIX`/`enable_thinking` phải khớp [1]. →
  dùng đúng hằng số `PROMPT_SUFFIX` từ infer_slice.py; viết 1 unit check so khớp prefix.
- **R6 — Submission không hợp lệ.** → sau [3b], verify `adapter_exp29/adapter_config.json` còn
  `r=32`; thử load bằng vllm LoRA (bước [4] chính là phép thử này).

---

## 8. Rollback

- adapter 0.86 **không bị động tới** (train vào thư mục MỚI `adapter_exp29/`).
- Nếu G3/G4 fail → submission = 0.86 nguyên bản. Xoá `adapter_exp29/`, `preds_*`, `dpo_pairs_*`.
- Patch `infer_slice.py` giữ default greedy → không cần revert (không phá exp khác).

---

## 9. Thứ tự thực thi (sau khi bạn duyệt plan)

1. Vá `infer_slice.py` (b) + 1 unit check prompt-prefix.
2. Viết `build_dpo_pairs.py` (c) + chạy thử trên `preds` giả (2-3 bài) để kiểm schema.
3. Viết `train_dpo_trl.py` (d) với chế độ `--smoke`.
4. Cập nhật `experiments/exp29.py` wiring + điền `ADAPTER_PATH`.
5. `ruff format/check`. (mypy hiện chưa cài trong env — bỏ qua hoặc `uv add --dev mypy`.)
6. Bàn giao runbook §4 để bạn chạy trên RTX PRO 6000.

**Không** đụng: `corpus.py`, `train_sft.py`, token format, reasoners/, generators/.

---

## Provenance
Viết 2026-06-05. Neo vào: `infer_slice.py` (greedy-only, PROMPT_SUFFIX), `build_eval_slice.py`
(schema eval_slice), `eval_slice.py` (compare_answer scoring), Tinker `LossFnType` (không có dpo),
`enhance_cot/redi/experiments_trl/open_r1_dpo.py` (mẫu DPOTrainer: beta=0.1, loss_type=sigmoid,
PEFT ref=None, LR 5e-7). Xem [[plan-batch-4.md]], runbook-batch-4-exp29-38.md.
