# Plan — Batch 5 (continue-train từ 0.86 + weight-space + optimizer), cho Codex

> **Audience: Codex.** Mỗi task self-contained: mục tiêu / file phải tạo-sửa (đường dẫn thật) /
> code tham khảo Ở ĐÂU & port THẾ NÀO / line-level edits trong marker / cách validate / rollback.
> Ý tưởng gốc: [batch-5.md](batch-5.md). Làm đúng thứ tự PHẦN 2 (Phase 1→4).
>
> **3 luật bất biến (từ kết quả batch-1..4, [tracker/leaderboard.md](../../tracker/leaderboard.md)):**
> 1. **Regime = continue-train từ 0.86** (`RESET_WEIGHTS=False`), KHÔNG from-scratch, KHÔNG continue-on-mix.
> 2. **Giữ corpus nguyên shape + order gốc**: dùng nhánh per-id + `TRAIN_ORDER_PATH`, `SHUFFLE_DATASET=False`.
>    KHÔNG dùng nhánh `KAGGLE_JSONL`, KHÔNG negatives, KHÔNG self-gen RFT.
> 3. **Liều nhẹ**: `LEARNING_RATE ≤ 1e-5`, ≤ 1 epoch. Gate 5-nhóm-mạnh không tụt > 0.5pp.
>
> **Deploy: CHỈ Kaggle, KHÔNG Modal.** Mỗi exp = 1 Kaggle notebook self-contained. Mọi code tham
> khảo trong `refs/` phải **INLINE vào file exp** (Kaggle không có `refs/`).

---

## PHẦN 0 — Giải phẫu base notebook (line-ref đã verify)

File gốc: [Continuer_Nemotron_Notebook.py](../../Continuer_Nemotron_Notebook.py). Mọi exp là **bản copy**
của file này, sửa bọc trong `# >>> EXP<N> START/END`.

| Vùng | Dòng | Nội dung |
|---|---|---|
| Config block | [6-40](../../Continuer_Nemotron_Notebook.py#L6-L40) | `LORA_RANK=32`, `NUM_EPOCHS=1.0`, `NUM_STEPS=None`, `LEARNING_RATE=2e-4`, `RESET_WEIGHTS`, `IN_PROJ_ONLY`, `MOE_TIE_WEIGHTS`, `SHUFFLE_DATASET=False`, `TARGET_MODULES` |
| `RESET_WEIGHTS` | [17-19](../../Continuer_Nemotron_Notebook.py#L17-L19) | =True (from-scratch). **Batch-5 đổi → False.** |
| `LEARNING_RATE` | [16](../../Continuer_Nemotron_Notebook.py#L16) | =2e-4 (from-scratch). **Batch-5 đổi → 1e-5.** |
| Kaggle CORPUS per-id + order | [115-116](../../Continuer_Nemotron_Notebook.py#L115-L116) | `CORPUS_PATH=.../sft/04-08-16-14/tokens`, `TRAIN_ORDER_PATH=.../logprobs/index.jsonl` (corpus GỐC + order) |
| Kaggle adapter 0.86 | [118-122](../../Continuer_Nemotron_Notebook.py#L118-L122) | `ADAPTER_SRC="/kaggle/input/datasets/ngoczhu/nemotron-086-adapter"` (đây là adapter 0.86) |
| Reader nhánh JSONL (mix) | [169-186](../../Continuer_Nemotron_Notebook.py#L169-L186) | **KHÔNG dùng trong batch-5** (mix → erosion) |
| Reader nhánh per-id (gốc) | [187-236](../../Continuer_Nemotron_Notebook.py#L187-L236) | corpus gốc theo `TRAIN_ORDER_PATH` — **dùng cái này** |
| Load adapter (continue) | [430-471](../../Continuer_Nemotron_Notebook.py#L430-L471) | nếu `RESET_WEIGHTS=False` → `load_peft_weights(ADAPTER_SRC)` nạp 0.86 |
| Freeze `IN_PROJ_ONLY` | [473-481](../../Continuer_Nemotron_Notebook.py#L473-L481) | freeze mọi LoRA trừ `.in_proj.` |
| MoE `_tie_grads` | [510-540](../../Continuer_Nemotron_Notebook.py#L510-L540) | sum grad qua 128 expert; gọi ở [686](../../Continuer_Nemotron_Notebook.py#L686) |
| Loss per-token + weight_sum | [650-662](../../Continuer_Nemotron_Notebook.py#L650-L662) | `weighted_loss=ce*weights`; `loss=loss_sum/weight_sum`; `(loss/n_accum).backward()` |
| Optimizer (AdamW) | [675-682](../../Continuer_Nemotron_Notebook.py#L675-L682) | `AdamW(betas=(0.9,0.95),eps=1e-8,wd=0)` tạo lazy ở step 1 |
| LR schedule | [683-685](../../Continuer_Nemotron_Notebook.py#L683-L685) | `lr=LEARNING_RATE*(1-step/num_steps)` — linear decay, KHÔNG warmup |
| Grad clip | [687-689](../../Continuer_Nemotron_Notebook.py#L687-L689) | `max_norm=1e9` (≈ TẮT clip) |
| Save + rename lm_head | [690-705](../../Continuer_Nemotron_Notebook.py#L690-L705) | `save_pretrained` → rename `lm_head.`→`backbone.lm_head.` |
| Zip submission | [716-725](../../Continuer_Nemotron_Notebook.py#L716-L725) | `ZIP_DEFLATED` → `submission.zip` |
| Trigger Kaggle | [857-858](../../Continuer_Nemotron_Notebook.py#L857-L858) | `if IS_KAGGLE: run_training()` |

**Refs đã clone cho batch-5:** [refs/muon/muon.py](../../refs/muon/muon.py) (D11),
[refs/lion-pytorch/lion_pytorch/lion_pytorch.py](../../refs/lion-pytorch/lion_pytorch/lion_pytorch.py)
(fallback), [soup_adapters.py](../../soup_adapters.py) (D7/D8), [exp16.py](../../exp16.py) (khung KL cho D9).

---

## PHẦN 1 — Clone base → exp + set continue-train-từ-0.86 (SHARED, mọi train-exp dùng)

### 1A. Tạo file exp
```bash
cp Continuer_Nemotron_Notebook.py exp<N>.py
```
Thêm header comment ngay sau dòng 1 (`# # Nemotron finetuning pipeline`), bọc marker:
```python
# >>> EXP<N> START
# EXP<N> — <tên idea> (Batch-5 D<k>)
# Ref: <file refs/... hoặc code path> | Knob: <knob=val> | Rollback: <cách tắt>
# >>> EXP<N> END
```

### 1B. SHARED continue-train config (BẮT BUỘC cho mọi train-exp D5/D9/D10/D11/D1/D3/D4)
Sửa config block [6-25](../../Continuer_Nemotron_Notebook.py#L6-L25), bọc trong `# >>> EXP<N>_CONT START/END`:
```python
# >>> EXP<N>_CONT START   (continue-train từ 0.86 — luật 1+3 batch-5)
NUM_EPOCHS = 1.0
NUM_STEPS = None
LEARNING_RATE = 1e-5          # <- từ 2e-4; liều nhẹ continue-train
RESET_WEIGHTS = False         # <- từ True; NẠP adapter 0.86 thay vì fresh init
SHUFFLE_DATASET = False       # <- giữ curated order (đã là False ở base, giữ nguyên)
# >>> EXP<N>_CONT END
```
**KHÔNG đụng** `CORPUS_PATH`/`TRAIN_ORDER_PATH` ([115-116](../../Continuer_Nemotron_Notebook.py#L115-L116))
và **KHÔNG** vào nhánh `KAGGLE_JSONL` — để reader chạy nhánh per-id gốc ([187-236](../../Continuer_Nemotron_Notebook.py#L187-L236)).
`ADAPTER_SRC` ([121](../../Continuer_Nemotron_Notebook.py#L121)) giữ nguyên `ngoczhu/nemotron-086-adapter`.

### 1C. Sanity assert (thêm 1 lần, phát hiện sai regime sớm)
Thêm ngay sau config block:
```python
# >>> EXP<N>_GUARD START
assert RESET_WEIGHTS is False, "batch-5 phải continue-train từ 0.86"
assert LEARNING_RATE <= 1e-5, "batch-5 liều nhẹ"
assert SHUFFLE_DATASET is False, "giữ curated order"
# >>> EXP<N>_GUARD END
```

---

## PHẦN 1.5 — Deploy Kaggle (BẮT BUỘC — KHÔNG Modal)

Mỗi exp = 1 Kaggle notebook (GPU on). **Attach các dataset/notebook sau** (y như base đang hardcode):
- Model: `metric/nemotron-3-nano-30b-a3b-bf16` (kagglehub, [123-125](../../Continuer_Nemotron_Notebook.py#L123-L125)).
- Wheels: `mayukh18/nemotron-packages` + `llkh0a/rtx-wheels` ([59-90](../../Continuer_Nemotron_Notebook.py#L59-L90)).
- **Corpus GỐC**: snapshot `huikang/huikang-nemotron-repository-snapshot` ([115-116](../../Continuer_Nemotron_Notebook.py#L115-L116)).
- **Adapter 0.86**: dataset `ngoczhu/nemotron-086-adapter` ([121](../../Continuer_Nemotron_Notebook.py#L121)) — **PHẢI attach** vì `RESET_WEIGHTS=False` nạp từ đây.
- Train.csv competition ([117](../../Continuer_Nemotron_Notebook.py#L117)).

Run notebook → tự ghi `submission.zip` ([716-725](../../Continuer_Nemotron_Notebook.py#L716-L725)) → submit.
- Phase-3 (D1/D3/D4 đổi corpus) cần upload corpus mới làm dataset — xem từng task.
- Phase-1 (soup) KHÔNG cần GPU notebook — chạy local `soup_adapters.py` rồi đóng `submission.zip` thủ công (xem D8/D7).

---

## PHẦN 2 — Tasks theo thứ tự ưu tiên (Phase 1→4)

Map idea → exp:

| Idea | exp / công cụ | Phase | Đổi corpus? |
|---|---|---|---|
| D8 soup 2×0.86 | `soup_adapters.py` (no train) | 1 | không |
| D7 WiSE-FT nội suy | `soup_adapters.py` mở rộng (no train) | 1 | không |
| D5 EMA+warmup+clip+accum-fix | **exp40** | 2 | không |
| D11 Muon | **exp41** | 2 | không |
| D9 anchored-L2→0.86 | **exp42** | 2 | không |
| D10 module-localized | **exp43** | 2 | không |
| D1 bit-shorten | data-gen + **exp44** | 3 | CÓ |
| D2 keep-boxed-tail | **exp45** (+corpus build) | 3 | một phần |
| ~~D3 cryptarithm coverage~~ | ~~exp46~~ **BỎ (no-op)** | 3 | — |
| D4 quality gate | data-gen + **exp47** | 3 | CÓ |
| D6 guess CSP | data-gen + **exp48** | 4 | CÓ |

---

### PHASE 1 — Weight-space (no GPU train, làm TRƯỚC, rẻ nhất)

#### D8 — Soup hai checkpoint 0.86 (baseline + exp21)
- **Code có sẵn:** [soup_adapters.py](../../soup_adapters.py) — average đều, assert key khớp.
- **Bước:**
  1. Lấy 2 thư mục adapter đã giải nén: `θ_base` (adapter 0.86 hiện dùng = `ngoczhu/nemotron-086-adapter`)
     và `θ_exp21` (giải nén `submission.zip` của exp21). Mỗi dir phải có `adapter_model.safetensors` + `adapter_config.json`.
  2. `python soup_adapters.py <θ_base_dir> <θ_exp21_dir> <out_dir>`.
  3. Đóng gói `out_dir` thành `submission.zip` (chỉ các file `adapter*`), submit.
- **Validate:** script in "souped 2 adapters"; mở `out_dir/adapter_model.safetensors` key-count = bản gốc.
- **Falsify:** điểm ≤ 0.86 → soup vô ích, bỏ.
- **Rollback:** không có side-effect (chỉ tạo file mới).

#### D7 — WiSE-FT nội suy θ=(1−α)·θ_0.86 + α·θ_lever
- **Sửa file:** thêm CLI weighted vào [soup_adapters.py](../../soup_adapters.py) (giữ hàm cũ). Code tham
  khảo logic average ở chính file đó ([dòng `avg = sum(...)/len`](../../soup_adapters.py)). Đổi thành:
  ```python
  # >>> EXP_WISEFT START
  # usage: python soup_adapters.py --alpha 0.2 θ0_dir θ1_dir out_dir
  # θ = (1-alpha)*θ0 + alpha*θ1 ; θ0 = 0.86 anchor, θ1 = lever (vd exp35/exp38)
  # >>> EXP_WISEFT END
  ```
  Khi có `--alpha`: `avg = {k: (1-a)*t0[k].float() + a*t1[k].float() for k in keys}`.
- **Bước:** scan `α ∈ {0.1, 0.2, 0.3, 0.5}` với `θ1` = adapter của exp35 (0.74, best lever) → 4 submission.
- **Falsify:** mọi α ≤ 0.86 → nội suy chết. Nếu có α > 0.86 → lever có coverage thật (justify D1).
- **Rollback:** giữ hàm `main()` cũ nguyên vẹn, chỉ thêm nhánh `--alpha`.

---

### PHASE 2 — Continue-train, KHÔNG đổi corpus (exp40–43)

> Tất cả dùng SHARED config PHẦN 1B/1C. Khác biệt duy nhất nằm trong marker riêng.

#### exp40 — D5: EMA + warmup + grad-clip thật + fix grad-accum
Bốn sửa nhỏ độc lập, mỗi cái 1 marker con để ablate được:

1. **Grad-clip thật** — sửa [687-689](../../Continuer_Nemotron_Notebook.py#L687-L689):
   ```python
   # >>> EXP40_CLIP START
   grad_norm = torch.nn.utils.clip_grad_norm_(
       [p for p in model.parameters() if p.requires_grad], max_norm=1.0)  # từ 1e9
   # >>> EXP40_CLIP END
   ```
2. **Warmup + giữ floor** — sửa [683](../../Continuer_Nemotron_Notebook.py#L683):
   ```python
   # >>> EXP40_WARM START
   warmup = max(1, int(0.03 * num_steps))
   if step < warmup:
       lr = LEARNING_RATE * (step + 1) / warmup
   else:
       prog = (step - warmup) / max(1, num_steps - warmup)
       lr = LEARNING_RATE * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))  # cosine→10% floor
   # >>> EXP40_WARM END
   ```
3. **Fix grad-accum normalization** — gom global thay vì mean-of-means. Sửa vùng [650-662](../../Continuer_Nemotron_Notebook.py#L650-L662):
   tích luỹ `loss_sum_t`/`weight_sum_t` qua các micro-batch, **backward 1 lần ở cuối batch** với
   `total_loss_sum/total_weight_sum`. (Nếu rủi ro cao, để LẠI làm ablation riêng — đánh dấu `EXP40_ACCUM` và có thể tắt.)
4. **EMA ship** — thêm sau khi tạo optimizer + trong loop + lúc save:
   ```python
   # >>> EXP40_EMA START
   EMA_DECAY = 0.999
   ema = {n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad}
   # ... sau optimizer.step():
   with torch.no_grad():
       for n, p in model.named_parameters():
           if p.requires_grad: ema[n].mul_(EMA_DECAY).add_(p.detach().float(), alpha=1-EMA_DECAY)
   # ... NGAY TRƯỚC save_pretrained (dòng 698): nạp EMA vào model
   with torch.no_grad():
       for n, p in model.named_parameters():
           if p.requires_grad and n in ema: p.copy_(ema[n].to(p.dtype))
   # >>> EXP40_EMA END
   ```
- **Validate:** log `grad_norm` < 1.0 sau clip; LR ramp lên rồi cosine; cuối in "shipped EMA".
- **Falsify:** exp40 ≥ 0.86 (giữ baseline) và ≥ continue-train-trần. Nếu < → tắt từng marker để định thủ phạm.
- **Rollback:** mỗi marker tắt độc lập (clip→1e9, bỏ warmup, bỏ EMA).

#### exp41 — D11: Muon cho LoRA 2D + AuxAdam cho phần còn lại
- **Port từ:** [refs/muon/muon.py](../../refs/muon/muon.py) — **INLINE** vào exp41 (Kaggle không có refs/).
  Copy 3 thứ: `zeropower_via_newtonschulz5` ([5-31](../../refs/muon/muon.py#L5-L31)), `muon_update`
  ([34-41](../../refs/muon/muon.py#L34-L41)), class `SingleDeviceMuonWithAuxAdam` ([228-286](../../refs/muon/muon.py#L228-L286)).
  **BỎ** `import torch.distributed as dist` (chỉ SingleDevice, không cần).
- **Thay optimizer** [675-682](../../Continuer_Nemotron_Notebook.py#L675-L682):
  ```python
  # >>> EXP41 START
  muon_params, adam_params = [], []
  for n, p in model.named_parameters():
      if not p.requires_grad: continue
      is_lora2d = (".lora_" in n) and (".experts." not in n) and p.ndim == 2  # LoRA 2D sạch
      (muon_params if is_lora2d else adam_params).append(p)   # expert-LoRA + router/bias → Adam
  optimizer = SingleDeviceMuonWithAuxAdam([
      dict(params=muon_params, lr=MUON_LR, momentum=0.95, weight_decay=0.0, use_muon=True),
      dict(params=adam_params, lr=LEARNING_RATE, betas=(0.9,0.95), eps=1e-8, weight_decay=0.0, use_muon=False),
  ])
  # >>> EXP41 END
  ```
  với `MUON_LR` đặt ở config (bắt đầu **0.5e-3**, KHÔNG dùng 1e-5 — Muon ở đơn vị spectral-norm/update).
- **2 cảnh báo (đã đọc ref):**
  - `MUON_LR` scale khác AdamW hoàn toàn ([muon.py:61](../../refs/muon/muon.py#L61)) → nhớ exp4 nổ vì LR sai; bắt đầu nhỏ.
  - **expert-LoRA để ở group Adam** (loại bằng `.experts. not in n`) → tránh tương tác `_tie_grads`/orthogonalize.
- **Lưu ý LR schedule:** [683-685](../../Continuer_Nemotron_Notebook.py#L683-L685) hiện set `pg["lr"]` cho mọi
  group theo `LEARNING_RATE`. Phải sửa để **chỉ scale theo group**: muon-group scale từ `MUON_LR`, adam-group từ `LEARNING_RATE`.
- **Falsify:** giữ 5-nhóm-mạnh trong 0.5pp và ≥0.86. Bất ổn kiểu exp4 → giảm `MUON_LR` ×0.3 hoặc revert.
- **Rollback:** header ghi "set về AdamW gốc" — xoá marker EXP41, khôi phục [675-682].

#### exp42 — D9: anchored-L2 về θ_0.86 (chống drift)
- **Tham khảo khung KL:** [exp16.py:615-643](../../exp16.py#L615-L643) — NHƯNG exp16 anchor về **base model**
  (sai đích). D9 anchor về **adapter 0.86**, và dùng **L2 trên weight** (rẻ, KHÔNG forward thêm) thay vì KL.
- **Bước:**
  1. Sau khi nạp adapter 0.86 ([467](../../Continuer_Nemotron_Notebook.py#L467)), snapshot reference:
     ```python
     # >>> EXP42 START
     ANCHOR_LAMBDA = 1e-3
     theta_ref = {n: p.detach().float().clone() for n, p in model.named_parameters() if p.requires_grad}
     # >>> EXP42 END
     ```
  2. NGAY TRƯỚC `optimizer.step()` ([690](../../Continuer_Nemotron_Notebook.py#L690)), thêm gradient kéo về ref
     (decoupled L2 toward θ_ref, sau `_tie_grads` để không phá tie):
     ```python
     # >>> EXP42 START
     with torch.no_grad():
         for n, p in model.named_parameters():
             if p.requires_grad and p.grad is not None and n in theta_ref:
                 p.grad.add_(ANCHOR_LAMBDA * (p.detach().float() - theta_ref[n]).to(p.grad.dtype))
     # >>> EXP42 END
     ```
- **Falsify:** exp42 trên corpus gốc giữ ≥0.86 (anchor không được làm tụt); khi ghép lên D1/D3 (đổi corpus)
  thì giữ 5-nhóm-mạnh tốt hơn bản không-anchor. Nếu λ làm cứng quá (không học được delta) → giảm λ.
- **Rollback:** `ANCHOR_LAMBDA=0`.

#### exp43 — D10: continue-train khu trú module
- **Dùng cơ chế sẵn:** `IN_PROJ_ONLY` ([20](../../Continuer_Nemotron_Notebook.py#L20), freeze [473-481](../../Continuer_Nemotron_Notebook.py#L473-L481)).
  Tổng quát hoá thành whitelist module:
  ```python
  # >>> EXP43 START
  TRAIN_ONLY_MODULES = ("in_proj", "out_proj")   # ví dụ: khu trú Mamba mixer cho bit
  for name, param in model.named_parameters():
      if param.requires_grad and not any(m in name for m in TRAIN_ONLY_MODULES):
          param.requires_grad = False
  # >>> EXP43 END
  ```
  Đặt block này NGAY SAU block `IN_PROJ_ONLY` ([481](../../Continuer_Nemotron_Notebook.py#L481)).
- **Falsify:** so với exp42, nếu khu-trú giữ 5-nhóm-mạnh tốt hơn mà vẫn cải thiện target → giữ; không thì bỏ.
- **Rollback:** `TRAIN_ONLY_MODULES = ()` (train tất cả).

---

### PHASE 3 — At-source corpus regen + continue-train (exp44–47)

> ⚠️ Phase này ĐỔI corpus ⇒ rủi ro cao nhất (xem L1/L2). BẮT BUỘC: corpus mới = **full corpus gốc +
> delta override**, giữ order gốc; upload làm Kaggle dataset; train với SHARED config + **ghép exp42 anchor**.

#### Data-gen chung (chạy trong `nemotron-master/`, `uv run`)
Pipeline: [reasoning.py](../../nemotron-master/reasoning.py) → [augmentation.py](../../nemotron-master/augmentation.py)
→ [corpus.py](../../nemotron-master/corpus.py). Đọc [nemotron-master/CLAUDE.md](../../nemotron-master/CLAUDE.md) trước.
Token-format nguồn DUY NHẤT: [corpus.py](../../nemotron-master/corpus.py) — completion
`"{reasoning}\n</think>\n\\boxed{{answer}}<|im_end|>"`. **KHÔNG đổi format.**

#### D1 / exp44 — Token-efficient bit_manipulation solver
- **Sửa:** [nemotron-master/reasoners/bit_manipulation.py](../../nemotron-master/reasoners/bit_manipulation.py)
  — rút **prose lặp**, giữ NGUYÊN mọi bước cột-bit (khác token-pruning của exp33). Tham khảo cấu trúc CoT
  hiện tại trong chính file đó; tham khảo ý "nén có verifier-gate" ở [offline/compress_traces.py](../../offline/compress_traces.py)
  + [refs/r1-compress](../../refs/r1-compress)/[refs/cot-valve](../../refs/cot-valve) (chỉ ĐỌC ý, port thủ công).
- **Mục tiêu đo được:** p95 completion bit < ~6000 (từ 7243); mọi trace mới verify đúng bằng solver.
- **Regenerate:** chạy reasoning.py→corpus.py → corpus full mới (bit ngắn), **giữ order gốc** (emit theo
  cùng training order). Upload làm Kaggle dataset `nemotron-corpus-bitshort`.
- **exp44:** copy base, SHARED config, trỏ corpus per-id sang dataset mới (giữ nhánh per-id + order),
  ghép EXP42 anchor. Marker `# >>> EXP44`.
- **Falsify:** held-out bit dài: p95 completion ↓ VÀ solve-rate bit ↑; 5-nhóm-mạnh không tụt. solve-rate bit ↓ → revert.

#### D2 / exp45 — Keep-boxed-tail / skip-overflow lúc build/train
- **Đếm trước (prerequisite):** số example > 8192 token trong corpus gốc. ~0 → BỎ D2.
- **Sửa reader per-id** [219-221](../../Continuer_Nemotron_Notebook.py#L219-L221): thay cắt-phải bằng
  **giữ đuôi boxed** (cắt giữa reasoning, luôn chừa `\boxed...<|im_end|>`) hoặc **skip** trace tràn.
  Marker `# >>> EXP45`.
- **Falsify:** nếu count>8192 đáng kể → exp45 ≥ baseline; nếu ~0 → không chạy.

#### D3 / exp46 — Cryptarithm_deduce coverage — ❌ ĐÃ BỎ (2026-06-10)
- **Kết quả Phase 3:** solver skip-1 đã chạy (reasoning+corpus regen fresh) nhưng sinh trace cryptarithm
  **GIỐNG HỆT** cũ (verified: per-id fresh==bitshort 300/300, bitshort==S_solver 823/823). D3 **no-op
  ở mức corpus** → corpus_crypto == corpus_bitshort, exp46 ≡ exp44. **Đã xoá exp46.py + corpus_crypto.jsonl.**
- Muốn D3 có giá trị: phải điều tra vì sao skip-1 không giải thêm (memory nói +17 nhưng không hiện thực
  hoá). Lever nhỏ (~1pp), ưu tiên thấp — chỉ làm sau khi exp44/47 cho kết quả.

#### D4 / exp47 — Quality gate (bỏ trace SAI)
- **Sửa:** [nemotron-master/corpus.py](../../nemotron-master/corpus.py) — loại trace `rule_unknown`/
  `hypothesis_formed` (chưa verify), **giữ TOÀN BỘ `rule_found`** (khác exp6: không cắt bài-dễ-đúng).
  Xem [research/data_status.md](../data_status.md).
- **Regenerate + upload**; exp47 trỏ tới.
- **Falsify:** giữ 5-nhóm-mạnh; nếu count-bỏ nhỏ hoặc điểm không đổi → neutral.

---

### PHASE 4 — Hedge (chỉ khi Phase 1–3 cạn)

#### D6 / exp48 — Guess categories bằng constraint-search
- Đã flag dead-end ([[cryptarithm-unsolved-levers]]), leverage 0.2–2%. Solver inductive/CSP trong
  [nemotron-master/reasoners/](../../nemotron-master/reasoners/), tham khảo [refs/python-constraint](../../refs/python-constraint).
  KHÔNG ưu tiên; chỉ làm nếu mọi thứ trên cạn.

---

## PHẦN 3 — Run matrix & gating

| Thứ tự | exp/tool | Bet | Chi phí | Đổi corpus |
|---|---|---|---|---|
| 1 | D8 soup | 2×0.86 average vượt 0.86 | ~0 (CPU) | không |
| 2 | D7 WiSE-FT α-scan | lever có coverage thật? | ~0 + 4 infer | không |
| 3 | exp40 (D5) | EMA/clip/warmup free win | 1 train | không |
| 4 | exp41 (D11 Muon) | optimizer-side (L4) | 1 train | không |
| 5 | exp42 (D9 anchor) | regularizer chống drift | 1 train | không |
| 6 | exp43 (D10) | khu trú giảm drift | 1 train | không |
| 7 | exp44+exp42 (D1) | bit truncation (đòn bẩy ~3.6pp) | data-gen+train | CÓ |
| 8 | exp45 (D2) | keep-boxed-tail | train | một phần |
| ~~9~~ | ~~exp46 (D3)~~ **BỎ — no-op (corpus_crypto==bitshort)** | — | — |
| 10 | exp47 (D4) | quality gate | data-gen+train | CÓ |
| 11 | exp48 (D6) | guess hedge | data-gen+train | CÓ |

**Gate chung mỗi run:** pass@1 của 5 nhóm mạnh (numeral/unit/gravity/cipher/eq_deduce) KHÔNG tụt > 0.5pp.
Tụt ⇒ giảm liều (LR/step) hoặc rollback delta/marker. Mỗi exp cho điểm → cập nhật
[tracker/leaderboard.md](../../tracker/leaderboard.md) + copy `tracker/rounds/round_template.md`.

**CẤM (từ kết quả batch-1..4):** continue-on-mix; nhánh `KAGGLE_JSONL`; negatives `sign=−1`; self-gen RFT;
`SHUFFLE=True` trên corpus; from-scratch (`RESET_WEIGHTS=True`); đổi PROMPT_SUFFIX/token-format; import
`refs/` lúc runtime (phải INLINE vào exp); Modal (`modal run`).

**Definition of done batch-5:** Phase-1 (D8/D7) submit xong + ghi điểm; ≥1 exp Phase-2/3 > 0.86 HOẶC
kết luận rõ (vd L5: coverage category-hiếm không dời điểm); mọi exp ghi tracker đầy đủ.
