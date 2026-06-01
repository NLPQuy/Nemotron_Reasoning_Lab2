# Implementation Plan — Batch 2 (exp11…exp19)

Mục tiêu: hiện thực hóa 9 ý tưởng trong [batch-2.md](batch-2.md) để đẩy điểm leaderboard **0.86 → 0.88+**. Mỗi ý tưởng là **một file riêng** `exp<N>.py`, **copy nguyên xi** từ [Continuer_Nemotron_Notebook.py](../../Continuer_Nemotron_Notebook.py) (thư mục gốc repo), rồi chỉ sửa đúng vùng được đánh dấu. Một số idea (exp13, exp15, exp19) có phần chính nằm **upstream** trong `nemotron-master/` (data-gen) — file `exp<N>.py` chỉ train trên corpus đã regenerate.

> File plan này nằm ở `research/ideation/`; file gốc và các `exp<N>.py` nằm ở **thư mục gốc repo** (`../../`). exp11 ↔ Idea 1, …, exp19 ↔ Idea 9 theo đúng thứ tự ranked trong batch-2.md.

> Tất cả thay đổi đều **training-time / data-time**. Inference vẫn 1 lần greedy, vLLM-loadable, rank ≤ 32. Ideas 8/9 chỉ dùng sampling **khi sinh data offline**, không đụng lúc nộp bài.

> ⚠️ **Cổng chặn lặp lại xuyên suốt:** DoRA (exp11) và PiSSA (exp14) **đổi format adapter** → phải qua **vLLM load-test** trước khi tin là deliverable. 7 exp còn lại xuất ra adapter LoRA vanilla chuẩn.

---

## 0. Setup chung (làm 1 lần)

```bash
cd /media/mlinh/DATA/projects/ML/Nemotron_Reasoning_Lab2
for i in $(seq 11 19); do cp Continuer_Nemotron_Notebook.py exp${i}.py; done
```

**Quy ước cho mọi exp file (giống batch-1):**
1. Ngay đầu file (sau dòng 2 `# # Nemotron finetuning pipeline`) thêm banner:
   ```python
   # ============================================================
   # EXP<N> — <title>  (Batch-2 Idea <K>)
   # Base: Continuer_Nemotron_Notebook.py (unmodified except marked blocks)
   # Change: <1 dòng>   |   Knob: <KNOB>=<value>   |   Rollback: set <KNOB> mặc định
   # ============================================================
   ```
2. Mỗi vùng sửa bọc bằng `# >>> EXP<N> START` … `# <<< EXP<N> END` (đúng style đang dùng ở exp1–10).
3. Knob mới đặt trong **block config dòng 6–38** cùng các knob hiện có.
4. Chỉ đổi **một** cơ chế / file. Không gộp nhiều idea (trừ run "combo" ở mục 11).

**Tham chiếu nhanh codepath trong file gốc (đã verify theo bản 804 dòng hiện tại):**
| Vùng | Dòng | Vai trò |
|------|------|---------|
| Config knobs | 6–38 | `LORA_RANK/ALPHA/DROPOUT`, `NUM_STEPS`, `BATCH_SIZE`, `MICRO_BATCH_SIZE`, `LEARNING_RATE`, `RESET_WEIGHTS`(=True), `MOE_TIE_WEIGHTS`, `SHUFFLE_DATASET`, `TARGET_MODULES` |
| `run_training()` | 88 | entry; `linear_cross_entropy` import 101, `LoraConfig` import 102 |
| `MODEL_PATH` | 120 (Kaggle) / 124 (else) | đường model |
| Corpus truncate + append (Kaggle) | 197–199 / 202 | `tokens[:MAX_SEQ_LEN]`, `examples.append({tokens,targets,weights})` |
| Corpus truncate + append (Modal) | 216–218 / 221 | tương tự |
| Tokenizer + model load | 253–263 | `FastLanguageModel.from_pretrained` → `model, tokenizer` |
| `get_peft_model` | 268–277 | `r, target_modules, lora_alpha, lora_dropout, …` |
| Mamba fast-path patch | 280–289 | `is_fast_path_available=True` |
| lm_head LoRA thủ công | 291–307 | `_cfg = LoraConfig(...)` dòng 297 |
| Cast LoRA → fp32 | 309–312 | `if ".lora_" in name` |
| CCE forward + per-token CE | 385–392 | `model._cached_per_token_ce` |
| Adapter load / RESET | 398–401 | `if RESET_WEIGHTS: …` |
| MoE tying | 458–510 | `moe_tied_params`, `_tie_param_init`(478), `_tie_grads`(485/507), `_tie_param_init()`(504) |
| Data order | 517–523 | `indices`, `SHUFFLE_DATASET` |
| Train loop | 546–644 | batch slice 549; micro-batch 560–620; `per_token_ce*padded_weights` 600; `backward` 607; `optimizer` 622–629; `lr=LR*(1-step/num_steps)` 630; `_tie_grads()` 633; `step` 637 |

---

## exp11 — DoRA: weight-decomposed LoRA  *(Idea 1, P3, T1)* — 🏆 top-1

**Hypothesis:** ở rank tối đa 32, LoRA gộp magnitude+direction → không biểu diễn được update geometry của full-FT. DoRA tách magnitude (vector học riêng) khỏi direction (LoRA) → khai thác thêm capacity, **không thêm inference overhead** (merge lại khi nộp).

**Edit (1 cờ × 2 chỗ):**
1. Knob (block config): `USE_DORA = True`.
2. `get_peft_model(...)` (sau `lora_dropout=LORA_DROPOUT,` dòng 273):
   ```python
   # >>> EXP11 START
   use_dora=USE_DORA,
   # <<< EXP11 END
   ```
3. lm_head `LoraConfig` (dòng 297) — phải mirror nếu không lm_head bị bỏ lại dạng LoRA thường:
   ```python
   # >>> EXP11 START
   _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                     lora_dropout=LORA_DROPOUT, use_dora=USE_DORA)
   # <<< EXP11 END
   ```
4. (Khuyến nghị) re-tune LR ngắn (DoRA đổi step hiệu dụng): thử `LEARNING_RATE ∈ {1e-4, 2e-4}`.

**Lưu ý plumbing:** nếu `FastLanguageModel.get_peft_model` không chuyển tiếp kwarg `use_dora` (Unsloth có thể lọc kwargs), fallback: dựng `LoraConfig(..., use_dora=True)` rồi `get_peft_model(model, peft_config)` của PEFT thuần, hoặc patch trực tiếp. Test ngay sau khi wrap: `print([n for n,_ in model.named_parameters() if "lora_magnitude" in n][:3])` để xác nhận DoRA đã bật.

**Validate rẻ:** (a) **vLLM load-test** adapter đã lưu (gate cứng); nếu vLLM không nhận magnitude tensor → fallback merge→extract vanilla LoRA. (b) DoRA vs LoRA cùng LR đã re-tune trên slice 200 bài, phải ≥ baseline. **Rollback:** `USE_DORA=False`.

---

## exp12 — NEFTune noisy-embedding regularization  *(Idea 2, P4, T1)* — ⚡ quick win (có caveat)

**Hypothesis:** adapter rank-32 trên corpus synthetic nhỏ dễ overfit bề mặt trace. Thêm noise vào embedding khi train = regularizer gần-free. **Caveat (devil's-advocate):** gain NEFTune tập trung ở conversational; reasoning thường chỉ "stable" → coi như probe rẻ, không kỳ vọng chắc.

**Edit:**
1. Knob: `NEFTUNE_ALPHA = 10.0` (0 = tắt; sweep {0, 5, 10, 15}).
2. Đăng ký forward hook trên input-embedding, **chỉ khi `training`**. Chèn sau `FastLanguageModel.for_training(model)` (dòng 278):
   ```python
   # >>> EXP12 START
   if NEFTUNE_ALPHA > 0:
       _emb = model.get_input_embeddings()
       def _neftune_hook(_mod, _inp, out):
           if not _mod.training:
               return out
           L, d = out.size(-2), out.size(-1)
           mag = NEFTUNE_ALPHA / (float(L * d) ** 0.5)
           return out + torch.empty_like(out).uniform_(-mag, mag)
       _emb.register_forward_hook(_neftune_hook)
       print(f"EXP12 NEFTune hook on {type(_emb).__name__}, alpha={NEFTUNE_ALPHA}")
   # <<< EXP12 END
   ```

**Không có rủi ro inference:** adapter lưu ra không chứa noise; vLLM nộp bài chạy bình thường (noise chỉ tồn tại lúc `model.training`).

**Validate rẻ:** train `α=10` vs `α=0` trên slice; nếu exact-match không +≥0.5pp ở mọi α → drop (rẻ nên falsify nhanh). **Rollback:** `NEFTUNE_ALPHA=0`.

---

## exp13 — Self-verification "check-then-box" traces  *(Idea 3, P6, T2)* — phần chính UPSTREAM

**Hypothesis:** dạy trace **verify-trước-khi-box** (thay số đáp án vào ràng buộc gốc, recompute) → bắt được lỗi off-by-one/sai dấu/nhớ ở greedy. Khác exp1 batch-1 (chỉ làm sạch *label*); đây đổi **nội dung** trace. Khác STaR (không sample model, không thêm bài).

**Edit chính (upstream, `nemotron-master/`):**
1. Trong từng `reasoners/<category>.py`, sau khi solver tính xong đáp án, emit 1–3 dòng verify dùng đại lượng solver **đã có** (không hallucination). Ví dụ category `equation_numeric`:
   ```python
   # cuối hàm solve, trước khi build CoT kết:
   verify = f"Check: substitute x={ans} into the equation → {lhs}={rhs} ✓"
   reasoning = reasoning + "\n" + verify
   ```
2. `corpus.py`: verify nằm **trong** `{reasoning}` (mask 1), trước scaffold `\n</think>\n\\boxed{...}`. Re-tokenize, kiểm tra độ dài (compose với batch-1 idea-3 cap).
3. Chạy lại pipeline: `uv run python3 reasoning.py && uv run python3 corpus.py` → corpus mới.

**Edit trong `exp13.py`:** chỉ banner + (tùy chọn) knob `VERIFY_TRACES_CORPUS = "<path corpus mới>"` để trỏ data load sang corpus đã regenerate (Kaggle dataset / Modal volume mới). Không sửa logic train.

**Validate rẻ:** train verify-traces vs plain trên slice; exact-match nhóm numeric/equation phải +≥0.5pp **và** truncation-rate không tệ đi. **Rollback:** trỏ lại corpus gốc; bỏ verify trong reasoners.

---

## exp14 — PiSSA / LoRA-GA principled init  *(Idea 4, P3, T1)* — 🛡️ safe bet

**Hypothesis:** thay init A~Gaussian/B=0 bằng init theo principal singular vectors (PiSSA) / first-step full-FT gradient (LoRA-GA) → adapter "tiêu" 32 rank vào hướng tín hiệu cao từ step 0, hội tụ tốt hơn cùng `NUM_STEPS`. **Bắt buộc `RESET_WEIGHTS=True`** (đang là mặc định, dòng 16).

**Edit:**
1. Knob: `LORA_INIT = "pissa_niter_4"` (SVD xấp xỉ nhanh; hoặc `"pissa"` cho SVD đầy đủ).
2. `get_peft_model(...)` (sau dòng 273):
   ```python
   # >>> EXP14 START
   init_lora_weights=LORA_INIT,
   # <<< EXP14 END
   ```
3. lm_head `LoraConfig` (dòng 297) mirror: thêm `init_lora_weights=LORA_INIT`.
4. **Export cho vLLM (quan trọng):** PiSSA sửa base (residual) khi train → khi lưu phải convert về LoRA vanilla:
   ```python
   # >>> EXP14 START  (trong khối save adapter)
   model.save_pretrained(SAVE_DIR,
       path_initial_model_for_weight_conversion=PISSA_INIT_DIR)
   # <<< EXP14 END
   ```
   `PISSA_INIT_DIR` = bản init PiSSA lưu **trước khi train** (dùng `model.peft_config[...].init_lora_weights` + `save_pretrained(PISSA_INIT_DIR)` ngay sau wrap). Không có bước này adapter sẽ lệch base lúc vLLM-load.

**Lưu ý plumbing:** nếu Unsloth không nhận `init_lora_weights`, dựng `LoraConfig(..., init_lora_weights="pissa_niter_4")` thủ công. SVD trên expert weights MoE tốn 1 lần init (chấp nhận). LoRA-GA cần 1 batch full-FT-grad pre-pass (biến thể nặng hơn — để dành nếu PiSSA dương).

**Validate rẻ:** (a) **vLLM load-test** adapter đã convert. (b) PiSSA-init vs default-init cùng step/LR trên slice; cần exact-match ≥ +0.5pp **hoặc** converged-loss thấp hơn. **Rollback:** `LORA_INIT="gaussian"` (mặc định PEFT) + bỏ bước convert.

---

## exp15 — Exact-arithmetic scratchpad cho category số  *(Idea 5, P5, T2)* — phần chính UPSTREAM

**Hypothesis:** 1 chữ số nhớ sai = 0 điểm cứng (grader khớp 1e-2 / binary exact). Render mỗi phép tính nhiều chữ số thành **scratchpad từng-chữ-số** (carry-by-carry, có thể viết ngược LSB-first) → greedy *tính* thay vì *đoán*. Đối nghịch exp3 batch-1 (rút ngắn) → phải co-tune, chỉ scratchpad ở bước số học.

**Edit chính (upstream, `nemotron-master/`):**
1. `reasoners/store_types.py`: thêm path render scratchpad cho long-multiplication / long-division (helper đã tồn tại) — bật bằng cờ `scratchpad=True`.
2. Wire vào các `reasoners/` số học (`equation_numeric`, `unit_conversion`, `numeral`, `gravity`). **Gate theo số chữ số** (vd > 3 chữ số mới scratchpad; phép nhỏ để inline).
3. `uv run python3 reasoning.py && uv run python3 corpus.py` → corpus mới.

**Edit trong `exp15.py`:** như exp13 — knob trỏ corpus mới, không sửa train.

**Validate rẻ:** scratchpad vs render gốc trên slice; exact-match category số +≥0.5pp **và** cap-hit-rate 7680 không tăng >2pp. **Rollback:** `scratchpad=False`, corpus gốc.

---

## exp16 — Anchored-SFT KL regularization  *(Idea 6, P4, T1)*

**Hypothesis:** base Nemotron đã reasoning tốt; SFT mạnh trên corpus hẹp làm trôi (drift) năng lực trên mix leaderboard ẩn. Thêm `β·KL(base ‖ adapted)` trên token completion để neo, giữ năng lực ngoài-corpus mà vẫn học format. Đây là bản "guardrail KL" cho cảnh báo λ của batch-1 idea-2.

**Edit (train loop):**
1. Knob: `KL_ANCHOR_BETA = 0.1` (sweep {0, 0.05, 0.1, 0.2}).
2. ⚠️ **Xung đột với path no-logits `linear_cross_entropy`:** KL cần phân phối base. Cách thực dụng — thêm 1 forward **tắt adapter** chỉ lấy logits ở vị trí completion (mask=1), giới hạn memory bằng top-k. Trong micro-batch (sau khi có `per_token_ce`, ~dòng 599), chèn:
   ```python
   # >>> EXP16 START
   if KL_ANCHOR_BETA > 0:
       with torch.no_grad(), model.disable_adapter():
           base_out = model(input_ids=padded_input,
                            attention_mask=attention_mask, use_cache=False)
           base_logits = base_out.logits            # cần path có logits cho lần base này
       pol_logits = model._cached_logits             # xem ghi chú dưới
       m = padded_weights > 0
       kl = torch.nn.functional.kl_div(
           torch.log_softmax(pol_logits[m], -1),
           torch.log_softmax(base_logits[m], -1),
           log_target=True, reduction="batchmean")
       loss = loss + KL_ANCHOR_BETA * kl
   # <<< EXP16 END
   ```
   **Ghi chú:** monkey-patch CCE hiện không trả logits của policy. Hai lựa chọn: (i) với KL bật, cho forward policy chạy path có logits (chấp nhận chậm/tốn hơn cho run này), hoặc (ii) xấp xỉ KL bằng top-k logits (lấy `topk` trên cả base & policy ở vị trí completion). Chọn (ii) nếu OOM.

**Validate rẻ:** `β=0.1` vs `β=0` trên slice; exact-match (đặc biệt trên **category held-out**) phải +≥0.5pp. **Rollback:** `KL_ANCHOR_BETA=0` (về CCE thuần).

---

## exp17 — LoRA seed-soup (weight averaging)  *(Idea 7, P1, T1)* — N-run + script gộp

**Hypothesis:** train N adapter rank-32 khác seed/LR/thứ-tự rồi **trung bình weight** ("model soup") → basin phẳng hơn, generalize tốt hơn run đơn, **0 chi phí inference** (vẫn 1 adapter).

**Edit (`exp17.py`, mỗi run đổi 1 seed):**
1. Knob: `SOUP_SEED = 42` (đổi {42, 1, 7, 13, 99} cho từng run). Sửa `random_state=SOUP_SEED` ở `get_peft_model` (dòng 276) và seed torch/np.
2. Giữ **nguyên** rank / target_modules / format / init-regime giữa các member (không trộn PiSSA-init với gaussian-init).

**Script gộp (mới, `soup_adapters.py`):**
```python
# >>> EXP17 START
import safetensors.torch as st, glob, sys, shutil, os
paths = sys.argv[1:-1]; out = sys.argv[-1]
tensors = [st.load_file(f"{p}/adapter_model.safetensors") for p in paths]
keys = set(tensors[0]); assert all(set(t)==keys for t in tensors), "key mismatch"
avg = {k: sum(t[k].float() for t in tensors) / len(tensors) for k in keys}
os.makedirs(out, exist_ok=True)
st.save_file(avg, f"{out}/adapter_model.safetensors")
shutil.copy(f"{paths[0]}/adapter_config.json", f"{out}/adapter_config.json")
print(f"souped {len(paths)} adapters → {out}")
# <<< EXP17 END
```
Thử **uniform soup** trước, rồi **greedy soup** (thêm member theo thứ tự exact-match slice giảm dần, chỉ giữ nếu cải thiện).

**Validate rẻ:** soup vs member tốt nhất trên slice; nếu không +≥0.3pp → nộp member tốt nhất. **Rollback:** dùng 1 adapter đơn. Lưu ý: feasibility 3/5 vì tốn **N× run**.

---

## exp18 — Offline preference optimization trên verified pairs  *(Idea 8, P12, T2)* — 2-phase

**Hypothesis:** SFT-on-correct (STaR) thiếu tín hiệu **âm**. Tạo cặp (chosen=correct, rejected=incorrect) từ chính adapter, train SimPO (reference-free, length-normalized) → đẩy mass khỏi đúng *các lỗi cận* của model. Khác STaR (chỉ correct) — đây contrastive.

**Phương án ưu tiên:** làm trong `nemotron-master/train_sft.py` (đã có `dro/cispo/ppo`). Sinh data: sample K trace/bài bằng adapter hiện tại (sampling **chỉ offline**), verify bằng `reasoners/`, build `pairs.jsonl`.

**Nếu phải ở single-file `exp18.py`:** SimPO khả thi vì ta **đã có `per_token_ce`** → reward = avg-logprob = `-mean(per_token_ce*mask)/sum(mask)`. 2-phase:
1. **Phase A (offline):** `pref_generate.py` (mới): sample K, verify, ghi `pairs.jsonl` mỗi dòng `{chosen:{tokens,mask}, rejected:{tokens,mask}}` (dedupe, cân bằng category).
2. **Phase B (train):** knob `SIMPO_BETA=2.0`, `SIMPO_GAMMA=0.5`, `PAIRS_PATH`. Thay nội dung micro-batch: forward chosen & rejected, lấy reward từ `per_token_ce`, loss:
   ```python
   # >>> EXP18 START
   r_chosen   = -(ce_c * w_c).sum() / w_c.sum().clamp(min=1)
   r_rejected = -(ce_r * w_r).sum() / w_r.sum().clamp(min=1)
   loss = -torch.nn.functional.logsigmoid(
              SIMPO_BETA * (r_chosen - r_rejected) - SIMPO_GAMMA)
   # (giữ thêm 1 số hạng SFT-on-chosen làm anchor chống collapse)
   # <<< EXP18 END
   ```

**Validate rẻ:** sau 1 vòng, greedy exact-match trên nhóm bài-từng-sai phải +≥1pp, tổng thể không giảm; ≤2 vòng (drift). **Rollback:** `PAIRS_PATH=None` (về CE thuần / STaR).

---

## exp19 — Stream-of-Search backtracking traces  *(Idea 9, P2, T3)* — phần chính UPSTREAM, category-gated

**Hypothesis:** category tổ hợp (`cryptarithm`, `cipher`) cần thử–loại–quay-lui. Trace chỉ-đáp-án không dạy backtrack; serialize **quá trình tìm kiếm** (try → check → fail → backtrack → fix → `\boxed`) dạy model tự sửa giữa chừng. Đối nghịch exp3 batch-1 → **gate chặt theo category**.

**Edit chính (upstream, `nemotron-master/reasoners/`):**
1. Trong solver `cryptarithm`/`cipher`, log các nhánh đã thử & bị prune; serialize thành "stream of search" tuyến tính, kết thúc bằng `\boxed{answer}` đã verify.
2. **Cap số nhánh** để trace nằm sâu dưới 7680 token (compose batch-1 idea-3). Chỉ áp cho category search-structured.
3. `uv run python3 reasoning.py && uv run python3 corpus.py` → corpus mới (chỉ category đích đổi).

**Edit trong `exp19.py`:** knob trỏ corpus mới (như exp13/15), không sửa train.

**Validate rẻ:** SoS-traces (chỉ cryptarithm/cipher) vs concise trên slice; exact-match 2 category đó +≥1pp **và** cap-hit-rate không tăng >3pp. **Rollback:** corpus gốc; bỏ SoS trong reasoners.

---

## 11. Thứ tự chạy đề xuất

> Theo "Next steps" của batch-2.md — bank capacity-win rẻ trước, đẩy big-bet sau.

0. **Prerequisite (không phải idea, làm trước):** phân loại lỗi held-out thành `{format-zero, truncation-zero, arithmetic-slip, method-wrong, search/explore}`. Phép đo này quyết định wave 2: bucket số học/method → exp15/exp13; bucket search → exp19; bucket capacity chung → exp11/exp14/exp17.

| Đợt | Chạy | Lý do |
|-----|------|-------|
| 1 (capacity, rẻ) | **exp11** (DoRA), **exp14** (PiSSA) — đều gate **vLLM load-test**; **exp12** (NEFTune) probe | đường ít rủi ro nhất tới 0.87; exp11/exp14 là 2 trục LoRA-config độc lập |
| 2 (đánh bucket lỗi lớn) | exp15 (scratchpad) / exp13 (self-verify) cho lỗi số; exp19 (SoS) cho tổ hợp | M-effort upstream, chọn theo đo ở bước 0 |
| 3 (đẩy 0.87→0.88+) | exp17 (soup), exp18 (preference), exp16 (anchored-KL) | headroom lớn nhưng tốn compute / có xung đột implement — chỉ sau khi chốt wave 1 |

**Combo cuối:** gộp các thay đổi *độc lập* dương tính vào `exp_combo2.py` (vd exp11 DoRA + exp12 NEFTune + corpus của exp13/exp15). **Lưu ý compose:**
- exp11 (DoRA, parameterization) ⟂ exp14 (PiSSA, init) ⟂ batch-1 exp4 (rsLoRA, scaling) ⟂ batch-1 exp5 (module realloc) — 4 trục LoRA-config độc lập, thử gộp được.
- exp15 / exp19 (kéo dài trace) phải co-tune length budget với batch-1 exp3 (concise).
- exp16 (anchor, kéo lại) vs batch-1 exp2 (up-weight, đẩy mạnh) — lực đối nghịch, co-tune.
- **Soup (exp17) chỉ trộn member cùng init-regime** — đừng trộn member DoRA với member vanilla.

**Kỷ luật đo:** mọi so sánh trên **cùng slice held-out cố định**, cùng seed, cùng `NUM_STEPS`. Re-tune LR khi đổi parameterization/scaling (exp11 DoRA, exp14 PiSSA, batch-1 exp4 rsLoRA). **vLLM load-test là cổng cứng cho exp11 & exp14** trước mọi full run. Mục tiêu 0.86→0.88+ nên ưu tiên xác suất-cao/rủi-ro-thấp (exp11/exp14/exp12) trước, để compute-heavy (exp17/exp18) sau cùng.
