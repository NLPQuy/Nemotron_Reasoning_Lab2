# Implementation Plan — Batch 1 (exp1…exp10)

Mục tiêu: hiện thực hóa 10 ý tưởng trong [batch-1.md](batch-1.md) để đẩy điểm leaderboard **0.86 → 0.87**. Mỗi ý tưởng là **một file riêng** `exp<N>.py`, **copy nguyên xi** từ [Continuer_Nemotron_Notebook.py](../../Continuer_Nemotron_Notebook.py) (ở thư mục gốc repo), rồi chỉ sửa đúng vùng được đánh dấu.

> Đường dẫn: file plan này nằm ở `research/ideation/`; file gốc và các `exp<N>.py` nằm ở **thư mục gốc repo** (`../../`).

> Tất cả thay đổi đều là **training-time / data-time**. Không đụng tới inference (vẫn 1 lần greedy, vLLM-loadable, rank ≤ 32). exp<N> ↔ Idea<N> theo đúng thứ tự ranked trong batch-1.md.

---

## 0. Setup chung (làm 1 lần)

```bash
cd "/Users/phu-quynguyen-lam/o D/NVIDIA_Nemotron_Reasoning_Challenge"
for i in $(seq 1 10); do cp Continuer_Nemotron_Notebook.py exp${i}.py; done
```

**Quy ước cho mọi exp file:**
1. Ngay đầu file (sau dòng `# # Nemotron finetuning pipeline`) thêm banner:
   ```python
   # ============================================================
   # EXP<N> — <title>  (Batch-1 Idea <N>)
   # Base: Continuer_Nemotron_Notebook.py (unmodified except marked blocks)
   # Change: <1 dòng>   |   Knob: <KNOB>=<value>   |   Rollback: set <KNOB> mặc định
   # ============================================================
   ```
2. Mỗi vùng sửa bọc bằng `# >>> EXP<N> START` … `# <<< EXP<N> END` để dễ diff/revert.
3. Knob mới đặt ở **block config dòng 6–38** cùng các knob hiện có.
4. Chỉ đổi **một** cơ chế / file. Không gộp nhiều idea (trừ run "combo" ở mục 11).

**Tham chiếu nhanh các codepath trong file gốc:**
| Vùng | Dòng | Vai trò |
|------|------|---------|
| Config knobs | 6–38 | `LORA_RANK/ALPHA`, `NUM_STEPS`, `BATCH_SIZE`, `LEARNING_RATE`, `MOE_TIE_WEIGHTS`, `SHUFFLE_DATASET`, `TARGET_MODULES` |
| Corpus load (Kaggle) | 186–209 | build `examples=[{tokens,targets,weights}]`, truncate `[:MAX_SEQ_LEN]` |
| Corpus load (Modal) | 210–228 | tương tự |
| Filter ví dụ | 230–240 | mẫu `ORIGINAL_PROBLEMS_ONLY` để bắt chước khi lọc data |
| Tokenizer | 253–263 | `FastLanguageModel.from_pretrained` trả `tokenizer` |
| LoRA config | 268–277 | `get_peft_model(r, target_modules, lora_alpha, …)` |
| lm_head LoRA thủ công | 296–304 | `LoraConfig(r, lora_alpha, …)` |
| CCE forward + per-token CE | 365–396 | `model._cached_per_token_ce` |
| MoE tying | 458–508 | `moe_tied_params`, `_tie_param_init`, `_tie_grads` |
| Thứ tự data | 517–523 | `indices`, `SHUFFLE_DATASET` |
| Vòng train + LR schedule | 546–644 | batch slicing, `padded_weights`, `loss`, `lr = LR*(1-step/num_steps)`, `_tie_grads()` |

---

## exp1 — Format-verified clean labels + truncation-robust completions  *(Idea 1, P6)*

**Hypothesis:** mỗi label thiếu/hỏng `\boxed{}` = 1 điểm 0 cứng. Hiện tại dòng 197–199 cắt `tokens[:MAX_SEQ_LEN]` → **xén mất đuôi `\boxed{}`** ở trace dài. Verify + giữ đuôi boxed sẽ thu hồi nhóm lỗi format/truncation.

**Edit:**
1. Thêm knob (block config): `TRUNCATION_KEEP_BOXED_TAIL = True`, `BOXED_TAIL_TOKENS = 48`.
2. Load tokenizer **sớm**, ngay đầu `run_training()` (trước vòng load corpus, ~dòng 162) để decode kiểm tra:
   ```python
   # >>> EXP1 START
   from transformers import AutoTokenizer
   _tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
   import re
   _BOX_RE = re.compile(r"\\boxed\{.*\}")
   def _has_valid_box(token_ids):
       text = _tok.decode(token_ids)
       return bool(_BOX_RE.search(text)) and "<|im_end|>" in text
   # <<< EXP1 END
   ```
   (Lưu ý: `MODEL_PATH` được set trong nhánh `if IS_KAGGLE` dòng 120 / `else` 124 — chuyển khối load tokenizer xuống **sau** khi `MODEL_PATH` đã có, hoặc tách `MODEL_PATH` lên đầu.)
3. Sửa truncation (dòng 197–199, và bản Modal 216–218) thành **giữ đuôi**:
   ```python
   # >>> EXP1 START
   if len(tokens) > MAX_SEQ_LEN:
       if TRUNCATION_KEEP_BOXED_TAIL:
           head = tokens[: MAX_SEQ_LEN - BOXED_TAIL_TOKENS]
           tail = tokens[-BOXED_TAIL_TOKENS:]          # giữ </think>\boxed{...}<|im_end|>
           tokens = head + tail
           mask   = mask[: MAX_SEQ_LEN - BOXED_TAIL_TOKENS] + mask[-BOXED_TAIL_TOKENS:]
       else:
           tokens = tokens[:MAX_SEQ_LEN]; mask = mask[:MAX_SEQ_LEN]
   # <<< EXP1 END
   ```
4. Sau khi build xong `examples` (sau dòng 247) thêm pass verify, **drop** ví dụ hỏng:
   ```python
   # >>> EXP1 START
   before = len(examples)
   examples = [e for e in examples if _has_valid_box(e["targets"])]
   print(f"EXP1 format-verify: dropped {before-len(examples)} bad-box examples")
   # <<< EXP1 END
   ```

**Validate rẻ:** trước khi train, in `dropped` count + đếm số ví dụ bị truncate. Nếu `dropped==0` và truncate-rate ~0 → idea không có gì để sửa (xem falsification trong batch-1.md). **Rollback:** `TRUNCATION_KEEP_BOXED_TAIL=False` + bỏ pass verify.

---

## exp2 — Up-weight `\boxed{}` / critical answer tokens  *(Idea 2, P3)*

**Hypothesis:** grader chỉ chấm giá trị trong box. Tăng nhẹ trọng số loss cho span câu trả lời (đã có sẵn cơ chế `per_token_ce * padded_weights`, dòng 600) → khớp loss với exact-match. Guardrail: λ nhỏ, **không bao giờ hạ token nào < 1**.

**Edit:**
1. Knob: `ANSWER_TOKEN_WEIGHT = 2.0` (sweep {1.0, 1.5, 2.0, 3.0}).
2. Load tokenizer sớm như exp1 (tái dùng `_tok`).
3. Sau khi build `examples` (sau dòng 247), scale weights trong span boxed:
   ```python
   # >>> EXP2 START
   def _boxed_token_range(token_ids):
       text = _tok.decode(token_ids)
       i = text.rfind("\\boxed{")
       if i < 0: return None
       # ánh xạ ký tự → token bằng offset mapping
       enc = _tok(text, return_offsets_mapping=True, add_special_tokens=False)
       j = text.find("}", i)
       lo, hi = i, (j if j >= 0 else len(text))
       idx = [k for k,(a,b) in enumerate(enc["offset_mapping"]) if b > lo and a < hi]
       return (min(idx), max(idx)) if idx else None

   if ANSWER_TOKEN_WEIGHT != 1.0:
       for e in examples:
           rng = _boxed_token_range(e["targets"])
           if rng:
               a,b = rng
               for k in range(a, b+1):
                   if k < len(e["weights"]) and e["weights"][k] > 0:
                       e["weights"][k] = max(1.0, e["weights"][k] * ANSWER_TOKEN_WEIGHT)
   # <<< EXP2 END
   ```
   (Nếu offset_mapping của tokenizer Nemotron không khả dụng, fallback: up-weight `BOXED_TAIL_TOKENS` token cuối có `weight>0`.)

**Validate rẻ:** train λ=2.0 vs λ=1.0 trên slice 200 bài, so exact-match. Nếu không tăng ≥0.5pp ở mọi λ → reject. **Rollback:** `ANSWER_TOKEN_WEIGHT=1.0`.

---

## exp3 — Difficulty-aware concise traces / anti-truncation  *(Idea 3, P3)*

**Hypothesis:** trace quá dài → vượt budget 7680 lúc inference → 0 điểm. Vì corpus đã token-hóa, bản in-file = **cap độ dài completion** và bỏ/ghi nhận ví dụ completion dài hơn budget (để model học sinh ngắn hơn).

**Edit:**
1. Knob: `COMPLETION_BUDGET = 5000` (chừa headroom dưới 7680), `DROP_OVERLONG = True`.
2. Trong vòng build (sau khi có `tokens, mask`, trước `examples.append`, ~dòng 200):
   ```python
   # >>> EXP3 START
   comp_len = sum(1 for m in mask if m)         # số token completion (mask=1)
   if DROP_OVERLONG and comp_len > COMPLETION_BUDGET:
       continue                                  # bỏ ví dụ trace quá dài
   # <<< EXP3 END
   ```
3. (Tùy chọn nâng cao, để TODO) Regenerate trace ngắn ở `nemotron-master/reasoning.py` — nằm ngoài file đơn; ghi chú là phase-2.

**Validate rẻ:** trên slice đo (a) % generation chạm trần 7680, (b) exact-match. Nếu cap-hit-rate < 2% và exact-match không đổi → reject. **Rollback:** `DROP_OVERLONG=False`.

---

## exp4 — rsLoRA √r scaling  *(Idea 4, P4)* — ⚡ quick win

**Hypothesis:** ở rank tối đa 32, scaling α/r làm gradient yếu; α/√r (rsLoRA) khai thác hết rank.

**Edit (1 dòng × 2 chỗ):**
1. Knob: `USE_RSLORA = True`.
2. `get_peft_model(...)` dòng 268–277 thêm `use_rslora=USE_RSLORA,`.
3. `LoraConfig(...)` của lm_head dòng 297 thêm `use_rslora=USE_RSLORA,`.
   ```python
   # >>> EXP4 START
   _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                     use_rslora=USE_RSLORA)
   # <<< EXP4 END
   ```
4. (Khuyến nghị) dò lại LR ngắn: thử `LEARNING_RATE ∈ {1e-4, 2e-4}` vì step hiệu dụng đổi.

**Validate rẻ:** rsLoRA vs baseline ở LR đã re-tune, slice. Phải ≥ baseline. **Rollback:** `USE_RSLORA=False`. Kiểm tra adapter vẫn vLLM-load (rsLoRA ghi flag trong `adapter_config.json`, vLLM hỗ trợ).

---

## exp5 — Reasoning-critical target-module & rank realloc  *(Idea 5, P8)*

**Hypothesis:** năng lực reasoning tập trung ở MLP/expert + `o_proj` + mixer (Mamba); rải đều rank-32 lên attention thuần là lãng phí.

**Edit:** chỉ sửa `TARGET_MODULES` (dòng 28–38) — bỏ q/k/v, giữ MLP/expert/mixer/o_proj/lm_head:
```python
# >>> EXP5 START
TARGET_MODULES = [
    "o_proj",
    "up_proj", "down_proj",     # MLP + expert FFN
    "in_proj", "out_proj",      # Mamba mixer
    "lm_head",
]
# <<< EXP5 END
```
**Quan trọng:** in `model.named_modules()` để xác nhận tên module expert thực tế của Nemotron-H trước khi chốt (tên sai sẽ no-op âm thầm). Thử thêm biến thể giữ q/k/v để so. Cẩn thận tương tác với fast-path Mamba khi LoRA vào `in_proj/out_proj` (test load + vài greedy gen).

**Validate rẻ:** MLP/expert/mixer-focus vs current list, slice. Cần +≥0.5pp. **Rollback:** dùng lại list gốc dòng 28–38.

---

## exp6 — LIMO/s1 difficulty + diversity curation  *(Idea 6, P4)*

**Hypothesis:** ít-mà-tinh: bỏ bài quá dễ/trùng, cân bằng category → mỗi step nhiều tín hiệu hơn.

**Edit:** thêm block lọc kiểu `ORIGINAL_PROBLEMS_ONLY` (chèn sau dòng 240, trước thống kê 242):
1. Knob: `CURATE = True`, `PER_CATEGORY_CAP = 300`, `MIN_COMPLETION_LEN = 64`.
2.
   ```python
   # >>> EXP6 START
   if CURATE:
       from collections import defaultdict
       def _cat(pid): return pid.split("-")[0] if "-" in pid else pid[:8]  # proxy
       buckets = defaultdict(list)
       for e in examples:
           comp_len = int(sum(e["weights"]))
           if comp_len < MIN_COMPLETION_LEN:        # bỏ bài trivial/ngắn
               continue
           buckets[_cat(e["problem_id"])].append((comp_len, e))
       curated = []
       for cat, items in buckets.items():
           items.sort(key=lambda x: -x[0])          # ưu tiên khó (dài) hơn
           curated += [e for _, e in items[:PER_CATEGORY_CAP]]
       print(f"EXP6 curation: {len(examples)} → {len(curated)} "
             f"({len(buckets)} categories)")
       examples = curated
   # <<< EXP6 END
   ```
3. Chỉnh `NUM_STEPS` cho hợp số ví dụ mới.

**Validate rẻ:** curated vs full, **bằng số step**; macro-avg exact-match theo category phải tăng và không category nào rớt >1pp (sợ mất coverage vì mix leaderboard ẩn). **Rollback:** `CURATE=False`.

---

## exp7 — STaR/RFT self-generated verified traces  *(Idea 7, P12)* — 🎯 big bet, 2-phase

**Hypothesis:** thêm trace **tự sinh + đã verify đúng** cho bài đang sai → tăng coverage/đa dạng.

**Đây là idea nặng nhất, làm 2 pha; không nhét hết vào 1 lần train.**
1. **Phase A (sinh data, offline):** dùng adapter hiện tại generate K=4 trace/bài cho tập bài khó (cho phép sampling temperature>0 **chỉ khi sinh data**; inference nộp bài vẫn greedy). Verify bằng solver trong `nemotron-master/reasoners/` + grader round-trip; dedupe. Ghi ra `star_extra.jsonl` (cùng format `{problem_id, tokens, mask}`).
2. **Phase B (train):** exp7.py thêm knob `STAR_EXTRA_PATH` và merge vào `examples` (chèn sau dòng 247):
   ```python
   # >>> EXP7 START
   if STAR_EXTRA_PATH and os.path.isfile(STAR_EXTRA_PATH):
       with open(STAR_EXTRA_PATH) as f:
           for line in f:
               rec = json.loads(line)
               t, m = rec["tokens"], rec["mask"]
               if len(t) > MAX_SEQ_LEN: t, m = t[:MAX_SEQ_LEN], m[:MAX_SEQ_LEN]
               if not any(m): continue
               examples.append({"problem_id": rec["problem_id"],
                                "tokens": t[:-1], "targets": t[1:],
                                "weights": [float(x) for x in m[1:]]})
       print(f"EXP7 STaR: merged extra traces from {STAR_EXTRA_PATH}")
   # <<< EXP7 END
   ```
3. Một script sinh data riêng (`star_generate.py`) — TODO, dùng vLLM/`model.generate` + verifier. Nên **compose với exp3** (cap độ dài) và **exp1** (verify box) cho trace tự sinh.

**Validate rẻ:** sau 1 vòng STaR, exact-match trên nhóm bài-từng-sai phải +≥1pp, tổng thể không giảm; giới hạn ≤2 vòng (drift). **Rollback:** `STAR_EXTRA_PATH=None`.

---

## exp8 — Hot-expert untying of `MOE_TIE_WEIGHTS`  *(Idea 8, P8)*

**Hypothesis:** giữ expert "nguội" tied (regularize), **mở trói** top-k expert "nóng" để chuyên hóa.

**Edit:** sửa block tying (dòng 458–508).
1. Knob: `UNTIE_HOT_EXPERTS = True`, `NUM_HOT = 8`.
2. **Profiling:** trước train, hook router `mixer.gate` đếm tần suất chọn expert trên ~vài batch → `hot_idx` (top-`NUM_HOT`). (Scaffold: thêm forward hook trên các module có `.mixer.gate`, cộng dồn argtop-k của logits.)
3. Sửa `_tie_grads` để **chỉ broadcast cho expert nguội**, giữ grad riêng cho expert nóng:
   ```python
   # >>> EXP8 START
   def _tie_grads() -> None:
       with torch.no_grad():
           for p in moe_tied_params:
               if p.grad is None: continue
               if UNTIE_HOT_EXPERTS:
                   cold = [i for i in range(p.shape[0]) if i not in hot_idx]
                   gsum = p.grad[cold].sum(dim=0, keepdim=True)
                   p.grad[cold] = gsum.expand(len(cold), *p.grad.shape[1:])
                   # hàng hot_idx giữ nguyên grad độc lập
               else:
                   gsum = p.grad.sum(dim=0, keepdim=True)
                   p.grad.copy_(gsum.expand_as(p.grad))
   # <<< EXP8 END
   ```
   Đồng thời `_tie_param_init` chỉ mean-broadcast cho hàng cold (đừng ép hot bằng nhau).

**Rủi ro (đã flag MED):** corpus nhỏ → expert nóng dễ overfit (chính lý do ban đầu tie hết). **Validate:** grouped-untie vs full-tie trên slice, cần +≥0.5pp, không thì giữ tie. **Rollback:** `UNTIE_HOT_EXPERTS=False`.

---

## exp9 — Spaced-repetition (forgetting-curve) data scheduling  *(Idea 9, P2, T3)*

**Hypothesis:** lặp lại bài model còn sai ở khoảng cách tăng dần (spacing effect) → cùng số step nhớ bền hơn. Đây là đổi **thứ tự sampler**, không đổi data.

**Edit:** thay phần dựng `indices` + cách lặp batch (dòng 517–549).
1. Knob: `SPACED_REPETITION = True`, `REVIEW_EVERY = 25` (step), `INTERVAL_BASE = 2`.
2. Dùng EMA per-example loss làm proxy "đã thuộc". Thay vòng `for batch_start in range(...)` bằng sampler rút theo "due":
   ```python
   # >>> EXP9 START
   import heapq
   ema_loss = {i: 1.0 for i in range(len(examples))}   # cao = chưa thuộc
   due = [(0.0, i) for i in range(len(examples))]; heapq.heapify(due)
   def _next_batch(bs):
       picked = [heapq.heappop(due)[1] for _ in range(min(bs, len(due)))]
       return picked
   def _reschedule(i, cur_step, correctish):
       # đúng → giãn khoảng (×INTERVAL_BASE); sai → ôn lại sớm
       interval = (INTERVAL_BASE ** max(0, -int(math.log(ema_loss[i]+1e-6))))
       heapq.heappush(due, (cur_step + (interval if correctish else 1), i))
   # <<< EXP9 END
   ```
   Trong loop: lấy `batch_indices = _next_batch(BATCH_SIZE)`; sau micro-batch cập nhật `ema_loss[i]` từ loss của từng seq và gọi `_reschedule`. (Cần lấy loss per-sequence — tách `per_token_ce*weights` theo hàng.)

**Validate rẻ:** spaced vs shuffle cùng step/seed. Trong ±0.3pp ⇒ không có hiệu ứng, drop. **Rollback:** `SPACED_REPETITION=False` (về nhánh `SHUFFLE_DATASET`).

---

## exp10 — Simulated-annealing difficulty curriculum  *(Idea 10, P2, T3)*

**Hypothesis:** anneal sampling easy→hard theo nhiệt độ T giảm dần (continuation method) → basin tốt hơn trong cùng số step. Khác exp9: sắp theo **độ khó**, không theo "quên".

**Edit:** thay phần dựng `indices`/lặp batch (517–549) bằng sampler theo độ khó.
1. Knob: `ANNEAL_CURRICULUM = True`, `T0 = 2.0`, `T1 = 0.3`.
2.
   ```python
   # >>> EXP10 START
   import numpy as np
   diff = np.array([sum(e["weights"]) for e in examples], dtype=np.float64)  # proxy khó = dài
   diff = (diff - diff.mean()) / (diff.std() + 1e-6)
   def _sample_batch(step, bs):
       T = T0 + (T1 - T0) * (step / num_steps)        # giảm dần
       logits = -diff / max(T, 1e-3)                  # T cao→đều(thiên dễ), T thấp→bài khó
       p = np.exp(logits - logits.max()); p /= p.sum()
       return list(np.random.choice(len(examples), size=bs, replace=False, p=p))
   # <<< EXP10 END
   ```
   Trong loop dùng `batch_indices = _sample_batch(step, BATCH_SIZE)`. Thêm **control đảo chiều** (hard→easy) để chắc chắn hiệu ứng curriculum là thật.

**Validate rẻ:** annealed easy→hard vs uniform cùng step; cần +≥0.3pp **và** chiều đảo phải tệ hơn (có tín hiệu curriculum). **Rollback:** `ANNEAL_CURRICULUM=False`.

---

## 11. Thứ tự chạy đề xuất

> Theo "Next steps" của batch-1.md — bank các quick-win trước, để big-bet sau.

0. **Prerequisite (không phải idea, làm trước):** phân loại lỗi held-out hiện tại thành `{format-zero, truncation-zero, reasoning-wrong}`. Một phép đo này quyết định exp1/exp3 (format/truncation) hay exp2/exp5/exp6/exp7 (reasoning) mới là đòn bẩy đúng cho +0.01.

| Đợt | Chạy | Lý do |
|-----|------|-------|
| 1 (quick win) | **exp4** (rsLoRA), **exp2** (boxed weight), **exp1** (format-verify) | đều S-effort, độc lập, tái dùng codepath sẵn có |
| 2 | exp3 (anti-truncation), exp5 (target modules), exp6 (curation) | M-effort, phụ thuộc kết quả đo ở bước 0 |
| 3 (big bet) | exp7 (STaR), exp8 (MoE untie) | headroom lớn nhưng variance cao — chỉ sau khi đã chốt quick-win |
| 4 (thử nghiệm) | exp9, exp10 | hiệu ứng scheduling dưới LoRA có thể nhỏ; làm cuối |

**Combo cuối:** sau khi biết idea nào dương, gộp các thay đổi *độc lập* (vd exp4 + exp2 + exp1) vào một file `exp_combo.py` để submit. Mỗi thay đổi vẫn giữ banner `# >>> EXP<N>` để truy vết.

**Kỷ luật đo:** mọi so sánh trên **cùng slice held-out cố định**, cùng seed, cùng `NUM_STEPS`. Đổi LR khi (và chỉ khi) đổi scaling (exp4) hoặc reweighting mạnh (exp2). Mục tiêu chỉ là **+0.01 chắc chắn**, nên ưu tiên thay đổi rủi ro thấp, xác suất cao (exp1/exp2/exp4) trước.
