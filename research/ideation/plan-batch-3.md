# Implementation Plan — Batch 3 (exp20…exp27)

Mục tiêu: hiện thực hóa 8 ý tưởng trong [batch-3.md](batch-3.md) để đẩy leaderboard **0.86 → 0.88+**, chia đúng **2 trục**:
- **Trục TRAINING** (root `Continuer_Nemotron_Notebook.py`): mỗi idea = một file `exp<N>.py` **copy nguyên xi** từ notebook gốc, chỉ sửa vùng đánh dấu. → exp20, exp21, exp23, exp25, exp27.
- **Trục AUGMENTATION** (`nemotron-master/`): idea = **data-generation / reweighting** trong pipeline `reasoning.py → augmentation.py → corpus.py`. **Training procedure giữ nguyên notebook gốc** — sau khi sinh/cân bằng corpus mới, train bằng `Continuer_Nemotron_Notebook.py` **không sửa** (chỉ trỏ tới corpus mới). → exp22, exp24, exp26.

> Đường dẫn: plan này ở `research/ideation/`; notebook gốc + `exp<N>.py` ở **gốc repo** (`../../`); pipeline data ở `../../nemotron-master/`; **source code tham khảo đã clone sẵn ở `../../refs/`** (xem mục 1).

> Tất cả thay đổi là **training-time / data-time**. Không đụng inference (1 lần greedy, vLLM-loadable, rank ≤ 32). exp20↔Idea1 … exp27↔Idea8 theo đúng thứ tự ranked trong batch-3.md.

---

## 0. Map exp ↔ Idea ↔ Trục ↔ Repo tham khảo

| exp | Idea (batch-3) | Trục | Pattern | Repo refs/ | File then chốt để tham khảo |
|-----|----------------|------|---------|-----------|------------------------------|
| exp20 | 1 — High-entropy "forking-token" loss weighting | Training | P4 | `refs/trl` | `trl/trainer/grpo_config.py:774` (`top_entropy_quantile`) + `grpo_trainer.py` (mask top-ρ entropy) |
| exp21 | 2 — LoRA+ (LR riêng cho A vs B) | Training | P3 | `refs/loraplus` | `lora_plus.py` → `create_loraplus_optimizer()` |
| exp22 | 3 — DoReMi category data-mixture reweighting | Aug (data) | P4 | `refs/doremi` | `doremi/trainer.py` (cập nhật domain weight kiểu Group-DRO), `doremi/dataloader.py` (weighted sampling) |
| exp23 | 4 — ESFT expert-specialized MoE LoRA | Training | P8 | `refs/ESFT` | `esft.py`, `utils.py` (chọn expert theo gate-score), `scripts/` |
| exp24 | 5 — CSP solver mở khóa *guess* tasks | Aug (data) | P2 | `refs/python-constraint`, `refs/Logic-LLM` | `constraint/solvers.py:114 BacktrackingSolver`, `constraint/constraints.py:209 AllDifferentConstraint`, `examples/wordmath` (SEND+MORE=MONEY) |
| exp25 | 6 — GroupDRO worst-category objective | Training | P8 | `refs/group_DRO` | `loss.py` → `LossComputer` (cập nhật trọng số nhóm online) |
| exp26 | 7 — HER forward-generation cho *guess* | Aug (data) | P2 | `refs/stable-baselines3` | `stable_baselines3/her/her_replay_buffer.py`, `her/goal_selection_strategy.py` |
| exp27 | 8 — GSPO sequence-level MoE-stable RL | Training | P12 | `refs/trl` | `trl/trainer/grpo_config.py:668` (`importance_sampling_level="sequence"`) + `grpo_trainer.py` |

---

## 1. Setup chung (làm 1 lần)

### 1a. Tạo file exp (trục training)
```bash
cd "/Users/phu-quynguyen-lam/o D/NVIDIA_Nemotron_Reasoning_Challenge"
for i in 20 21 23 25 27; do cp Continuer_Nemotron_Notebook.py exp${i}.py; done
```
(exp22/24/26 **không** copy notebook — chúng là script data trong `nemotron-master/`; xem từng mục.)

### 1b. Source code tham khảo — ĐÃ CLONE sẵn ở `refs/`
```
refs/
├── loraplus/            # exp21  (github.com/nikhil-ghosh-berkeley/loraplus, ICML'24)
├── doremi/              # exp22  (github.com/sangmichaelxie/doremi, NeurIPS'23)
├── ESFT/                # exp23  (github.com/deepseek-ai/ESFT, EMNLP'24)
├── group_DRO/           # exp25  (github.com/kohpangwei/group_DRO, ICLR'20)
├── python-constraint/   # exp24  (github.com/python-constraint/python-constraint)
├── Logic-LLM/           # exp24  (github.com/teacherpeterpan/Logic-LLM, EMNLP'23, arXiv:2305.12295)
├── stable-baselines3/   # exp26  (github.com/DLR-RM/stable-baselines3 — HER reference)
└── trl/                 # exp20 + exp27 (github.com/huggingface/trl)
```
> **Cách dùng `refs/`**: KHÔNG import trực tiếp các repo này vào training (khác framework — chúng dùng HF Trainer/PPO, ta dùng vòng train thủ công + Unsloth + cut_cross_entropy). Chúng là **bản tham chiếu thuật toán**: đọc đúng file đã chỉ, lấy **công thức / thứ tự phép tính / cách build optimizer-group / cách cập nhật trọng số**, rồi **port** vào codepath của notebook. Mỗi exp dưới đây ghi rõ "Tham chiếu" = (file trong refs) → (port vào đâu trong notebook).

**Quy ước cho mọi exp file (training):**
1. Banner đầu file:
   ```python
   # ============================================================
   # EXP<N> — <title>  (Batch-3 Idea <k>)
   # Base: Continuer_Nemotron_Notebook.py (unmodified except marked blocks)
   # Ref:  refs/<repo>/<file>   |   Knob: <KNOB>=<value>   |   Rollback: <KNOB> mặc định
   # ============================================================
   ```
2. Mỗi vùng sửa bọc `# >>> EXP<N> START` … `# <<< EXP<N> END`.
3. Knob mới đặt ở block config **dòng 6–38**.
4. Chỉ đổi **một** cơ chế.

**Codepath notebook gốc (line numbers đã verify trên bản hiện tại, 805 dòng):**
| Vùng | Dòng | Vai trò |
|------|------|---------|
| Config knobs | 6–38 | `LORA_RANK/ALPHA`, `NUM_STEPS`, `BATCH_SIZE`, `LEARNING_RATE`, `MOE_TIE_WEIGHTS`, `SHUFFLE_DATASET`, `TARGET_MODULES` |
| Corpus load (Kaggle) | 186–209 | build `examples=[{problem_id,tokens,targets,weights}]`, truncate `[:MAX_SEQ_LEN]` |
| Corpus load (Modal) | 210–228 | tương tự |
| Stats print | 242–247 | `total_unmasked`, `total_tokens` |
| get_peft_model | 268–277 | `r, target_modules, lora_alpha, lora_dropout` |
| lm_head LoRA thủ công | 291–307 | `LoraConfig(...)` dòng 297 |
| fp32 cast + router check | 309–354 | `.lora_` → fp32; `.mixer.gate.` → fp32 |
| CCE forward + per-token CE | 365–396 | `lm_weight` (383), `per_token_ce` (385–388), `model._cached_per_token_ce` (392) |
| MoE tying | 451–509 | `moe_tied_params` (458), `_tie_param_init` (478), `_tie_grads` (485–499) |
| indices / shuffle | 517–523 | thứ tự data |
| Training loop | 546–644 | batch slice (549–553), micro-batch (560–620), `per_token_ce*padded_weights` (599–605), optimizer init (622–629), `lr` schedule (630–632), `_tie_grads()` (633), `clip_grad_norm_` (634), `step()` (637) |
| Save adapter | 650–665 | `save_pretrained` + rename `lm_head`→`backbone.lm_head` |

---

# TRỤC TRAINING

## exp20 — High-entropy "forking-token" loss weighting  *(Idea 1, P4)* — 🥇 top composite

**Hypothesis:** chỉ ~20% token entropy cao ("forking tokens") thực sự lèo lái reasoning; nhân thêm trọng số loss cho nhóm này (và/hoặc zero-out đuôi entropy thấp) → adapter rank-32 dồn capacity vào điểm quyết định. Hook có sẵn: dòng 599–600 đã nhân `per_token_ce * padded_weights`.

**Tham chiếu (refs/trl):**
- `refs/trl/trl/trainer/grpo_config.py:774` — đọc docstring `top_entropy_quantile` (ý nghĩa "giữ top-ρ quantile token theo entropy, mask phần còn lại").
- `refs/trl/trl/trainer/grpo_trainer.py` — `grep -n "entropy" grpo_trainer.py`: xem cách TRL tính per-token entropy từ logits và dựng mask `entropy >= quantile(ρ)`. **Port công thức mask** (không port loop PPO).

**Edit (exp20.py):**
1. Knob (block config): `ENTROPY_WEIGHTING = True`, `ENTROPY_TOP_QUANTILE = 0.2` (giữ top 20%), `ENTROPY_HIGH_W = 2.0`, `ENTROPY_LOW_W = 1.0` (đặt 0.0 để bắt chước "chỉ train forking tokens").
2. **Nguồn entropy** — KHÔNG materialize logits live (xung đột `linear_cross_entropy`). Hai cách, ưu tiên (a):
   - **(a) Precompute offline (khuyến nghị):** chạy 1 pass base model (hoặc tận dụng `nemotron-master/training/.../logprobs/` đã có) để lấy per-token entropy, lưu kèm corpus thành trường `entropy` song song `mask`. Khi đó exp20 chỉ cần đọc thêm `e["entropy"]` và dựng hệ số.
   - **(b) Proxy rẻ:** entropy ≈ `per_token_ce` của chính base model ở pass đầu (token CE cao ↔ model phân vân ↔ entropy cao). Tận dụng `model._cached_per_token_ce` ở **step 0** để xếp hạng, cache lại.
3. Dựng hệ số entropy và **nhân vào `padded_weights`** ngay trước dòng 600:
   ```python
   # >>> EXP20 START
   if ENTROPY_WEIGHTING:
       # ent: [n_micro, max_len] entropy per token (từ precompute hoặc proxy), đã pad
       thr = torch.quantile(ent[padded_weights > 0], 1.0 - ENTROPY_TOP_QUANTILE)
       ent_factor = torch.where(ent >= thr, ENTROPY_HIGH_W, ENTROPY_LOW_W)
       padded_weights = padded_weights * ent_factor          # giữ mask gốc (completion-only)
   # <<< EXP20 END
   weighted_loss = per_token_ce * padded_weights              # dòng 600 nguyên bản
   ```
   (Nếu dùng (a): thêm `padded_entropy` vào khối padding 582–589 y như `padded_weights`.)

**Validate rẻ:** train slice 200 bài: `ENTROPY_TOP_QUANTILE=0.2, HIGH_W=2` vs baseline uniform. Cần +≥0.5pp exact-match. Thử thêm `LOW_W=0` (chỉ-forking). **Rollback:** `ENTROPY_WEIGHTING=False`.
**Rủi ro:** kết quả gốc là cho RL policy-gradient, không phải SFT-CE → transfer là điểm bất định chính (slice-test gate). Entropy cần logits → bắt buộc dùng precompute/proxy, đừng phá fast-path.

---

## exp21 — LoRA+ : learning-rate riêng cho A vs B  *(Idea 2, P3)* — ⚡ quick win

**Hypothesis:** single-LR cho cả A và B là dưới tối ưu ở width lớn; đặt `lr_B = λ·lr_A` (λ≈4–16) khai thác feature-learning hiệu quả hơn — cùng compute, chỉ đổi cách nhóm optimizer.

**Tham chiếu (refs/loraplus):**
- `refs/loraplus/lora_plus.py` — đọc `create_loraplus_optimizer(model, optimizer_cls, lr, loraplus_lr_ratio, loraplus_lr_embedding=...)`. **Lấy đúng logic phân nhóm tham số**: gom `lora_B` (và `lora_embedding_B`) vào group có `lr = lr * ratio`; `lora_A` + còn lại giữ `lr`. Đây chính là phần thay cho `torch.optim.AdamW(...)` ở dòng 622–629.

**Edit (exp21.py):** sửa **khởi tạo optimizer** (dòng 622–629).
1. Knob: `LORAPLUS_LR_RATIO = 8.0` (sweep {1.0(=baseline),4,8,16}).
2.
   ```python
   # >>> EXP21 START
   if optimizer is None:
       if LORAPLUS_LR_RATIO != 1.0:
           # port từ refs/loraplus/lora_plus.py: tách lora_B ra group lr cao hơn
           groupB, groupA_other = [], []
           for n, p in model.named_parameters():
               if not p.requires_grad: continue
               (groupB if ".lora_B." in n else groupA_other).append(p)
           optimizer = torch.optim.AdamW(
               [{"params": groupA_other, "lr": LEARNING_RATE},
                {"params": groupB,       "lr": LEARNING_RATE * LORAPLUS_LR_RATIO}],
               betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
       else:
           optimizer = torch.optim.AdamW(
               [p for p in model.parameters() if p.requires_grad],
               lr=LEARNING_RATE, betas=(0.9,0.95), eps=1e-8, weight_decay=0.0)
   # <<< EXP21 END
   ```
3. **LR schedule (dòng 630–632) phải scale theo group**, đừng ghi đè bằng 1 `lr`:
   ```python
   # >>> EXP21 START
   decay = (1 - step / num_steps)
   optimizer.param_groups[0]["lr"] = LEARNING_RATE * decay
   if len(optimizer.param_groups) > 1:
       optimizer.param_groups[1]["lr"] = LEARNING_RATE * LORAPLUS_LR_RATIO * decay
   # <<< EXP21 END
   ```

**Validate rẻ:** slice, λ∈{4,8,16} vs **single-LR đã re-tune** (control bắt buộc — xem devil's-advocate batch-3.md). Cần +≥0.3pp so với control. **Rollback:** `LORAPLUS_LR_RATIO=1.0`.
**Rủi ro:** gain có thể chỉ là "tune LR" → control sweep bắt buộc; λ lớn trên LoRA fp32 + `MOE_TIE_WEIGHTS` (grad-sum dòng 498) có thể gây loss spike — theo dõi `grad_norm`.

---

## exp23 — ESFT : LoRA tập trung vào expert có affinity cao  *(Idea 4, P8)*

**Hypothesis:** routing MoE tập trung vào tập nhỏ expert cho mỗi task; dồn rank-32 vào top-affinity expert (untie/giữ riêng), freeze/tie phần đuôi → đỡ loãng capacity trên 128 expert.

**Tham chiếu (refs/ESFT):**
- `refs/ESFT/utils.py` + `refs/ESFT/scripts/` — đọc cách ESFT **tính expert affinity** (chạy forward trên sample, cộng gate-score/ tần suất chọn expert theo layer) và **chọn tập expert** theo ngưỡng (`top_p`/`gate_score`).
- `refs/ESFT/esft.py` — cách build adapter chỉ trên expert được chọn (ta thay bằng: chọn expert nào **untie** trong `moe_tied_params`).
- `refs/ESFT/configs/` — format file lưu danh sách expert per-layer (tham khảo để lưu `hot_experts.json`).

**Edit (exp23.py):** sửa block MoE tying (dòng 451–509).
1. Knob: `ESFT_SELECT = True`, `ESFT_TOP_EXPERTS = 16` (trên tổng số expert/layer), `ESFT_PROFILE_BATCHES = 8`, `ESFT_AFFINITY_PATH = "hot_experts.json"`.
2. **Profiling** (chèn sau model load, trước training loop ~dòng 510): hook `mixer.gate` cộng dồn gate-score qua `ESFT_PROFILE_BATCHES` batch → per-layer affinity → chọn top-`ESFT_TOP_EXPERTS` → lưu `ESFT_AFFINITY_PATH`. (Port cách cộng score từ `refs/ESFT/utils.py`.)
3. Sửa `_tie_grads` (485–499): expert **được chọn** → giữ grad riêng (untie); expert còn lại → grad-sum như cũ:
   ```python
   # >>> EXP23 START
   def _tie_grads() -> None:
       with torch.no_grad():
           for p in moe_tied_params:
               if p.grad is None: continue
               if ESFT_SELECT:
                   cold = [i for i in range(p.shape[0]) if i not in HOT_EXPERTS]
                   if cold:
                       gsum = p.grad[cold].sum(dim=0, keepdim=True)
                       p.grad[cold] = gsum.expand(len(cold), *p.grad.shape[1:])
                   # hàng HOT_EXPERTS giữ grad độc lập
               else:
                   p.grad.copy_(p.grad.sum(dim=0, keepdim=True).expand_as(p.grad))
   # <<< EXP23 END
   ```
   `_tie_param_init` (478) cũng chỉ mean-broadcast cho hàng cold.

**Phân biệt với batch-1 exp8 (hot-expert untie):** exp8 untie heuristic; exp23 **đo affinity từ data theo gate-score** (port ESFT) và có thể freeze hẳn đuôi. **In `model.named_modules()`** xác nhận tên `mixer.gate`/expert của Nemotron-H trước khi hook.
**Validate rẻ:** top-16 affinity vs tie-all (slice). Cần ≥ tie-all (trong noise) với cost thấp hơn. **Rollback:** `ESFT_SELECT=False`.

---

## exp25 — GroupDRO : tối thiểu hóa loss worst-category  *(Idea 6, P8)*

**Hypothesis:** tối ưu **worst-category** thay vì trung bình → không "ăn xổi" ở category dễ/đông token; khớp với grader chấm rộng theo category.

**Tham chiếu (refs/group_DRO):**
- `refs/group_DRO/loss.py` → class `LossComputer`: đọc **đúng** quy tắc cập nhật trọng số nhóm online:
  `q[g] *= exp(η · loss_g); q /= q.sum()` (exponentiated-gradient), rồi `loss = Σ_g q[g]·loss_g`. Lấy `step_size`/`adj` (generalization adjustment) làm tham khảo.
- `refs/group_DRO/run_expt.py` — cách truyền `group_idx` mỗi sample.

**Edit (exp25.py):**
1. Knob: `GROUP_DRO = True`, `DRO_STEP_SIZE = 0.01`, `DRO_ADJ = 0.0`.
2. Cần **category mỗi example**: thêm `e["category"]` khi build `examples` (suy từ `problem_id` qua `problems.jsonl`/`train.csv`, hoặc prefix). Map category→`group_id`.
3. Khởi tạo `q = torch.ones(n_groups)/n_groups` trước training loop. Trong loop, cần **loss theo group trong batch** (tách `per_token_ce*padded_weights` theo group của từng seq), rồi:
   ```python
   # >>> EXP25 START  (port từ refs/group_DRO/loss.py: LossComputer)
   # group_loss[g] = mean weighted-CE của các seq thuộc group g trong micro-batch
   q = q * torch.exp(DRO_STEP_SIZE * (group_loss.detach() - DRO_ADJ))
   q = q / q.sum()
   loss = (q[present_groups] * group_loss[present_groups]).sum()
   # <<< EXP25 END
   ```
   (Thay cho `loss = loss_sum_t/weight_sum_t` ở 603–605 khi `GROUP_DRO`.)

**Phân biệt:** đây là **objective online** (minimax theo nhóm); khác exp22/DoReMi (đổi **tỉ lệ data offline**). Hai cái **compose** (DoReMi = prior, GroupDRO = hiệu chỉnh động).
**Validate rẻ:** worst-category held-out exact-match phải +≥1pp mà trung bình không tụt >0.5pp. **Rollback:** `GROUP_DRO=False`.
**Rủi ro (ICLR'20 cảnh báo):** worst-group overfit trên mạng over-param → giữ `DRO_STEP_SIZE` nhỏ + sàn kích thước nhóm; group nhiễu/nhỏ dễ thống trị.

---

## exp27 — GSPO : sequence-level RL ổn định cho MoE  *(Idea 8, P12)* — 🎯 big bet, ⚠️ KHÓ NHẤT BATCH (feas 2/5)

> **Đây là exp khó nhất** (verification report batch-3.md: feasibility 2/5, risk HIGH). Vì vậy phần này viết **đầy đủ cho codex**: kiến trúc 2 pha, định dạng dữ liệu, công thức tensor có shape, điểm chèn theo dòng trong notebook, knob, xử lý số học, và **bản tối giản (MVP)** để giảm rủi ro hạ tầng. Đọc hết trước khi code.

**Hypothesis:** Với bài có đáp án verify được, reward (đúng/sai từ `reasoners/`) là **không nhiễu** → RL rất hợp. Nhưng GRPO/PPO tính importance-ratio **per-token**; trên MoE, ~10% expert được kích hoạt thay đổi giữa policy cũ/mới → ratio token-level nhiễu → training phân kỳ. **GSPO** tính ratio ở **mức sequence** (length-normalized) → khử nhiễu routing → ổn định MoE (đã chứng minh trên Qwen3-MoE).

### 0. Quyết định kiến trúc — LÀM TRONG NOTEBOOK, KHÔNG dùng train_sft.py
- `nemotron-master/train_sft.py` chạy trên **Tinker** (`trainer/client.py ServiceClient`): loss tính **server-side**, chỉ nhận `advantages/logprobs` qua `build_datum` (train_sft.py:139) và **bộ loss-fn cố định** (`cross_entropy/importance_sampling/ppo/cispo/dro` trong `loss_config.py`). GSPO **không** nằm trong bộ primitive đó → không thêm được nếu không sửa Tinker (ngoài tầm). ⇒ **bỏ hướng train_sft.py.**
- **Notebook `Continuer_Nemotron_Notebook.py` có toàn quyền autograd** (vòng train thủ công, dòng 546–644). Quan trọng: monkey-patch CCE (dòng 365–396) cho ra `per_token_ce` = **−logπ(token mục tiêu)**. Vậy **logπ per-token = `−per_token_ce`** — đã có sẵn, đúng thứ cần cho importance ratio. ⇒ GSPO khả thi ngay trong notebook. **exp27.py = copy notebook + sửa.**

### 1. Tham chiếu (refs/trl) — port CÔNG THỨC, không port Trainer
- `refs/trl/trl/trainer/grpo_trainer.py:2496–2533` — **đây là khối loss chuẩn để port**. Trích đúng:
  - `:2496` `log_ratio = per_token_logps - old_per_token_logps`
  - `:2499–2501` (GSPO = sequence-level):
    ```python
    log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)   # (B,)
    log_importance_weights = log_importance_weights.unsqueeze(-1)                         # (B,1)
    ```
  - `:2508` `coef_1 = torch.exp(log_importance_weights)`        # ratio mức-sequence, broadcast theo token
  - `:2526` `coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)`
  - `:2531–2533` `per_token_loss = -min(coef_1·adv, coef_2·adv)`
  - `:2567` reduce: `loss = ((per_token_loss*mask).sum(-1)/mask.sum(-1).clamp(min=1)).mean()`
- `refs/trl/trl/trainer/grpo_config.py` — defaults: `epsilon=0.2` (`:601`), `num_generations=8` (`:392`), `beta` KL (`:589`). **GSPO dùng clip CHẶT hơn nhiều** vì ratio sequence-level phương sai nhỏ — paper GSPO dùng cỡ `3e-4`. Đặt `epsilon_low=epsilon_high≈3e-4` (đọc lại `:601,:613` để khớp tên).

### 2. Định dạng dữ liệu giữa 2 pha (`rollouts.jsonl`)
Mỗi dòng = 1 completion đã sample:
```json
{"problem_id": "...", "category": "...",
 "tokens":   [int, ...],          // prompt + completion (prompt sẽ bị mask)
 "mask":     [0/1, ...],          // 1 = token completion (giống corpus.py)
 "old_logp": [float, ...],        // logπ_old per token completion (= −CE dưới policy sample); len = len(tokens)-1
 "reward":   0 or 1}              // verifier: 1 nếu \boxed khớp compare_answer
```

### 3. Phase A — rollout + reward (script riêng `gspo_rollout.py`, OFFLINE)
1. Nạp adapter hiện tại; với mỗi problem trong tập đã chọn, **sample G=8** completion (temperature 0.8–1.0, top_p 0.95) — **chỉ sample lúc sinh data**, submission vẫn greedy.
2. Chấm reward: trích `\boxed{}` (reasoning.py `extract_answer`) + `compare_answer(answer, pred)` (reasoning.py:60) → `reward∈{0,1}`. Thêm **format-check** (có đúng 1 `\boxed{}`, có `<|im_end|>`) để chống reward-hack.
3. Lưu `old_logp` ngay lúc sample (logπ của chính policy sinh ra nó). Nếu dùng vLLM: bật `logprobs`; nếu dùng `model.generate`: chạy 1 forward lấy `−per_token_ce`.
4. Ghi `rollouts.jsonl` đúng schema trên. **Bỏ nhóm reward toàn-0 hoặc toàn-1** (advantage = 0, vô ích).

### 4. Phase B — cập nhật GSPO (sửa exp27.py)
**Knob (block config 6–38):**
```python
GSPO_ENABLE     = True
GSPO_ROLLOUTS   = "rollouts.jsonl"
GSPO_GROUP_SIZE = 8          # G; phải khớp Phase A
GSPO_EPS_LOW    = 3e-4       # clip chặt (sequence-level)
GSPO_EPS_HIGH   = 3e-4
GSPO_BETA_KL    = 0.0        # 0 = bỏ KL (giữ đơn giản); >0 nếu cần neo
```
**(a) Nạp rollouts thay corpus** (sau khối build `examples`, ~dòng 247): mỗi record → `{tokens[:-1], targets=tokens[1:], weights=mask[1:], old_logp, advantage}`.
**(b) Advantage group-normalized** (GRPO-style, tính 1 lần khi nạp): cho mỗi nhóm G completion cùng problem:
```python
# >>> EXP27 START
adv = (reward - mean_group_reward) / (std_group_reward + 1e-4)   # scalar/sequence
# gán adv cho mọi token completion của sequence đó
# <<< EXP27 END
```
**(c) Tính logπ_θ per-token hiện tại**: **dùng lại `per_token_ce` có sẵn** (dòng 599): `cur_logp = -per_token_ce`. (Không cần materialize logits — đây là lợi thế của CCE patch.)
**(d) Thay khối loss CE (dòng 600–605) bằng GSPO** — port từ TRL:
```python
# >>> EXP27 START  (port refs/trl grpo_trainer.py:2496–2533, importance_sampling_level="sequence")
mask        = padded_weights                              # (B,T) 1=completion
cur_logp    = -per_token_ce                               # (B,T)
old_logp    = padded_old_logp                             # (B,T) từ rollouts (pad như padded_weights)
adv         = padded_adv                                  # (B,T) advantage broadcast theo token
log_ratio   = (cur_logp - old_logp)                       # (B,T)
seq_logw    = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)   # (B,)  ← MỨC SEQUENCE
seq_logw    = torch.clamp(seq_logw, -20.0, 20.0).unsqueeze(-1)          # (B,1) ổn định số học
coef_1      = torch.exp(seq_logw)                          # (B,1) ratio sequence, broadcast
coef_2      = torch.clamp(coef_1, 1 - GSPO_EPS_LOW, 1 + GSPO_EPS_HIGH)
per_tok     = -torch.min(coef_1 * adv, coef_2 * adv)       # (B,T)
loss        = ((per_tok * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
# <<< EXP27 END
```
(Thêm `padded_old_logp`, `padded_adv` vào khối padding 582–589 y hệt `padded_weights`.)
**(e) MoE tying GIỮ NGUYÊN**: `_tie_grads()` (dòng 633) vẫn chạy bình thường — GSPO không đụng grad-flow của expert; chính ratio mức-sequence là thứ làm ổn định MoE, nên **không tắt** `MOE_TIE_WEIGHTS`.

### 5. MVP để giảm rủi ro (KHUYẾN NGHỊ làm trước)
- **On-policy 1 bước**: dùng `cur_logp == old_logp` ở step đầu ⇒ `coef_1≈1`, loss ≈ `-adv·logp` (REINFORCE có baseline). Xác nhận đường ống (advantage, reward, format-check, mask) đúng **trước** khi bật ratio off-policy nhiều step.
- Bắt đầu **1 category** (vd `equation_numeric_deduce`) đã có verifier chắc, G=8, ≤1 epoch.

### 6. Validate rẻ & rollback
- **Sanity**: in `mean(coef_1)` (~1.0 ± nhỏ), `mean(reward)`, `clip_fraction`. Nếu `coef_1` bùng nổ / NaN → clip/eps sai.
- **Hiệu quả**: exact-match held-out **+≥1pp** sau 1 pha so với SFT; nếu (a) MoE oscillate/loss-spike dù đã sequence-level, hoặc (b) không cải thiện → **dừng**, giữ SFT + offline-pref (batch-2 idea8).
- **Rollback**: `GSPO_ENABLE=False` → exp27.py về đúng notebook SFT gốc.

### 7. Failure modes (đọc kỹ)
- **Reward-hack format**: model học nhồi token rác quanh box → bắt buộc format-check trong reward (mục 3.2).
- **Off-policy drift**: nếu rollouts cũ so với policy hiện tại quá xa, `coef_1` lệch mạnh → giữ ≤1–2 pha, re-rollout giữa các pha.
- **Group toàn-0/toàn-1**: advantage=0 → đã lọc ở Phase A.
- **Tương tác `MOE_TIE_WEIGHTS` × on-policy chưa kiểm chứng**: theo dõi `grad_norm`; nếu spike, hạ `GSPO_EPS_*` hoặc giảm LR.
- **Chi phí**: rollout G×problems là phần nặng nhất — chạy `gspo_rollout.py` trên 1 category nhỏ trước. **Chỉ làm exp27 sau khi đã bank exp20/21/22.**

---

# TRỤC AUGMENTATION  (training = notebook gốc, KHÔNG sửa)

> Cả 3 exp dưới đây chạy trong `nemotron-master/` (env `uv` riêng): sinh/chỉnh data → regenerate `corpus.jsonl` + `corpus/<id>/synthetic.jsonl` (đúng format `corpus.py`: completion `"{reasoning}\n</think>\\boxed{{answer}}<|im_end|>"`, prompt mask 0 / completion mask 1) → **train bằng `Continuer_Nemotron_Notebook.py` không đổi**, chỉ trỏ tới snapshot corpus mới (đường dẫn `CORPUS_PATH` dòng 109/125). Mọi format token mới phải đồng bộ `corpus.py` / `metric_reference.py`.

## exp24 — Deterministic CSP solver mở khóa `cryptarithm_guess` + `equation_numeric_guess`  *(Idea 5, P2)* — 🎯 big bet

**Hypothesis:** 300 bài (164+136) đang `rule_unknown` 100% → **0 trace**. Viết solver backtracking + constraint-propagation → `reasoning.py` xuất `rule_found` cho chúng → thêm ~300 ví dụ token-cao.

**Tham chiếu (refs/python-constraint, refs/Logic-LLM):**
- `refs/python-constraint/examples/wordmath/` — **đây chính là cryptarithm** (SEND+MORE=MONEY): đọc cách model hóa bằng `Problem`, `addVariables(letters, range(10))`, `AllDifferentConstraint`, ràng buộc leading-digit≠0 và phương trình cột. Đây là **khung trực tiếp cho `cryptarithm_guess`**.
- `refs/python-constraint/constraint/solvers.py:114 BacktrackingSolver` (và `:234 OptimizedBacktrackingSolver`) — đọc thứ tự **propagate→assign→conflict→backtrack** để **render thành CoT** (mỗi bước = 1 câu trace), giống các reasoner hiện có.
- `refs/python-constraint/constraint/constraints.py:209 AllDifferentConstraint` — ràng buộc gán khác nhau.
- `refs/Logic-LLM/models/` + `solver_examples/` — cách dịch bài ngôn ngữ tự nhiên → constraint program; lấy ý tưởng cho `equation_numeric_guess` (suy hệ ràng buộc số học từ các example).

**Edit (trong `nemotron-master/`):**
1. **KHÔNG dùng `python-constraint` làm dependency runtime** (giữ pipeline thuần). Port một backtracking-CSP **gọn** vào reasoner, mượn cấu trúc từ `examples/wordmath`.
2. `reasoners/cryptarithm.py` (hiện chỉ xử lý concat, trả `None` cho phần còn lại — xem `reasoning_cryptarithm` đầu file): thêm nhánh **CSP guess**:
   - Từ `problem.examples` suy quan hệ chữ↔số; dựng biến + `AllDifferent` + ràng buộc cột; backtracking tìm nghiệm; nếu nhất quán với mọi example và giải được `problem.question` → render trace propagate/backtrack rồi `\boxed{answer}`.
3. `reasoners/equation_numeric.py`: thêm nhánh guess — suy hệ phương trình/ràng buộc từ các (input,output) example, giải, verify, render.
4. `reasoning.py`: bỏ `cryptarithm_guess` khỏi diện skip (docstring dòng 1–3 nói "skipping cryptarithm_guess"; kiểm tra `SKIP_CATEGORIES`/nhánh tương ứng) để generator mới được gọi; chỉ ghi trace khi `compare_answer` (dòng 60) xác nhận đúng → `status="rule_found"` (dòng 180).
5. `uv run python3 reasoning.py` → `uv run python3 corpus.py` để tái sinh corpus.

**Validate rẻ:** solver phải verify đúng ≥90% trong 300 bài (in `rule_found` count cho 2 category). Train slice: exact-match 2 category guess phải từ ~0 → +≥5pp (absolute). **Rollback:** trả `None` ở nhánh guess (về trạng thái cũ).
**Rủi ro:** một số bài có thể ill-posed → để `rule_unknown`, **đừng bịa**; trace guess dài → cap độ sâu nhánh (compose batch-1 exp3 concise).

## exp26 — HER-style forward generation cho *guess*  *(Idea 7, P2)*

**Hypothesis:** thay vì chỉ giải 300 bài cho sẵn, **sinh bài mới từ đáp án đã chọn trước** (giống HER relabel theo "achieved goal") → ví dụ luôn giải được + có trace verified, mở rộng guess vượt 300.

**Tham chiếu (refs/stable-baselines3):**
- `refs/stable-baselines3/stable_baselines3/her/her_replay_buffer.py` — đọc ý tưởng cốt lõi: **relabel goal = outcome đã đạt** (hàm sample + `_sample_her_transitions`). Ta không dùng replay buffer; **lift khái niệm**: chọn nghiệm trước (= "achieved goal"), build bài quanh nó.
- `refs/stable-baselines3/stable_baselines3/her/goal_selection_strategy.py` — các chiến lược chọn goal (`FUTURE/FINAL/EPISODE`) → tham khảo để **đa dạng độ khó** khi sinh.

**Edit (trong `nemotron-master/`):** thêm generator forward (đặt cạnh `augmenters/` hoặc `reasoners/`):
1. `equation_numeric_guess`: sample hệ số/biến → tính output → phát biểu bài "guess" + render forward-derivation CoT (verified by construction).
2. `cryptarithm_guess`: sample phép gán chữ↔số hợp lệ (AllDifferent, leading≠0) → sinh phương trình → render.
3. Knob: `HER_N_PER_CATEGORY = 500`, tham số độ khó calibrate theo 300 bài gốc.
4. `uv run python3 augmentation.py`/`corpus.py` để hợp nhất vào corpus.

**Validate rẻ:** sinh 500/category, train slice; exact-match trên **300 bài gốc** phải +≥3pp (nếu chỉ tăng trên synthetic mà không lên gốc ⇒ phân phối lệch). **Rollback:** không nạp file synthetic. **Compose:** exp24 (giải bài có sẵn) + exp26 (sinh bài mới) là cặp **coverage** mạnh nhất.
**Rủi ro:** phân phối synthetic lệch leaderboard (calibrate theo 300 gốc); model học artifact thay vì reasoning (gate bằng held-out trên bài gốc, không phải synthetic).

## exp22 — DoReMi category data-mixture reweighting  *(Idea 3, P4)* — 🛡️ safe bet

**Hypothesis:** tỉ lệ category hiện tại là tình cờ (bit_manipulation = 26.4% token); reweight theo DoReMi dồn step budget vào category dưới-phục-vụ giá trị cao.

**Tham chiếu (refs/doremi):**
- `refs/doremi/doremi/trainer.py` — class `DoReMiTrainer`: đọc **vòng cập nhật domain weight kiểu Group-DRO**: `α_t ← α_{t-1} · exp(η·excess_loss_per_domain)`, normalize, smoothing với uniform (`(1-c)·α + c·u`). Đây là thuật toán cốt lõi.
- `refs/doremi/doremi/dataloader.py` — `weighted sampling` theo domain (cách hiện thực sampler có trọng số/ domain).
- `refs/doremi/scripts/` + `configs/` — format `domain_weights` xuất ra.

**Edit (2 phần):**
1. **Phase A — tính weights (proxy rẻ, trong `nemotron-master/`):** port công thức từ `doremi/trainer.py` ở mức nhẹ: dùng per-category **excess loss** (loss adapter hiện tại − loss tham chiếu) để ra `category_weights` (hoặc bản đơn giản: weight ∝ loss-proportional, có sàn/trần để bit_manipulation không bị bóp quá). Lưu `category_weights.json`.
2. **Phase B — resample corpus:** trong `corpus.py` (hoặc bước build `examples`), **nhân bản/giảm mẫu** mỗi category theo `category_weights` để ra `corpus.jsonl` mới. **Training = notebook gốc**, chỉ trỏ `CORPUS_PATH` tới snapshot mới.
   - (Phương án không sửa corpus: đưa weight vào notebook bằng cách lặp index theo trọng số ở dòng 517 — nhưng để **giữ notebook gốc nguyên** đúng yêu cầu trục aug, ưu tiên resample ở `corpus.py`.)
3. Knob ở data script: `DOREMI_SMOOTHING_C = 0.1`, `WEIGHT_FLOOR = 0.5`, `WEIGHT_CEIL = 2.0` (so với uniform-per-category).

**Validate rẻ:** macro-avg (per-category) exact-match ở weights DoReMi vs uniform, **cùng số step**. Cần +≥0.5pp; không category nào rớt >1pp. **Rollback:** dùng corpus gốc.
**Rủi ro:** proxy run thêm cost (dùng proxy nhỏ / heuristic loss-proportional làm bản đầu); bóp bit_manipulation quá tay → rớt 26% leaderboard share → giữ trong band `[FLOOR, CEIL]`.

---

## 2. Thứ tự chạy đề xuất

> Theo "Run-order" của batch-3.md. Prerequisite (không phải idea): **đo phân loại lỗi held-out theo category** {format-zero, truncation, arithmetic-slip, method-wrong, unsolved-category} — quyết định nên ưu tiên weighting (exp20/22/25) hay coverage (exp24/26).

| Đợt | Chạy | Lý do |
|-----|------|-------|
| 1 (quick win, training) | **exp21** (LoRA+, kèm control single-LR) + **exp20** (entropy weighting) | S-effort, vLLM-neutral, độc lập |
| 2 (fix data) | **exp22** (DoReMi) → **exp24** (CSP solver) + **exp26** (HER gen) | cặp coverage giá trị cao nhất; train = notebook gốc |
| 3 (MoE-specific) | **exp23** (ESFT) → **exp25** (GroupDRO) | sau khi mixture đã chốt |
| 4 (hold, big bet) | **exp27** (GSPO online RL) | ceiling cao nhất, cost cao nhất; chỉ sau khi đã bank trên |

**Combo cuối:** gộp các thay đổi *độc lập* dương tính (vd exp21 + exp20 + corpus của exp22/24/26) vào 1 file submit, giữ banner `# >>> EXP<N>` để truy vết.

**Kỷ luật đo:** mọi so sánh trên **cùng slice held-out cố định**, cùng seed, cùng `NUM_STEPS`. Re-tune LR khi đổi scaling/LR-geometry (exp21) hoặc reweighting mạnh (exp20/25). Mục tiêu **+0.01..0.02 chắc chắn** trước, big-bet (exp24/27) sau.

## 3. Lưu ý format & vLLM
- 5 exp training (20/21/23/25/27) **không đổi format adapter** → vẫn vanilla rank-32, vLLM-loadable, **không cần load-test** (khác DoRA/PiSSA batch-2). exp23 đổi *expert nào* mang LoRA nhưng dạng lưu vẫn chuẩn.
- 3 exp aug (22/24/26) chỉ đổi **data**; token format phải đồng bộ `corpus.py` (completion scaffold + mask) và `metric_reference.py`. Verify lại bằng `compare_answer` round-trip trước khi train.
