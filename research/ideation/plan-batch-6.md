# Plan — Batch 6 (kiến trúc adapter LoRA: chỉ code cái CHẮC deploy được), cho Codex

> **Audience: Codex.** Mỗi task self-contained: mục tiêu / file thật phải tạo-sửa / **code tham khảo
> Ở ĐÂU trong `refs/` (đã clone thật) & port THẾ NÀO** / line-level edits trong marker / validate / rollback.
> Ý tưởng gốc: [batch-6.md](batch-6.md). Refs đã clone: xem [refs/README.md](../../refs/README.md) mục "Batch-6".
>
> **3 luật bất biến (từ batch-1..5, [tracker/leaderboard.md](../../tracker/leaderboard.md)):**
> 1. **Regime = continue-train từ 0.86** (`RESET_WEIGHTS=False`), giữ corpus + order gốc
>    (`SHUFFLE_DATASET=False`, nhánh per-id + `TRAIN_ORDER_PATH`). KHÔNG mix, KHÔNG negatives.
> 2. **Liều nhẹ**: `LEARNING_RATE ≤ 1e-5`, ≤ 1 epoch. Gate 5-nhóm-mạnh không tụt > 0.5pp.
> 3. **Mỗi exp PHẢI dò LR riêng** (prereq "LR Matters" trong batch-6) — đừng so sánh dưới 1 LR cứng.
>
> **Deploy: CHỈ Kaggle, KHÔNG Modal.** Mỗi exp = 1 Kaggle notebook self-contained → **mọi code tham
> khảo trong `refs/` phải INLINE vào file exp** (Kaggle runtime không có `refs/`).

---

## ⚠️ PHẦN 0 — LUẬT SỐNG/CHẾT của batch-6: "deployable as a rank-32 vLLM LoRA adapter"

Submission = **LoRA adapter** (`adapter_config.json` + `adapter_model.safetensors`), nạp dưới **vLLM**,
ràng buộc **`max_lora_rank=32`**. vLLM chỉ áp **delta cộng low-rank chuẩn** `ΔW = scaling · B@A`
(`B∈ℝ^{out×r}`, `A∈ℝ^{r×in}`, `r≤32`) lên **base gốc nguyên vẹn**, cho một **tập module cố định**
(q/k/v/o, gate_up/down, in_proj/out_proj, embed/lm_head). **Bất cứ thứ gì không khớp khuôn này thì
không deploy được** — train tốn GPU rồi cũng vô nghĩa.

Mình đã **đọc code thật** từng repo (quote trong PHẦN 4/5) và phân loại theo khuôn này. Đây là điểm
mấu chốt khiến thứ tự batch-6.md **đảo lại**: 3 idea top của batch-6.md (HiRA, SSM-tuning, MiLoRA)
**vướng khuôn deploy** → **KHÔNG code**.

| Idea | Train cho ra gì | Deploy = rank-32 LoRA chuẩn? | Quyết định |
|---|---|---|---|
| **B6-4 Freeze-A** | A đóng băng + B chuẩn | ✅ LoRA chuẩn, base gốc | **CODE (exp48)** |
| **B6-7 Flat-LoRA** | A/B chuẩn, base bị nhiễu *rồi khôi phục* | ✅ base cuối = gốc | **CODE (exp49)** |
| **B6-11 Partial-LoRA SSM** | A/B chuẩn, B zero ngoài slice dt/B/C | ✅ vẫn rank-32 LoRA | **CODE (exp50→56)** |
| **B6-19 embed_tokens** | LoRA-embedding chuẩn | ✅ `embedding_modules` (xác nhận tĩnh v0.12.0) | **CODE (exp55)** — free coverage |
| **B6-21 HP-sweep** | plain LoRA, grid LR×batch | ✅ LoRA chuẩn | **CODE (exp57)** — baseline thật |
| **B6-8 LoRA-Pro** | A/B chuẩn (grad chỉnh lúc train) | ✅ LoRA chuẩn | **CODE (exp52)** — khó, để sau |
| **B6-10 LoRA-Dropout** | A/B chuẩn + dropout lúc train | ✅ LoRA chuẩn | **CODE (exp53)** — knob rẻ |
| **B6-14 MoSLoRA** | mixer fold→B' | ✅ rank-32 LoRA | 🟡 **CODE (exp54)** — NO-GO/near-free, để cuối |
| **B6-2 Router-LoRA** | LoRA trên `mixer.gate` | ❌ gate KHÔNG trong supported set (v0.12.0) | **KILLED (exp51)** |
| **B6-6 HiRA** | `ΔW = W₀ ⊙ (BA)` **high-rank** | ❌ không phải rank-32 cộng; vLLM không có path HiRA | **BLOCK** |
| **B6-1 SSM state-offset** | param mới `(d_state,d_inner)` cộng vào state | ❌ vLLM chạy Mamba gốc, **không gọi param này** | **BLOCK** |
| **B6-1 additional-scan** | mở rộng `d_state` | ❌ phá khuôn kernel | **BLOCK** |
| **B6-3 MiLoRA** | trừ delta khỏi base + init minor | ❌ base bị sửa; convert→LoRA **rank 2r=64 > 32** | **BLOCK** (xem biến thể r=16) |
| **B6-13 BitFit/Norm** | bias + RMSNorm gains (`modules_to_save`) | ❌ không phải LoRA, vLLM không nạp norm/bias | **BLOCK** |
| **B6-5 HDMoLE, B6-12 Affix** | — | ❌ không có code (HDMoLE) / không phải LoRA (Affix) | **BLOCK** |

> **Nguyên tắc viết plan này:** chỉ viết line-level code cho hàng **CODE** (đã chắc khuôn deploy).
> Hàng **BLOCK** ghi rõ **lý do kỹ thuật + điều kiện gỡ block**, KHÔNG đưa code train để tránh đốt GPU.

---

## PHẦN 1 — Giải phẫu base notebook (line-ref đã verify 2026-06-10)

File gốc: [Continuer_Nemotron_Notebook.py](../../Continuer_Nemotron_Notebook.py) (858 dòng). Mọi exp là
**bản copy**, sửa bọc trong `# >>> EXP<N> START/END`.

| Vùng | Dòng | Nội dung |
|---|---|---|
| Config | [6-40](../../Continuer_Nemotron_Notebook.py#L6-L40) | `LORA_RANK=32`, `LORA_ALPHA=32`, `LORA_DROPOUT=0.0`, `LEARNING_RATE=2e-4`, `RESET_WEIGHTS`, `IN_PROJ_ONLY`, `MOE_TIE_WEIGHTS`, `SHUFFLE_DATASET=False`, `TARGET_MODULES` |
| `LEARNING_RATE` | [16](../../Continuer_Nemotron_Notebook.py#L16) | =2e-4 (from-scratch). **Continue-0.86 đổi → ≤1e-5.** |
| `RESET_WEIGHTS` | [17-19](../../Continuer_Nemotron_Notebook.py#L17-L19) | =True. **Continue-0.86 đổi → False.** |
| `TARGET_MODULES` | [30-40](../../Continuer_Nemotron_Notebook.py#L30-L40) | q/k/v/o, up/down, in_proj/out_proj, lm_head |
| Model load (Unsloth) | [285-295](../../Continuer_Nemotron_Notebook.py#L285-L295) | `FastLanguageModel.from_pretrained` bf16 eager |
| `get_peft_model` | [300-309](../../Continuer_Nemotron_Notebook.py#L300-L309) | wrap LoRA, `r=LORA_RANK`, `target_modules=TARGET_MODULES` |
| Mamba fast-path patch | [312-321](../../Continuer_Nemotron_Notebook.py#L312-L321) | `is_fast_path_available=True` |
| lm_head LoRA thủ công | [323-339](../../Continuer_Nemotron_Notebook.py#L323-L339) | `_create_and_replace` cho lm_head (Unsloth bỏ) |
| Cast LoRA→fp32 | [341-350](../../Continuer_Nemotron_Notebook.py#L341-L350) | mọi `.lora_` → fp32 |
| Gate giữ fp32 | [353-384](../../Continuer_Nemotron_Notebook.py#L353-L384) | `.mixer.gate.` để fp32 (router) |
| CCE forward patch | [397-428](../../Continuer_Nemotron_Notebook.py#L397-L428) | `linear_cross_entropy`; stash `model._cached_per_token_ce` |
| Load adapter 0.86 | [430-471](../../Continuer_Nemotron_Notebook.py#L430-L471) | nếu `RESET_WEIGHTS=False` → `load_peft_weights(ADAPTER_SRC)` |
| Freeze `IN_PROJ_ONLY` | [473-481](../../Continuer_Nemotron_Notebook.py#L473-L481) | mẫu freeze theo tên param |
| MoE tie | [483-540](../../Continuer_Nemotron_Notebook.py#L483-L540) | `_tie_param_init` [510-515], `_tie_grads` [517-531] (sum grad qua 128 expert) |
| Micro-batch forward+backward | [642-663](../../Continuer_Nemotron_Notebook.py#L642-L663) | `model(...)` → `per_token_ce` → `loss` → `(loss/n_accum).backward()` |
| Optimizer (lazy) | [675-682](../../Continuer_Nemotron_Notebook.py#L675-L682) | `AdamW(betas=(0.9,0.95),eps=1e-8,wd=0)` từ `requires_grad` params |
| LR decay | [683-685](../../Continuer_Nemotron_Notebook.py#L683-L685) | `lr=LEARNING_RATE*(1-step/num_steps)` |
| `_tie_grads()` + clip + step | [686-691](../../Continuer_Nemotron_Notebook.py#L686-L691) | tie → clip(1e9≈off) → `optimizer.step()` → `zero_grad()` |
| Save + rename lm_head | [703-718](../../Continuer_Nemotron_Notebook.py#L703-L718) | `save_pretrained` → rename `lm_head.`→`backbone.lm_head.` |
| Kaggle trigger | [857-858](../../Continuer_Nemotron_Notebook.py#L857-L858) | `if IS_KAGGLE: run_training()` |

**Quan sát then chốt cho code:** PEFT LoRA module có `.base_layer.weight`, `.lora_A["default"].weight`,
`.lora_B["default"].weight`, `.scaling["default"]` (xem CCE patch [410-414](../../Continuer_Nemotron_Notebook.py#L410-L414)
dùng đúng các attr này cho lm_head). Mọi helper dưới đây dựa vào khuôn đó.

---

## PHẦN 2 — SETUP CHUNG (mọi exp train của batch-6 áp dụng)

### 2A. Tạo file
```bash
cp Continuer_Nemotron_Notebook.py exp<N>.py
```
Thêm header ngay sau dòng 1, bọc marker:
```python
# >>> EXP<N> START
# EXP<N> — <tên idea> (Batch-6 B6-<k>)
# Ref: <refs/...:line đã đọc> | Knob: <knob=val> | Rollback: <cách tắt> | Deploy: rank-32 LoRA chuẩn
# >>> EXP<N> END
```

### 2B. Bật regime continue-train-từ-0.86 (SỬA config, mọi exp)
- [16](../../Continuer_Nemotron_Notebook.py#L16): `LEARNING_RATE = 2e-4` → **`1e-5`** (rồi dò {5e-6, 1e-5, 2e-5}).
- [17-19](../../Continuer_Nemotron_Notebook.py#L17-L19): `RESET_WEIGHTS = True` → **`False`**.
- [25](../../Continuer_Nemotron_Notebook.py#L25): giữ `SHUFFLE_DATASET = False`.
- Giữ nhánh đọc corpus per-id + `TRAIN_ORDER_PATH`, `ADAPTER_SRC` = adapter 0.86 (như batch-5 PHẦN 0).

### 2C. Validate chung (mọi exp, TRƯỚC khi tốn full GPU)
1. **Smoke 3 step**: chạy `NUM_STEPS=3` → in `trainable / total` ([388-390](../../Continuer_Nemotron_Notebook.py#L388-L390)),
   `grad_norm` hữu hạn, loss giảm.
2. **Deploy check (BẮT BUỘC)**: sau khi smoke, mở `adapter_model.safetensors` đã save, xác nhận **chỉ
   chứa key `lora_A`/`lora_B`** (+ rename lm_head), rank ≤ 32, **không có** key lạ (norm/bias/state-offset),
   **không** key base weight. Nếu có key lạ → idea đó **không deploy được** → dừng.

---

## PHẦN 3 — HÀNG "CODE NGAY" (chắc khuôn deploy)

### exp48 — B6-4 Freeze-A (train chỉ B; rẻ nhất, neo exp21) ✅
- **Cơ sở:** Hayou [arXiv:2406.08447] — B chi phối update, frozen-A ≈ full-LoRA; cùng mạch A/B-asymmetry
  của exp21 (thứ DUY NHẤT giữ 0.86). Repo gốc paper = lý thuyết, **dùng** [refs/loraplus](../../refs/loraplus)
  làm tham chiếu A/B-asymmetry (không có code riêng cho paper này — xác nhận trong refs/README).
- **Deploy:** A,B vẫn LoRA chuẩn, base gốc → **rank-32 chuẩn**. ✅
- **Edit:** chèn **sau** [481](../../Continuer_Nemotron_Notebook.py#L481) (sau in trainable/frozen),
  **trước** block MoE tie [483](../../Continuer_Nemotron_Notebook.py#L483) — để `_tie_grads` (lọc
  `param.requires_grad` ở [495](../../Continuer_Nemotron_Notebook.py#L495)) tự bỏ qua A đã freeze:
```python
# >>> EXP48 START — B6-4 Freeze-A (train B only). Rollback: FREEZE_LORA_A=False
FREEZE_LORA_A = True
if FREEZE_LORA_A:
    _froz = 0
    for _n, _p in model.named_parameters():
        if ".lora_A." in _n and _p.requires_grad:
            _p.requires_grad = False
            _froz += 1
    print(f"EXP48 Freeze-A: froze {_froz} lora_A params (B-only training)")
# >>> EXP48 END
```
- **Validate:** trainable param ~giảm nửa; smoke loss vẫn giảm. **Deploy check** PHẦN 2C.
- **Falsify (batch-6):** ≥ exp21 cùng setup. Nếu < → A-init là thủ phạm (thử biến thể re-init A orthogonal,
  thêm trong cùng marker bằng `torch.nn.init.orthogonal_` trên `.lora_A` trước khi freeze).
- **Rollback:** `FREEZE_LORA_A = False`.

### exp49 — B6-7 Flat-LoRA (flat minimum cho greedy) ✅
- **Repo thật:** [refs/flat-lora/logTrainer.py](../../refs/flat-lora/logTrainer.py) — đã đọc:
  - Nhiễu cộng **vào base weight `W₀`** (KHÔNG vào A/B), std theo filter-norm hàng:
    `filter_norm = factor·rho/√(d_in)·‖W_eff‖_row` với `W_eff = W₀ + scaling·B@A`
    (logTrainer.py:160-161); `factor = 0.5·(1−cos(progress·π))` cosine schedule (logTrainer.py:148).
  - **Add trước forward, SUBTRACT lại bằng cùng seed sau backward, rồi mới step** (logTrainer.py:151-186)
    → base cuối = gốc. **1 forward**, không double-cost như SAM.
- **Deploy:** base được khôi phục, chỉ A/B lưu → **rank-32 chuẩn**. ✅
- **Port vào loop thủ công của notebook** (perturb 1 lần mỗi optimizer-step, bao quanh cụm micro-batch):
  1. Định nghĩa helper cạnh `_tie_grads` (chèn sau [540](../../Continuer_Nemotron_Notebook.py#L540)):
```python
# >>> EXP49 START — B6-7 Flat-LoRA. Rollback: FLAT_RHO=0.0
FLAT_RHO = 0.05  # dò {0.0(off), 0.02, 0.05, 0.1}
_flat_mods = [m for m in model.modules()
              if hasattr(m, "lora_A") and hasattr(m, "base_layer")
              and "default" in getattr(m, "lora_A", {})]
_flat_saved = []  # (module, noise) để khôi phục đúng tensor đã cộng

def _flat_perturb(progress: float) -> None:
    """progress in [0,1] = step/num_steps. Cộng nhiễu Gaussian vào base weight."""
    _flat_saved.clear()
    if FLAT_RHO <= 0:
        return
    factor = 0.5 * (1.0 - math.cos(progress * math.pi))
    with torch.no_grad():
        for m in _flat_mods:
            W0 = m.base_layer.weight
            A = m.lora_A["default"].weight
            B = m.lora_B["default"].weight
            scaling = m.scaling["default"]
            W_eff = W0.float() + scaling * (B.float() @ A.float())
            d_in = W_eff.shape[1]
            row_norm = W_eff.norm(dim=1, keepdim=True)            # ‖W_eff‖_row
            std = factor * (FLAT_RHO + 1e-16) / math.sqrt(d_in) * row_norm
            noise = torch.normal(0.0, std.expand_as(W0).to(torch.float32)).to(W0.dtype)
            W0.add_(noise)
            _flat_saved.append((W0, noise))

def _flat_restore() -> None:
    with torch.no_grad():
        for W0, noise in _flat_saved:
            W0.sub_(noise)
    _flat_saved.clear()
# >>> EXP49 END
```
   2. Gọi **trước** cụm micro-batch của step (ngay trước vòng for micro-batch, ~[600](../../Continuer_Nemotron_Notebook.py#L600)):
      `# >>> EXP49 perturb` → `_flat_perturb(step / num_steps)`.
   3. Gọi **sau** cụm micro-batch, **trước** `_tie_grads()` ở [686](../../Continuer_Nemotron_Notebook.py#L686):
      `# >>> EXP49 restore` → `_flat_restore()`.
- **Lưu ý chính xác:** lưu `noise` thật để subtract (không re-sinh bằng seed như repo gốc — chắc ăn hơn,
  tránh lệch RNG dưới autocast). Chỉ perturb các module LoRA-target (không toàn 30B). `math` đã import? nếu
  chưa, thêm `import math` trong marker.
- **Validate:** smoke 3 step, in `‖noise‖`; xác nhận sau `_flat_restore` thì `W0` bằng trước (so 1 phần tử).
  **Deploy check** PHẦN 2C (base khôi phục → adapter sạch).
- **Falsify:** ≥ baseline + variance 5-nhóm-mạnh thấp hơn qua seed. **Rollback:** `FLAT_RHO=0.0`.

### exp50 — B6-11 Partial-LoRA trên SSM-slice của in_proj ✅ (gated: recon split-order)
- **Repo thật:** [refs/mambapeft/.../tuners/mamba_peft.py](../../refs/mambapeft/language/commonsense_reasoning/mamba_peft/src/peft/tuners/mamba_peft.py)
  — đã đọc: Partial-LoRA = đặt LoRA chỉ lên **một slice hàng output** của in_proj/x_proj bằng **zero-pad B**
  (mamba_peft.py:404-431); variants `lora_d(dt)/lora_B/lora_C` map vào các slice `[dt | B | C]`
  (setup 966-1020). **CẢNH BÁO TỪ REPO:** thứ tự split của MambaPEFT là Mamba-1 (`x_proj → [dt,B,C]`);
  **Nemotron-H là Mamba-2**, in_proj gộp `[z, x, B, C, dt]` với `dt` per-head — **dims KHÁC**.
- **GATE bắt buộc (recon, KHÔNG tốn train-GPU) — làm trước khi code:**
  1. Đọc `modeling_nemotron_h.py` (transformers) tìm class mixer Mamba-2 của Nemotron-H, lấy công thức
     `d_in_proj` và thứ tự `torch.split(...)` của output in_proj (thường
     `2*intermediate + 2*n_groups*d_state + nheads`). Ghi lại offset chính xác của slice `[B, C, dt]`.
  2. Nếu không xác định được offset chắc chắn → **KHÔNG code exp50** (đổi sang exp51/52).
- **Deploy:** vẫn `lora_A/lora_B` chuẩn, chỉ một số **hàng của B bị ép = 0** → **rank-32 chuẩn**, vLLM nạp
  bình thường. ✅
- **Cách port (mask B-rows thay vì custom layer — giữ LoRA chuẩn):**
  1. Đặt `IN_PROJ_ONLY` tinh thần: chỉ để LoRA `in_proj` (+ giữ các module mạnh khác như baseline tùy chọn).
  2. Sau khi xác định `SLICE = slice(off_lo, off_hi)` (vùng B+C+dt trong out-dim của in_proj), chèn:
     - **Init mask**: zero các hàng B ngoài slice ngay sau load adapter ([471](../../Continuer_Nemotron_Notebook.py#L471)).
     - **Grad mask**: trong `_tie_grads`-style hook gọi ở [686](../../Continuer_Nemotron_Notebook.py#L686),
       zero `lora_B.grad` các hàng ngoài slice cho mọi module in_proj → B ngoài slice giữ nguyên 0.
```python
# >>> EXP50 START — B6-11 Partial-LoRA SSM-slice. Cần SLICE đúng từ recon Nemotron-H!
PARTIAL_SLICE = (None, None)  # (off_lo, off_hi) — ĐIỀN sau recon; (None,None) = tắt
def _partial_inproj_mods():
    return [m for m in model.modules()
            if hasattr(m, "lora_B") and hasattr(m, "base_layer")
            and getattr(m, "_exp50_inproj", False)]
# đánh dấu module in_proj: lặp named_modules, set m._exp50_inproj = name.endswith("in_proj")
def _partial_zero_B(grad: bool) -> None:
    lo, hi = PARTIAL_SLICE
    if lo is None:
        return
    with torch.no_grad():
        for m in _partial_inproj_mods():
            Bw = m.lora_B["default"].weight
            tgt = Bw.grad if grad else Bw
            if tgt is None:
                continue
            mask = torch.zeros(Bw.shape[0], 1, device=Bw.device, dtype=Bw.dtype)
            mask[lo:hi] = 1.0
            tgt.mul_(mask)
# >>> EXP50 END
```
     gọi `_partial_zero_B(grad=False)` 1 lần sau load; `_partial_zero_B(grad=True)` ngay trước
     `optimizer.step()` ([690](../../Continuer_Nemotron_Notebook.py#L690)) (sau `_tie_grads`).
- **Validate:** in số hàng B ≠ 0 = `hi-lo`; smoke loss giảm. **Deploy check** PHẦN 2C.
- **Falsify:** vượt in_proj-LoRA generic (exp43-style) trên bit. **Rollback:** `PARTIAL_SLICE=(None,None)`.

### exp51 — B6-2 Router-LoRA (LoRA trên `mixer.gate`) — ❌ KILLED (deep-research 2026-06-10)
> **KILL — KHÔNG code.** Deep-research (Trục 3) xác nhận `mixer.gate` **KHÔNG nằm** trong
> `supported_lora_modules` của class Nemotron-H trong vLLM (list từ vLLM #38085 không có gate/router).
> LoRA trên router sẽ bị **warn-and-ignore (load-but-never-apply)** → train xong vô tác dụng lúc inference.
> Giữ lại mục này chỉ để ghi verdict; **không tạo exp51**. (Refs adamoe/ld-mole vẫn lưu cho tham khảo lý thuyết.)

<details><summary>Code cũ (KHÔNG dùng — router không vLLM-applied)</summary>
- **Edit (chỉ khi gate verify OK):**
  - Thêm `"gate"` (hoặc tên đúng `mixer.gate`) vào `TARGET_MODULES` [30-40](../../Continuer_Nemotron_Notebook.py#L30-L40)
    **với rank/LR riêng** — KHÔNG dùng `LORA_RANK=32` cho gate. Vì `get_peft_model` dùng 1 rank chung,
    cách sạch: để gate ra ngoài `TARGET_MODULES`, **thêm LoRA thủ công cho gate** y hệt cơ chế lm_head
    [323-336](../../Continuer_Nemotron_Notebook.py#L323-L336) nhưng `LoraConfig(r=8, lora_alpha=8)`:
```python
# >>> EXP51 START — B6-2 Router-LoRA (rank 8 trên mixer.gate). Rollback: ROUTER_LORA=False
ROUTER_LORA = True
ROUTER_RANK = 8
if ROUTER_LORA:
    from peft.tuners.lora import LoraConfig as _LC
    _gcfg = _LC(r=ROUTER_RANK, lora_alpha=ROUTER_RANK, lora_dropout=0.0)
    _n_gate = 0
    for _name, _mod in list(model.named_modules()):
        if _name.endswith(".mixer.gate"):  # NemotronHTopkRouter Linear
            _parent = model.get_submodule(_name.rsplit(".", 1)[0])
            model.base_model._create_and_replace(_gcfg, "default", target=_mod,
                                                  target_name="gate", parent=_parent)
            _n_gate += 1
    print(f"EXP51 Router-LoRA: added rank-{ROUTER_RANK} LoRA to {_n_gate} gates")
# >>> EXP51 END
```
  - **Giữ LoRA-gate ở fp32** (router là fp32, xem [353-384](../../Continuer_Nemotron_Notebook.py#L353-L384));
    cast fp32 ở [341-344](../../Continuer_Nemotron_Notebook.py#L341-L344) đã cover `.lora_` nên OK.
  - **LR cực thấp cho gate + anchor**: dùng param-group riêng trong AdamW [675-682](../../Continuer_Nemotron_Notebook.py#L675-L682)
    với `lr = LEARNING_RATE * 0.1` cho param gate (router rất nhạy).
- (validate cũ — bỏ qua, router không vLLM-applied)
</details>

### exp54 — B6-14 MoSLoRA: learnable subspace-mixer — 🟡 NO-GO/near-free (deep-research 2026-06-10)
> **HẠ ƯU TIÊN — chạy cuối, kỳ vọng ~0.** Deep-research (Trục 2): gain +2.8pp của paper là **single-config
> commonsense, KHÔNG test text-math**; **PEFT maintainer reproduce 20-run → KHÔNG khác LoRA** (p≈1.0, PR
> đóng không merge); 3 paper 2026 (LR/Batch Matters) cho thấy variants hội tụ 1–2% khi tune LR. Fold đúng
> toán nên **không hại**, nhưng "win" trên 0.86 nhiều khả năng = noise LR. Chỉ chạy nếu rảnh GPU, và PHẢI
> **dò LR riêng** trước khi tin delta. **2 ĐÍNH CHÍNH so với code dưới:** (a) scaling official = **`alpha/r`
> cố định, KHÔNG phải `alpha/√r`**; (b) init mixer = **Kaiming** (identity = plain LoRA, vô ích — Table 3:
> kaiming 85.6 > orthogonal 84.4 > identity 82.6 ≈ LoRA 82.8).
- **Repo thật:** [refs/moslora/.../lora/layer.py](../../refs/moslora/subject_driven_generation/peft/tuners/lora/layer.py)
  — đã đọc: mixer `lora_AB = nn.Linear(r, r, bias=False)` (L113); forward
  `lora_B(lora_AB(lora_A(x)))·scaling` (L347); init mixer kaiming `a=√5` (L166).
- **Toán:** `ΔW = scaling·B·M·A`, `M∈ℝ^{r×r}`. `rank(BMA) ≤ r ≤ 32`. **Fold lúc save**: `B' = B·M`
  (out×r) → adapter lưu = `lora_A, lora_B'` **chuẩn** → vLLM nạp rank-32. (M biến mất sau fold.)
- **Deploy:** sau fold = LoRA chuẩn trên base gốc → **rank-32 chuẩn**. ✅ (đây là điểm khác HiRA.)
- **Port (notebook dùng PEFT LoraLinear, KHÔNG thay PEFT — patch forward + thêm mixer param):**
  1. Sau lm_head-add + cast fp32 ([350](../../Continuer_Nemotron_Notebook.py#L350)), gắn mixer cho mỗi
     module target + bọc forward:
```python
# >>> EXP54 START — B6-14 MoSLoRA mixer. Rollback: MOSLORA=False. Fold B'=B·M lúc save.
import types
MOSLORA = True
_moslora_mods = []
if MOSLORA:
    for _m in model.modules():
        if hasattr(_m, "lora_A") and hasattr(_m, "lora_B") and "default" in getattr(_m, "lora_A", {}):
            _r = _m.lora_A["default"].weight.shape[0]
            _mix = torch.nn.Linear(_r, _r, bias=False).to(
                _m.lora_A["default"].weight.device, torch.float32)
            torch.nn.init.kaiming_uniform_(_mix.weight, a=math.sqrt(5))  # ref layer.py:166
            _m.lora_AB = torch.nn.ModuleDict({"default": _mix})
            _moslora_mods.append(_m)
    # patch forward của từng LoraLinear: chèn mixer giữa A và B
    def _moslora_forward(self, x, *args, **kwargs):
        result = self.base_layer(x, *args, **kwargs)
        for adp in self.active_adapters:
            if adp not in self.lora_A:
                continue
            A, B = self.lora_A[adp], self.lora_B[adp]
            M = self.lora_AB[adp]
            drop = self.lora_dropout[adp]
            xd = drop(x).to(A.weight.dtype)
            result = result + B(M(A(xd))) * self.scaling[adp]
        return result.to(x.dtype)
    for _m in _moslora_mods:
        _m.forward = types.MethodType(_moslora_forward, _m)
    print(f"EXP54 MoSLoRA: mixer added to {len(_moslora_mods)} modules")
# >>> EXP54 END
```
  2. **MoE-tie**: thêm `lora_AB` vào diện tie nếu module là expert (mixer cũng phải tie 128 slice) —
     bổ sung mixer params vào `moe_tied_params` ([490-508](../../Continuer_Nemotron_Notebook.py#L490-L508))
     cùng cách. Nếu phức tạp, **chạy MoSLoRA chỉ trên non-expert module** trước (q/k/v/o/in/out/lm_head).
  3. **Fold trước save** ([711](../../Continuer_Nemotron_Notebook.py#L711) `model.save_pretrained`): chèn
     ngay TRƯỚC:
```python
# >>> EXP54 fold — B' = B·M để adapter lưu thành LoRA chuẩn (vLLM-safe)
if MOSLORA:
    with torch.no_grad():
        for _m in _moslora_mods:
            Bw = _m.lora_B["default"].weight        # out×r
            Mw = _m.lora_AB["default"].weight        # r×r
            Bw.copy_((Bw.float() @ Mw.float()).to(Bw.dtype))
            del _m.lora_AB                            # bỏ mixer khỏi state_dict
# >>> EXP54 fold END
```
- **Validate:** (a) smoke 3 step, loss giảm; (b) **fold-equivalence test**: trước fold lưu `y0 = B(M(A(x)))`
  cho 1 x ngẫu nhiên, sau fold tính `B'(A(x))`, assert ≈ (sai số <1e-3); (c) **Deploy check PHẦN 2C**:
  adapter chỉ có `lora_A/lora_B`, **không** key `lora_AB`, rank ≤32.
- **Lưu ý fp32:** mixer để fp32 (như mọi `.lora_`). `import math` nếu chưa có.
- **Falsify:** vượt baseline rank-32, 5-nhóm-mạnh giữ. **Rollback:** `MOSLORA=False`.

---

## PHẦN 4 — HÀNG "CODE SAU / KHÓ"

### exp52 — B6-8 LoRA-Pro (grad bám full-FT) ⚠️ khó, deploy ✅
- **Repo thật:** [refs/lora-pro/DeepSpeed-0.15.1/deepspeed/runtime/zero/stage_1_and_2.py](../../refs/lora-pro)
  — công thức `lorapro_full_adjustment()` (dòng **1828-1895**) + `solve_sylvester()` (dòng **2611-2632**).
  Đã đọc; **KHÔNG có bản standalone ngoài DeepSpeed** — phải bê 2 hàm này ra.
- **Cốt lõi đã trích (port theo, KHÔNG bịa):**
  - bước>0: `B_TB_inv=pinv(BᵀB+δI)`, `AA_T_inv=pinv(AAᵀ+δI)`,
    `grad_A = (1/s²)·B_TB_inv·grad_A_orin`,
    `grad_B = (1/s²)·(I − B·B_TB_inv·Bᵀ)·grad_B_orin·AA_T_inv` (stage_1_and_2.py:1850-1857),
    rồi `equiv_grad = s·B·grad_A + s·grad_B·A` (1858), Adam trên `equiv_grad`, back-project + Sylvester
    `X=solve_sylvester(BᵀB, AAᵀ, …)` (1888). `s = alpha/r` (rslora: `alpha/√r`).
- **Deploy:** chỉ chỉnh **gradient** lúc train; A,B lưu vẫn LoRA chuẩn → **rank-32 chuẩn**. ✅
- **Rủi ro implement (CAO) — phải xử đúng:**
  1. **Cặp A/B đồng thời:** adjustment **ghép grad_A & grad_B**; phải có **cả hai** trước khi chỉnh.
  2. **Thứ tự với MoE tie:** notebook `_tie_grads` **sum grad qua 128 expert** ([517-531](../../Continuer_Nemotron_Notebook.py#L517-L531))
     rồi mới step. LoRA-Pro adjust phải chạy **SAU `_tie_grads`, TRƯỚC `optimizer.step`** ([686-690](../../Continuer_Nemotron_Notebook.py#L686-L690))
     và **bỏ qua các param expert đã tie** (chiều expert làm hỏng `BᵀB`) — chỉ áp cho LoRA non-expert
     (q/k/v/o/in_proj/out_proj/lm_head 2-D thuần). Áp riêng từng cặp (A,B) cùng module.
  3. Vì notebook dùng AdamW thuần (không DeepSpeed), bê **chỉ phần toán** của `lorapro_full_adjustment`
     thành 1 hàm `_lorapro_adjust(pairs)` chạy trên `p.grad` tại chỗ; **không** bê Adam của họ (để AdamW
     notebook xử). Đây là sai khác có chủ đích — ghi rõ trong header.
- **Để cuối** (effort cao). **Falsify:** vượt exp21. **Rollback:** không gọi `_lorapro_adjust`.

### exp53 — B6-10 LoRA-Dropout (chống overfit continue-train) ✅ rẻ
- **Không repo riêng** (paper 2404.09610 **withdrawn**, repo tác giả rỗng — xác nhận refs/README). Đây là
  **knob có sẵn trong PEFT**, không cần port.
- **Edit:** [8](../../Continuer_Nemotron_Notebook.py#L8) `LORA_DROPOUT = 0.0` → **`0.05`** (dò {0.05, 0.1}).
  Dropout chỉ tác động lúc train; A,B lưu vẫn chuẩn → **rank-32 chuẩn**. ✅
- **Lưu ý:** lm_head LoRA thủ công [329](../../Continuer_Nemotron_Notebook.py#L329) cũng nhận `LORA_DROPOUT`
  → tự đồng bộ. **Falsify:** vượt baseline, 5-nhóm-mạnh không tụt. **Rollback:** `LORA_DROPOUT=0.0`.

---

## PHẦN 4bis — ĐỢT 5: coverage có-kiểm-chứng (module-map TĨNH, KHÔNG cần GPU)

> Nguồn: 2 deep-research + đọc source vLLM `v0.12.0`. **Không probe runtime** (không có máy chạy vLLM) —
> dùng **module-map tĩnh** dưới đây làm chân lý. Grader = `vllm/vllm-openai:v0.12.0` (recipes page chính
> thức; Blackwell sm_120 ép cao).

**Module-map TĨNH cho `NemotronHForCausalLM` @ vLLM 0.12.0** (đã đọc `models/nemotron_h.py` +
`lora/layers/fused_moe.py`):

| Module | Áp LoRA? | 0.86 đã target? |
|---|---|---|
| q/k/v/o_proj, in_proj, out_proj, lm_head | ✅ | có |
| **embed_tokens** (`embedding_modules`) | ✅ | **CHƯA → free coverage** |
| experts gate_up/down (`SharedFusedMoE ⊂ FusedMoE`, cần `use_ep=False` = đúng trên 1-GPU) | ✅ | có (LIVE) |
| router `mixer.gate`, conv1d, A_log/D/dt_bias | ❌ | — |

**Kết luận:** expert-LoRA của 0.86 **đang được áp** (không có "expert chết"). Module applicable **duy nhất**
0.86 chưa chạm = **embed_tokens**. ⇒ Đợt-5 = coverage tới embed + tinh chỉnh in_proj + HP-sweep.

### exp55 — B6-19 embed_tokens coverage (free, đã xác nhận tĩnh) ⭐ rẻ nhất
- **Cơ sở:** `embedding_modules: {embed_tokens, lm_head}` trong `nemotron_h.py@v0.12.0` → embed_tokens
  **áp được**; 0.86 chỉ dùng lm_head ⇒ embed_tokens = capacity mới, robust-apply, **rank-32 chuẩn**.
- **Edit:** thêm `"embed_tokens"` vào `TARGET_MODULES` [30-40](../../Continuer_Nemotron_Notebook.py#L30-L40):
```python
# >>> EXP55 START — B6-19 embed_tokens free coverage. Rollback: gỡ "embed_tokens" khỏi TARGET_MODULES
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj",
    "in_proj", "out_proj", "lm_head", "embed_tokens",
]
# >>> EXP55 END
```
- **Lưu ý:** PEFT bọc embedding bằng `Embedding`-LoRA (`lora_embedding_A/B`) — KHÔNG `modules_to_save`
  (full-weight embed → vLLM **hard ValueError**). Kiểm save chỉ ra key `lora_embedding_A/B` cho embed_tokens.
  Unsloth có thể bỏ embed như nó bỏ lm_head → nếu vậy, **thêm thủ công** y hệt lm_head
  [323-336](../../Continuer_Nemotron_Notebook.py#L323-L336).
- **Validate:** trainable tăng đúng phần embed; **Deploy check PHẦN 2C** (key `lora_embedding_*`, không
  `modules_to_save`). **Falsify:** macro pass@1 ↑; nếu phẳng → embed ít trọng số cho task này. **Rollback:** gỡ tên.

### exp56 — B6-18 in_proj 3-slice fan-out (bản đầy đủ của exp50)
- **Cơ sở:** vLLM tách `in_proj` thành `in_proj_z/qkv/ba`. Để **3× capacity SSM**, train **3 LoRA rank-32 độc
  lập** lên 3 dải output của in_proj (thay vì 1 rank-32 phủ cả khối, hay exp50 mask 1 slice).
- **Khác exp50:** exp50 = 1 LoRA, mask về 1 slice (tập trung rank-32 vào dt/B/C). exp56 = **3 LoRA riêng**,
  mỗi slice rank-32 → tổng 3×32. Cần **đúng offset z/qkv/ba** (cùng recon `modeling_nemotron_h` như exp50).
- **Code path:** tạo 3 LoRA-on-in_proj thủ công, mỗi cái mask grad/weight về dải của nó (mở rộng cơ chế
  `_partial_zero_B` exp50 thành 3 mask z/qkv/ba). Effort cao hơn exp50.
- **Quan hệ:** làm exp50 trước (rẻ, 1 slice); chỉ lên exp56 nếu exp50 cho tín hiệu dương trên bit.
- **Falsify:** vượt exp50 trên bit; nếu = → tách 3 slice không thêm giá trị (rank-32 gộp đã đủ).

### exp57 — B6-21 HP-sweep plain rank-32 LoRA (không method mới) ⭐ song song
- **Cơ sở:** [LR-Matters 2602.04998], [Batch-Size-Matters 2602.09492], [unified 2601.22708] — gain thật ở
  **LR × batch**, không phải method; plain LoRA tune kỹ vượt PiSSA/MiLoRA.
- **Edit:** KHÔNG sửa cấu trúc — chỉ grid config [14-16](../../Continuer_Nemotron_Notebook.py#L14-L16):
  `LEARNING_RATE ∈ {5e-6, 1e-5, 2e-5, 5e-5}` × `BATCH_SIZE ∈ {16, 32, 64}` trên **plain continue-0.86**.
- **Vì sao đáng:** rẻ về code, đo trước khi tin bất kỳ method-delta nào. Đây là **baseline thật** cho mọi exp khác.
- **Falsify:** nếu best-of-grid = 0.86 → recipe đã ở optimum HP (method mới mới đáng); nếu > → đó là cú free.

### B6-20 expert-format hygiene (không exp riêng — kiểm lúc save)
- Experts LIVE nhưng cần layout **PEFT-3D** đúng (`experts.gate_up_proj/down_proj.lora_A/B` stacked) để
  `FusedMoEWithLoRA` không đọc nhầm → garbage. Khi save adapter, **đối chiếu format expert** với chuẩn PEFT;
  notebook đã emit per-expert copy ([488-489](../../Continuer_Nemotron_Notebook.py#L488-L489)) — xác nhận tên
  khớp. Low-risk hygiene, gộp vào Deploy-check PHẦN 2C.

---

## PHẦN 5 — HÀNG "BLOCK" (KHÔNG code train; lý do + điều kiện gỡ)

> Mình **đã đọc code repo** từng cái; block vì **khuôn deploy** (PHẦN 0), không phải vì lười.

### B6-6 HiRA — BLOCK (rank/format)
- **Code thật:** [refs/hira/hira/tuners/lora.py:555,562](../../refs/hira/hira) — forward
  `x @ (W₀ ⊙ (AᵀBᵀ+1))ᵀ`; `merge()` raise `NotImplementedError` (lora.py:524), vài attr lỗi (`self.r`).
- **Vì sao block:** `ΔW = W₀ ⊙ (BA)` là **high-rank** theo thiết kế → **không biểu diễn được bằng
  rank-32 cộng** mà vLLM áp. vLLM **không có path HiRA**; nếu nạp A,B như LoRA thường sẽ áp **sai công thức**
  (cộng thay vì Hadamard). Merge vào base → ra **full-weight 30B**, không phải adapter để submit.
- **Gỡ block khi:** (a) vLLM-Nemotron-H hỗ trợ peft_type HiRA (hiện không), HOẶC (b) competition cho submit
  merged full-weight (không — yêu cầu `adapter_config.json`). ⇒ giữ block.

### B6-1 SSM-tuning (state-offset / additional-scan) — BLOCK (vLLM không chạy param)
- **Code thật:** [refs/ssm-state-tuning/modules/ssm_peft.py:133-159](../../refs/ssm-state-tuning) — param
  `ssmpeft_bias (d_state,d_inner)` cộng vào state `h` **sau** selective-scan (PyTorch, kernel gốc OK lúc
  *train*). `dim_extend.py:163-216` (additional-scan) nối thêm `d_state` → **phá khuôn kernel**.
- **Vì sao block:** param này **không phải LoRA** và nằm trong **forward của Mamba mixer**. Lúc *inference*,
  vLLM chạy **modeling Mamba của riêng nó**, **không gọi** `StateToYWithTransform` → state-offset **không
  có tác dụng**. additional-scan đổi `d_state` → vLLM kernel không tương thích.
- **Gỡ block khi:** patch được model code vLLM dùng lúc final inference (không khả thi trên Kaggle final).

### B6-3 MiLoRA — BLOCK (base-mod + rank doubling > 32)
- **Code thật:** [refs/milora/svd_init.py:139-153](../../refs/milora/svd_init.py) — **trừ `delta` khỏi
  `base_layer.weight`** rồi init A,B từ minor-SVD (mode="min", svd_init.py:27-30).
- **Vì sao block:** (a) base bị sửa → để deploy trên **base gốc**, adapter hiệu dụng =
  `scaling·(B_f@A_f − B_i@A_i)` có **rank ≤ 2r = 64 > 32** (như PiSSA convert) → **vi phạm `max_lora_rank=32`**.
- **Biến thể gỡ block (cân nhắc sau):** train MiLoRA ở **r=16** rồi convert → LoRA r=32 ≤ cap (PEFT
  `convert_pissa_to_lora`-style). Capacity giảm nửa; chỉ thử nếu hàng CODE cạn. Khi đó mới viết exp.

### B6-13 BitFit / Norm-tuning — BLOCK (modules_to_save không vLLM)
- **Code thật:** [refs/bitfit/glue_evaluator.py](../../refs/bitfit) (`--bias-terms`). RMSNorm-gains/bias là
  **full-param `modules_to_save`**, không phải `lora_A/B`.
- **Vì sao block:** vLLM LoRA-adapter format mong `lora_A/B`; norm/bias trong `modules_to_save` **không được
  vLLM nạp/áp** cho Nemotron-H → train xong không có tác dụng inference.
- **Gỡ block khi:** verify vLLM nạp `modules_to_save` cho norm/bias (rủi ro cao là không).

### B6-5 HDMoLE / B6-12 Affix — BLOCK
- HDMoLE (2409.19878): **không có code public** (xác nhận refs/README, hunt 2026-06-10). Affix-tuning
  (MambaPEFT): **không phải LoRA-adapter**, áp token lúc inference → vLLM không nạp. ⇒ không code.

---

## PHẦN 6 — Thứ tự chạy & cổng chốt

**Coverage đã chốt TĨNH (không probe — xem PHẦN 4bis module-map).** Grader = `vllm/vllm-openai:v0.12.0`.
Expert-LoRA của 0.86 **đang được áp** (không có "expert chết" để thu hồi). Module applicable **duy nhất**
0.86 chưa dùng = **embed_tokens** (exp55). Router/conv1d/SSM-param **không áp** → đã bỏ khỏi danh sách code.

**Run order (rẻ→đắt, đã lọc theo deploy + module-map tĩnh):**
1. **exp55 embed_tokens** (1 dòng `TARGET_MODULES`) — free coverage **đã xác nhận tĩnh**; rẻ & chắc nhất.
2. **exp48 Freeze-A** (≈5 dòng) — bank A/B-asymmetry, neo exp21.
3. **exp57 HP-sweep plain LoRA** (grid config, không method) — baseline thật; chạy song song.
4. **exp53 LoRA-Dropout** (1 dòng) — regularizer rẻ.
5. **exp49 Flat-LoRA** (helper sạch) — robustness cho greedy.
6. **exp50 Partial-LoRA SSM** — coverage in_proj (1 slice); nếu dương → **exp56** (3-slice fan-out).
7. **exp52 LoRA-Pro** — cuối (khó, xử MoE-tie cẩn thận).
8. **exp54 MoSLoRA** — chỉ nếu rảnh GPU (NO-GO, kỳ vọng ~0; dò LR riêng).

> **Trạng thái cuối (2026-06-10):**
> - ✅ **CODE**: exp55 (embed) · exp48 (freeze-A) · exp57 (HP-sweep) · exp53 (dropout) · exp49 (flat) ·
>   exp50→exp56 (in_proj coverage) · exp52 (LoRA-Pro). B6-20 = format hygiene lúc save.
> - 🟡 **near-free, để cuối**: exp54 MoSLoRA (maintainer reproduce = ngang LoRA).
> - ❌ **KILLED/BLOCK**: exp51 Router-LoRA (gate không áp) · HiRA · SSM-offset · MiLoRA · BitFit · HDMoLE · Affix.

**Mỗi exp dò LR riêng** {5e-6, 1e-5, 2e-5} (prereq LR-Matters) trước khi kết luận.

**Cổng chốt mọi exp (như batch-5 + deploy):**
- Regime continue-0.86, LR ≤1e-5, order gốc, `SHUFFLE_DATASET=False`.
- **Deploy check PHẦN 2C bắt buộc**: adapter chỉ có `lora_A/B` (+lm_head rename), rank ≤32, không key lạ.
- 5-nhóm-mạnh (bit/gravity/cipher/unit/numeral) không tụt > 0.5pp; macro pass@1 ↑ mới ghi tracker.

**Khi có điểm:** copy `tracker/rounds/round_template.md` → `round_<N>.md`, append vào
[tracker/leaderboard.md](../../tracker/leaderboard.md).
</content>
</invoke>
