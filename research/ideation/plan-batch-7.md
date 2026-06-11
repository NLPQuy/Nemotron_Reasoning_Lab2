# Plan — Batch 7 (LoRA merging / soup), cho Codex

> **Audience: Codex.** Mỗi task self-contained: mục tiêu / file thật / **code tham khảo Ở ĐÂU trong `refs/`**
> & port THẾ NÀO / validate / rollback. Ý tưởng gốc: [batch-7.md](batch-7.md). Paper venue-verified:
> [batch-7-search-log.md](batch-7-search-log.md). Refs đã clone: [refs/README.md](../../refs/README.md) mục Batch-7.
>
> **Cấu trúc plan:** PHẦN 0 luật deploy · PHẦN 1 **lõi fuse dùng chung** · PHẦN 2 validation KHÔNG-vLLM ·
> PHẦN 3 danh sách exp · PHẦN 4 pitfalls · PHẦN 5 refs map · **PHẦN 6 thuật toán cho method codeless**
> (SeedLoRA, LoRA-LEGO, Model Stock — đọc từ paper).
>
> ## SCOPE (chốt với user 2026-06-11): **ZERO-TRAIN, chỉ đổi CÁCH FUSE.**
> Ingredients đã đủ = 5 adapter 0.86 sẵn trong Kaggle dataset. **KHÔNG train lại bất cứ gì.** Mọi idea cần
> train/re-seed/activation **BỎ khỏi scope**: ~~B7-7 re-seed~~, ~~B7-9 tail-ckpt~~, ~~B7-10 repair-anneal~~,
> ~~B7-15 IterIS~~ (cần forward-activation), ~~B7-16 LoRI freeze-A~~. SeedLoRA (B7-12) **vẫn giữ** vì nó là
> *fusion method* — chạy thẳng trên 5 adapter sẵn có (diversity đã có nhờ exp32 lineage khác). Còn lại **13 idea
> fusion zero-train**, mỗi cái đọc 5 adapter → fuse → ghi `submission.zip`.
>
> ## DELIVERABLE (theo yêu cầu user — tiện submit Kaggle):
> **Mỗi idea = 1 file `expN.py` SELF-CONTAINED** (đúng quy ước exp1–57), chạy thẳng trong Kaggle notebook CPU:
> đọc `/kaggle/input/<slug>/expN-0-86/...` → fuse → ghi `/kaggle/working/submission/` → deploy_check inline → zip.
> Để KHÔNG copy-paste lệch 11 lần: viết **lõi fuse PHẦN 1 một lần** (đã test local), rồi mỗi `expN.py` **inline y
> nguyên lõi đó** + một block `# >>> EXP<N> START/END` chứa CONFIG (adapter nào, mode nào, tham số) + lời gọi mode.
> Phần khác nhau giữa các file CHỈ nằm trong block marker — giống hệt cách exp1–57 đang làm. Mapping số: **PHẦN 3**.

## Môi trường THẬT — Kaggle Dataset (ảnh user 2026-06-11)

Các adapter nằm trong **một Kaggle Dataset** (21.3 GB, "Data Explorer Version 1"), attach vào notebook tại
`/kaggle/input/<dataset-slug>/`. **Đặc điểm quan trọng — KHÁC giả định cũ:**

```
/kaggle/input/<slug>/
  exp32-0-86/  exp32-0-86/  adapter_config.json  adapter_model.safetensors   # ⚠️ DOUBLE-NEST
  exp40-0-86/  exp40-0-86/  adapter_config.json  adapter_model.safetensors
  exp43-0-86/  exp43-0-86/  adapter_config.json  adapter_model.safetensors
  exp49-0-86/  exp49-0-86/  adapter_config.json  adapter_model.safetensors
  exp53-0-86/  exp53-0-86/  adapter_config.json  adapter_model.safetensors
```

1. **DOUBLE-NEST:** adapter thật ở `<name>/<name>/adapter_*`, KHÔNG phải `<name>/adapter_*`. `load_adapter` phải
   auto-resolve (xem 1B sửa lại). Tên dùng **dấu gạch** (`exp40-0-86`), không phải `exp40`.
2. **Pool thật = {exp32, exp40, exp43, exp49, exp53}** — KHÔNG có baseline/exp21/exp42 như plan cũ giả định.
   Suffix `-0-86` = "exp này đạt 0.86". Nếu muốn thêm baseline/exp21/exp42 làm ingredient → phải upload chúng vào dataset.
3. **Pool đều 0.86, có diversity tốt:** cả 5 folder đạt 0.86. **`exp32-0-86` là adapter của bạn-của-user, train bằng
   recipe/lineage KHÁC repo này** (đừng nhầm với exp32=0.70 step-localized REDI trong tracker batch-4 — khác nguồn).
   Đây là **lợi thế cho soup**: ingredient train khác recipe = diversity chức năng cao hơn (đúng điều kiện DiWA/SeedLoRA
   cần để soup vượt best-single). exp40/exp43 = batch-5 cùng họ (continue-train từ 0.86); exp49/exp53 = batch-6 (Flat-LoRA/
   dropout); exp32 = ngoài-repo. **Khuyến nghị:** dùng cả 5 cho B7-1, và ưu tiên các cặp/tập **gồm exp32** ở B7-3/diag vì
   nó xa các adapter còn lại nhất (diversity cao nhất → soup có cái để trung bình).
4. **Merge chạy Ở ĐÂU:** trong **Kaggle notebook CPU** (engine không cần GPU/vLLM) — đọc `/kaggle/input/<slug>/`,
   ghi `/kaggle/working/submission/`, rồi `deploy_check` + zip thành `submission.zip`. Đúng ràng buộc Kaggle-only.
   (Hoặc tải dataset về chạy local nếu tiện; engine giống hệt.)

Quy ước plan: `ROOT = /kaggle/input/<slug>`; một "adapter dir" truyền cho engine = đường tới folder NGOÀI
(`$ROOT/exp40-0-86`) — engine tự xuống lớp trong.

---

## PHẦN 0 — LUẬT SỐNG/CHẾT (đọc trước khi viết 1 dòng)

Verify 2026-06-11 trên adapter thật [submission/adapter_config.json](../../submission/adapter_config.json) +
header safetensors (12,011 keys):

**L0.1 — Config thật:** `peft_type=LORA, r=32, lora_alpha=32, use_rslora=False` ⇒ **scaling α/r = 1.0** ⇒
**ΔW = B@A** trực tiếp. Engine vẫn tính `s_i=α_i/r_i` tổng quát rồi gấp vào B (phòng ingredient khác config).

**L0.2 — Key shape (verify):** mỗi module `P`: `P.lora_A.weight=[r,in]`, `P.lora_B.weight=[out,r]` ⇒ `B@A=[out,in]`.
Module: `in_proj(46) out_proj(46) down_proj(5934) up_proj(5934) q/k/v/o_proj(12) lm_head(2)`. down/up_proj đông vì
**per-expert** (`...experts.<E>.down_proj.lora_A`). Engine xử lý **mỗi cặp A/B độc lập** → expert tự đúng.

**L0.3 — LUẬT MERGE ĐÚNG (lý do tồn tại batch):** merge trên **tích ΔW=B@A**, KHÔNG average `A`,`B` rời.
`B@A` bất biến với `(B·g, g⁻¹·A)` ⇒ average factor tạo cross-term `B_iA_j` phá ΔW. Verify 3-0:
[knots/merging_functions.py](../../refs/knots/merging_functions.py), [mergekit](../../refs/mergekit), SeedLoRA Eq.5-8.
**`soup_adapters.py` hiện tại làm SAI** (average từng tensor key) → giữ chỉ để rollback.

**L0.4 — Giữ rank-32 + deploy:** mọi mode kết thúc bằng **SVD-truncate ΔW về r=32** → cặp A/B mới; set
`lora_alpha=r=32` ở config ra ⇒ scaling=1 ⇒ `B_new@A_new=ΔW_merge`. Output PHẢI qua
[offline/deploy_check.py](../../offline/deploy_check.py) (peft_type=LORA, r≤32, key chỉ `lora_*` hoặc
`backbone.lm_head.base_layer.weight`).

**L0.5 — Key không-factor:** key KHÔNG phải `lora_A/lora_B` (vd `lm_head.base_layer.weight` full-weight,
`__metadata__`) → **average thẳng** (đúng, không factorization). `lora_embedding_A/B` nếu có → product-merge như A/B.

> **Nguyên tắc:** engine đúng L0.3/L0.4 một lần → mọi idea là 1 flag.

---

## PHẦN 1 — Lõi fuse (viết 1 lần, test local, rồi INLINE vào mỗi expN.py)

> CPU thuần, deps `safetensors` + `torch`(CPU). Hai cách tồn tại — viết cả hai:
> 1. **`offline/merge_lora.py`** — module nguồn-sự-thật để **dev + test local** (`submission/` làm mẫu). Có `argparse`.
> 2. **Mỗi `expN.py`** (chạy Kaggle) **inline y nguyên các hàm lõi này** (load_adapter/_resolve/product_svd/svd32/
>    lora_pairs/_write + hàm mode tương ứng) — vì Kaggle notebook KHÔNG có `offline/` của repo. Phần khác nhau giữa
>    các expN.py CHỈ là block `# >>> EXP<N>` (POOL + mode + params + ghi submission.zip). Giống hệt quy ước exp1–57.

### 1A — Product-SVD qua QR factor xếp chồng (KHÔNG tạo `[out,in]` cho soup)
`ΔW_merge = Σ w_i s_i B_iA_i = B_cat @ A_cat`, `B_cat=[w_1s_1B_1|…]` (out×nr), `A_cat=[A_1;…]` (nr×in), nr≤160.
```
Q1,R1 = qr_reduced(B_cat)        # Q1:[out,nr] R1:[nr,nr]
Q2,R2 = qr_reduced(A_cat.T)      # Q2:[in,nr]  R2:[nr,nr]
U,S,Vt = svd(R1 @ R2.T)          # [nr,nr] rẻ
sq=sqrt(S[:32]); B_new=Q1@(U[:,:32]*sq); A_new=(sq[:,None]*Vt[:32])@Q2.T
```
Element-wise (TIES/DARE/SeedLoRA-stage1/LEGO) thì **buộc** tạo `[out,in]` per-module — nặng RAM, làm tuần tự.

### 1B — Khung file
```python
#!/usr/bin/env python3
"""Batch-7 LoRA merge engine (product-space SVD, L0.3/L0.4). CPU-only.
Modes:
  soup       ADIRS...                      uniform product-SVD soup            (B7-1)
  wsoup      --weights .. ADIRS...          weighted product-SVD soup          (B7-8)
  wiseft     --alpha a A B                  pairwise interp [1-a,a]            (B7-3)
  scale      --s S ADIR                     mul every lora_B by S (no SVD)     (B7-4)
  ties|dare  --density D [--rate P] ADIRS.. elementwise trim/sign/drop+SVD32   (B7-5)
  graft      --spec spec.json ADIRS...      per-module-group source select    (B7-6)
  modelstock ADIRS... --anchor BASE0_or_zero closed-form center-by-angle      (B7-11)
  seedlora   --sigma Q --substage ties ADIRS...   two-stage SeedLoRA          (B7-12)
  lego       --k 32 ADIRS...               MSU rank-wise k-means              (B7-13)
  normeq     --target med ADIRS...          DisTaC norm-equalize (pre-step)    (B7-14)
  diag       --diag DIR --anchor A [--sources ..]   offline CPU diagnostics
All modes write OUT/{adapter_model.safetensors, adapter_config.json} (alpha=r=32)."""
import json, sys, argparse
from pathlib import Path
import torch, safetensors.torch as st
R_OUT = 32

def _resolve(d):                              # Kaggle double-nest: <name>/<name>/adapter_* hoặc <name>/adapter_*
    d = Path(d)
    if (d/"adapter_config.json").exists(): return d
    inner = d/d.name
    if (inner/"adapter_config.json").exists(): return inner
    hits = list(d.glob("*/adapter_config.json"))   # fallback: bất kỳ subdir nào có config
    if hits: return hits[0].parent
    raise FileNotFoundError(f"no adapter_config.json under {d}")

def load_adapter(d):
    d = _resolve(d)
    cfg = json.load(open(f"{d}/adapter_config.json")); t = st.load_file(f"{d}/adapter_model.safetensors")
    r = cfg.get("r", R_OUT); s = cfg.get("lora_alpha", R_OUT)/(r**0.5 if cfg.get("use_rslora") else r)
    return cfg, t, s

def lora_pairs(keys):
    bases, passth = {}, []
    for k in keys:
        if k.endswith(".lora_A.weight"): bases.setdefault(k[:-14], {})["A"] = k
        elif k.endswith(".lora_B.weight"): bases.setdefault(k[:-14], {})["B"] = k
        elif k != "__metadata__": passth.append(k)
    return {p:(v["A"],v["B"]) for p,v in bases.items() if "A" in v and "B" in v}, passth

def delta(t, ak, bk, s):                      # ΔW = s·B@A  [out,in]
    return s * (t[bk].float() @ t[ak].float())

def svd32(dW, r=R_OUT):                        # dense ΔW -> (B,A) rank r
    U,S,Vt = torch.linalg.svd(dW, full_matrices=False); r=min(r,S.shape[0]); sq=S[:r].clamp_min(0).sqrt()
    return (U[:,:r]*sq).contiguous(), (sq[:,None]*Vt[:r,:]).contiguous()

def product_svd(B_blocks, A_blocks, r=R_OUT): # soup nhanh, không tạo [out,in]
    Bc=torch.cat(B_blocks,1); Ac=torch.cat(A_blocks,0)
    Q1,R1=torch.linalg.qr(Bc,mode="reduced"); Q2,R2=torch.linalg.qr(Ac.T,mode="reduced")
    U,S,Vt=torch.linalg.svd(R1@R2.T); r=min(r,S.shape[0]); sq=S[:r].clamp_min(0).sqrt()
    return (Q1@(U[:,:r]*sq)).contiguous(), ((sq[:,None]*Vt[:r])@Q2.T).contiguous()

def soup(adirs, weights=None, out="merged"):
    cfgs,tens,scal = zip(*[load_adapter(d) for d in adirs])
    keys=set(tens[0]); assert all(set(t)==keys for t in tens), "key mismatch"
    n=len(adirs); w=weights or [1.0/n]*n; pairs,passth=lora_pairs(keys); out_t={}
    for p,(ak,bk) in pairs.items():
        Bs=[(w[i]*scal[i])*tens[i][bk].float() for i in range(n)]; As=[tens[i][ak].float() for i in range(n)]
        out_t[bk],out_t[ak]=product_svd(Bs,As)
    for k in passth: out_t[k]=sum(w[i]*tens[i][k].float() for i in range(n))
    _write(out, cfgs[0], out_t)

def _write(out, cfg, tens):
    Path(out).mkdir(parents=True, exist_ok=True); cfg=dict(cfg)
    cfg["r"]=R_OUT; cfg["lora_alpha"]=R_OUT; cfg["use_rslora"]=False; cfg["inference_mode"]=True
    json.dump(cfg, open(f"{out}/adapter_config.json","w"), indent=2); st.save_file(tens, f"{out}/adapter_model.safetensors")
    print(f"wrote {out} ({len(tens)} tensors)")
# argparse dispatch: gọi hàm tương ứng từng mode (scale/wiseft/ties/dare/graft/modelstock/seedlora/lego/normeq/diag)
```

### 1C — `scale`/`wiseft` (rẻ nhất)
- `wiseft(--alpha a, A, B)` = `soup([A,B], weights=[1-a,a])`. Một dòng.
- `scale(--s S, ADIR)`: nhân mọi `*.lora_B.weight`×S (KHÔNG nhân `lm_head.base_layer.weight`), giữ A, ghi ra. Không SVD.

### 1D — `ties`/`dare` (element-wise, nặng RAM — chỉ chạy nếu B7-1 hòa 0.86)
Port [refs/knots/merging_functions.py](../../refs/knots/merging_functions.py) (`topk_values_mask` L22, `resolve_sign`
L133, `disjoint_merge` L102) + DARE [refs/mergelm-dare/.../mask_weights_utils.py](../../refs/mergelm-dare) (`mask_input_with_mask_rate`, rescale `1/(1-rate)`). Per module tuần tự:
```
dW_i = delta(t_i)                            # [out,in]
(dare) dW_i = drop_rescale(dW_i, rate)
mask_i = topk_by_|magnitude|(dW_i, density); elected = sign(Σ mask_i·dW_i)
dW = disjoint_merge(mask_i·dW_i, elected);  B_new,A_new = svd32(dW)
```

### 1E — `graft` (B7-6, không SVD)
`--spec {"attn":"exp40-0-86","in_out_proj":"exp43-0-86","experts":"exp49-0-86","lm_head":"exp40-0-86"}`. Phân loại module theo
suffix → **copy nguyên A,B từ nguồn của nhóm đó** (rank-32 tự nhiên). passthrough theo nguồn nhóm lm_head.

### 1F — `modelstock` (B7-11), `normeq` (B7-14) → xem code đầy đủ PHẦN 6.

### 1G — `seedlora` (B7-12), `lego` (B7-13) → thuật toán + pseudocode đầy đủ PHẦN 6.

### 1H — `diag` (cổng offline 2A, CPU, không model)
```python
def diag(merged, anchor, sources):
    _,tm,sm=load_adapter(merged); _,ta,sa=load_adapter(anchor); src=[load_adapter(s) for s in (sources or [])]
    pm,_=lora_pairs(set(tm)); pa,_=lora_pairs(set(ta)); te=dv=di=[]; te,dv,di=[],[],[]
    for p,(ak,bk) in list(pm.items())[:200]:                      # mẫu 200 module đủ trung vị
        dW=delta(tm,ak,bk,sm); sv=torch.linalg.svdvals(dW)
        te.append((sv[32:]**2).sum()/(sv**2).sum().clamp_min(1e-12))   # truncation_energy
        if p in pa:
            dA=delta(ta,*pa[p][::-1],sa) if False else sa*(ta[pa[p][1]].float()@ta[pa[p][0]].float())
            dv.append((dW-dA).norm()/dA.norm().clamp_min(1e-12))       # drift_vs_anchor
        if len(src)>=2:
            d0=src[0][2]*(src[0][1][bk].float()@src[0][1][ak].float()); d1=src[1][2]*(src[1][1][bk].float()@src[1][1][ak].float())
            di.append((d0*d1).sum()/(d0.norm()*d1.norm()).clamp_min(1e-12))   # ingredient cosine
    med=lambda x: float(torch.tensor(x).median()) if x else float("nan")
    print(f"trunc_energy={med(te):.4f} drift={med(dv):.4f} ingr_cos={med(di):.4f}")
```

---

## PHẦN 2 — Validation KHÔNG có vLLM (ràng buộc THẬT)

**Sự thật:** grader chạy vLLM greedy, nhưng **bạn không chạy được vLLM** (Kaggle không cài được, torch 2.10 không
wheel; vLLM chỉ trên RunPod/Modal trả phí). ⇒ **KHÔNG có local accuracy eval.** `sample_rollouts.py`/`measure_yield.py`
cần vLLM → KHÔNG dùng. Kênh accuracy DUY NHẤT = **submit leaderboard Kaggle** (~5 lượt/ngày).

### 2A — Cổng offline BẮT BUỘC (CPU, không model) — cho MỌI artifact trước khi nghĩ submit
1. `deploy_check.py <DIR>` phải in **PASS**. Fail → vứt.
2. `merge_lora.py diag --diag <DIR> --anchor $P/exp43-0-86 --sources $P/exp40-0-86 $P/exp43-0-86`
   (anchor = MỘT adapter 0.86 đã biết tốt, vd exp43 — không có "baseline" trong dataset):
   - **trunc_energy** `Σ_{k>32}σ²/Σσ²` — <~0.05 = SVD-32 gần lossless. Cao → giảm ingredient / pairwise.
   - **drift** `‖ΔW_merge−ΔW_anchor‖/‖ΔW_anchor‖` — ≈0 ⇒ merge≈adapter-0.86 anchor ⇒ **submit vô nghĩa**. Quá lớn ⇒ rủi ro tụt.
   - **ingr_cos** cosine giữa ΔW nguồn — ~1.0 ⇒ ingredient gần trùng ⇒ soup≈identity ⇒ KHÔNG submit (gate cả nhánh fusion).

### 2B — Ngân sách submit
Sinh **nhiều** ứng viên offline → chạy 2A hết → xếp hạng → **submit tối đa ~3–5 cho cả batch**, best-first, ghi `tracker/`.
Offline-diag để **chọn** 1 α / 1 s / 1 spec đáng submit, KHÔNG submit cả grid.

**Đóng gói submit (Kaggle):** engine ghi `--out /kaggle/working/submission` → chạy `python offline/deploy_check.py
/kaggle/working/submission` (PASS) → `cd /kaggle/working && zip -r submission.zip submission/` (đúng format thi:
`submission.zip` chứa `adapter_config.json`). Verify zip có `adapter_config.json` ở đúng tầng trước khi nộp.

> **PROBE-M0 = chạy 2A cho `soup` 5×0.86.** ingr_cos~1.0 → cảnh báo cả nhánh soup ≈0.86, đừng tốn submit.

---

## PHẦN 3 — Wiring 16 exp

> Cột **Validate** = lọc offline 2A. **Submit** chỉ cho ứng viên sống sót, theo 2B. KHÔNG "probe" — accuracy chỉ biết sau leaderboard.

### 13 file fusion zero-train (mỗi idea = 1 `expN.py` self-contained Kaggle)

> `$P = /kaggle/input/<slug>`. **POOL = cả 5 folder (đều 0.86):** `exp32-0-86 exp40-0-86 exp43-0-86 exp49-0-86 exp53-0-86`
> (exp32 = lineage ngoài-repo → diversity tốt, xem mục môi trường 3). Mỗi `expN.py` hard-code POOL + mode + params
> trong block `# >>> EXP<N>`. "Code" = hàm lõi PHẦN 1; "PHẦN 6.x" = thuật toán codeless.

| File | Idea | Mode + tham số (trong block EXP) | Code | Validate (2A) |
|---|---|---|---|---|
| **exp58** ⭐⭐ | B7-1 uniform soup | `soup(POOL5)` | 1A | **submit #1**; PASS+drift≠0+trunc thấp |
| **exp59** ⭐ | B7-11 Model Stock | `modelstock(POOL5, anchor=zero)` | PHẦN 6.3 | t theo góc, data-free |
| **exp60** ⭐ | B7-13 LoRA-LEGO | `lego(POOL5, k=32)` | PHẦN 6.2 (codeless) | rank-32 by construction |
| **exp61** ⭐⭐ | B7-12 SeedLoRA | `seedlora(POOL5, sigma=0.9, substage=ties)` | PHẦN 6.1 (codeless) | exp32-diversity giúp ở đây |
| **exp62** ⭐ | B7-3 WiSE-FT | `wiseft(alpha, A, B)` — quét cặp **gồm exp32** (xa nhất) | 1C | diag chọn 1 α |
| **exp63** ⭐ | B7-4 scale | `scale(s, exp43-0-86)` — quét s∈{0.9,0.95,1.05,1.1} | 1C | diag drift theo s |
| **exp64** | B7-14 DisTaC norm-eq | `normeq(POOL5)` → `soup` | PHẦN 6.4 | pre-step trước soup |
| **exp65** | B7-6 graft | `graft(POOL5, spec.json)` | 1E | bắt đầu graft exp40/43 |
| **exp66** | B7-5 TIES/DARE-SVD | `ties(POOL5, density)` / `dare(rate,density)` | 1D | chỉ nếu exp58≥0.86 |
| **exp67** | B7-2 subset-select | vòng greedy chọn tập con POOL theo diag | 1A+diag | ra 1 tập con |
| **exp68** | B7-8 small-α | `wsoup(soup, 0.84-dirs)` — **chỉ khi user upload thêm adapter 0.84** | 1B | chỉ sau exp58≥0.86 |
| **exp69** ⭐ | B7-17 KnOTS | `knots(POOL5, substage=ties)` | PHẦN 6.5 ([refs/knots](../../refs/knots)) | joint-SVD align → TIES → submit 1 |
| **exp70** ⭐ | B7-18 Core Space | `corespace(POOL5, substage=ties)` | PHẦN 6.6 ([refs/core-space](../../refs/core-space)) | LoRA-native, output low-rank |

> Mỗi file kết thúc: ghi `/kaggle/working/submission/` → `deploy_check` inline (PASS) → `zip submission.zip`.
> **PROBE-M0** = 1 cell `diag(...)` chạy đầu notebook exp58 (đọc ingr_cos/trunc/drift), không phải file riêng.

### ❌ OUT OF SCOPE (cần train/activation — user chốt KHÔNG train lại)
~~B7-7 re-seed~~ · ~~B7-9 tail-ckpt~~ · ~~B7-10 repair-anneal~~ · ~~B7-16 LoRI freeze-A~~ — đều cần TRAIN.
**B7-15 IterIS** = zero-train (least-squares closed-form) **nhưng cần forward 30B lấy activation** → hoãn vì
infra (RAM/GPU Kaggle), KHÔNG vì train. Mở lại nếu Kaggle nạp nổi model. Không viết file cho các idea này.

**Thứ tự chạy (submit best-first):** PROBE-M0 (offline) → sinh **exp58, exp59, exp60, exp61** + chạy 2A cho cả 4 →
**submit #1 = exp58** (uniform soup, phép thử nền) → nếu ≥0.86: submit thêm 1–2 cái **diversity-cao nhất theo diag**
(exp59 Model Stock / exp60 LEGO / exp61 SeedLoRA) → còn ngân sách thì exp62–66. Nếu exp58 < 0.86 và mọi diag báo
ingr_cos~1 → đóng hướng merging.

---

## PHẦN 4 — Pitfalls

- **P1** đừng average factor (L0.3). `soup_adapters.py` cũ SAI; engine mới thay.
- **P2** scaling: pool α/r=1 nên s_i=1; vẫn để công thức tổng quát. Output luôn `alpha=r=32`.
- **P3** lm_head: adapter này là cặp lora_A/B (2 key) → product-merge; checkpoint khác có thể là `base_layer.weight`
  full-weight → nhánh passthrough average. Engine robust cả hai. `scale` KHÔNG nhân base_layer.
- **P4** key mismatch: `soup` assert cùng keyset; đừng trộn 2 dạng lm_head.
- **P5** expert tied: product-merge per-key giữ tied; SVD-32 không phá tied.
- **P6** KHÔNG local accuracy eval — đừng gọi `sample_rollouts`/`measure_yield`. Validate = đại số offline (2A).
- **P7** RAM element-wise (TIES/DARE/SeedLoRA-stage1/LEGO): `in_proj` ΔW=[10304,2688] fp32 ≈110MB×n → tuần tự, `del`.
- **P8 (LEGO/SeedLoRA)** sign ambiguity của rank-1 unit `b_iaᵢᵀ` bất biến với `(a_i,b_i)→(-a_i,-b_i)` →
  **canonical-hoá dấu** mỗi MSU trước khi cluster/so sánh (vd ép `a_i[argmax|a_i|]≥0`), nếu không cluster ra rác.
- **P9 (SeedLoRA/LEGO same-rank)** ta merge các rank-32 về k=32 ⇒ hệ số variance LEGO `√r/√k=1` (bỏ qua);
  SeedLoRA Stage-2 SVD rank r=32. Khác paper (họ k=2r) — đừng copy mù hằng số của họ.

## PHẦN 5 — Refs map
- Product-SVD / full-ΔW skeleton: [refs/knots](../../refs/knots) (`merging_functions.py`, `task_merger.py`).
- TIES primitives: [refs/ties-merging/src](../../refs/ties-merging) (gọn trong knots). DARE: [refs/mergelm-dare](../../refs/mergelm-dare).
- Greedy-soup: [refs/model-soups/main.py](../../refs/model-soups) (`--greedy-soup` L81). WiSE-FT: [refs/wise-ft/src/wise_ft.py](../../refs/wise-ft).
- Model Stock notebook: [refs/model-stock/notebooks](../../refs/model-stock). DisTaC: [refs/distac/src/distac.py](../../refs/distac) (L59 norm-rescale).
- LoRI merge: [refs/lori/src/merge_4_loras.py](../../refs/lori) (= PEFT `add_weighted_adapter`; novelty ở train freeze-A).
- IterIS: [refs/iteris/IterIS.py](../../refs/iteris) (`solution_matrix` L67). Cross-check method tên: [refs/mergekit](../../refs/mergekit).
- **NEGATIVE**: [refs/lorahub/lorahub/algorithm.py](../../refs/lorahub) (sum-A/sum-B-rồi-nhân = cross-term).
- Codeless (implement từ PHẦN 6): SeedLoRA, LoRA-LEGO.

---

## PHẦN 6 — Thuật toán đầy đủ cho method KHÔNG có code (đọc từ paper 2026-06-11)

### 6.1 — SeedLoRA two-stage fusion (ICML 2025, PMLR v267:38384) — `seedlora` mode
**Đọc trực tiếp từ PDF (phương trình gốc).** Operate trên các **ΔW = τ_i = s_i·B_iA_i** (dense [out,in]) per module.
n adapter. Hai stage, mỗi entry/dimension j của ΔW phân loại 1 lần:

**Stage 1 — Robust + Conflicting (element-wise, ngưỡng σ):**
- Quét mọi dimension j qua n adapter. Đặt ngưỡng "large" `σ` (paper dùng phân vị; ta dùng **σ = quantile |τ| ở 0.9** per-module).
- **Robust** (Eq.4): nếu một tập I_j adapter có `|τ_i(j)| ≥ σ` **và cùng dấu** → `τ_robust(j) = mean_{i∈I_j} τ_i(j)`.
  (core shared knowledge — average để giữ.)
- **Conflicting** (TIES-style): nếu các nhóm có |giá trị| lớn **trái dấu** ở cùng j → giữ **chỉ dấu đa số** (majority-sign),
  bỏ entry thiểu số (chống interference).
- Các j được robust/conflict xử lý xong = "settled", **không vào Stage 2**.

**Stage 2 — Subspace Fusion cho Residual dimensions (per layer):**
- Step 1 (Eq.5): `M_avg = (1/n) Σ_i τ_i` — chỉ trên residual dims (dim không settled ở Stage 1; thực hành: mask).
- Step 2 (Eq.6): truncated SVD `M_avg ≈ U Σ Vᵀ`, `U∈[out,r]`, `V∈[in,r]`, **r=32** (ta), shared basis.
- Step 3 (Eq.7): project mỗi adapter `Z_i = Uᵀ τ_i V` (coordinate matrix [r,r] trong subspace chung).
- Step 4 (Eq.8): fuse `{Z_i}` → `Z̃` bằng **TIES/DARE/weighted-avg** (port từ 1D; mặc định weighted-avg hoặc ties trên Z̃
  — rất rẻ vì Z là [32,32]), rồi reconstruct `τ_fused = U Z̃ Vᵀ`.

**Kết hợp + deploy:** `ΔW_final = (robust/conflict entries từ Stage 1) ⊕ τ_fused (residual)` → **svd32(ΔW_final)** → cặp A/B.
**Hyperparam:** `--sigma` (quantile, default 0.9), `--substage` (ties|dare|wavg cho Step 4, default ties). r cố định 32.
**Lưu ý (P8/P9):** thuật toán này là **merge thuần, ZERO-TRAIN** — chỉ nhận 5 adapter sẵn có rồi fuse (KHÔNG cần
train seed mới; paper tự train seed chỉ là cách họ TẠO ingredient, không bắt buộc). Nó **ăn diversity**: paper claim
+4.9% GSM8K / +6.6% HumanEval vs single LoRA nhờ ingredient đa dạng. **5 adapter của ta đa dạng hơn cả literal-seeds**
(khác recipe: exp32 ngoài-repo, exp40 EMA, exp43 localized, exp49 Flat-LoRA, exp53 dropout) → đây là input lý tưởng.
PROBE-M0 ingr_cos sẽ xác nhận diversity; nếu ~1.0 (gần trùng) thì cả nhánh soup yếu, không riêng SeedLoRA.
**Pseudocode:**
```
for module P:
  T = [delta(t_i, P) for i]                  # list n × [out,in]
  sigma = quantile(|stack(T)|, 0.9)
  robust = mask(|τ|≥sigma & same-sign across subset);  conflict = mask(|τ|≥sigma & opposite-sign)
  W = zeros[out,in]
  W[robust] = mean_i T_i[robust]
  W[conflict] = mean over majority-sign group only
  res = ~(robust|conflict)
  Mavg = mean_i T_i ; U,_,Vt = svd(Mavg)[:, :32]
  Z = [U.T @ T_i @ V.T for i]                # [32,32] each
  Ztil = ties_or_wavg(Z)                     # fuse small matrices
  W[res] = (U @ Ztil @ Vt)[res]
  B_new,A_new = svd32(W)
```

### 6.2 — LoRA-LEGO MSU rank-wise clustering (ICLR 2025, 2409.16167) — `lego` mode
**Đọc từ arXiv HTML v3 (Def.1 + Eq + Theorem 3.1).** Output **rank-k by construction** (ta k=32).
- **MSU (Def.1):** đơn vị ngữ nghĩa nhỏ nhất `s_i = [a_i, b_i]` với `a_i` = **hàng i của A** (∈ℝ^in), `b_i` = **cột i của B** (∈ℝ^out).
  Mỗi LoRA rank-32 = 32 MSU. MSU là vector ghép dài `in+out`.
- **Step 1 — Pool:** `Φ = ⋃_j {s_{j,1..32}}` qua n adapter → `32n` MSU.
- **Step 2 — Cluster:** k-means Euclidean trên Φ về **k=32 cluster** (`minimize Σ_i Σ_{s∈C_i} ‖s−μ_i‖²`).
- **Step 3 — Centroid:** `μ_i = mean_{s∈C_i} s` (MSU trung bình mỗi cluster).
- **Reweight (Eq dual):**
  - *Norm-decay* (∞-norm): `μ_i ← (mean_{s∈C_i}‖s‖_∞ / ‖μ_i‖_∞) · μ_i` (bù triangle-inequality co norm).
  - *Variance* (Thm 3.1): scale output `×√r/√k`. **Ta r=k=32 ⇒ hệ số = 1 ⇒ bỏ qua** (P9).
- **Reconstruct:** tách `μ_i=[a_i,b_i]` → `A' = stack rows a_i` [32,in], `B' = stack cols b_i` [out,32]. Đã rank-32, KHÔNG cần svd32.
**Hyperparam:** `--k` (default 32). **P8:** canonical-hoá dấu MSU trước k-means (rank-1 `b_iaᵢᵀ` bất biến đổi dấu cặp).
**Pseudocode:**
```
for module P:
  MSU = []
  for i in adapters:
    A=t_i[ak]; B=t_i[bk]; s=scal_i
    for j in range(32):
      a=A[j,:]; b=(s*B)[:,j]                  # fold scaling vào b
      a,b = sign_canon(a,b)                   # P8
      MSU.append(concat(a,b))
  C = kmeans(stack(MSU), k=32)                # sklearn KMeans
  for cluster: mu=mean(members); mu*= mean(inf_norm(members))/inf_norm(mu)
  a_parts, b_parts = split(centroids, [in,out])
  A_new = stack(a_parts)            # [32,in]
  B_new = stack(b_parts, axis=1)    # [out,32]
```

### 6.3 — Model Stock (ECCV 2024 Oral, 2403.19522) — `modelstock` mode
**Closed-form, data-free** (đọc notebook [refs/model-stock/notebooks/model_stock_example.ipynb](../../refs/model-stock/notebooks)).
Cho N fine-tune `w_i` + anchor `w_0` (ở ta = **base ⇒ ΔW=0**, vì ta làm trên delta nên `w_0=0`, `w_i=ΔW_i`):
- Per layer: `w_avg = mean_i w_i`; góc `cosθ` = cosine trung bình từng cặp `(w_i−w_0)·(w_j−w_0)`.
- Hệ số nội suy `t = N·cosθ / ((N−1)·cosθ + 1)`; merged `w_H = t·w_avg + (1−t)·w_0`.
- Ở delta-space (`w_0=0`): **`ΔW_H = t · mean_i ΔW_i`** với `t = N·cosθ/((N−1)cosθ+1)` → tức **soup uniform nhân thêm hệ số t**
  (t<1 co về 0/base khi các delta lệch hướng — chống over-shoot). Sau đó **svd32(ΔW_H)** hoặc dùng product_svd với weight `t/N`.
**Khác B7-1:** B7-1 dùng t=1; Model Stock chọn t theo góc giữa ingredient → nếu ingredient gần cùng hướng (cosθ→1) thì t→1
(=soup thường); nếu lệch (cosθ nhỏ) thì t<1 co về base. **Chi phí ≈ B7-1 + tính cosθ.** N=2: `t=2cosθ/(cosθ+1)`.

### 6.4 — DisTaC norm-equalize (ICLR 2026, 2508.01148) — `normeq` mode (PRE-STEP)
**Phần data-free dùng được** (code [refs/distac/src/distac.py](../../refs/distac/src/distac.py) L59:
`zs + norm_lambda*(param − zs)` = rescale task-vector theo norm). Chẩn đoán: các adapter trong POOL train kiểu khác nhau
(exp40 EMA vs exp43 localized vs exp49 Flat-LoRA vs exp53 dropout) có thể có `‖ΔW_i‖` lệch → norm-lớn lấn át soup.
- **`normeq`:** với mỗi module, tính `target = median_i ‖ΔW_i‖_F`; rescale mỗi adapter `ΔW_i ← (target/‖ΔW_i‖_F)·ΔW_i`
  (thực hành: nhân `lora_B_i` của module đó với `target/‖ΔW_i‖_F`). Ghi ra các adapter đã equalize → **rồi soup như B7-1**.
- Là **pre-step** ghép trước mọi mode soup; rẻ, data-free. (Phần self-distillation của DisTaC bỏ — cần data/train.)

### 6.5 — KnOTS (ICLR 2025, 2410.19735) — `knots` mode — ĐỌC TỪ CODE [refs/knots/task_merger.py](../../refs/knots/task_merger.py)
**Zero-train, data-free.** Ý chính (`apply_svd` L294): align các ΔW vào **basis chung** bằng joint-SVD TRƯỚC khi
TIES → tránh "misaligned factorization" (đúng L0.3). Per module:
```
T = [delta(t_i, P) for i]                    # n × [out,in]
M = concat(T, dim=1)                          # [out, n·in]  (nối ngang)
U, s, Vh = svd(M, full_matrices=False)        # U:[out,R] shared basis; Vh:[R, n·in]
keep = s > 1e-5; U=U[:,keep]; s=s[keep]; Vh=Vh[keep]
Vs = split(Vh, n, dim=1)                       # n × [R, in]
sV_i = diag(s) @ Vs[i]                         # toạ độ adapter i trong basis chung, [R,in]
sV_merged = TIES_or_wavg(sV_i...)              # merge ĐÃ ALIGN (port topk/sign/disjoint từ knots/merging_functions.py)
dW = U @ sV_merged                             # [out,in]
B_new, A_new = svd32(dW)
```
**Tham số:** `--substage` (ties|dare|wavg, default ties), `density` cho ties. **Khác SeedLoRA stage-2:** KnOTS align
TOÀN BỘ ΔW (không tách robust/conflict trước); đơn giản hơn, ít hằng số. **P7** (RAM): `M`=[out, n·in], n=5 → in_proj
[10304, 5·2688] fp64 SVD nặng → làm tuần tự per-module, dùng fp32, `del`.

### 6.6 — Core Space (NeurIPS 2025, 2509.17786) — `corespace` mode — [refs/core-space/task_merger.py](../../refs/core-space/task_merger.py)
**Zero-train, data-free, LoRA-NATIVE (output low-rank trực tiếp).** Khác KnOTS: dựng basis chung CẢ HAI phía từ
**factor A, B** (không cần tạo dense [out,in]), merge trong **core Tr×Tr** (lossless theo paper). Per module:
```
# build basis hai phía từ factor (rẻ, nr≤160)
U_basis = orth(concat([s_i·B_i], dim=1))       # [out, Ru]  từ các B (cột) — QR/SVD của [B_1|…|B_n]
V_basis = orth(concat([A_i],     dim=0).T)     # [in,  Rv]  từ các A (hàng)
Z_i = U_basis.T @ (s_i·B_i @ A_i) @ V_basis    # core [Ru,Rv] mỗi adapter — KHÔNG tạo [out,in] (gộp qua factor)
Z_merged = TIES_or_wavg(Z_i...)                # merge trong core nhỏ
dW = U_basis @ Z_merged @ V_basis.T            # reconstruct
B_new, A_new = svd32(dW)                        # hoặc B=U_basis@..., A=...@V_basis.T nếu Z_merged đã rank≤32
```
(`Z_i = U_basisᵀ B_i A_i V_basis` tính bằng `(U_basisᵀ B_i)(A_i V_basis)` — chỉ nhân ma trận nhỏ, **không** dựng
[out,in]; rẻ hơn KnOTS + đúng tinh thần "core space".) **Tham số** giống KnOTS. Bám `apply_svd` (L320) + cách dựng
core của refs/core-space để khớp chi tiết. **Cảnh báo:** Core Space/KnOTS thiết kế cho **cross-task interference** —
trên same-task gain so với soup thường (exp58) CHƯA chứng minh; thử vì zero-cost, đừng kỳ vọng cao hơn exp58/61.

> **Kỳ vọng (trung thực):** **SeedLoRA fuse trên 5 adapter sẵn có (exp61)** là exp có bằng chứng top-tier MẠNH NHẤT
> cho "same-task LoRA merge > best single trên GSM8K/MATH" — và nó ZERO-TRAIN (5 adapter đã đóng vai "nhiều seed",
> còn đa dạng hơn vì khác recipe). Cả nhánh fusion phụ thuộc vào **diversity của 5 adapter**: nếu PROBE-M0 ingr_cos~1
> (gần trùng) thì mọi merge ≈ 0.86 và hướng này hết cửa → quay lại coverage-at-source (memory `next-step-original-corpus`).
> Nếu ingr_cos đủ thấp → exp61 (SeedLoRA) + exp58 (soup) + exp59 (Model Stock) là 3 ứng viên submit ưu tiên.
