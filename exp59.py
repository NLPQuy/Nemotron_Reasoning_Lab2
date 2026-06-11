# %% [markdown]
# # Batch-7 LoRA fusion (ZERO-TRAIN) — exp59
# Self-contained Kaggle CPU notebook: attach the 5x0.86 adapter dataset, Run All -> submission.zip.
# >>> EXP59 START
# EXP59 — B7-11 Model Stock — closed-form center-by-angle, data-free.
# Mode: modelstock(auto_pool(), anchor=zero): dW_H = t*mean(dW), t = N*cos/((N-1)cos+1).
# Note: ⭐ submit candidate (interpolation coeff shrinks toward base when deltas disagree).
# Run on Kaggle CPU: reads /kaggle/input/<slug>/expN-0-86/(double-nest) -> /kaggle/working/submission -> submission.zip
# Rollback: none (zero-train fusion). Deploy: rank-32 additive LoRA, deploy_check PASS.
# >>> EXP59 END

# %% [code]
import glob
import json
import os
import struct
import zipfile
from pathlib import Path

import torch
import safetensors.torch as st

R_OUT = 32


def _quantile_large(x, q, cap=1_000_000):
    """torch.quantile() rejects >~16M-element inputs; sample evenly to stay under the cap."""
    x = x.reshape(-1)
    if x.numel() > cap:
        x = x[torch.linspace(0, x.numel() - 1, cap, dtype=torch.long)]
    return torch.quantile(x, q)


# ----------------------------------------------------------------------------- core io
def _resolve(d):
    """Kaggle double-nest: adapter lives at <name>/<name>/adapter_* or <name>/adapter_*."""
    d = Path(d)
    if (d / "adapter_config.json").exists():
        return d
    inner = d / d.name
    if (inner / "adapter_config.json").exists():
        return inner
    hits = list(d.glob("*/adapter_config.json"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(f"no adapter_config.json under {d}")


def _norm_key(k):
    """Canonicalise key naming across lineages to submission's deploy-correct 'backbone' scheme.
    Unsloth saves '...backbone.layers...'/'backbone.lm_head'; Tinker saves '...model.layers...'/
    'lm_head' for the SAME modules. Rename so cross-lineage adapters share one keyset."""
    if "base_model.model.backbone." in k:
        return k
    k = k.replace("base_model.model.model.layers.", "base_model.model.backbone.layers.")
    pre = "base_model.model.lm_head."
    if k.startswith(pre):
        k = "base_model.model.backbone.lm_head." + k[len(pre):]
    return k


class _LazyTensors:
    """Dict-like view over a safetensors file: loads each tensor on demand (low RAM, scales to
    many adapters). Keys are normalised; values fetched via mmap only when indexed."""

    def __init__(self, path):
        self.f = safe_open(path, framework="pt", device="cpu")
        self.keymap = {_norm_key(k): k for k in self.f.keys()}

    def __getitem__(self, k):
        return self.f.get_tensor(self.keymap[k])

    def __contains__(self, k):
        return k in self.keymap

    def __iter__(self):
        return iter(self.keymap)

    def keys(self):
        return self.keymap.keys()

    def items(self):
        for k in self.keymap:
            yield k, self[k]


def load_adapter(d):
    d = _resolve(d)
    cfg = json.load(open(f"{d}/adapter_config.json"))
    t = _LazyTensors(f"{d}/adapter_model.safetensors")
    r = cfg.get("r", R_OUT)
    s = cfg.get("lora_alpha", R_OUT) / ((r ** 0.5) if cfg.get("use_rslora") else r)
    return cfg, t, s


def lora_pairs(keys):
    """Split keys into {prefix:(A_key,B_key)} lora pairs and passthrough (non-factor) keys."""
    bases, passth = {}, []
    for k in keys:
        if k.endswith(".lora_A.weight"):
            bases.setdefault(k[:-14], {})["A"] = k
        elif k.endswith(".lora_B.weight"):
            bases.setdefault(k[:-14], {})["B"] = k
        elif k != "__metadata__":
            passth.append(k)
    pairs = {p: (v["A"], v["B"]) for p, v in bases.items() if "A" in v and "B" in v}
    return pairs, passth


def delta(t, ak, bk, s):
    """dW = s * B @ A  -> [out, in]."""
    return s * (t[bk].float() @ t[ak].float())


def svd32(dW, r=R_OUT):
    """Dense dW -> (B[out,r], A[r,in]) with B@A == rank-r truncation of dW."""
    U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
    r = min(r, S.shape[0])
    sq = S[:r].clamp_min(0).sqrt()
    return (U[:, :r] * sq).contiguous(), (sq[:, None] * Vt[:r, :]).contiguous()


def product_svd(B_blocks, A_blocks, r=R_OUT):
    """Soup without materialising [out,in]: dW = cat(B,1) @ cat(A,0), thin via QR+SVD."""
    Bc = torch.cat(B_blocks, 1)
    Ac = torch.cat(A_blocks, 0)
    Q1, R1 = torch.linalg.qr(Bc, mode="reduced")
    Q2, R2 = torch.linalg.qr(Ac.T, mode="reduced")
    U, S, Vt = torch.linalg.svd(R1 @ R2.T)
    r = min(r, S.shape[0])
    sq = S[:r].clamp_min(0).sqrt()
    return (Q1 @ (U[:, :r] * sq)).contiguous(), ((sq[:, None] * Vt[:r]) @ Q2.T).contiguous()


def _factored_uvt(B_blocks, A_blocks, r=R_OUT):
    """U[out,r], S[r], Vt[r,in] of mean/weighted sum already folded into B_blocks."""
    Bc = torch.cat(B_blocks, 1)
    Ac = torch.cat(A_blocks, 0)
    Q1, R1 = torch.linalg.qr(Bc, mode="reduced")
    Q2, R2 = torch.linalg.qr(Ac.T, mode="reduced")
    U, S, Vt = torch.linalg.svd(R1 @ R2.T)
    r = min(r, S.shape[0])
    return (Q1 @ U[:, :r]).contiguous(), S[:r].contiguous(), (Vt[:r] @ Q2.T).contiguous()


def _write(out, cfg, tens):
    Path(out).mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg)
    cfg["r"] = R_OUT
    cfg["lora_alpha"] = R_OUT
    cfg["use_rslora"] = False
    cfg["inference_mode"] = True
    json.dump(cfg, open(f"{out}/adapter_config.json", "w"), indent=2)
    st.save_file(tens, f"{out}/adapter_model.safetensors")
    print(f"wrote {out} ({len(tens)} tensors)")


# ----------------------------------------------------------------------------- shared driver
def _load_all(adirs):
    cfgs, tens, scal = zip(*[load_adapter(d) for d in adirs])
    return list(cfgs), list(tens), list(scal)


def _cast_like(x, ref):
    return x.to(ref.dtype)


def _pad_rank(B, A, r=R_OUT):
    """Zero-pad/trim (B[out,k], A[k,in]) to exactly rank r so every module is uniformly
    rank-32 like the 0.86 baseline (a rank-0/1 factor from SVD truncation is malformed)."""
    k = B.shape[1]
    if k == r:
        return B, A
    if k > r:
        return B[:, :r].contiguous(), A[:r].contiguous()
    Bp = torch.zeros(B.shape[0], r, dtype=B.dtype)
    Ap = torch.zeros(r, A.shape[1], dtype=A.dtype)
    Bp[:, :k] = B
    Ap[:k] = A
    return Bp.contiguous(), Ap.contiguous()


def run_pairwise(adirs, pair_fn, out, passth_weights=None):
    """Generic loop: merge lora pairs present in ALL adapters; passthrough keys = union (averaged
    over the adapters that have each one). Keys are pre-normalised so lineages share one scheme."""
    cfgs, tens, scal = _load_all(adirs)
    n = len(tens)
    w = passth_weights or [1.0 / n] * n
    per = [lora_pairs(set(t)) for t in tens]
    pairsets = [set(p) for p, _ in per]
    common = sorted(set.intersection(*pairsets))
    dropped = set.union(*pairsets) - set(common)
    if dropped:
        print(f"WARN: {len(dropped)} module(s) not present in every adapter -> skipped from merge")
    p0 = per[0][0]
    out_t = {}
    for prefix in common:
        ak, bk = p0[prefix]
        B, A = pair_fn(tens, scal, ak, bk)
        B, A = _pad_rank(B, A)  # uniform rank-32 (knots/corespace may truncate below 32)
        out_t[bk] = _cast_like(B.contiguous(), tens[0][bk])
        out_t[ak] = _cast_like(A.contiguous(), tens[0][ak])
    passth = set().union(*[set(pt) for _, pt in per])
    for k in passth:
        have = [i for i in range(n) if k in tens[i]]
        sw = sum(w[i] for i in have) or 1.0
        out_t[k] = _cast_like(sum((w[i] / sw) * tens[i][k].float() for i in have), tens[have[0]][k])
    _write(out, cfgs[0], out_t)


# ----------------------------------------------------------------------------- TIES / DARE primitives
def ties_merge(S, density=1.0):
    """S: [n, ...]; standard TIES (trim top-density, elect majority sign, disjoint mean)."""
    n = S.shape[0]
    flat = S.reshape(n, -1).clone()
    d = flat.shape[1]
    if density < 1.0 and d > 1:
        k = max(1, int(round(d * density)))
        thr = flat.abs().kthvalue(d - k + 1, dim=1, keepdim=True).values
        flat = flat * (flat.abs() >= thr)
    gamma = torch.sign(flat.sum(0))
    gamma[gamma == 0] = torch.sign(flat.sum()) or 1.0
    agree = (torch.sign(flat) == gamma) & (flat != 0)
    cnt = agree.sum(0).clamp(min=1)
    merged = (flat * agree).sum(0) / cnt
    return merged.reshape(S.shape[1:])


def dare_merge(S, rate=0.5, density=1.0, seed=0):
    """DARE: bernoulli drop (rate) + rescale 1/(1-rate) per task, then TIES."""
    g = torch.Generator().manual_seed(seed)
    keep = (torch.rand(S.shape, generator=g) >= rate).float() / (1.0 - rate)
    return ties_merge(S * keep, density)


def fuse_small(Z_list, substage="ties", density=1.0, rate=0.5):
    """Fuse small same-shape matrices (subspace coords) by substage."""
    S = torch.stack(Z_list, 0)
    if substage == "wavg":
        return S.mean(0)
    if substage == "dare":
        return dare_merge(S, rate, density)
    return ties_merge(S, density)


# ----------------------------------------------------------------------------- modes
def soup(adirs, weights=None, out="merged"):
    n = len(adirs)
    w = weights or [1.0 / n] * n

    def pair_fn(tens, scal, ak, bk):
        Bs = [(w[i] * scal[i]) * tens[i][bk].float() for i in range(n)]
        As = [tens[i][ak].float() for i in range(n)]
        return product_svd(Bs, As)

    run_pairwise(adirs, pair_fn, out, passth_weights=w)


def wiseft(alpha, A, B, out="merged"):
    soup([A, B], weights=[1.0 - alpha, alpha], out=out)


def scale(s, adir, out="merged"):
    cfg, t, _ = load_adapter(adir)
    out_t = {}
    for k, v in t.items():
        if k.endswith(".lora_B.weight"):
            out_t[k] = (s * v.float()).to(v.dtype)
        else:
            out_t[k] = v
    _write(out, cfg, out_t)


def normeq(adirs, out="merged"):
    """DisTaC norm-equalize each module's per-adapter ||dW||_F to the median, then soup."""
    n = len(adirs)

    def pair_fn(tens, scal, ak, bk):
        norms = []
        for i in range(n):
            d = delta(tens[i], ak, bk, scal[i])
            norms.append(d.norm())
        med = torch.median(torch.stack(norms))
        Bs, As = [], []
        for i in range(n):
            f = (med / norms[i].clamp_min(1e-12)) * scal[i] / n
            Bs.append(f * tens[i][bk].float())
            As.append(tens[i][ak].float())
        return product_svd(Bs, As)

    run_pairwise(adirs, pair_fn, out)


def modelstock(adirs, anchor="zero", out="merged"):
    """Model Stock (delta-space, anchor=base=>dW=0): dW_H = t * mean dW, t by angle."""
    assert anchor == "zero", "only delta-space (anchor=zero) supported"
    n = len(adirs)

    def pair_fn(tens, scal, ak, bk):
        Bs = [scal[i] * tens[i][bk].float() for i in range(n)]
        As = [tens[i][ak].float() for i in range(n)]
        # pairwise cosine via [r,r] inner products (no dense [out,in])
        ip = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                P = Bs[i].T @ Bs[j]
                Q = As[i] @ As[j].T
                ip[i, j] = (P * Q.T).sum()
        cs = []
        for i in range(n):
            for j in range(i + 1, n):
                den = (ip[i, i] * ip[j, j]).clamp_min(1e-24).sqrt()
                cs.append(ip[i, j] / den)
        cos = torch.stack(cs).mean() if cs else torch.tensor(1.0)
        cos = cos.clamp(-0.999, 1.0)
        t = (n * cos) / ((n - 1) * cos + 1.0)
        Bw = [(t / n) * Bs[i] for i in range(n)]
        return product_svd(Bw, As)

    run_pairwise(adirs, pair_fn, out)


def _row_chunk(in_dim, budget=4_000_000):
    return max(1, budget // max(1, in_dim))


def _ties_dense(Bs, As, density, rate=0.0, seed=0):
    """TIES/DARE on the dense delta, computed in output-row chunks (never stacks full [n,out,in]).
    Element-wise (trim-by-task / elect-sign / disjoint-mean) is local per element -> single pass."""
    n = len(Bs)
    out_dim, in_dim = Bs[0].shape[0], As[0].shape[1]
    thr = []
    for i in range(n):
        di = (Bs[i] @ As[i]).abs()
        thr.append(_quantile_large(di, 1.0 - density) if density < 1.0 else di.new_tensor(-1.0))
        del di
    thr = torch.stack(thr)
    g = torch.Generator().manual_seed(seed)
    ch = _row_chunk(in_dim)
    W = torch.empty(out_dim, in_dim)
    for s in range(0, out_dim, ch):
        e = min(s + ch, out_dim)
        Tc = torch.stack([Bs[i][s:e] @ As[i] for i in range(n)], 0)  # [n, e-s, in]
        if rate > 0.0:
            keep = (torch.rand(Tc.shape, generator=g) >= rate).float() / (1.0 - rate)
            Tc = Tc * keep
        Tc = Tc * (Tc.abs() >= thr[:, None, None])
        gamma = torch.sign(Tc.sum(0))
        agree = (torch.sign(Tc) == gamma) & (Tc != 0)
        W[s:e] = (Tc * agree).sum(0) / agree.sum(0).clamp(min=1)
        del Tc, agree
    return W


def ties(adirs, density=0.7, out="merged"):
    n = len(adirs)

    def pair_fn(tens, scal, ak, bk):
        Bs = [scal[i] * tens[i][bk].float() for i in range(n)]
        As = [tens[i][ak].float() for i in range(n)]
        return svd32(_ties_dense(Bs, As, density))

    run_pairwise(adirs, pair_fn, out)


def dare(adirs, rate=0.5, density=0.7, out="merged"):
    n = len(adirs)

    def pair_fn(tens, scal, ak, bk):
        Bs = [scal[i] * tens[i][bk].float() for i in range(n)]
        As = [tens[i][ak].float() for i in range(n)]
        return svd32(_ties_dense(Bs, As, density, rate=rate, seed=abs(hash(bk)) % (2 ** 31)))

    run_pairwise(adirs, pair_fn, out)


def seedlora(adirs, sigma=0.9, substage="ties", out="merged"):
    """SeedLoRA two-stage: Stage1 robust/conflict (element-wise, row-chunked), Stage2 subspace.
    Stage-2 basis is factored (cheap); the dense [out,in] work is done in row chunks to bound RAM."""
    n = len(adirs)

    def pair_fn(tens, scal, ak, bk):
        Bs = [scal[i] * tens[i][bk].float() for i in range(n)]
        As = [tens[i][ak].float() for i in range(n)]
        out_dim, in_dim = Bs[0].shape[0], As[0].shape[1]
        # Stage 1 per-task threshold (sampled quantile of |B@A|, formed one task at a time)
        sig = []
        for i in range(n):
            di = (Bs[i] @ As[i]).abs()
            sig.append(_quantile_large(di, sigma))
            del di
        sig = torch.stack(sig)
        # Stage 2 subspace basis of mean dW (factored, no dense [out,in])
        U, _, Vt = _factored_uvt([b / n for b in Bs], As)
        V = Vt.T
        Z = [(U.T @ Bs[i]) @ (As[i] @ V) for i in range(n)]
        Ztil = fuse_small(Z, substage)
        ch = _row_chunk(in_dim)
        W = torch.empty(out_dim, in_dim)
        for s in range(0, out_dim, ch):
            e = min(s + ch, out_dim)
            Tc = torch.stack([Bs[i][s:e] @ As[i] for i in range(n)], 0)  # [n, e-s, in]
            large = Tc.abs() >= sig[:, None, None]
            pos = large & (Tc > 0)
            neg = large & (Tc < 0)
            cnt_p = pos.sum(0)
            cnt_n = neg.sum(0)
            val = torch.where(cnt_p >= cnt_n,
                              (Tc * pos).sum(0) / cnt_p.clamp(min=1),
                              (Tc * neg).sum(0) / cnt_n.clamp(min=1))
            settled = (cnt_p + cnt_n) > 0
            recon = U[s:e] @ Ztil @ Vt  # low-rank residual reconstruction for these rows
            W[s:e] = torch.where(settled, val, recon)
            del Tc, large, pos, neg
        return svd32(W)

    run_pairwise(adirs, pair_fn, out)


def _sign_canon(a, b):
    """Canonicalise rank-1 unit sign (b a^T invariant to (a,b)->(-a,-b)); P8."""
    idx = int(a.abs().argmax())
    if a[idx] < 0:
        return -a, -b
    return a, b


def lego(adirs, k=32, out="merged"):
    """LoRA-LEGO: pool MSU (rank-1 units), k-means to k clusters, centroids -> rank-k A,B."""
    from sklearn.cluster import KMeans
    import numpy as np
    n = len(adirs)

    def pair_fn(tens, scal, ak, bk):
        msu = []
        in_dim = tens[0][ak].shape[1]
        out_dim = tens[0][bk].shape[0]
        for i in range(n):
            A = tens[i][ak].float()
            B = (scal[i] * tens[i][bk].float())
            r = A.shape[0]
            for j in range(r):
                a = A[j, :]
                b = B[:, j]
                a, b = _sign_canon(a, b)
                msu.append(torch.cat([a, b]).numpy())
        X = np.stack(msu)
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X)
        labels = km.labels_
        a_parts, b_parts = [], []
        for c in range(k):
            members = X[labels == c]
            if len(members) == 0:
                a_parts.append(np.zeros(in_dim, dtype=X.dtype))
                b_parts.append(np.zeros(out_dim, dtype=X.dtype))
                continue
            mu = members.mean(0)
            inf_mu = np.abs(mu).max()
            inf_mean = np.abs(members).max(1).mean()
            if inf_mu > 1e-12:
                mu = mu * (inf_mean / inf_mu)
            a_parts.append(mu[:in_dim])
            b_parts.append(mu[in_dim:])
        A_new = torch.from_numpy(np.stack(a_parts)).float()           # [k, in]
        B_new = torch.from_numpy(np.stack(b_parts, axis=1)).float()   # [out, k]
        return B_new, A_new

    run_pairwise(adirs, pair_fn, out)


def graft(adirs, spec, out="merged"):
    """Per-module-group source select (no SVD): copy native rank-32 A,B from chosen adapter."""
    cfgs, tens, scal = _load_all(adirs)
    # robust source map: spec values may be a full path, the outer name, or the inner name
    name_idx = {}
    for i, d in enumerate(adirs):
        name_idx[str(d)] = i
        name_idx[Path(d).name] = i
        name_idx[Path(_resolve(d)).name] = i

    def group_of(prefix):
        if "lm_head" in prefix:
            return "lm_head"
        if ".experts." in prefix or "shared_experts" in prefix or \
           "up_proj" in prefix or "down_proj" in prefix:
            return "experts"
        if any(x in prefix for x in ("q_proj", "k_proj", "v_proj", "o_proj")):
            return "attn"
        if "in_proj" in prefix or "out_proj" in prefix:
            return "in_out_proj"
        return "attn"

    per = [lora_pairs(set(t)) for t in tens]
    common = sorted(set.intersection(*[set(p) for p, _ in per]))
    p0 = per[0][0]
    out_t = {}
    for prefix in common:
        ak, bk = p0[prefix]
        i = name_idx.get(spec.get(group_of(prefix)), 0)
        if ak not in tens[i]:  # chosen source lacks this module -> first that has it
            i = next(j for j in range(len(tens)) if ak in tens[j])
        out_t[ak] = tens[i][ak]
        out_t[bk] = tens[i][bk]
    lm_src = name_idx.get(spec.get("lm_head"), 0)
    passth = set().union(*[set(pt) for _, pt in per])
    for k in passth:
        i = lm_src if k in tens[lm_src] else next(j for j in range(len(tens)) if k in tens[j])
        out_t[k] = tens[i][k]
    _write(out, cfgs[0], out_t)


def knots(adirs, substage="ties", density=0.7, out="merged"):
    """KnOTS: joint-SVD align all dW into a shared basis (concat across in), then TIES."""
    n = len(adirs)

    def pair_fn(tens, scal, ak, bk):
        Bs = [scal[i] * tens[i][bk].float() for i in range(n)]
        As = [tens[i][ak].float() for i in range(n)]
        in_dim = As[0].shape[1]
        Bcat = torch.cat(Bs, 1)                       # [out, n*r]
        Q1, R1 = torch.linalg.qr(Bcat, mode="reduced")
        nr = R1.shape[0]
        # Small = R1 @ blockdiag(As) -> [n*r, n*in]
        Small = torch.zeros(nr, n * in_dim)
        col = 0
        off = 0
        for i in range(n):
            Small[:, col:col + in_dim] = R1[:, off:off + As[i].shape[0]] @ As[i]
            col += in_dim
            off += As[i].shape[0]
        U2, s, Vh = torch.linalg.svd(Small, full_matrices=False)
        keep = s > 1e-5
        U = (Q1 @ U2[:, keep])                          # [out, K]
        s = s[keep]
        Vh = Vh[keep]                                   # [K, n*in]
        sV = [s[:, None] * Vh[:, i * in_dim:(i + 1) * in_dim] for i in range(n)]
        sV_m = fuse_small(sV, substage, density)        # [K, in]
        # B@A == U @ sV_m, truncate to rank 32 via small SVD (U orthonormal)
        Um, S2, Vt2 = torch.linalg.svd(sV_m, full_matrices=False)
        r = min(R_OUT, S2.shape[0])
        sq = S2[:r].clamp_min(0).sqrt()
        B_new = U @ (Um[:, :r] * sq)
        A_new = sq[:, None] * Vt2[:r]
        return B_new, A_new

    run_pairwise(adirs, pair_fn, out)


def corespace(adirs, substage="ties", density=0.7, out="merged"):
    """Core Space: shared bases from factors A,B; merge in small core [Ru,Rv]; reconstruct."""
    n = len(adirs)

    def pair_fn(tens, scal, ak, bk):
        Bs = [scal[i] * tens[i][bk].float() for i in range(n)]
        As = [tens[i][ak].float() for i in range(n)]
        U_basis, _ = torch.linalg.qr(torch.cat(Bs, 1), mode="reduced")   # [out, Ru]
        V_basis, _ = torch.linalg.qr(torch.cat(As, 0).T, mode="reduced")  # [in, Rv]
        Z = [(U_basis.T @ Bs[i]) @ (As[i] @ V_basis) for i in range(n)]   # [Ru, Rv]
        Z_m = fuse_small(Z, substage, density)
        M = Z_m @ V_basis.T                                              # [Ru, in]
        Um, S2, Vt2 = torch.linalg.svd(M, full_matrices=False)
        r = min(R_OUT, S2.shape[0])
        sq = S2[:r].clamp_min(0).sqrt()
        B_new = U_basis @ (Um[:, :r] * sq)
        A_new = sq[:, None] * Vt2[:r]
        return B_new, A_new

    run_pairwise(adirs, pair_fn, out)


# ----------------------------------------------------------------------------- diagnostics
def diag(merged, anchor, sources=None):
    _, tm, sm = load_adapter(merged)
    _, ta, sa = load_adapter(anchor)
    src = [load_adapter(s) for s in (sources or [])]
    pm, _ = lora_pairs(set(tm))
    pa, _ = lora_pairs(set(ta))
    te, dv, di = [], [], []
    for p, (ak, bk) in list(pm.items())[:200]:
        dW = delta(tm, ak, bk, sm)
        sv = torch.linalg.svd(dW, full_matrices=False)[1]  # svdvals() is flaky on some LAPACK
        te.append(float((sv[R_OUT:] ** 2).sum() / (sv ** 2).sum().clamp_min(1e-12)))
        if p in pa:
            aak, abk = pa[p]
            dA = sa * (ta[abk].float() @ ta[aak].float())
            dv.append(float((dW - dA).norm() / dA.norm().clamp_min(1e-12)))
        if len(src) >= 2:
            d0 = src[0][2] * (src[0][1][bk].float() @ src[0][1][ak].float())
            d1 = src[1][2] * (src[1][1][bk].float() @ src[1][1][ak].float())
            di.append(float((d0 * d1).sum() / (d0.norm() * d1.norm()).clamp_min(1e-12)))
    med = lambda x: float(torch.tensor(x).median()) if x else float("nan")
    print(f"trunc_energy={med(te):.4f} drift={med(dv):.4f} ingr_cos={med(di):.4f}")
    return med(te), med(dv), med(di)



# ----------------------------------------------------------------------------- Kaggle plumbing
# Dataset mount root (edit if your dataset mounts elsewhere). Direct path -> no slow recursive glob.
DATASET_ROOT = "/kaggle/input/datasets/mlinhbng/adapter-86"


def auto_pool(root=DATASET_ROOT):
    """Auto-discover EVERY adapter in the dataset (add more later -> swept automatically).
    Bounded-depth globs (<=3 levels) so it stays fast and never deep-walks a base-model dir."""
    roots = [root] if os.path.isdir(root) else (
        sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")))
    seen, pool = set(), []
    for r in roots:
        for pat in (f"{r}/adapter_config.json",
                    f"{r}/*/adapter_config.json",
                    f"{r}/*/*/adapter_config.json"):
            for cfgp in sorted(glob.glob(pat)):
                d = os.path.dirname(cfgp)
                if not os.path.exists(f"{d}/adapter_model.safetensors"):
                    continue
                rp = os.path.realpath(d)
                if rp in seen:
                    continue
                seen.add(rp)
                pool.append(d)
    assert pool, f"no adapters found (root={root}); check the dataset is attached"
    print(f"auto_pool: discovered {len(pool)} adapters:")
    for p in pool:
        print("  ", p)
    return pool


def find_pool(names, root=DATASET_ROOT):
    """Pick SPECIFIC adapters by folder name (direct path; shallow-glob fallback)."""
    pool = []
    for n in names:
        p = f"{root}/{n}"
        if os.path.exists(p):
            pool.append(p)
            continue
        hits = sorted(set(glob.glob(f"/kaggle/input/*/{n}") + glob.glob(f"/kaggle/input/*/*/{n}")
                          + glob.glob(f"/kaggle/input/*/*/*/{n}")), key=len)
        assert hits, f"adapter {n} not found (root={root})"
        pool.append(hits[0])
    print("POOL:", pool)
    return pool


def deploy_check(adapter_dir):
    """Inline deploy-check (mirror offline/deploy_check.py): PASS => clean rank-32 vLLM LoRA."""
    ALLOWED = {"base_model.model.backbone.lm_head.base_layer.weight"}
    cfg = json.load(open(f"{adapter_dir}/adapter_config.json"))
    with open(f"{adapter_dir}/adapter_model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr = {k: v for k, v in hdr.items() if k != "__metadata__"}
    problems, ranks = [], set()
    if cfg.get("peft_type") != "LORA":
        problems.append(f"peft_type != LORA ({cfg.get('peft_type')})")
    if (cfg.get("r") or 0) > 32:
        problems.append(f"r > 32 ({cfg.get('r')})")
    if cfg.get("modules_to_save"):
        problems.append(f"modules_to_save set: {cfg.get('modules_to_save')}")
    for k, meta in hdr.items():
        import re as _re
        m = _re.search(r"(lora_embedding_[AB]|lora_[AB])", k)
        if m:
            ranks.add(min(meta["shape"]))
        elif k not in ALLOWED:
            problems.append(f"unexpected key: {k}")
    if ranks and max(ranks) > 32:
        problems.append(f"lora rank > 32: {sorted(ranks)}")
    print(f"deploy-check r={cfg.get('r')} ranks={sorted(ranks)} use_rslora={cfg.get('use_rslora')}")
    if problems:
        print("FAIL — not deploy-clean:")
        for p in problems:
            print("  x", p)
        raise SystemExit(1)
    print("PASS — clean rank-32 vLLM LoRA adapter.")


def zip_submission(out_dir="/kaggle/working/submission", zip_path="/kaggle/working/submission.zip"):
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            z.write(f"{out_dir}/{f}", f"submission/{f}")
    print(f"wrote {zip_path}")


def subset_select(pool, thresh=0.95, sample=24):
    """Greedy diversity subset by ingredient cosine (no accuracy signal; diag-driven, P:2A)."""
    tens = [load_adapter(d) for d in pool]
    pairs, _ = lora_pairs(set(tens[0][1]))
    keys = list(pairs.items())[:sample]
    n = len(pool)
    cos = torch.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            cs = []
            for _, (ak, bk) in keys:
                di = delta(tens[i][1], ak, bk, tens[i][2])
                dj = delta(tens[j][1], ak, bk, tens[j][2])
                cs.append(float((di * dj).sum() / (di.norm() * dj.norm()).clamp_min(1e-12)))
            cos[i, j] = cos[j, i] = sum(cs) / len(cs)
    # start from the most distant pair, add any adapter whose max cos to the set < thresh
    i0, j0 = divmod(int((cos + torch.eye(n)).argmin()), n)
    sel = [i0, j0]
    for k in range(n):
        if k in sel:
            continue
        if max(float(cos[k, s]) for s in sel) < thresh:
            sel.append(k)
    sel = sorted(set(sel))
    print(f"subset cos matrix=\n{cos}\nselected idx={sel} -> {[pool[i] for i in sel]}")
    return [pool[i] for i in sel]


# %% [code]
# >>> EXP59 START
OUT = "/kaggle/working/submission"
POOL = auto_pool()
modelstock(POOL, anchor="zero", out=OUT)
# >>> EXP59 END

# %% [code]
# ---- validate (offline, no vLLM) + package ----
deploy_check(OUT)
zip_submission(OUT)
print("exp59 done: /kaggle/working/submission.zip")
