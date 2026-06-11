# Leaderboard Tracker

Target: **0.88+** | Current best: **0.86** (baseline; tied by exp21, exp40, exp42, exp43)

| Round | Exp | Score | Δ vs baseline | Config summary | Notes |
|-------|-----|-------|---------------|----------------|-------|
| baseline | — | 0.86 | — | `Continuer_Nemotron_Notebook.py` default | Starting point |
| 2 | exp1 | 0.84 | −0.02 | `TRUNCATION_KEEP_BOXED_TAIL=True` + box-filter | Mostly neutral; safe to compose |
| 3 | exp2 | 0.76 | −0.10 | Answer-token loss up-weighting | Weight multiplier too aggressive |
| 4 | exp3 | 0.58 | −0.28 | Concise anti-truncation traces | Severe regression; trace compression hurts reasoning |
| 5 | exp4 | 0.50 | −0.36 | rsLoRA `use_rslora=True` | Worst result; LR too high for rsLoRA at rank 32 |
| 6 | exp5 | 0.84 | −0.02 | Target-module reallocation (narrow set) | Default module set already near-optimal |
| 7 | exp6 | 0.79 | −0.07 | LIMO/s1 hard-subset curation | Coverage loss hurts more than difficulty gain |
| 8 | exp7 | — | — | STaR/RFT self-generated traces | Not evaluated yet |
| 9 | exp8 | — | — | Hot-expert untying | Not evaluated yet |
| 10 | exp9 | DNF | — | Spaced-repetition scheduling | Training timeout — too slow to complete |
| 11 | exp10 | 0.84 | −0.02 | SA difficulty curriculum | Small regression; step budget too short for curriculum |
| 12 | exp21 | 0.86 | 0.00 | LoRA+ split A/B LR (`LORAPLUS_LR_RATIO=8.0`) | **Ties baseline** — only Batch-3 config holding 0.86; safe continue-from base |

---

## Batch-3 status (exp20–exp27)

exp21 (LoRA+) is the **only Batch-3 idea that holds baseline (0.86)**. Every other Batch-3 exp
**regressed** (tụt ít → tụt nhiều). Exact per-exp scores not yet individually recorded — fill in
when available.

| Exp | Idea | Score | Verdict |
|---|---|---|---|
| exp20 | High-entropy forking-token loss weighting | regressed (TBD) | < baseline |
| **exp21** | **LoRA+ split A/B learning rates** | **0.86** | **= baseline (best)** |
| exp22 | DoReMi-style category mixture reweighting | regressed (TBD) | < baseline |
| exp23 | ESFT hot-expert MoE LoRA specialization | regressed (TBD) | < baseline |
| exp24 | Self-contained CSP traces for guess tasks | regressed (TBD) | < baseline |
| exp25 | GroupDRO worst-category objective | regressed (TBD) | < baseline |
| exp26 | HER-style forward generation for guess tasks | regressed (TBD) | < baseline |
| exp27 | GSPO sequence-level policy objective | regressed (TBD) | < baseline (off-policy dead-end, see CLAUDE.md) |

**Takeaway:** single-stage LoRA/objective tweaks have now been exhausted across Batch-1/2/3 —
**none beat 0.86, best case ties.** The bottleneck is not the LoRA recipe. Pivot to bottleneck-
targeted directions (selective-token / reward-weighted SFT, CoT compression, coverage-expansion
via STaR / augment-solved→harder) — see [research/offline_rl_cot_sota.md](../research/offline_rl_cot_sota.md).

---

## Batch-4 status (exp29–exp39) — data-mix on top of 0.86 (post shuffle/epoch fix)

All exps continue-train the 0.86 adapter (`RESET_WEIGHTS=False`, `LR=1e-5`, 1 epoch,
`SHUFFLE_DATASET=True`) on a rebuilt `build_corpus.py` mix. Scores below are **post**-shuffle-fix
(2026-06-10); they supersede the pre-fix artifacts (exp30=0.65 / exp34=0.64).

| Exp | Idea | Corpus | Score | Δ vs baseline |
|---|---|---|---|---|
| exp29 | EXP-C REDI negatives (λ=0.8, sign=−1) | mix_redi | **0.37** | −0.49 |
| exp33 | EXP-B TokenSkip compress bit_manip | mix_tokenskip | 0.62 | −0.24 |
| exp30 | EXP-D A★-PO reward-weighted | mix_apo | 0.70 | −0.16 |
| exp32 | EXP-F Step-localized REDI | mix_stepneg | 0.70 | −0.16 |
| exp34 | EXP-G self-correction | mix_correction | 0.70 | −0.16 |
| exp38 | EXP-A RFT-mined (clean additive) | mix_rft | 0.72 | −0.14 |
| exp39 | EXP-J AdaSTaR difficulty-reweight | mix_adastar | 0.72 | −0.14 |
| exp35 | EXP-H length curriculum (best) | mix_length | 0.74 | −0.12 |
| exp36 | EXP-I SwS weakness synthesis | mix_weakness | pending | — |
| exp37 | EXP-E evolve/SAND harder variants | mix_evolve | pending | — |

**Takeaway — the mix-corpus *pipeline* regresses, not just individual ideas.** Even the safest
pure-additive levers with no negatives/compression (exp38 RFT, exp39 AdaSTaR reweight) land at
**0.72**, and the best of the whole batch (exp35) is **0.74** — all far below 0.86, *after* the
shuffle/epoch bug was fixed. Continue-training the 0.86 adapter for 1 epoch on a rebuilt mix
replaces the full curated ~50.5M-token corpus + curated training order with a smaller/re-ordered
mix → **coverage erosion** dominates any lever gain. exp29's catastrophic **0.37** confirms
`sign=−1` negatives at λ=0.8 poison on top (gradient-ascent on wrong traces). **Verdict: data-mix-
on-top-of-0.86 via this `build_corpus` path is a dead direction as built.** Next: either (a) verify
the mix contains the FULL original corpus untouched (token-count vs 50.5M) with the lever strictly
*additive*, or (b) abandon continue-on-mix and move coverage to the solver source (Tier-0
cryptarithm/guess in `nemotron-master/reasoners/`).

---

## Batch-5 status (exp40–exp47) — continue-train from 0.86 (RESET_WEIGHTS=False, LR=1e-5, 1 epoch, order kept)

Observation-driven batch (drawn from exp1–39): all continue-train the 0.86 adapter while keeping the
full ~50.5M-token corpus + curated order. **Levers that DON'T touch the corpus tie 0.86; every
corpus-modifying lever regresses** — same coverage-erosion pattern as Batch-4, seen from a new angle.

| Exp | Idea (Batch-5) | Touches corpus? | Score | Δ vs baseline | Verdict |
|---|---|---|---|---|---|
| **exp43** | **D10 localized continue-train (in_proj/out_proj only)** | no | **0.86** | 0.00 | **= baseline (best of batch)** |
| exp40 | D5 EMA + warmup/cosine-floor + clip=1.0 + grad-accum fix | no | 0.86 | 0.00 | = baseline |
| exp42 | D9 anchored-L2 toward θ_0.86 (λ=1e-3) | no | 0.86 | 0.00 | = baseline |
| exp41 | D11 Muon optimizer (LoRA 2D) + AuxAdam | no | 0.78 | −0.08 | regress |
| exp47 | D4 quality-gate corpus + anchored-L2 | YES | 0.70 | −0.16 | regress |
| exp44 | D1 bit-shorten corpus + anchored-L2 | YES | 0.66 | −0.20 | regress |

**Takeaway:** re-confirms the 0.86 plateau from a fresh angle. The three levers that keep the corpus
intact and only change the *optimization* (EMA package, anchored-L2, localized freeze) all **hold
0.86 exactly — none exceed it**. The two that **edit the corpus** (bit-shorten D1, quality-gate D4)
regress hard (0.66 / 0.70) *even with* the anchored-L2 safety net — coverage erosion dominates again.
Muon (exp41, 0.78) is the only non-corpus lever that hurt: swapping the optimizer off AdamW is net
negative at this scale. **Net: still no config > 0.86 across Batch-1/2/3/4/5.**

---

## Score history (chronological)

- **2026-06-01** — baseline `0.86` (pretrained adapter, default config)
- **2026-06-02** — exp1 `0.84` (format-verified labels + truncation tail-keeping)
- **2026-06-02** — exp2 `0.76` (answer-token loss up-weighting)
- **2026-06-02** — exp3 `0.58` (concise anti-truncation traces)
- **2026-06-02** — exp4 `0.50` (rsLoRA √r scaling)
- **2026-06-02** — exp5 `0.84` (target-module reallocation)
- **2026-06-02** — exp6 `0.79` (LIMO/s1 curation)
- **2026-06-02** — exp9 `DNF` (spaced-repetition — training timeout)
- **2026-06-02** — exp10 `0.84` (SA difficulty curriculum)
- **2026-06-03** — exp21 `0.86` (LoRA+ split A/B LR — ties baseline; rest of Batch-3 regressed)
- **2026-06-10** — Batch-4 data-mix (post shuffle/epoch fix): exp35 `0.74` (best), exp38/exp39 `0.72`, exp30/exp32/exp34 `0.70`, exp33 `0.62`, exp29 `0.37` — all regress; mix-corpus pipeline erodes baseline coverage (exp36/exp37 pending)
- **2026-06-10** — Batch-5 continue-train from 0.86: exp43/exp40/exp42 `0.86` (tie — no corpus change: localized-freeze / EMA-package / anchored-L2), exp41 `0.78` (Muon optimizer), exp47 `0.70` (quality-gate corpus) / exp44 `0.66` (bit-shorten corpus) — corpus-editing levers regress even with anchored-L2; plateau holds at 0.86

---

## Batch-1 summary

All 8 evaluated ideas from batch-1 **regressed vs baseline**. Key takeaways:

| Idea | Exp | Score | Verdict |
|---|---|---|---|
| rsLoRA √r scaling | exp4 | 0.50 | Hard fail — LR must be reduced significantly if retried |
| Concise traces | exp3 | 0.58 | Hard fail — reasoning quality drop too severe |
| Answer up-weight | exp2 | 0.76 | Fail — weight multiplier too aggressive; retry with ×1.5 |
| LIMO/s1 curation | exp6 | 0.79 | Fail — coverage matters more than difficulty |
| Format-verified filter | exp1 | 0.84 | Soft fail (−0.02) — mostly neutral, safe to compose |
| Module reallocation | exp5 | 0.84 | Soft fail (−0.02) — default set already near-optimal |
| SA curriculum | exp10 | 0.84 | Soft fail (−0.02) — needs more steps to take effect |
| Spaced repetition | exp9 | DNF | Infeasible — needs efficiency rewrite |
| STaR/RFT | exp7 | — | Pending |
| Hot-expert untying | exp8 | — | Pending |

**Next**: single-stage LoRA tweaks are exhausted (Batch-1/2/3 all ≤ 0.86). Pivot to second-stage
on top of exp21 (the 0.86 base): selective-token / reward-weighted SFT, CoT compression, and
coverage-expansion (STaR / augment-solved→harder). See [research/offline_rl_cot_sota.md](../research/offline_rl_cot_sota.md).
