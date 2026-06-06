# Leaderboard Tracker

Target: **0.88+** | Current best: **0.86** (baseline, tied by exp21)

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
