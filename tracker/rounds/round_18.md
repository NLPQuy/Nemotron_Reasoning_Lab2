# Round 18 — exp47 Quality-gate corpus + anchored-L2 (Batch-5 D4 + D9)

**Date**: 2026-06-10
**Score**: 0.70
**Δ vs previous best**: −0.16

---

## Hypothesis

Apply a **quality gate** to the corpus (D4): drop unverified `hypothesis_formed` / `rule_unknown`
traces, keep only `rule_found` + augmentation (~16,830 rows / ~46.2M tokens), to train on cleaner
labels — protected by anchored-L2 toward θ_0.86 (D9, λ=1e-3). Continue-train regime. Expected the
cleaner subset to help or hold, with the anchor preventing drift.

## Config changes

```python
# vs baseline, continue-train regime
RESET_WEIGHTS = False
LEARNING_RATE = 1e-5
# >>> EXP47 (D4 + D9)
# corpus: quality-gated JSONL (drop hypo/unknown; keep rule_found+aug)
ANCHOR_LAMBDA = 1e-3      # D9 anchored-L2 safety net
# FULL-SIZE GUARD: ~16,830 rows / ≥90% of 46.2M tokens
```

## Training run

| Field | Value |
|-------|-------|
| Platform | Kaggle |
| GPU | — |
| Steps | 1 epoch (auto-sized) |
| LR | 1e-5 |
| Train time | — |

## Result

| Split | Score |
|-------|-------|
| Public LB | 0.70 |
| Private LB | — |

## Insights

- **Regress −0.16.** Even "cleaning" the corpus by dropping unverified traces **erodes coverage** —
  the dropped hypo/unknown traces still carried useful signal, and the smaller/re-ordered set
  underperforms the full curated corpus. Anchored-L2 again could not save a corpus edit.
- Together with exp44 (bit-shorten, 0.66): **both corpus-editing Batch-5 levers regress hard**,
  while the three non-corpus levers (exp40/exp42/exp43) all held 0.86. Clean separation.
- **Verdict reinforced:** do not edit/subset the corpus for continue-train. Add coverage at the
  solver source and keep the full ~50.5M-token corpus + order intact.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
