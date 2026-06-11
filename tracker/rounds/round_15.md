# Round 15 — exp42 Anchored-L2 toward θ_0.86 (Batch-5 D9)

**Date**: 2026-06-10
**Score**: 0.86
**Δ vs previous best**: 0.00 (ties baseline)

---

## Hypothesis

Continue-train the 0.86 adapter with an **L2 anchor** on the LoRA weights toward their loaded 0.86
values (`grad += λ·(θ − θ_ref)`, λ=1e-3). This is a weight-space proximal penalty (not KL/forward
like exp16) meant to let the model adapt while preventing drift away from the good 0.86 solution.
No corpus change. Expected to protect baseline and possibly nudge up.

## Config changes

```python
# vs baseline, continue-train regime
RESET_WEIGHTS = False
LEARNING_RATE = 1e-5
# >>> EXP42 (D9)
ANCHOR_LAMBDA = 1e-3      # snapshot theta_ref at load; grad += λ·(θ−θ_ref)
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
| Public LB | 0.86 |
| Private LB | — |

## Insights

- **Holds baseline exactly (0.86).** The anchor successfully prevents regression — but with no
  corpus/coverage change to learn from, there is nothing to gain either; it converges back to 0.86.
- Confirms anchored-L2 is a **safe guard rail** for continue-train (reused in exp44/exp47 as the
  safety net for corpus-editing experiments — though it could not save those, see rounds 17/18).
- Next: the guard rail works; the missing ingredient is *new coverage* added without eroding the
  existing corpus.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
