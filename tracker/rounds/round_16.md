# Round 16 — exp43 Localized continue-train (in_proj/out_proj only) (Batch-5 D10)

**Date**: 2026-06-10
**Score**: 0.86 (best of Batch-5)
**Δ vs previous best**: 0.00 (ties baseline)

---

## Hypothesis

Continue-train the 0.86 adapter but **freeze all LoRA except `in_proj`/`out_proj`** (the Mamba
mixer projections), via the `IN_PROJ_ONLY`-style mechanism generalized to `TRAIN_ONLY_MODULES`.
Idea: concentrate the limited continue-train budget on the SSM mixer where coverage gaps (bit/SSM
reasoning) live, leaving the strong attention/MLP/lm_head LoRA untouched. No corpus change.

## Config changes

```python
# vs baseline, continue-train regime
RESET_WEIGHTS = False
LEARNING_RATE = 1e-5
# >>> EXP43 (D10)
TRAIN_ONLY_MODULES = ("in_proj", "out_proj")   # freeze everything else
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

- **Ties baseline (0.86) — highest of Batch-5.** Restricting training to the SSM projections neither
  helps nor hurts: the localized lever is safe but does not unlock new capability on its own in a
  light 1-epoch continue pass.
- Motivates the Batch-6 **exp50 Partial-LoRA SSM-slice** (focus capacity inside in_proj's B/C/dt
  rows) and **exp56** (3-slice fan-out) — a finer-grained version of this localization.
- Next: localization holds 0.86; pair it with *added SSM coverage* (from-scratch exp50) rather than
  light continue, since continue alone has no new signal to fit.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
