# Round 6 — exp5: Reasoning-critical target-module & rank reallocation

**Date**: 2026-06-02  
**Score**: 0.84  
**Δ vs previous best**: −0.02 (baseline: 0.86)  

---

## Hypothesis

Route the rank-32 budget to MLP/expert, `o_proj`, and Mamba `mixer` projections where reasoning concentrates, rather than spreading thinly across low-value modules.

## Config changes

```python
# >>> EXP5
TARGET_MODULES = [
    # Focused subset: MLP experts + mixer projections + o_proj
    # Removed low-value modules (e.g. q_proj, k_proj) from the default list
]
```

## Training run

| Field | Value |
|-------|-------|
| Platform | Modal |
| GPU | RTX-PRO-6000 |
| Steps | 1000 |
| LR | 2e-4 |
| Train time | — |

## Result

| Split | Score |
|-------|-------|
| Public LB | 0.84 |
| Private LB | — |

## Insights

Small regression vs baseline (−0.02). Narrowing target modules does not improve over the broad default set. The default `TARGET_MODULES` covering q/k/v/o/up/down/in/out/lm_head may already be well-chosen. Module reallocation alone is insufficient — the capacity is already near-optimal in the baseline.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
