# Round 11 — exp10: Simulated-annealing difficulty curriculum

**Date**: 2026-06-02  
**Score**: 0.84  
**Δ vs previous best**: −0.02 (baseline: 0.86)  

---

## Hypothesis

Anneal the sampling distribution from easy to hard over training — high "temperature" early (broad, easy-weighted) cooling to hard problems late — mirroring simulated annealing's coarse-to-fine optimization.

## Config changes

```python
# >>> EXP10
# SA curriculum: sampling temperature schedule over NUM_STEPS
# Early steps: oversample easy problems; late steps: oversample hard problems
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

Small regression vs baseline (−0.02). The SA curriculum does not improve over standard corpus ordering. The 1000-step budget may be too short for the curriculum schedule to have meaningful effect — easy→hard annealing needs enough steps for the "easy" phase to build a foundation. Possible to retry with more steps or a steeper annealing schedule.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
