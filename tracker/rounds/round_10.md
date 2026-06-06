# Round 10 — exp9: Spaced-repetition data scheduling

**Date**: 2026-06-02  
**Score**: —  
**Δ vs previous best**: — (training timeout — did not complete)  

---

## Hypothesis

Re-present problems the model still gets wrong at expanding intervals (spaced repetition) instead of i.i.d. ordering, so limited steps buy more durable mastery of hard categories.

## Config changes

```python
# >>> EXP9
# Spaced-repetition scheduler: re-queues hard/unsolved problems at expanding intervals
# Requires per-problem loss tracking across steps → significant overhead
```

## Training run

| Field | Value |
|-------|-------|
| Platform | Modal |
| GPU | RTX-PRO-6000 |
| Steps | Did not complete |
| LR | 2e-4 |
| Train time | Exceeded time limit |

## Result

| Split | Score |
|-------|-------|
| Public LB | — |
| Private LB | — |

## Insights

Training did not finish within the compute budget. Per-problem loss tracking for spaced repetition adds significant per-step overhead that makes the full training run too slow. **This approach requires a more efficient implementation or a reduced step count before it can be evaluated.** Deprioritize until a lightweight tracking mechanism is designed.

## Status

- [ ] Submitted
- [ ] Result recorded in leaderboard.md
