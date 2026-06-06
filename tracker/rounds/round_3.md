# Round 3 — exp2: Up-weight \boxed{} / critical answer tokens in the loss

**Date**: 2026-06-02  
**Score**: 0.76  
**Δ vs previous best**: −0.10 (baseline: 0.86)  

---

## Hypothesis

Give the answer-bearing span (`\boxed{...}` and the `</think>\n\boxed{` scaffold) a mildly higher loss weight, aligning the objective closer to the exact-match grader.

## Config changes

```python
# >>> EXP2
ANSWER_TOKEN_WEIGHT = <higher than 1.0>
# Completion mask weights boosted for \boxed{} span tokens
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
| Public LB | 0.76 |
| Private LB | — |

## Insights

Significant regression (−0.10). Up-weighting answer tokens hurts reasoning quality — focusing the loss on the boxed span comes at the cost of the reasoning trace signal. The weight multiplier is too aggressive; if retried, use a much smaller boost (e.g. ×1.5 instead of a large multiplier).

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
