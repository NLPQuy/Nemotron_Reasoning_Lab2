# Round 2 — exp1: Format-verified labels + truncation-robust completions

**Date**: 2026-06-02  
**Score**: 0.84  
**Δ vs previous best**: −0.02 (baseline: 0.86)  

---

## Hypothesis

Guarantee every training label is byte-exact in the canonical boxed scaffold and grader-verifiable; on truncation, keep the `\boxed{}` tail instead of hard-cutting. Removes any training signal from malformed/missing-box completions.

## Config changes

```python
# >>> EXP1
TRUNCATION_KEEP_BOXED_TAIL = True
BOXED_TAIL_TOKENS = 48
# Also: drops examples where decoded targets contain no valid \boxed{} + <|im_end|>
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

Small regression vs baseline (−0.02). Truncation tail-keeping + box-format filtering is mostly neutral — the filter removes relatively few truly malformed examples, so the corpus size barely changes. This idea is safe to compose with others but does not independently improve score.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
