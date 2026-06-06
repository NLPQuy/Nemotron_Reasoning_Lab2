# Round 7 — exp6: LIMO/s1 difficulty + diversity corpus curation

**Date**: 2026-06-02  
**Score**: 0.79  
**Δ vs previous best**: −0.07 (baseline: 0.86)  

---

## Hypothesis

Down-select the corpus to fewer, harder, category-diverse verified traces ("less is more") instead of training on many near-trivial repeats, following the LIMO/s1 finding that quality > quantity for SFT.

## Config changes

```python
# >>> EXP6
# Corpus filtered to hard/diverse subset before training
# Fewer examples, higher average difficulty per training step
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
| Public LB | 0.79 |
| Private LB | — |

## Insights

Moderate regression (−0.07). The LIMO "less is more" effect does not transfer here — the synthetic corpus is already fairly uniform in quality, so removing "easy" examples just reduces coverage. The model needs breadth across all categories, not just the hardest subset.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
