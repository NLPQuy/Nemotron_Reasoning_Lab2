# Round 4 — exp3: Difficulty-aware concise reasoning traces (anti-truncation)

**Date**: 2026-06-02  
**Score**: 0.58  
**Δ vs previous best**: −0.28 (baseline: 0.86)  

---

## Hypothesis

Shorten/compress training reasoning traces proportional to problem difficulty so `\boxed{}` reliably fits inside the 7680-token greedy budget, reducing truncation losses.

## Config changes

```python
# >>> EXP3
# Concise trace generation: traces compressed/shortened in nemotron-master/
# Anti-truncation: enforces trace length cap before tokenization
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
| Public LB | 0.58 |
| Private LB | — |

## Insights

Severe regression (−0.28). Aggressively compressing traces removes crucial intermediate reasoning steps; the model loses the ability to reason correctly even when the answer fits in context. Concise traces trade reasoning quality for length budget — the trade-off is very unfavorable here. **Do not use aggressive trace compression.**

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
