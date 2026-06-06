# Round 9 — exp8: Hot-expert untying of MOE_TIE_WEIGHTS

**Date**: —  
**Score**: —  
**Δ vs previous best**: — (not evaluated)  

---

## Hypothesis

Keep cold experts tied (regularized, data-efficient) but untie the top-k "hot" experts so they can specialize, recovering MoE capacity that full tying currently suppresses.

## Config changes

```python
# >>> EXP8
# MOE_TIE_WEIGHTS modified: identify top-k hot experts by activation frequency,
# untie their LoRA slices; keep remaining cold experts tied
```

## Training run

| Field | Value |
|-------|-------|
| Platform | — |
| GPU | — |
| Steps | — |
| LR | — |
| Train time | — |

## Result

| Split | Score |
|-------|-------|
| Public LB | — |
| Private LB | — |

## Insights

Not yet evaluated.

## Status

- [ ] Submitted
- [ ] Result recorded in leaderboard.md
