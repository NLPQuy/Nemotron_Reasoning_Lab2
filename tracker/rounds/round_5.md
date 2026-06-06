# Round 5 — exp4: rsLoRA √r scaling

**Date**: 2026-06-02  
**Score**: 0.50  
**Δ vs previous best**: −0.36 (baseline: 0.86)  

---

## Hypothesis

Replace vanilla α/r LoRA scaling with rank-stabilized α/√r so gradients don't collapse at the high (32) rank forced by the competition constraint.

## Config changes

```python
# >>> EXP4
use_rslora = True  # α/√r instead of α/r in LoraConfig
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
| Public LB | 0.50 |
| Private LB | — |

## Insights

Worst single-change result so far (−0.36). rsLoRA's α/√r scaling significantly increases the effective learning rate at rank 32 — combined with the existing LR of 2e-4 this likely causes training instability or overshooting. **If rsLoRA is retried, reduce LR substantially (try 5e-5 or lower) to compensate for the larger effective step size.**

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
