# Round 8 — exp7: STaR/RFT self-generated verified-correct traces

**Date**: —  
**Score**: —  
**Δ vs previous best**: — (not evaluated)  

---

## Hypothesis

Use the current adapter to sample multiple traces per problem, keep only those whose `\boxed{}` answer is verified correct, and add deduped correct traces to the corpus before retraining — bootstrapping harder verified reasoning data.

## Config changes

```python
# >>> EXP7
# Self-play data generation: sample K traces/problem, verify, add correct ones to corpus
# Retrain on augmented corpus
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
