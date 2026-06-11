# Round 17 — exp44 Bit-shorten corpus + anchored-L2 (Batch-5 D1 + D9)

**Date**: 2026-06-10
**Score**: 0.66
**Δ vs previous best**: −0.20

---

## Hypothesis

Shorten the verbose `bit_manipulation` traces (D1, the largest token bucket at 26% of corpus) to
free token budget / reduce truncation, while protecting the 0.86 solution with anchored-L2 (D9,
λ=1e-3). Continue-train regime. Expected: shorter bit traces help truncation without losing accuracy,
anchor prevents drift.

## Config changes

```python
# vs baseline, continue-train regime
RESET_WEIGHTS = False
LEARNING_RATE = 1e-5
# >>> EXP44 (D1 + D9)
# corpus: bit_manipulation traces shortened (D1)
ANCHOR_LAMBDA = 1e-3       # D9 anchored-L2 safety net
```

## Training run

| Field | Value |
|-------|-------|
| Platform | Kaggle |
| GPU | — |
| Steps | 1 epoch (auto-sized) |
| LR | 1e-5 |
| Train time | — |

## Result

| Split | Score |
|-------|-------|
| Public LB | 0.66 |
| Private LB | — |

## Insights

- **Hard regress −0.20**, the worst of Batch-5. **Editing the corpus erodes coverage** even with the
  anchored-L2 guard rail — the anchor cannot compensate for changed/lost training signal.
- Bit_manipulation is the dominant category (26%); shortening its traces removes exactly the
  reasoning detail the model relies on → accuracy collapse on the largest bucket.
- **Re-confirms the standing verdict** (Batch-4): any continue-train that *modifies the corpus*
  regresses. Coverage at the source, via the `nemotron-master` solver pipeline, not trace surgery.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
