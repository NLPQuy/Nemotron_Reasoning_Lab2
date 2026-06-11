# Round 13 — exp40 EMA + warmup/cosine + clip + grad-accum fix (Batch-5 D5)

**Date**: 2026-06-10
**Score**: 0.86
**Δ vs previous best**: 0.00 (ties baseline)

---

## Hypothesis

Stabilize the continue-train of the 0.86 adapter with a package of optimization fixes: EMA of
trainable params (ship EMA weights), 3% warmup + cosine→10% floor LR schedule, grad clip lowered
1e9→1.0, and a grad-accumulation correction (backward raw `loss_sum`, scale by `1/total_weight`
after the batch to remove the mean-of-means bias). No corpus change. Expected a small, safe uplift.

## Config changes

```python
# vs baseline, continue-train regime
RESET_WEIGHTS = False   # load 0.86 adapter
LEARNING_RATE = 1e-5
SHUFFLE_DATASET = False
# >>> EXP40 (D5)
EMA_DECAY = 0.999          # ship EMA weights
warmup = 0.03 * num_steps  # + cosine → 10% floor
clip max_norm = 1.0        # from 1e9
# grad-accum: backward loss_sum, scale 1/total_weight
```

## Training run

| Field | Value |
|-------|-------|
| Platform | Kaggle |
| GPU | — |
| Steps | 1 epoch (auto-sized) |
| LR | 1e-5 (warmup+cosine→10% floor) |
| Train time | — |

## Result

| Split | Score |
|-------|-------|
| Public LB | 0.86 |
| Private LB | — |

## Insights

- **Holds baseline exactly (0.86), no gain.** The optimization package neither hurts nor helps —
  the 0.86 recipe was not bottlenecked by EMA / LR-schedule / clip / accum-bias.
- Useful as a **safe, stable continue-train substrate**: the D5 package is now reused on top of all
  Batch-6 idea exps (added to exp48–55).
- Next: levers that keep the corpus intact tie 0.86 (see exp42/exp43); the bottleneck is elsewhere.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
