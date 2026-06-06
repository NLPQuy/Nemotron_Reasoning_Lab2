# Round 12 — exp21 LoRA+ (Batch-3 Idea 2)

**Date**: 2026-06-03  
**Score**: 0.86  
**Δ vs previous best**: 0.00 (ties baseline)  

---

## Hypothesis

LoRA+ assigns a higher learning rate to the LoRA **B** matrix than the **A** matrix
(`LORAPLUS_LR_RATIO=8.0`), which the LoRA+ paper argues corrects the imbalanced dynamics of
the two factors and improves feature learning at no extra parameter cost. Expected a small
uplift over the default single-LR LoRA baseline at rank 32.

## Config changes

```python
# vs baseline Continuer_Nemotron_Notebook.py
LORA_RANK = 32
LORA_ALPHA = 32
NUM_STEPS = 1000
BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MOE_TIE_WEIGHTS = True
# >>> EXP21
LORAPLUS_LR_RATIO = 8.0   # B-matrix LR = 8× A-matrix LR
```

## Training run

| Field | Value |
|-------|-------|
| Platform | — (Modal / Kaggle) |
| GPU | — |
| Steps | 1000 |
| LR | 2e-4 (A), 1.6e-3 effective (B via ×8) |
| Train time | — |

## Result

| Split | Score |
|-------|-------|
| Public LB | 0.86 |
| Private LB | — |

## Insights

- **No regression, no gain.** LoRA+ at ratio 8.0 lands exactly on the baseline (0.86) — the
  split-LR does not hurt, but the default LoRA dynamics were not the bottleneck either.
- This makes exp21 the **only Batch-3 config that holds baseline**; all other Batch-3 ideas
  regressed (see leaderboard Batch-3 section). So exp21 is the current **safe continue-from base**
  for any second-stage work (offline-RL / STaR / selective-token SFT — see
  [research/offline_rl_cot_sota.md](../../research/offline_rl_cot_sota.md)).
- Next: rather than chase more single-stage LoRA tweaks (all ≤ baseline so far), pivot to the
  coverage-expansion / selective-loss directions that target the actual bottleneck
  (verbose-trace token budget + 0%-pass categories).

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
</content>
