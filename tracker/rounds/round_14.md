# Round 14 — exp41 Muon optimizer for LoRA-2D + AuxAdam (Batch-5 D11)

**Date**: 2026-06-10
**Score**: 0.78
**Δ vs previous best**: −0.08

---

## Hypothesis

Replace AdamW with **Muon** (Newton-Schulz orthogonalized momentum) for the 2D LoRA matrices, with
an auxiliary AdamW ("AuxAdam") for everything else (router, 1D/embedding-like params). Muon's
orthogonalized updates are reported to improve conditioning and convergence for matrix params at no
extra memory. Continue-train regime, no corpus change. Expected ≥ baseline.

## Config changes

```python
# vs baseline, continue-train regime
RESET_WEIGHTS = False
LEARNING_RATE = 1e-5
# >>> EXP41 (D11) — refs/muon/muon.py inlined
MUON_LR = 0.5e-3           # Muon for 2D LoRA params
# AuxAdam (AdamW) for the rest
```

## Training run

| Field | Value |
|-------|-------|
| Platform | Kaggle |
| GPU | — |
| Steps | 1 epoch (auto-sized) |
| LR | MUON_LR=0.5e-3 (2D) + AuxAdam |
| Train time | — |

## Result

| Split | Score |
|-------|-------|
| Public LB | 0.78 |
| Private LB | — |

## Insights

- **Regress −0.08.** Swapping the optimizer off AdamW is **net negative** at this scale/regime —
  the only non-corpus Batch-5 lever that actually hurt (EMA/anchor/localized all held 0.86).
- Likely the Muon LR / Newton-Schulz step count is not tuned for rank-32 continue-train of a
  pretrained adapter; the orthogonalization perturbs the already-good 0.86 LoRA directions.
- **Verdict: drop Muon for this task.** AdamW (with the D5 package) is the right base optimizer.

## Status

- [x] Submitted
- [x] Result recorded in leaderboard.md
