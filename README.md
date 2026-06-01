# NVIDIA Nemotron Model Reasoning Challenge

Fine-tune a rank-32 LoRA adapter for `Nemotron-3-Nano-30B-A3B` (Mamba/MoE hybrid) to solve structured reasoning problems under greedy vLLM inference. The model must emit every answer inside `\boxed{...}`; the grader accepts an exact string match or relative error ≤ 1e-2. Binary strings are matched exactly.

**Current best: 0.86 | Target: 0.87 | Deadline: 15 June 2026**

## Hard inference constraints (baked into all training decisions)

| Param | Value |
|-------|-------|
| `max_lora_rank` | 32 |
| `temperature` | 0.0 (greedy) |
| `max_tokens` | 7680 |
| `max_model_len` | 8192 |
| Inference engine | vLLM |
| Deliverable | `submission.zip` containing `adapter_config.json` |

## Repository layout

```
Continuer_Nemotron_Notebook.py   # Unsloth single-file trainer (Kaggle / Modal)
exp1.py … exp10.py               # One experiment per batch-1 idea (copy + one change)
nemotron-master/                 # Reference Tinker-based pipeline (own uv env)
research/ideation/               # batch-N.md + plan-batch-N.md
tracker/
  leaderboard.md                 # Score history
  rounds/round_N.md              # Per-experiment notes
competition_info.md              # Full rules (Vietnamese)
```

## Approach 1 — `Continuer_Nemotron_Notebook.py` (primary)

Self-contained Unsloth trainer. Three runtime modes selected by env vars:

| Mode | Trigger | What it does |
|------|---------|--------------|
| `IS_MODAL_LAUNCHER` | default (neither below) | Defines Modal app; `modal run` submits the remote job |
| `IS_MODAL_WORKER` | `MODAL_TASK_ID` set | Runs on RTX-PRO-6000, uploads adapter to Kaggle dataset |
| `IS_KAGGLE` | `KAGGLE_KERNEL_RUN_TYPE` set | Installs wheels offline, trains, writes `submission.zip` |

**Launch (Modal):**
```bash
modal run Continuer_Nemotron_Notebook.py
```

Config knobs are at the top of the file: `LORA_RANK`, `NUM_STEPS`, `BATCH_SIZE`, `LEARNING_RATE`, `RESET_WEIGHTS`, `MOE_TIE_WEIGHTS`, `SHUFFLE_DATASET`, `TARGET_MODULES`.

### Running an experiment

Each `exp<N>.py` is a copy of the base script with exactly one idea from batch-1 applied. Changes are bracketed by `# >>> EXP<N> START` / `# <<< EXP<N> END` markers.

```bash
modal run exp1.py   # Format-verified labels + truncation-robust completions
modal run exp2.py   # Up-weight \boxed{} tokens in loss
# … etc.
```

See [research/ideation/batch-1.md](research/ideation/batch-1.md) for hypotheses and expected gains, and [research/ideation/plan-batch-1.md](research/ideation/plan-batch-1.md) for line-level edit instructions.

## Approach 2 — `nemotron-master/` (reference pipeline)

Tinker-based SFT pipeline with deterministic CoT data generation. Run everything with `uv` from inside `nemotron-master/`:

```bash
uv run python3 reasoning.py      # CoT traces → reasoning/<id>.txt
uv run python3 augmentation.py   # Synthetic aux tasks → augmentations/<id>.txt
uv run python3 corpus.py         # Tokenize + mask → corpus.jsonl
uv run python3 train_sft.py      # SFT (CE / importance_sampling / ppo / cispo / dro)
uv run modal run upload_adapter.py
```

See [nemotron-master/CLAUDE.md](nemotron-master/CLAUDE.md) and [nemotron-master/README.md](nemotron-master/README.md) before working in this directory.

## Tracking results

After each submission:
1. Copy `tracker/rounds/round_template.md` → `tracker/rounds/round_<N>.md` and fill in hypothesis, config diff, training details, and LB score.
2. Add a row to `tracker/leaderboard.md`.
