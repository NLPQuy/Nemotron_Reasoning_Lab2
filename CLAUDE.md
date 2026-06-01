# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research / ideation outputs

Benchmark-climbing ideation batches (training ideas to improve the leaderboard score) are stored in [research/ideation/](research/ideation/). When asked for ideas to raise the score, **read the latest batch there first** (currently [research/ideation/batch-1.md](research/ideation/batch-1.md) — 10 verified, training-time ideas to go from 0.86 → 0.87) before re-deriving, and append any new batch as `batch-<N>.md`. Each batch lists per-idea hypothesis, the exact knob/code path in `Continuer_Nemotron_Notebook.py`, expected gain, risk, and a cheap falsification test.

The matching **implementation plan** is [research/ideation/plan-batch-1.md](research/ideation/plan-batch-1.md): it maps each idea to a standalone `exp<N>.py` (a copy of `Continuer_Nemotron_Notebook.py` at repo root) with line-level edits, validation, rollback, and a recommended run order. Keep a `plan-batch-<N>.md` alongside each batch.

## Competition context

This repo is for the **NVIDIA Nemotron Model Reasoning Challenge** (Kaggle). The deliverable is a **LoRA adapter** for the open base model `Nemotron-3-Nano-30B-A3B` (a Mamba/MoE hybrid, `modeling_nemotron_h`), packaged as `submission.zip` (must contain `adapter_config.json`). See [competition_info.md](competition_info.md) for full rules (in Vietnamese).

Hard constraints any adapter must satisfy — these drive most design decisions in the training code:
- `max_lora_rank: 32` (so `LORA_RANK = 32`).
- Inference is greedy (`temperature: 0.0`), `max_tokens: 7680`, `max_model_len: 8192`.
- The model must emit its final answer inside a LaTeX `\boxed{...}`. The grader extracts the boxed value and accepts a string match or relative error within `1e-2`. Binary strings are matched exactly (no numeric tolerance) — see `compare_answer` in [nemotron-master/reasoning.py](nemotron-master/reasoning.py).
- Final inference runs under vLLM, so the adapter must be vLLM-loadable.

## Two independent training approaches

The repo holds **two separate codebases** that pursue the same goal but do not share code:

### 1. `Continuer_Nemotron_Notebook.py` (root) — Unsloth single-file trainer

A self-contained fine-tuning script that runs in **three modes from the same file**, selected at runtime by environment variables (top of file):
- `IS_KAGGLE` (`KAGGLE_KERNEL_RUN_TYPE` set) — installs wheels from Kaggle datasets, reads the pre-tokenized corpus from a Kaggle dataset snapshot, calls `run_training()` directly, and writes `submission.zip`.
- `IS_MODAL_WORKER` (`MODAL_TASK_ID` set) — runs inside the Modal container (`gpu="RTX-PRO-6000"`, asserts `sm_120`), reads the corpus from a Modal volume, and uploads the trained adapter to a Kaggle dataset.
- `IS_MODAL_LAUNCHER` (neither) — defines the Modal `app`/image/volumes and `main()` submits `train_remote.remote()`.

Non-obvious mechanics in `run_training()`:
- Forward is **monkey-patched** to use `cut_cross_entropy.linear_cross_entropy` (no logits materialization); per-token CE is stashed on `model._cached_per_token_ce` and the training loop applies the token `weights` (the corpus mask) itself.
- `lm_head` LoRA is **added manually** because Unsloth drops it for MoE; on save, `lm_head` keys are renamed to `backbone.lm_head` to match the base model.
- LoRA params are cast to fp32; base stays bf16 **except the MoE router** (`mixer.gate`), which Nemotron-H keeps in fp32 on purpose.
- `MOE_TIE_WEIGHTS` (Tinker-style): keeps all 128 expert LoRA slices identical by mean-init + **summing** grads across the expert dim before each step.
- The Mamba CUDA fast path (`is_fast_path_available`) is force-enabled after model load.

Config knobs live at the very top (`LORA_RANK`, `NUM_STEPS`, `BATCH_SIZE`, `RESET_WEIGHTS` to train fresh vs. continue from a pretrained adapter, etc.).

### 2. `nemotron-master/` — reference Progress-Prize-winning submission

A complete data-generation + Tinker-based SFT pipeline (a vendored copy of the winning solution). **It has its own [nemotron-master/CLAUDE.md](nemotron-master/CLAUDE.md) and [nemotron-master/README.md](nemotron-master/README.md) — read those before working inside it.** Work in this directory using its own `uv` environment (`pyproject.toml`, `uv.lock` live there, not at the root).

Pipeline (run in order, all `uv run` from inside `nemotron-master/`):
```
uv run python3 reasoning.py      # deterministic CoT traces -> reasoning/<id>.txt
uv run python3 augmentation.py   # synthetic aux tasks -> augmentations/<id>.txt
uv run python3 corpus.py         # tokenize + mask -> corpus.jsonl + corpus/<id>/synthetic.jsonl
uv run python3 train_sft.py      # SFT via Tinker (trainer/client.py); supports CE / importance_sampling / ppo / cispo / dro
uv run modal run upload_adapter.py
```

Architecture of the data side:
- `reasoners/<category>.py` — one deterministic solver per problem category (`bit_manipulation`, `cipher`, `cryptarithm`, `equation_numeric`, `gravity`, `numeral`, `unit_conversion`). Each emits a natural-language CoT mirroring the solver and ends with `\boxed{answer}`. `reasoning.py` only writes a trace when the solver's answer is verified correct (`status = rule_found`); otherwise the problem is left `rule_unknown` / `hypothesis_formed`.
- `reasoners/store_types.py` — the `Problem`/`Example` datatypes and decimal long-multiplication/long-division helpers used to make arithmetic CoT exact.
- `augmenters/` — generators for auxiliary string tasks (spelling, splitting, matching, concatenation, lstrip), assembled by `augmentation.py`.
- `investigators/` — analysis/one-off scripts for categories where the rule isn't yet found.
- `corpus.py` — the **single source of truth for the training token format**. Completion is `"{reasoning}\n</think>\n\\boxed{{answer}}<|im_end|>"`; the prompt (with `PROMPT_SUFFIX`) is tokenized via the chat template with `enable_thinking=True` and **masked out** (mask `0`); only the completion contributes to loss (mask `1`). This format must stay in sync with the grader's `metric_reference.py` / `query.py`.
- Static HTML dashboards (`synthetic.html`, `corpus.html`, `training.html`, `metrics.html`) visualize each stage; serve them with `./serve.sh` (http://localhost:33304/).

## Tooling (applies to `nemotron-master/`)

Per [nemotron-master/CLAUDE.md](nemotron-master/CLAUDE.md):
- **Use `uv` only — never `pip`** (and never `uv pip install` or `@latest`). Add deps with `uv add <pkg>`, run with `uv run <tool>`.
- Format/lint/types: `uv run --frozen ruff format *.py`, `uv run --frozen ruff check *.py [--fix]`, `uv run --frozen mypy *.py`.
- Tests live under `.claude/` (`[tool.pytest.ini_options] testpaths = [".claude"]`): `uv run pytest`. Run a single test with `uv run pytest <path>::<test>`.
- HTML changes are validated via the `puppeteer` MCP server (configured in `nemotron-master/.mcp.json`).

The root `Continuer_Nemotron_Notebook.py` has no local `uv` project — it is meant to run on Kaggle or be launched against Modal (`modal run Continuer_Nemotron_Notebook.py`), not executed locally.

## Experiment files (`exp<N>.py`)

`exp1.py` through `exp10.py` at the repo root are **standalone experiment scripts** — each is a copy of `Continuer_Nemotron_Notebook.py` with exactly one idea from batch-1 applied. Changes are bracketed by `# >>> EXP<N> START` / `# >>> EXP<N> END` markers to make the diff obvious. They launch identically to the base script (`modal run exp<N>.py`). Each file's header comment names the batch idea, the knob changed, and the rollback instruction.

## Experiment tracking

When an experiment yields a leaderboard score:
1. Copy `tracker/rounds/round_template.md` → `tracker/rounds/round_<N>.md` and fill in every field.
2. Append the result row to `tracker/leaderboard.md`.

Current state: **baseline 0.86** (pretrained adapter, default `Continuer_Nemotron_Notebook.py`). Target: **0.87**. All 10 exp files for batch-1 are already generated.
