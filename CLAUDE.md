# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research / ideation outputs

Benchmark-climbing ideation batches (training ideas to improve the leaderboard score) are stored in [research/ideation/](research/ideation/). When asked for ideas to raise the score, **read the latest batch there first** before re-deriving, and append any new batch as `batch-<N>.md`. Each batch lists per-idea hypothesis, the exact knob/code path in `Continuer_Nemotron_Notebook.py`, expected gain, risk, and a cheap falsification test.

The matching **implementation plan** lives alongside each batch as `plan-batch-<N>.md`: it maps each idea to a standalone `exp<N>.py` with line-level edits, validation, rollback, and a recommended run order.

**Current batches:**

| Batch | File | Ideas | Target | Status |
|---|---|---|---|---|
| 1 | [batch-1.md](research/ideation/batch-1.md) | 10 (format-verified labels, answer-upweight, concise traces, rsLoRA, module realloc, LIMO curation, STaR/RFT, hot-expert untying, spaced-repetition, SA-curriculum) | 0.86 → 0.87 | exp1–exp10 generated, evaluated |
| 2 | [batch-2.md](research/ideation/batch-2.md) | 9 (DoRA, NEFTune, self-verify traces, PiSSA init, arithmetic scratchpad, anchored-KL SFT, LoRA seed-soup, preference opt, Stream-of-Search) | 0.86 → 0.88+ | exp11–exp19 generated |
| 3 | [batch-3.md](research/ideation/batch-3.md) | 8 (forking-token loss weighting, LoRA+ split LR, DoReMi mixture reweight, ESFT expert-specialized LoRA, CSP guess-solver, GroupDRO worst-category, HER forward-generation, GSPO) | 0.86 → 0.88+ | exp20–exp27 generated, evaluated |
| 4 | [plan-batch-4-data-mix.md](research/ideation/plan-batch-4-data-mix.md) | 10 (RFT-mined, A★-PO reward-weight, REDI negatives, step-localized REDI, TokenSkip compress, self-correction, length curriculum, SwS weakness, evolve/SAND, AdaSTaR reweight) | 0.86 → 0.88+ | exp29–exp39 generated, **evaluated — all regressed (best exp35 0.74 ≪ 0.86)** |
| 5 | [batch-5.md](research/ideation/batch-5.md) + [plan-batch-5.md](research/ideation/plan-batch-5.md) | 11 (D1 bit-shorten, D2 keep-boxed-tail, D3 cryptarithm coverage, D4 quality-gate, D5 EMA/warmup/clip, D6 guess-CSP, D7 WiSE-FT interp, D8 2×0.86 soup, D9 anchored-L2→0.86, D10 module-localized, D11 Muon) | 0.86 → 0.88+ | **observation-driven** (rút từ exp1–39); regime = continue-train từ 0.86; exp40–48 + soup. refs: `muon`, `lion-pytorch` cloned |
| 6 | [batch-6.md](research/ideation/batch-6.md) + [plan-batch-6.md](research/ideation/plan-batch-6.md) | LoRA-architecture + Đợt-4 deployable replacements (B6-14 MoSLoRA, B6-15 coverage, B6-16 SBoRA) | 0.86 → 0.88+ | **10 repos cloned to `refs/`** (hira, mambapeft, ssm-state-tuning, adamoe, ld-mole, lora-pro, flat-lora, milora, bitfit, **moslora**). **plan triaged by vLLM-deploy constraint** (submission = rank-32 additive LoRA on pristine base): CODE = exp48 Freeze-A / exp49 Flat-LoRA / exp50 partial-SSM / exp52 LoRA-Pro / exp53 dropout; **exp54 MoSLoRA = NO-GO/near-free** (deep-research: PEFT-maintainer 20-run repro = no diff vs LoRA, paper never tested text-math); **exp51 router-LoRA = KILLED** (mixer.gate not in vLLM supported_lora_modules → silently ignored); **BLOCK = HiRA, MoRA, ReLoRA/COLA, LoRA-GA/CorDA/KaSA/PiSSA, nonlinear-LoRA, SSM-offset, MiLoRA, BitFit**. **MANDATORY PROBE-0 first**: load the 0.86 adapter under grader's exact vLLM, check which modules actually apply — vLLM applies expert-LoRA only ≥v0.11.2 (PR #21229), so the 0.86 adapter's expert deltas MAY be silently dropped. Coverage = only surviving capacity lever; vLLM exposes in_proj as in_proj_z/qkv/ba |
| 7 | [batch-7.md](research/ideation/batch-7.md) + [plan-batch-7.md](research/ideation/plan-batch-7.md) + [batch-7-search-log.md](research/ideation/batch-7-search-log.md) | **Scope chốt 2026-06-11: ZERO-TRAIN, fusion-only.** Ingredients đủ = **5 adapter 0.86 trong 1 Kaggle dataset** (double-nested `expN-0-86/expN-0-86/`; exp32 = bạn-của-user, lineage ngoài-repo → diversity tốt). **Deliverable = 13 file fusion self-contained `exp58.py`–`exp70.py`** (đúng quy ước exp1–57: inline lõi fuse + block `# >>> EXP<N>` chứa POOL/mode/params → ghi `/kaggle/working/submission` → zip): exp58 uniform-soup, exp59 Model-Stock, exp60 LoRA-LEGO, exp61 SeedLoRA, exp62 WiSE-FT, exp63 scale, exp64 DisTaC-normeq, exp65 graft, exp66 TIES/DARE-SVD, exp67 subset-select, exp68 small-α, **exp69 KnOTS (joint-SVD align→TIES, refs/knots), exp70 Core-Space (LoRA-native core-Tr×Tr merge, refs/core-space)** — cả 2 zero-train data-free. Train-based ideas (re-seed/tail-ckpt/anneal/IterIS/LoRI) **DROP**. Lõi cũng có bản `offline/merge_lora.py` để test local. | 0.86 → 0.88+ | **deep-research + venue-verified 4-agent sweep**. Key finding: `soup_adapters.py` averages lora_A/lora_B factors separately = "Direct Merging" — WRONG (cross-terms; B·A invariant to (B·s, s⁻¹·A)); correct = **avg BA products then SVD-truncate to rank 32**. exp17 regression confounded by this. **Validate offline only** (deploy_check + `diag`: trunc-energy/drift/ingredient-cosine); **Kaggle có NO vLLM nên accuracy CHỈ qua leaderboard** (~5/day → submit best-first, exp58 trước). **PHẦN 6 of plan = full from-paper pseudocode** (SeedLoRA Eq.4-8, LoRA-LEGO MSU k-means, Model Stock `t=N·cosθ/((N-1)cosθ+1)`, DisTaC norm-rescale). refs/: model-soups, wise-ft, mergekit, knots, ties-merging, mergelm-dare, core-space, lorahub, model-stock, iteris, lori, distac, coto. **SeedLoRA & LoRA-LEGO repos confirmed codeless** |

Batch 3's plan is [plan-batch-3.md](research/ideation/plan-batch-3.md); its live-search provenance is in [batch-3-search-log.md](research/ideation/batch-3-search-log.md).

**Empirical verdict so far (see [tracker/leaderboard.md](tracker/leaderboard.md)):** every evaluated single-stage LoRA/objective tweak across Batch-1/2/3 has been ≤ 0.86 — `exp21` (LoRA+ split A/B LR, `LORAPLUS_LR_RATIO=8.0`) is the **only** non-baseline config that *ties* 0.86; everything else regressed. The conclusion recorded in the tracker is that the LoRA recipe is **not** the bottleneck.

**Batch-4 verdict (2026-06-10) — data-mix-on-top-of-0.86 is a dead direction as built.** All 8 evaluated batch-4 mixes regressed (best exp35 0.74, even clean-additive exp38 RFT / exp39 reweight only 0.72; exp29 REDI negatives catastrophic 0.37) — *after* the shuffle/epoch bug was fixed. Continue-training the 0.86 adapter for 1 epoch on a rebuilt `build_corpus` mix replaces the full curated ~50.5M-token corpus + curated training order with a smaller/re-ordered mix → **coverage erosion** dominates any lever gain. See [tracker/leaderboard.md](tracker/leaderboard.md) Batch-4 section.

**→ Next direction (decided 2026-06-10): go back to the ORIGINAL corpus.** Do *not* continue-train on `build_corpus` subset mixes. Any improvement must keep the **full curated ~50.5M-token corpus + its curated training order intact** and add value at the **source** — i.e. grow/fix the corpus inside `nemotron-master/` (`reasoners/` solvers, `augmenters/`), especially Tier-0 coverage of the 100%-unsolved `cryptarithm_guess` / `equation_numeric_guess` and the cryptarithm/bit zero-problems — then regenerate the full corpus via the `nemotron-master/` pipeline (`reasoning.py → corpus.py`), not a mix. Solver-source coverage is the only direction with a positive datapoint (see [research/cryptarithm_gap_plan.md](research/cryptarithm_gap_plan.md), [research/offline_rl_cot_sota.md](research/offline_rl_cot_sota.md)). Do not re-run exhausted single-knob LoRA tweaks or build_corpus mixes expecting a different result.

**Batch-2 recommended run order** (from [plan-batch-2.md](research/ideation/plan-batch-2.md)):
1. **First** — bank cheap LoRA-capacity wins: `exp11` (DoRA) + `exp14` (PiSSA init); also probe `exp12` (NEFTune) as near-free regularizer.
2. **Then** — attack dominant loss bucket: `exp15` (arithmetic scratchpad) / `exp13` (self-verify) for numeric slips, or `exp19` (Stream-of-Search) for combinatorial categories.
3. **Hold for 0.87→0.88+** — `exp17` (LoRA seed-soup), `exp18` (preference opt), `exp16` (anchored-KL) — highest headroom but carry compute/implementation-conflict risk.

## Known dead ends — do NOT reproduce

### On-policy RL methods (GSPO, GRPO, PPO, online RLHF, …)

**Do not propose any on-policy RL method** for this project. The infrastructure cost is prohibitive and the current setup cannot support the core requirement:

- On-policy RL *requires* regenerating rollouts from the **current policy** every N gradient steps. That means running the full 30B model under vLLM between training steps — on the same machine that is training. Kaggle cannot install vLLM (torch 2.10 has no compatible wheel). Modal could in principle interleave vLLM inference and training, but the orchestration complexity is very high and eats into the competition timeline.
- exp27 (Batch-3 GSPO) attempted this by generating rollouts **once** then training for 1000 steps — this is off-policy and the importance-weight clipping (eps=3e-4) is so tight that the loss degenerates to reward-weighted SFT anyway. The exp exists but produces no real RL signal, and it regressed on the leaderboard. The supporting `generate_rollouts_vllm.py` + `setup_runpod.sh` (see *Root scripts* below) generate those one-shot rollouts on RunPod/Modal; they do **not** make on-policy RL feasible — they only feed the off-policy `GSPO_ROLLOUTS` path.

**If the goal is learning from reward signal**, use **offline preference optimization** (exp18 / SimPO): generate chosen/rejected pairs once with `pref_generate.py`, upload as a Kaggle dataset, train normally. Honest about being off-policy, zero extra infrastructure.

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
- `investigators/` — analysis/one-off scripts for categories where the rule isn't yet found. The `crypto_*_probe.py` set (`opset`, `extended_op`, `families`, `glyph_prior`, `nonuniq`, `robust_solver`) probes the still-weak cryptarithm/*guess* categories; findings feed [research/cryptarithm_gap_plan.md](research/cryptarithm_gap_plan.md).
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

`exp1.py` through `exp27.py` at the repo root are **standalone experiment scripts** — each is a copy of `Continuer_Nemotron_Notebook.py` with exactly one idea applied. Changes are bracketed by `# >>> EXP<N> START` / `# >>> EXP<N> END` markers to make the diff obvious. They launch identically to the base script (`modal run exp<N>.py`). Each file's header comment names the batch idea, the knob changed, and the rollback instruction.

| Range | Batch | Ideas |
|---|---|---|
| exp1–exp10 | Batch 1 | format-verified labels, answer-upweight, concise traces, rsLoRA, module realloc, LIMO curation, STaR/RFT, hot-expert untying, spaced-repetition, SA-curriculum |
| exp11–exp19 | Batch 2 | DoRA, NEFTune, self-verify traces, PiSSA init, arithmetic scratchpad, anchored-KL, LoRA seed-soup, preference opt, Stream-of-Search |
| exp20–exp27 | Batch 3 | forking-token loss weighting, LoRA+ split LR (exp21 = best, ties 0.86), DoReMi mixture reweight, ESFT expert-specialized LoRA, CSP guess-solver traces, GroupDRO worst-category, HER forward-generation, GSPO (off-policy, dead end) |

**Utility / data-gen scripts** (also added in the `ml` branch):
- `soup_adapters.py` — averages multiple trained adapter checkpoints into one (supports Idea 7 / LoRA seed-soup).
- `pref_generate.py` — samples traces from the current adapter and labels them via the verifier to build preference pairs (supports Idea 8 / offline preference optimization).
- `generate_rollouts_vllm.py` — generates one-shot GSPO rollouts (`G` samples/problem) with vLLM on RunPod/Modal, writing `rollouts.jsonl` consumed by exp27 via `GSPO_ROLLOUTS=`. CLI: `--model_path / --adapter_path / --train_csv / --output / --group_size / --max_problems`.
- `setup_runpod.sh` — provisions a RunPod box (torch 2.6 + cu124, vLLM, deps) to run the rollout generator; usage `bash setup_runpod.sh [HF_TOKEN]`.

## Experiment tracking

When an experiment yields a leaderboard score:
1. Copy `tracker/rounds/round_template.md` → `tracker/rounds/round_<N>.md` and fill in every field (rounds run `round_1`…`round_12` so far).
2. Append the result row to `tracker/leaderboard.md` (the canonical scoreboard — read it before proposing the next experiment).

Current state: **best score still 0.86**. baseline (pretrained adapter, default `Continuer_Nemotron_Notebook.py`) is tied by `exp21` (LoRA+) **and now `exp40` (D5 EMA-package), `exp42` (D9 anchored-L2), `exp43` (D10 localized in_proj/out_proj continue-train)** — all 0.86, none exceed. Every other evaluated exp regressed: all of Batch-1/2/3 (27 files), all evaluated Batch-4 data-mixes (exp29–39, best exp35 0.74 ≪ 0.86), **and Batch-5: corpus-editing levers regress (exp47 quality-gate 0.70, exp44 bit-shorten 0.66) while Muon (exp41) 0.78 — only the three non-corpus optimization levers held 0.86**. Target: **0.88+**. **Standing guidance (2026-06-10): single-stage LoRA tweaks, `build_corpus` data-mixes, AND optimizer/regularizer levers are all exhausted (≤0.86); the clean Batch-5 split (non-corpus levers tie 0.86, corpus-edits regress) re-confirms coverage is the binding constraint. Next direction = go back to the ORIGINAL full corpus and add coverage at the source** (regenerate via the `nemotron-master/` pipeline with improved `reasoners/` solvers, esp. Tier-0 `cryptarithm_guess` / `equation_numeric_guess`) — keep the full ~50.5M-token corpus + curated order intact; do NOT continue-train on subset mixes.

## Research notes (`research/`)

Long-form analysis that informs but isn't an ideation batch:
- [research/data_status.md](research/data_status.md) — current corpus/category token breakdown and the flag that `corpus.py` currently trains on **unverified** CoT (`rule_unknown` / `hypothesis_formed` included).
- [research/cryptarithm_gap_plan.md](research/cryptarithm_gap_plan.md) — plan for closing the cryptarithm / unsolved-*guess* gap; pairs with the `investigators/crypto_*_probe.py` analysis scripts.
- [research/offline_rl_cot_sota.md](research/offline_rl_cot_sota.md) — survey of offline-RL / reward-weighted-SFT / CoT-compression methods to layer on top of exp21; the recommended pivot after single-stage LoRA tweaks were exhausted.

## Upstream reference: tonghuikang/nemotron (Progress Prize winner)

GitHub: https://github.com/tonghuikang/nemotron  
Blog post: https://blog.huikang.dev/2026/05/02/nemotron-progress-prize.html

`nemotron-master/` is a local copy of this repo. Studying it is the fastest way to understand what the winning pipeline does and where gaps remain.

### Corpus stats (as generated by the winning pipeline)

**Total: 17,963 examples / 50,544,389 tokens** (40.9M unmasked / 9.7M masked)

| Category | Entries | Unmasked tokens | % of corpus |
|---|---|---|---|
| bit_manipulation | 1,602 | 10,778,410 | 26.4% |
| gravity | 1,597 | 5,694,239 | 13.9% |
| cipher | 1,576 | 4,916,239 | 12.0% |
| unit_conversion | 1,594 | 3,882,764 | 9.5% |
| concatenation (aug) | 1,500 | 3,549,069 | 8.7% |
| splitting (aug) | 1,500 | 3,543,247 | 8.7% |
| equation_numeric_deduce | 596 | 3,455,149 | 8.4% |
| spelling (aug) | 648 | 2,575,296 | 6.3% |
| equation_numeric_guess | 136 | 822,948 | 2.0% |
| matching (aug) | 4,515 | 516,676 | 1.3% |
| lstrip (aug) | 300 | 441,726 | 1.1% |
| cryptarithm_deduce | 659 | 415,042 | 1.0% |
| numeral | 1,576 | 180,202 | 0.4% |
| cryptarithm_guess | 164 | 100,863 | 0.2% |

Local file sizes: `corpus.jsonl` = 41 MB, `corpus/` directory = 291 MB.

### Problem solve status per category

Current LOCAL status — source of truth = [nemotron-master/problems.jsonl](nemotron-master/problems.jsonl)
(`{id, category, status}`), re-joined 2026-06-10. Totals: rule_found 8,367 / hypothesis_formed 217 /
rule_unknown 916 (+8,463 augmentation rows with no status). Local solvers are stricter than the
upstream huikang snapshot, so several rule_found counts are lower than that snapshot's.

| Category | Total | rule_found | hypothesis_formed | rule_unknown |
|---|---|---|---|---|
| bit_manipulation | 1,602 | 1,364 | 121 | 117 |
| gravity | 1,597 | 1,597 | 0 | 0 |
| unit_conversion | 1,594 | 1,594 | 0 | 0 |
| cipher | 1,576 | 1,576 | 0 | 0 |
| numeral | 1,576 | 1,576 | 0 | 0 |
| cryptarithm_deduce | 659 | 85 | 15 | 559 |
| equation_numeric_deduce | 596 | 540 | 22 | 34 |
| **cryptarithm_guess** | **164** | **14** | **24** | **126** |
| **equation_numeric_guess** | **136** | **21** | **35** | **80** |

`cryptarithm_guess` / `equation_numeric_guess` now have 14 / 21 verified traces (small progress from
0); `cryptarithm_deduce` only 85 verified (most of its 659 traces are still unverified). These are the
weakest-coverage categories — but tiny % of the eval (see batch-5 L3 on leverage).

### Augmentation types (in `augmenters/`)

| Augmenter | N problems | Rows/problem | Technique |
|---|---|---|---|
| concatenation | 1,500 | 100 | Merge individually-bracketed symbols → single bracket |
| splitting | 1,500 | 100 | Split single bracket → individually-bracketed symbols |
| lstrip | 300 | 100 | Strip leading spaces from bracketed strings |
| spelling | ~648 | 100 | Spell out tokens char-by-char with en-dashes; source = tokenizer vocab |
| matching | ~4,515 | variable | Extract bit-column matching sections from cryptarithm reasoning files |

`matching` applies heavy downsampling: `all_absent` keeps 10%, `both_none` keeps 10%, `few_matches` (<4 matches) keeps 20%.

### Data enrichment opportunities (prioritized)

1. **Scale augmentation constants (easiest, immediate)** — increase `N_PROBLEMS` in concatenation/splitting (1,500 → 5,000+) and lstrip (300 → 2,000+); no logic changes needed.
2. **Remove matching downsampling** — relaxing the 10–20% filters could grow matching 2–5x.
3. **Solve cryptarithm_guess / equation_numeric_guess** — implement deterministic solvers in `reasoners/`; these are 100% unsolved and would add ~300 high-token examples.
4. **Add new augmentation types** — base conversion, long multiplication/division CoT, string reversal, or other step-by-step tasks that teach the model to reason sequentially.
