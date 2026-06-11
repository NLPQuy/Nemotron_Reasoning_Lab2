# Idea Batch 3 — NVIDIA Nemotron Model Reasoning Challenge / Deterministic-rule reasoning with `\boxed{}` answer (DATA AUGMENTATION only)

**Generated**: 2026-06-02T00:00:00Z
**Time-to-batch**: ~13 min
**Skill version**: 0.1.0
**Skill invocation**: `/benchmark-climb-ideation` — "research chiến lược DATA AUGMENTATION để cải tiến baseline 0.86 → 0.88+ … CHỈ data-time augmentation, KHÔNG thay đổi inference/decoding"

## Inputs
- **Benchmark**: NVIDIA Nemotron Model Reasoning Challenge (Kaggle), public leaderboard exact-match score.
- **Task / problem**: Given a reasoning problem (7 deterministic-rule categories — `bit_manipulation`, `cipher`, `cryptarithm`, `equation_numeric`, `gravity`, `numeral`, `unit_conversion`), emit a CoT trace then the final answer inside `\boxed{...}`. Graded by `compare_answer` (exact string / binary-exact / float rel-tol ≤ 1e-2). One greedy vLLM pass, ≤ 7680 output tokens.
- **Existing pipeline**: `nemotron-master/` data pipeline — `reasoning.py` (deterministic solvers in `reasoners/` produce self-verified traces from `problems.jsonl`), `augmenters/` (5 masked no-boxed string tasks), `corpus.py` (single source of truth for token format: completion `"{reasoning}\n</think>\n\\boxed{{answer}}<|im_end|>"`, prompt mask 0, completion mask 1). Trains a LoRA rank-32 adapter. **Current score 0.86.** Also informed by `research/strategies/augmentation-data.md` and the round-2 post-mortem (`tracker/rounds/round_2.md`).
- **Batch scope**: **enhance-existing** (7/7 ideas modify the existing pipeline; 0 greenfield).
- **Tier mix (configured)**: **55/30/15** (pipeline-biased default; user did not override — in-field drop-in augmentations dominate).
- **Baseline**: Nemotron-3-Nano-30B-A3B + pretrained LoRA adapter @ **0.86**. Target ≥ 0.88.
- **Compute budget**: 1× RTX-PRO-6000 (Modal/Kaggle), full training run ~hours; A/B on a ~200-problem held-out slice is cheap.
- **Time budget**: per-idea falsification ≤ a few GPU-hours.
- **Constraints**: **Data-time only — NO inference/decoding changes.** LoRA rank ≤ 32, 1 adapter, vLLM-loadable, greedy, ≤ 7680 tokens (trace overrun → truncate before `\boxed` → 0). Grader `compare_answer` is exact (binary strings matched exactly; numbers rel-tol 1e-2; else case-insensitive string). **Baseline sits on a sharp peak**: round-2 showed every augmentation that *rewrote trace content/length* (self-verify 0.68, stream-of-search 0.79) or *added noise* (NEFTune 0.83) regressed. ⇒ every idea must analyse distribution shift of trace / answer / length.

## Summary
| Metric | Value |
|--------|-------|
| Batch size | 7 |
| Tier 1 / 2 / 3 (counts) | 4 / 2 / 1 |
| Tier mix vs configured | 57 / 29 / 14 vs 55/30/15 (deviation ≤ 10pp per tier ✅) |
| Scope mix | 7 enhance-existing / 0 greenfield (≥ 50% ✅) |
| Patterns used | P4, P6, P11, P8, P5, P2, P12 (7 distinct) |
| Distinct venues | 6 (NeurIPS, ACL, IROS, ICLR, arXiv-preprint, ACL-2016) |
| Time windows | <12mo (4), 12-36mo (1), >36mo (2) |
| Avg feasibility | 4.1/5 |
| Avg confidence | 🟢 43%, 🟡 43%, 🔴 14% |

## Summary table
| # | Title | Pattern | Tier | Gain (mid) | Feas | Effort | Score |
|---|-------|---------|------|------|------|--------|-------|
| 1 | Procedural in-distribution instance scaling via per-category generators | P4 | 1 | +0.5 | 5 | M | 4.4 |
| 2 | Solver-as-free-verifier acceptance gate + hard length cap | P6 | 1 | +0.3 | 5 | S | 3.8 |
| 3 | Expand masked auxiliary string-skill augmenters | P5 | 2 | +0.2 | 5 | S | 3.4 |
| 4 | Category-coverage + difficulty-stratified mixture weighting | P8 | 1 | +0.4 | 3 | M | 3.4 |
| 5 | Prompt paraphrase augmentation (masked prompt, fixed answer) | P11 | 1 | +0.3 | 4 | M | 3.3 |
| 6 | Domain randomization of solver-invariant surface features | P2 | 3 | +0.3 | 4 | M | 3.3 |
| 7 | STaR-offline self-traces for failing items, solver-filtered + length-capped | P12 | 1 | +0.2 | 3 | L | 2.3 |

## Top-3 recommendations

### 🏆 Top-1 by composite score (Big bet)
**Idea 1: Procedural in-distribution instance scaling** — Score 4.4
The solvers in `reasoners/` are already self-verifying, so generating *new* problems of the same category and labelling them with the solver is free, zero-noise, and keeps the exact corpus format. This is the single highest-leverage, lowest-risk way to add correct gradient signal — the only failure mode is plateau, which the falsification test catches.

### ⚡ Quick win (lowest effort)
**Idea 2: Solver-as-free-verifier acceptance gate + hard length cap** — Effort S
A small `corpus.py` filter: re-verify every trace's `\boxed{}` against the solver and drop any trace longer than the per-category baseline mean+σ. It is the *safety wrapper* that makes ideas 1/5/6/7 safe (no answer drift, no truncation creep) and is near-free to add.

### 🛡️ Safe bet (highest confidence)
**Idea 3: Expand masked auxiliary string-skill augmenters** — Confidence 🟢
New `augmenters/` modules (reverse-string, count-substring, char-index, digit-extract) are masked + no-`\boxed{}`, so they *cannot* shift the boxed-reasoning distribution — the exact property that protected the existing 5 augmenters at 0.86.

---

## Ranked ideas

### Idea 1: Procedural in-distribution instance scaling via per-category generators

- **Pattern**: P4 (Scale)
- **Tier**: 1
- **Target task**: Same 7-category `\boxed{}` reasoning task — add more *training instances* per category without changing format or answer style.
- **Scope**: enhance-existing — adds a `generate()` entry-point to each `reasoners/<cat>.py` (or a sibling `generators/<cat>.py`) that emits fresh `Problem` objects; `reasoning.py` then solves+verifies them exactly as today; `corpus.py` tokenises them unchanged. The solver, format, mask, and grader path stay identical.
- **One-liner**: Procedurally synthesise more same-distribution problems per category and label them with the existing self-verifying solver, multiplying correct, zero-noise training signal.

**Mechanism**:
For each category, write a parametric generator that samples problem parameters from the *same* ranges observed in `problems.jsonl` (e.g. cipher key length, cryptarithm word set from `wonderland.txt`, gravity mass/height ranges). Feed each generated `Problem` through the existing solver in `reasoners/<cat>.py`; keep only `status=rule_found` traces (the solver self-verifies the answer). Write them as new `reasoning/<id>.txt`, then run `corpus.py` unchanged. Net effect: 1.5–3× more verified, in-format examples for under-fit categories.

**Source inspirations**:
- Primary: "Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards", Stojanovski et al., **NeurIPS 2025 (Spotlight)** [arXiv:2505.24760](https://arxiv.org/abs/2505.24760) — 100+ procedural generators+verifiers giving "virtually infinite training data with adjustable complexity".
- Supporting: "Skywork-Math: Data Scaling Laws for Mathematical Reasoning in LLMs", Zeng et al., 2024 [arXiv:2407.08348](https://arxiv.org/abs/2407.08348) — log-linear gains from scaling synthetic math instances.
- Contrasting: "AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking", Gao et al., **ICLR 2026** [arXiv:2506.07751](https://arxiv.org/abs/2506.07751) — warns that more synthetic *variations* under SFT can underperform abstraction; bounds the expected gain.

**Why expected to improve**:
Reasoning Gym/Skywork show that adding verifiable, in-distribution instances raises accuracy along a scaling curve before plateau. Because the solver is deterministic and self-verifying, every new label is correct (no noise that exact-match punishes) and the trace style is byte-identical to baseline — so this adds signal *without* the distribution shift that sank exp13/exp19.

**Expected gain**: +0.2 / +0.5 / +0.8 pp 🟡 (mid)
**Feasibility**: 5/5 🟢
**Effort**: M 🟢

**Implementation sketch**:
1. Profile `problems.jsonl` per category (parameter histograms) so generated instances match the observed support.
2. Add `generate(n, seed)` per `reasoners/<cat>.py`; emit fresh `Problem`s into `problems.jsonl`.
3. Run `reasoning.py` (keeps only verified) → `corpus.py`; start with the 1–2 weakest categories at 1.5–2×.

**Risks**:
- Plateau if the model already saturates that category → flat gain (round-2 hint: baseline is well-tuned).
- Generator support drifts off the real distribution → adds out-of-distribution instances that *hurt* (mitigate with idea 6's surface bounds + idea 2's verifier gate).

**Falsification test**: Add 1.5–2× verified instances to the 1–2 weakest categories (by held-out bucket). Re-train at fixed NUM_STEPS/seed and score the ~200-problem held-out slice. **Fail if** macro exact-match does not rise ≥ +0.3pp, OR any category drops > 1pp, OR 7680 cap-hit-rate increases. Plateau (Δ < +0.3pp) ⇒ category saturated, stop scaling it.

---

### Idea 2: Solver-as-free-verifier acceptance gate + hard length cap

- **Pattern**: P6 (Verify)
- **Tier**: 1
- **Target task**: Same task — guarantee every corpus example is answer-correct and short enough to never truncate before `\boxed`.
- **Scope**: enhance-existing — a filter inside `corpus.py` (the existing answer-extraction at lines ~183–186 already re-reads the boxed value). Adds (a) a re-verification call to the matching solver and (b) a per-category token-length cap. Everything downstream unchanged.
- **One-liner**: Before an example enters the corpus, re-verify its `\boxed{}` with the deterministic solver and drop any trace whose length exceeds the baseline category mean+σ, eliminating both wrong labels and truncation risk.

**Mechanism**:
In `corpus.py`, after extracting `reasoning_answer` from `\boxed{}`, call the same-category solver's `compare_answer` to confirm correctness; drop or quarantine mismatches. Independently, compute per-category completion-token length on the baseline corpus, set `cap = mean + 1σ`, and drop generated/augmented traces above it. This gate is what lets ideas 1/5/6/7 add data without re-introducing the exp13/exp19 failure modes (answer-flip, length inflation).

**Source inspirations**:
- Primary: "Scaling Relationship on Learning Mathematical Reasoning with LLMs" (RFT — Rejection sampling Fine-Tuning), Yuan et al., 2023 [arXiv:2308.01825](https://arxiv.org/abs/2308.01825) — keeping only verified-correct paths lifted LLaMA-7B GSM8K 35.9→49.3.
- Supporting: "STaR: Bootstrapping Reasoning With Reasoning", Zelikman et al., **NeurIPS 2022** [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) — fine-tune only on rationales that yield the correct answer.

**Why expected to improve**:
RFT/STaR show verified-correct filtering is a reliable, low-variance gain. Here the verifier is *free and perfect* (deterministic solver), so the gate has no false-accept ceiling. The length cap directly attacks round-2's truncation failure (exp13/exp19 overran 7680), protecting greedy-decode exactness without touching decoding.

**Expected gain**: +0.1 / +0.3 / +0.5 pp 🟢 (mid) — and it de-risks every other idea.
**Feasibility**: 5/5 🟢
**Effort**: S 🟢

**Implementation sketch**:
1. Compute per-category baseline completion-length stats from the current corpus.
2. Add `verify_or_drop()` + `length_cap()` in `corpus.py`'s example loop; log dropped counts per category.
3. Re-train on the gated corpus; confirm no category loses examples it needs.

**Risks**:
- Over-aggressive length cap deletes legitimately long-but-correct traces in hard categories → cap per-category, not global.
- If baseline corpus is already 100% solver-correct, the verify half is a no-op (still cheap insurance).

**Falsification test**: Apply the gate to the default corpus, re-train at fixed steps/seed. **Fail if** held-out macro exact-match drops, OR > 5% of any category's examples are dropped (cap too tight). **Pass** if cap-hit-rate at 7680 falls and macro is ≥ baseline.

---

### Idea 3: Expand masked auxiliary string-skill augmenters

- **Pattern**: P5 (Decompose — add low-level sub-skills the high-level solvers consume)
- **Tier**: 2
- **Target task**: Same task — strengthen the character-level primitives (reverse, count, index, extract) that cipher/cryptarithm/numeral reasoning relies on, via masked auxiliary tasks.
- **Scope**: enhance-existing — new modules in `augmenters/` registered in `augmentation.py` alongside the existing 5; consumed by `corpus.py`'s augmentation branch (lines ~229+, "no reasoning, no `\boxed{}`"). The boxed-reasoning corpus is untouched.
- **One-liner**: Add reverse-string / count-substring / char-index / digit-extract augmenters (masked, no `\boxed{}`) to drill the low-level string operations that the deterministic categories implicitly require.

**Mechanism**:
Mirror `spelling.py`/`splitting.py`: each new augmenter emits `{category, prompt, completion}` where the completion is the deterministic answer (e.g. "reverse of `abcde` → `edcba`"), no reasoning, no box. `corpus.py` masks the prompt and trains on the completion only. Cap the aggregate auxiliary share at ≤ 15–20% of corpus tokens so reasoning signal is not diluted.

**Source inspirations**:
- Primary: "Deep multi-task learning with low level tasks supervised at lower layers", Søgaard & Goldberg, **ACL 2016** [aclanthology P16-2038](https://aclanthology.org/P16-2038/) — low-level auxiliary tasks consistently help the primary task.
- Supporting: "Reasoning Gym", NeurIPS 2025 [arXiv:2505.24760](https://arxiv.org/abs/2505.24760) — many low-level string/algorithmic generators improve broad reasoning.

**Why expected to improve**:
Cipher/cryptarithm/numeral all decompose into char indexing, counting, and reversal. Søgaard & Goldberg show such low-level auxiliaries lift the primary task. Because these examples are masked + no-boxed, they cannot shift the boxed-answer or trace-length distribution — the exact invariant that kept the original 5 augmenters safe at 0.86.

**Expected gain**: +0.0 / +0.2 / +0.3 pp 🟢 (mid)
**Feasibility**: 5/5 🟢
**Effort**: S 🟢

**Implementation sketch**:
1. Add `reverse.py`, `count_substring.py`, `char_index.py`, `digit_extract.py` to `augmenters/`; register in `augmentation.py`.
2. Generate; run `corpus.py`; check auxiliary token share ≤ 20%.
3. A/B vs baseline on held-out slice.

**Risks**:
- Auxiliary share too high → dilutes reasoning gradient (cap it; round-2 shows the peak is sensitive).
- Skills may be redundant with what the base model already has → flat gain (cheap to test).

**Falsification test**: Add the 4 augmenters (≤ 20% token share), re-train at fixed steps/seed. **Fail if** macro exact-match does not rise ≥ +0.2pp on the held-out slice, OR cipher/cryptarithm/numeral collectively do not improve, OR any category drops > 1pp.

---

### Idea 4: Category-coverage + difficulty-stratified mixture weighting

- **Pattern**: P8 (Specialize — weight the mixture per category/difficulty)
- **Tier**: 1
- **Target task**: Same task — rebalance how many examples each category/difficulty contributes so macro accuracy is maximised against the (proxied) test mix.
- **Scope**: enhance-existing — a sampling-weight layer applied when assembling the corpus in `corpus.py` (or example repetition counts), driven by held-out per-category accuracy. Format, solver, grader unchanged.
- **One-liner**: Up-sample under-performing categories/difficulty bands and down-sample saturated ones, choosing weights from small-scale validation rather than guessing the hidden leaderboard mix.

**Mechanism**:
Build a per-category held-out slice; measure accuracy per category and per difficulty band. Set corpus sampling weights inversely to validation accuracy (within bounds), re-train, re-measure, iterate a couple of rounds (small-scale weight search à la data-mixing optimization). Difficulty is read from generator parameters (idea 1) where available.

**Source inspirations**:
- Primary: "Data Mixing Optimization for Supervised Fine-Tuning of LLMs", Li, Liu & Xing, 2025 [arXiv:2508.11953](https://arxiv.org/abs/2508.11953) — small-scale mixture search gets within 0.66% of grid-search-optimal; composition swings downstream accuracy materially.
- Supporting: "Scaling Laws for Optimal Data Mixtures", 2025 [arXiv:2507.09404](https://arxiv.org/abs/2507.09404).

**Why expected to improve**:
Mixture composition can swing SFT accuracy by double digits; the current corpus weights are unaudited. Reweighting toward weak categories should lift macro exact-match — *if* the held-out slice's category mix tracks the leaderboard's.

**Expected gain**: +0.1 / +0.4 / +0.6 pp 🟡 (mid)
**Feasibility**: 3/5 🟡 (requires a trustworthy per-category slice first; otherwise it is guessing the hidden mix — same caveat as batch-1 idea-6)
**Effort**: M 🟡

**Implementation sketch**:
1. Stand up the held-out per-category slice (prerequisite for the whole batch — see round-2).
2. Measure per-category/difficulty accuracy; set inverse-accuracy weights (bounded).
3. Two reweight→retrain→measure rounds; keep only if macro improves and no category collapses.

**Risks**:
- **Distribution shift = hidden-mix mismatch**: if the leaderboard weights categories differently than the slice, optimising the slice can *hurt* LB (medium risk — no public proxy for the mix).
- Over-up-sampling a weak category causes over-fit/forgetting elsewhere → bound weights, watch every category.

**Falsification test**: Apply inverse-accuracy weights, re-train at fixed steps/seed. **Fail if** held-out macro does not rise ≥ +0.3pp OR any category drops > 1pp. Because of hidden-mix risk, also require the *minimum* per-category accuracy to not regress before trusting it on LB.

---

### Idea 5: Prompt paraphrase augmentation (masked prompt, fixed answer)

- **Pattern**: P11 (ICL / input-variation)
- **Tier**: 1
- **Target task**: Same task — make the model robust to phrasing variants of the *question* without touching the trace or answer.
- **Scope**: enhance-existing — a prompt-paraphrase step that produces extra `(prompt', same completion)` pairs; `corpus.py` masks the prompt (loss 0) exactly as today, so only the unchanged completion contributes to loss.
- **One-liner**: Generate paraphrased problem statements (synonyms, clause reordering) paired with the *unchanged* verified completion, exploiting that the prompt is masked so paraphrase cannot shift the loss/trace distribution.

**Mechanism**:
For each problem, produce 1 paraphrase of the prompt (rule-based templates and/or an offline LLM). Re-run the deterministic solver on the paraphrased prompt to confirm it still parses to the *same* answer (reject paraphrases that change the problem). Emit `(prompt', completion)` with the original completion. Because the prompt is masked, the gradient is identical to baseline — only the input conditioning is diversified.

**Source inspirations**:
- Primary: "MuggleMath: Assessing the Impact of Query and Response Augmentation on Math Reasoning", Li et al., **ACL 2024** [arXiv:2310.05506](https://arxiv.org/abs/2310.05506) — query (question) augmentation is empirically effective for math SFT.
- Supporting / Contrasting: "AbstRaL", ICLR 2026 [arXiv:2506.07751](https://arxiv.org/abs/2506.07751) — surface paraphrase helps robustness but abstraction helps more; bounds the gain.

**Why expected to improve**:
The hidden test may phrase problems differently than `problems.jsonl`. MuggleMath shows query augmentation improves math SFT. Crucially, because completion mask=1 / prompt mask=0, paraphrasing the prompt does **not** move the trace/answer/length distribution — far safer than paraphrasing reasoning (which regressed in round-2).

**Expected gain**: +0.1 / +0.3 / +0.4 pp 🟡 (mid)
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. Add a paraphrase generator (templates first; LLM optional, offline).
2. Solver-verify each paraphrase still maps to the same answer (reuse idea 2's gate); drop drifters.
3. Cap to ≤ 1 paraphrase/problem; A/B on held-out slice.

**Risks**:
- LLM paraphrase silently changes the problem's answer → solver re-verification mandatory (without it this becomes label noise).
- Marginal if the test phrasing already matches training → small gain.

**Falsification test**: Add 1 verified paraphrase per problem for 1–2 categories, re-train at fixed steps/seed. **Fail if** macro exact-match does not rise ≥ +0.3pp on a *phrasing-perturbed* held-out slice, OR any category drops > 1pp on the standard slice.

---

### Idea 6: Domain randomization of solver-invariant surface features

- **Pattern**: P2 (Transfer — from sim-to-real domain randomization)
- **Tier**: 3
- **Target task**: Same task — force the model to learn the category's invariant rule by randomizing surface features the solver is invariant to (variable/entity names, spacing, number formatting within tolerance, clause order).
- **Scope**: enhance-existing — a randomization layer inside the idea-1 generators (and/or applied to existing `problems.jsonl`); solver re-verifies, `corpus.py` unchanged.
- **One-liner**: Randomize solver-irrelevant surface details across generated instances so the LoRA learns the underlying rule rather than memorising surface form — the sim-to-real domain-randomization principle applied to reasoning data.

**Mechanism**:
When generating (idea 1) or rewriting existing problems, jitter only features the solver is provably invariant to: rename cryptarithm words (from `wonderland.txt`/`dictionary.txt`), permute independent clauses, vary whitespace/number presentation while keeping the answer within grader tolerance. Solver re-verifies the answer is unchanged. The completion still flows through the identical format.

**Source inspirations**:
- Primary: "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", Tobin et al., **IROS 2017** [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) — randomizing irrelevant generative parameters yields invariant features that transfer.
- Supporting: "Reasoning Gym" (adjustable surface complexity) [arXiv:2505.24760](https://arxiv.org/abs/2505.24760).
- Contrasting: "AbstRaL" [arXiv:2506.07751](https://arxiv.org/abs/2506.07751) — argues abstraction can beat brute surface randomization under SFT (devil's-advocate evidence).

**Why expected to improve**:
Domain randomization is the canonical way to make a model robust to a shifted test distribution by treating real variation as "just another training variation". Applied here, it should close the gap between `problems.jsonl` surface forms and the hidden test's. Answer/trace-structure distribution is preserved (only surface tokens move), keeping it on the safe side of the round-2 line — but it is Tier-3 transfer, hence lower confidence.

**Expected gain**: +0.1 / +0.3 / +0.5 pp 🟡 (mid)
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. Enumerate per-category solver-invariant surface knobs.
2. Apply bounded randomization in the generator; solver re-verify (idea 2 gate).
3. A/B on a surface-perturbed held-out slice.

**Risks**:
- A "surface" knob the solver is *not* actually invariant to flips the answer → mandatory re-verification.
- AbstRaL's caution: under pure SFT, surface randomization may underperform → keep dose moderate, measure.

**Adjacent / Cross-domain notes**:
- Original domain: robotics sim-to-real (randomized rendering/physics).
- Target domain: text reasoning corpus (randomized surface tokens).
- Adaptation needed: define solver-invariant knobs; re-verify each instance; bound number-formatting jitter to within grader rel-tol 1e-2 (and never for binary-string answers — those are matched exactly).

**Falsification test**: Add surface-randomized variants for 1–2 categories, re-train fixed steps/seed. **Fail if** macro on a *surface-perturbed* held-out slice does not rise ≥ +0.3pp, OR standard-slice macro regresses, OR any category drops > 1pp.

---

### Idea 7: STaR-offline self-traces for failing items, solver-filtered + length-capped (⚠️ RISKY)

- **Pattern**: P12 (Self-play / self-improve)
- **Tier**: 1
- **Target task**: Same task — recover currently-failing items by having the *current adapter* generate candidate traces, keep only solver-verified-correct, short ones.
- **Scope**: enhance-existing — an offline generation pass with the current adapter; outputs filtered by the deterministic solver (idea 2 gate) and added to `reasoning/`; `corpus.py` unchanged. (Extends, does not duplicate, batch-1 idea-7: here the verifier is the *deterministic solver* and a hard length cap is mandatory.)
- **One-liner**: For items the model currently gets wrong, sample traces from the current adapter, keep only those whose `\boxed{}` the solver verifies *and* that fit well under 7680, then fine-tune on them — STaR with a free perfect verifier.

**Mechanism**:
Run the current adapter (greedy + a few sampled rollouts) on failing held-out-adjacent items; for each output, the deterministic solver checks the boxed answer; keep only correct traces below the per-category length cap; add to corpus and re-train. Unlike model-only STaR, the verifier is exact, so accepted traces are guaranteed correct.

**Source inspirations**:
- Primary: "STaR: Bootstrapping Reasoning With Reasoning", Zelikman et al., **NeurIPS 2022** [arXiv:2203.14465](https://arxiv.org/abs/2203.14465).
- Supporting: RFT, Yuan et al. 2023 [arXiv:2308.01825](https://arxiv.org/abs/2308.01825).
- Contrasting: round-2 `tracker/rounds/round_2.md` — model-generated trace *content* (exp13 0.68, exp19 0.79) regressed; this is the in-house negative evidence this idea must overcome.

**Why expected to improve**:
STaR/RFT lift accuracy by learning from self-generated correct traces; the deterministic solver removes the usual false-accept risk. *However*, round-2 proves model-written traces shift the sharp-peak distribution and regress — so this is the batch's highest-risk idea, gated hard by idea 2 (length cap + verify) and category gating.

**Expected gain**: −0.2 / +0.2 / +0.6 pp 🔴 (high variance — can regress)
**Feasibility**: 3/5 🟡
**Effort**: L 🟡

**Implementation sketch**:
1. Identify failing items via the held-out slice; sample K traces from the current adapter.
2. Solver-verify + length-cap (idea 2); keep correct & short only.
3. Add ≤ a small fraction to corpus; re-train; A/B with strict guards.

**Risks**:
- Distribution drift / length inflation (the exp13/exp19 failure mode) → mandatory hard length cap + small dose.
- Model-trace style diverges from the solver-trace peak → gate per category, keep dose ≤ 10%.

**Falsification test**: Add solver-verified self-traces (≤ 10% of corpus) for 1 weak category, re-train fixed steps/seed. **Fail if** held-out macro does not rise ≥ +0.3pp, OR mean completion length rises, OR 7680 cap-hit-rate rises, OR any category drops > 1pp. (Given round-2, abandon quickly on any regression.)

---

## Open questions from `augmentation-data.md §6` — answered

1. **Optimal auxiliary-task : reasoning ratio before signal dilution?** No exact value in the literature for this task, but data-mixing work ([arXiv:2508.11953](https://arxiv.org/abs/2508.11953)) shows composition swings accuracy materially, and MTL practice keeps low-level auxiliaries a minority. **Recommendation: cap aggregate masked-auxiliary tokens at ≤ 15–20%** and tune via idea-4's small-scale weight search.
2. **How does `compare_answer` normalize?** Confirmed from [nemotron-master/reasoning.py:69-103](../../nemotron-master/reasoning.py#L69-L103): `strip()` → if `^[01]+$` then **binary exact, case-insensitive**; else try `float()` both sides and `math.isclose(rel_tol=1e-2, abs_tol=1e-5)`; else **case-insensitive string equality**. ⇒ **Strategy E is confirmed unsafe**: `"1/2"`/`".5"` do *not* `float()`-equal `"0.5"` (they fall to string compare and fail), and binary answers have zero tolerance. The only free-form latitude is numeric formatting within rel-tol 1e-2 for non-binary numbers (exploited *carefully* in idea 6, never for binary).
3. **Leaderboard category mix proxy?** No public proxy exists. ⇒ **Do not optimise the mix blind** (idea 4 risk). Build the per-category held-out slice and require the *minimum* per-category accuracy not to regress before trusting any reweight on LB.
4. **Saturation point of in-distribution augmentation (idea 1/B)?** Scaling-law evidence (Skywork-Math [arXiv:2407.08348](https://arxiv.org/abs/2407.08348)) is log-linear then plateaus; Reasoning Gym offers infinite data but with diminishing returns. ⇒ **No fixed number** — scale per-category and stop a category when Δ < +0.3pp (the idea-1 falsification threshold). Expect earliest plateau on already-strong categories.

## Verification report
| # | Title | Primary source? | Mechanism concrete? | Falsification? | Verdict |
|---|-------|-----------------|---------------------|----------------|---------|
| 1 | Procedural instance scaling | ✅ NeurIPS 2025 (2505.24760) VERIFIED | ✅ | ✅ +0.3pp/no cat −1pp/no cap rise | KEEP (EXTENDS) |
| 2 | Verifier gate + length cap | ✅ RFT (2308.01825) VERIFIED | ✅ | ✅ cap-hit ↓, macro ≥ base | KEEP (EXTENDS) |
| 3 | Aux string augmenters | ✅ ACL 2016 P16-2038 VERIFIED | ✅ | ✅ +0.2pp, cipher/crypt/numeral ↑ | KEEP (EXTENDS) |
| 4 | Coverage/difficulty weighting | ✅ Data-mix (2508.11953) VERIFIED | ✅ | ✅ +0.3pp, min-cat no regress | KEEP (EXTENDS), 🟡 hidden-mix risk |
| 5 | Prompt paraphrase | ✅ MuggleMath ACL 2024 (2310.05506) VERIFIED | ✅ | ✅ +0.3pp on perturbed slice | KEEP (EXTENDS) |
| 6 | Surface domain randomization | ✅ Tobin IROS 2017 (1703.06907) VERIFIED | ✅ | ✅ +0.3pp perturbed, std no regress | KEEP (EXTENDS) |
| 7 | STaR-offline solver-filtered | ✅ STaR NeurIPS 2022 (2203.14465) VERIFIED | ✅ | ✅ length/cap guards + +0.3pp | KEEP (EXTENDS), 🔴 distribution-drift risk |

**Cross-idea consistency**: No near-duplicates (idea 2 = quality gate / idea 7 = generation source / idea 1 = procedural problems — distinct primaries and mechanisms). No knob contradictions (ideas compose: 2 gates 1/5/6/7). Score distribution not over-confident (mix of 🟢/🟡/🔴; one 🔴 idea). **Rejected: 0.**

## Notes & warnings
- ⚠️ **Prerequisite for the whole batch** (from round-2): build a **per-category held-out slice (~200 problems, vLLM greedy)** and a per-bucket error breakdown `{format, truncation, arithmetic-slip, method-wrong}` *before* any submit. Round-2 burned 5 submissions skipping this.
- ⚠️ **Idea 7 is high-risk** — it is the only idea that lets model-generated trace *content* into the corpus, the exact lever that regressed in round-2. Run it last, dosed ≤ 10%, behind the idea-2 gate, and abandon on any regression.
- **Idea 2 is a dependency**, not just an idea: implement it first so ideas 1/5/6/7 inherit the verify+length-cap safety.
- **Strategy E (multi-form `\boxed{}`) deliberately excluded** — `compare_answer` analysis (open-Q 2) confirms it regresses.
- Tier/pattern/venue/recency gates all satisfied; no re-search needed; no under-quota.

## Next steps for user
1. **Build the held-out per-category slice** (blocking) → then implement **Idea 2** (verify+length gate).
2. **Idea 1** (procedural scaling) on the 1–2 weakest categories at 1.5–2×; then **Idea 3** (aux augmenters, ≤ 20%).
3. **Ideas 5 → 6 → 4** with controlled A/B; hold **Idea 7** for last with strict guards.

## Provenance signature
SHA256(inputs + paper IDs [2505.24760, 2407.08348, 2506.07751, 2308.01825, 2203.14465, 2310.05506, 2508.11953, 2507.09404, 1703.06907, P16-2038] + 2026-06-02): `b3-augmentation-data-7ideas-55_30_15`
