# Idea Batch 4 — NVIDIA Nemotron Model Reasoning Challenge / Few-shot rule inference (REASONING TRAJECTORY + TRAINING SIGNAL)
**Generated**: 2026-06-04T00:00:00Z
**Time-to-batch**: ~18 min
**Skill version**: 0.1.0
**Skill invocation**: `/benchmark-climb-ideation` — synthesized from data-distribution-analysis.md + tong-hui-kang-approach.md + research-overcome-cryptarithm.md; target 0.86 → 0.87+; scope: reasoning trajectory coverage + training signal improvements

## Inputs
- **Benchmark**: NVIDIA Nemotron Model Reasoning Challenge (Kaggle), public leaderboard exact-match score (9,800 test problems, greedy vLLM, `compare_answer` binary/float).
- **Task / problem**: Fine-tune Nemotron-3-Nano-30B-A3B (Mamba/MoE, LoRA rank-32) to solve 9-category few-shot rule inference puzzles (gravity, unit_conversion, cipher, numeral, bit_manipulation, equation_numeric_{deduce,guess}, cryptarithm_{deduce,guess}). For each problem: 4–8 input→output examples, infer hidden rule, output answer inside `\boxed{...}`. One-shot greedy decode, ≤ 7,680 output tokens.
- **Existing pipeline**: `nemotron-master/` — `reasoning.py` (deterministic solvers in `reasoners/` → verified CoT traces) → `augmentation.py` (9 off-distribution aux tasks: matching 4,515, splitting 1,500, concatenation 1,500, …) → `corpus.py` (tokenize + mask, completion = `{reasoning}\n</think>\n\boxed{answer}<|im_end|>`) → `train_sft.py` (Tinker SFT, stratified batching, CE / IS / PPO / CISPO / DRO loss variants). Corpus: 18,292 examples, 50.7M tokens, 51% off-distribution augmentation. Solver coverage: gravity/cipher/numeral/unit_conversion 100%; bit_manipulation 85.1% (238 unknown); equation_numeric_deduce 90.6% (56 unknown); cryptarithm_deduce 8.2% (605 unknown); cryptarithm_guess 6.7% (153 unknown); equation_numeric_guess 15.4% (115 unknown). **1,167 bài (11.9%) không có correct trajectory. Current score: 0.86.**
- **Batch scope**: mixed — 9 enhance-existing / 1 greenfield (Idea 7: offline RLVR/DPO pairs requires new sampling infrastructure beyond existing pipeline).
- **Tier mix (configured)**: 55/30/15 (pipeline-biased; user did not override).
- **Baseline**: Nemotron-3-Nano-30B-A3B + rank-32 LoRA @ **0.86**. Target ≥ **0.87**.
- **Compute budget**: 1× RTX-PRO-6000 (Modal), full train per run; offline slice inference (~170 problems) < 30 min.
- **Time budget**: falsification via eval_slice inference ≤ 1 GPU-hour; full corpus retrain ≤ 6 h.
- **Constraints**: LoRA rank=32 (hard), greedy inference, off-policy preferred (no vLLM-in-the-loop training loop), max_tokens=7,680 (truncation → loss of `\boxed` → 0 score), vLLM-loadable adapter. No ideas from prior batches 1–3 (see no-overlap check at end).
- **No-overlap mandate**: batch-2 already proposed DoRA (ERROR), NEFTune (−0.03), self-verify traces (−0.18), PiSSA (ERROR), scratchpad (pending), anchored-KL (not run), seed-soup (not run), offline-preference pairs (exp18 not run), stream-of-search (−0.07). Batch-3 proposed procedural instance scaling, verifier gate, expand augmenters, static mixture weighting, prompt paraphrase, domain randomization, STaR-offline self-traces. All 10 ideas below are mechanism-level distinct.

---

## Summary
| Metric | Value |
|--------|-------|
| Batch size | 10 |
| Tier 1 / 2 / 3 (counts) | 6 / 3 / 1 |
| Tier mix vs configured | 60/30/10 vs 55/30/15 (T1 +5pp, T3 −5pp — within ±10pp ✅) |
| Scope mix | 9 enhance-existing / 1 greenfield (≥ 50% enhance ✅) |
| Patterns used | P1×2, P2, P3×2, P4, P6, P8, P12×2 (8 distinct; P2 ✅ P6 ✅) |
| Distinct venues | arXiv, NeurIPS 2025, EMNLP 2025, OpenReview (≥ 3 ✅) |
| Time windows | <12mo: 4 papers; 12-36mo: 5 papers; >36mo: 1 paper (STaR 2022) ✅ |
| Avg feasibility | 3.2/5 |
| Avg confidence | 🟢 10%, 🟡 70%, 🔴 20% |

## Summary table
| Rank | Title | Pattern | Tier | Gain mid (pp) | Feas | Effort | Score |
|------|-------|---------|------|---------------|------|--------|-------|
| 1 | Extend bit_manipulation solver: rotation, shift, majority | P3 | 1 | +2.0 | 4 | M | 6.3 |
| 2 | Procedural cryptarithm generation + offline RFT | P12 | 2 | +3.0 | 3 | L | 6.2* |
| 3 | LogicPuzzleRL-style offline DPO pairs for cryptarithm | P2 | 3 | +3.5 | 2 | XL | 5.8 |
| 4 | Drop off-distribution augmentation, add in-dist mini-tasks | P3 | 1 | +1.25 | 4 | M | 5.7 |
| 5 | OXA offline exploration-aware SFT | P1 | 1 | +1.75 | 3 | M | 5.6 |
| 6 | LLM-paraphrase completion traces (Shape-of-Thought) | P12 | 1 | +1.0 | 3 | M | 5.2 |
| 7 | REDI offline negative-trace training | P1 | 1 | +1.5 | 3 | L | 4.9 |
| 8 | AdaSTaR dynamic per-category adaptive sampling | P8 | 2 | +1.0 | 3 | M | 4.9 |
| 9 | Iterative offline DPO on eval_slice rollouts | P6 | 1 | +1.0 | 3 | M | 4.9 |
| 10 | VCORE key-token loss weighting | P4 | 2 | +1.0 | 3 | M | 4.9 |
| 11 | GeoRA geometry-aware LoRA init trên adapter 0.86 | P3 | 2 | +1.0 | 2 | L | 3.8 |
| 12 | RL→SFT ordering: DPO trước, SFT sau | P1 | 2 | +1.0 | 2 | M | 4.0 |

*Idea 4 raw composite 6.4 → downgraded 1 slot by devil's-advocate pass (sparse reward concern); see Notes.

---

## Top-3 recommendations

### 🏆 Top-1 by composite score
**Idea 1: Extend bit_manipulation solver with rotation, shift, majority** — Score: 6.3
Pure solver engineering: add LEFT_ROTATE(k), RIGHT_ROTATE(k), LEFT_SHIFT(k), RIGHT_SHIFT(k), MAJORITY across column-pairs to `reasoners/bit_manipulation.py`. Covers an estimated 50–70% of the 238 unknown problems (likely ~120–170 new correct traces). Zero training-code change, zero risk of trace-format drift, verifiable via `compare_answer` before any training. The cleanest, highest-confidence lift available.

### ⚡ Quick win (lowest effort with deterministic ceiling lift)
**Idea 4: Drop off-distribution augmentation** — Effort: M (pure data edit)
Delete 9,663 augmentation examples (51% of corpus), regenerate 2,000–3,000 simpler in-distribution mini-rule tasks for the same 9 categories. This is a file-level corpus edit with no code change to solver/training logic. Eliminates the known off-distribution confusion and frees capacity for categories that actually appear in the test set. Risk: if augmentation was load-bearing for regularization, score could dip — cheap to falsify on eval_slice before full run.

### 🛡️ Safe bet (strongest published evidence)
**Idea 5: OXA offline exploration-aware SFT** — Confidence 🟡, NOVEL
OXA (Mar 2026, arXiv:2603.16206) reports +6 Pass@1 / +5 Pass@k vs. conventional SFT on Qwen2.5-1.5B-Math across 6 benchmarks, using the same offline, off-policy setting we have. The two-objective approach (promote hard correct traces, suppress easy incorrect ones) requires only a new `OXALossConfig` in `loss_config.py` and per-entry logprob computation — no new data generation, no RL infrastructure. Strongest directly-applicable evidence of any idea in this batch.

---

## Ranked Ideas

---

### Idea 1: Extend bit_manipulation solver — rotation, shift, majority

- **Pattern**: P3 (Replace — swap one component in existing pipeline)
- **Tier**: 1 (In-field)
- **Target task**: bit_manipulation category only (238 currently-unsolved problems)
- **Scope**: enhance-existing — modifies `reasoners/bit_manipulation.py` only; all other pipeline stages unchanged.
- **One-liner**: Add rotation, shift, and majority operators to the exhaustive per-column search so the 238 unknown bit_manipulation problems get correct training traces.

**Mechanism**:
The current solver in `reasoners/bit_manipulation.py` tests 9 operators (Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT) per 8 bit-columns. The 238 unsolved problems likely use LEFT_ROTATE(k), RIGHT_ROTATE(k), LEFT_SHIFT(k for 1–7), RIGHT_SHIFT(k), and 3-input MAJORITY(a,b,c). Add these to the operator search space, re-run `reasoning.py`, and re-run `corpus.py`. For each newly solved problem, a deterministic correct trace is emitted and verified via `compare_answer` before entering corpus.

**Source inspirations**:
- Primary: "How Much Backtracking is Enough? Exploring the Interplay of SFT and RL in Enhancing LLM Reasoning", Chen et al., arXiv May 2025, OpenReview NeurIPS 2025 [arXiv:2505.24273] — demonstrates that SFT gain is upper-bounded by coverage of correct traces; every additional correct trace for a previously-unsolved problem yields positive gradient signal.
- Supporting: "SynLogic: Synthesizing Verifiable Reasoning Data at Scale", Liu et al., NeurIPS 2025 [arXiv:2505.19641] — principle of extending verifiable task coverage to reduce trajectory gaps.

**Why expected to improve**:
Each of the 238 unknown bit_manipulation test problems represents ~0.01% of the leaderboard score directly. If rotation/shift/majority cover 50–70% of unknowns (~120–170 problems), that adds +1.2–1.7% to the bit_manipulation accuracy. With ~95% model recall on solver-provided traces, this translates to +1.1–1.6 pp leaderboard gain. Structure is identical to existing traces so no distribution shift risk.

**Expected gain**: +1.5 / +2.0 / +2.5 pp 🟡 (low / mid / high)
**Feasibility**: 4/5 🟢
**Effort**: M 🟢

**Implementation sketch**:
1. In `reasoners/bit_manipulation.py`, extend the `OPERATORS` dict with `LEFT_ROTATE_k`, `RIGHT_ROTATE_k` (k=1..7), `LEFT_SHIFT_k`, `RIGHT_SHIFT_k`, `MAJORITY` (majority of 3 input columns for each bit position).
2. Re-run `uv run python3 reasoning.py` — solver now attempts new operators; only writes trace when `compare_answer` passes.
3. Re-run `corpus.py` with `VERIFY_GATE=True` — newly solved problems enter corpus; nothing else changes.
4. Retrain. Falsify on eval_slice before full submit.

**Risks**:
- Operator search space grows exponentially → `reasoning.py` runtime increases by ~5–10×. Mitigation: add early-exit when unique per-column match found.
- Multi-operator compositions (e.g., ROTATE then XOR) still unsolved — sets a hard ceiling on this idea.
- Traces become longer if multi-operator path is needed → check LENGTH_GATE enforcement.

**Falsification test**: After adding operators, run `reasoning.py` on 238 known-unknown problems. If newly solved ≤ 30 problems (< 13%), the operator hypothesis is wrong and the idea yields < +0.3 pp — abandon before corpus rebuild.

---

### Idea 2: Procedural cryptarithm generation + offline RFT

- **Pattern**: P12 (Self-play / Self-improve)
- **Tier**: 2 (Adjacent — symbolic reasoning / program induction)
- **Target task**: cryptarithm_deduce and cryptarithm_guess (758 unsolved, ~700 test bài estimated)
- **Scope**: enhance-existing — adds `generators/cryptarithm_procedural.py` to existing pipeline; `corpus.py` and `train_sft.py` unchanged.
- **One-liner**: Generate 1,000+ verified cryptarithm problems procedurally, run adapter at temperature > 0 to collect rollouts, keep only correct completions (RFT), add to corpus.

**Mechanism**:
Write `generators/cryptarithm_procedural.py`: randomly sample symbolic operator alphabets (3–6 chars), assign operator semantics (concat, reverse-concat, take-left, take-right, etc.), generate equation triples `(left, op, right) → result` with verified answers, format into the Wonderland prompt template. Generate 1,000–2,000 diverse problems. Run the current adapter at temperature=0.5 (10 rollouts per problem) via `infer_slice.py`-style inference. Keep rollouts where `compare_answer()` passes (STaR/RFT filtering). Add as new corpus entries in `corpus/` with category=`cryptarithm_deduce`. Retrain including these self-verified traces.

**Source inspirations**:
- Primary: "SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond", Liu et al., NeurIPS 2025 [arXiv:2505.19641] — 35-task procedural synthesis with verifiable rewards; cryptarithm-class tasks included.
- Supporting: "AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners", Koh et al., NeurIPS 2025 [arXiv:2505.16322] — adaptive sampling prevents over-training on easy procedural problems.
- Supporting (classic): "STaR: Bootstrapping Reasoning With Reasoning", Zelikman et al., NeurIPS 2022 [arXiv:2203.14465] — foundational RFT: generate rationales, keep correct ones, fine-tune. **>36mo window ✅.**

**Why expected to improve**:
SynLogic demonstrates that procedural synthesis of verifiable reasoning tasks at scale improves out-of-distribution performance on harder instances of the same task family. With only 65 correct cryptarithm examples, even 100–200 additional self-verified traces would increase training density by 2–4×, reducing the training/test ratio from 0.04× to 0.1–0.2×. If adapter pass@10 temperature=0.5 achieves even 5–15% on procedurally-generated cryptarithms (vs. 8% greedy on original), RFT produces net-positive signal.

**Expected gain**: +2.0 / +3.0 / +4.5 pp 🟡 (high uncertainty due to sparse initial pass rate)
**Feasibility**: 3/5 🟡
**Effort**: L 🟡

**Implementation sketch**:
1. Write `nemotron-master/generators/cryptarithm_procedural.py`: generate problems using `itertools` to sample operator alphabets + semantics, verify answer with Python, format to Wonderland prompt (mirror `problems/<id>.jsonl` structure).
2. Generate 1,500 problems; verify each answer with deterministic evaluation (no LLM needed here).
3. Run 10-sample temperature=0.5 inference via `infer_slice.py`-style loop for each procedural problem. Filter by `compare_answer()`.
4. Write passing completions to `reasoning/<new_id>.txt` format; regenerate `corpus.jsonl` including new entries. Retrain.

**Risks**:
- Sparse reward: if adapter pass@10 < 2%, RFT produces too few correct traces (<30 items) → insufficient signal. Mitigation: use paraphrased human-written solutions from the 65 existing cryptarithm examples as additional seeds.
- Procedural problems may differ in distribution from original test problems (different operator alphabets) → risk of memorizing new distribution without transfer.
- Requires temperature-sampling infrastructure (Tinker may not support this out-of-the-box).

**Falsification test**: Generate 200 procedural cryptarithm problems. Run current adapter temperature=0.5, 10 samples each. If pass@10 < 2% on procedural problems, the bootstrap signal is too sparse and the idea fails before corpus integration.

> *⚠️ Devil's-advocate: Curriculum learning lit (arXiv:2603.27226, "Rethinking Easy-to-Hard") showed curriculum from procedural easy→hard yields no consistent gains over standard sampling in some settings. Concern: adapter's ~8% base accuracy on cryptarithm may be too sparse to bootstrap. Mitigate by using hint-augmented prompts (show a few correct examples) to push pass@10 > 5% before RFT.*

---

### Idea 3: LogicPuzzleRL-style offline DPO pairs for cryptarithm

- **Pattern**: P2 (Transfer from RL-on-logic-games domain → rule-inference fine-tuning)
- **Tier**: 3 (Cross-domain)
- **Target task**: cryptarithm_deduce + equation_numeric_guess (873 combined unsolved problems)
- **Scope**: greenfield — requires new sampling + DPO training stage beyond existing `train_sft.py`. Justification: the existing pipeline has no mechanism to generate (chosen, rejected) pairs for the same prompt; retrofitting would require new inference infrastructure and a new loss function not available in Tinker's current loss menu (CE/IS/PPO/CISPO/DRO — none is DPO). A new two-stage script is cleaner than patching `train_sft.py`.
- **One-liner**: Offline analog of LogicPuzzleRL — sample multiple rollouts per cryptarithm problem, pair correct vs. incorrect, train with DPO using `compare_answer` as the binary reward.

**Mechanism**:
Stage-1 (offline rollout collection): run current SFT adapter at temperature=0.8, N=20 samples per cryptarithm/equation_numeric_guess problem. For each problem, collect all rollout completions; verify with `compare_answer()`. Stage-2 (pair construction): for each problem with ≥1 correct AND ≥1 incorrect rollout, form (chosen=first correct, rejected=first incorrect) pair. Stage-3 (DPO fine-tuning): apply DPO loss with reference-KL anchor (reference = SFT adapter) on these pairs. Run for 100–200 steps with low LR (1e-5) to prevent format collapse.

**Source inspirations**:
- Primary: "LogicPuzzleRL: Cultivating Robust Mathematical Reasoning in LLMs via Reinforcement Learning", Wong et al., arXiv Jun 2025 [arXiv:2506.04821] — binary feedback on cryptarithms + logic puzzles improves out-of-distribution reasoning through iterative play-to-learn.
- Supporting: "Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation", Tu et al., arXiv Mar 2025 [arXiv:2503.12854] — single round of DPO with coarse filtering enhances reasoning for strong base models; recommends SFT → DPO recipe.
- Contrasting: "Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training" [arXiv:2603.27226] — caution on curriculum; not directly applicable but signals that RL-based approaches need careful reward design.

**Why expected to improve**:
LogicPuzzleRL achieves "significant improvements in out-of-distribution performance on a range of mathematical benchmarks" specifically including cryptarithm as one of its strongest-gain puzzle types. The offline DPO analog replicates the binary reward signal without requiring an online RL training loop. Iterative DPO literature shows 1 round of offline DPO + coarse filtering is sufficient for strong models. The 873 target problems represent ~8.9% of the test set; even partial improvement (+30–50% on this slice) yields +2.7–4.5 pp.

**Expected gain**: +2.0 / +3.5 / +5.0 pp 🔴 (high variance — depends on pass rate of rollouts)
**Feasibility**: 2/5 🔴
**Effort**: XL 🔴

**Implementation sketch**:
1. Write `generate_dpo_pairs.py`: load current adapter via vLLM (temperature=0.8, n=20), run on all cryptarithm + equation_numeric_guess problems, collect rollouts.
2. For each problem: verify rollouts with `compare_answer()`; filter to problems with ≥1 correct + ≥1 incorrect.
3. Build DPO dataset in HuggingFace format. Apply DPO with reference-KL (β=0.1) for 150 steps, LR=1e-5, mix 20% SFT corpus as anchor to prevent format collapse.
4. Evaluate on eval_slice before submission.

**Risks**:
- Requires vLLM temperature sampling infrastructure (not currently in `train_sft.py`). High engineering effort.
- If pass rate < 1% on cryptarithm rollouts, no (chosen, rejected) pairs form → no training signal.
- DPO can catastrophically forget the SFT format → always mix SFT corpus as anchor.
- KL penalty needs careful tuning; too high = no update, too low = format collapse.

**Falsification test**: After Stage-1, count problems with ≥1 correct rollout out of 20. If count < 20 problems (< 2% of 873), the reward is too sparse for DPO — abort Stage-2.

**Adjacent / Cross-domain notes**:
- Original domain: Logic puzzle RL (LogicPuzzleRL — cryptarithm/magic square games)
- Target domain: Rule inference SFT fine-tuning (Nemotron Reasoning Challenge)
- Adaptation needed: (1) replace online GRPO loop with one-shot offline rollout collection; (2) replace continuous reward with `compare_answer()` binary signal; (3) replace GRPO with DPO loss (reference-free constraint not available in Tinker → write standalone script).

---

### Idea 4: Drop off-distribution augmentation, add in-distribution mini-tasks

- **Pattern**: P3 (Replace — swap augmentation component)
- **Tier**: 1 (In-field)
- **Target task**: All 9 categories (indirect: free model capacity currently spent on off-distribution tasks)
- **Scope**: enhance-existing — replaces `augmentation.py` output only; `corpus.py`, `train_sft.py`, solvers all unchanged.
- **One-liner**: Delete the 9,663 off-distribution augmentation examples (51% of corpus), replace with 2,000–3,000 simpler in-distribution mini-rule tasks covering the same 9 test categories.

**Mechanism**:
Current augmentation (matching 4,515, splitting 1,500, concatenation 1,500, spelling 648, lstrip 300, reverse 300, count_substring 300, char_index 300, digit_extract 300) = 9,663 examples with **no \boxed{} answer format** and no representation in the test set. Delete all from corpus. Generate replacements: for each of the 9 test categories, create simplified instances (e.g., 3-bit manipulation, 2-step unit conversion, 5-character cipher) using existing `reasoners/` solvers with narrowed parameter ranges. These "mini" instances use identical trace format and are verified by `compare_answer()`. Add 2,000–3,000 such mini-instances to corpus, preserving 100% in-distribution signal.

**Source inspirations**:
- Primary: "Rethinking Data Quality for LLM Reasoning", arXiv/EMNLP 2025 Findings [aclanthology.org/2025.findings-emnlp.616] — data aligned to target task distribution dramatically outperforms generic off-distribution training data; paraphrasing to model distribution improves performance.
- Supporting: OXA [arXiv:2603.16206] — demonstrates that off-distribution training signal hurts exploration/entropy of the policy.

**Why expected to improve**:
51% of training capacity (9,663 examples, ~26M tokens) is currently spent on tasks with 0% test-set overlap. Eliminating this: (a) reduces task-confusion between matching (binary strings) and bit_manipulation (binary strings with different semantics); (b) allows LoRA capacity to concentrate on the 9 real categories; (c) in-distribution mini-tasks provide additional gradient signal for hard categories (cryptarithm mini-instances with shorter operator chains are easier bootstraps).

**Expected gain**: +0.5 / +1.25 / +2.0 pp 🟡
**Feasibility**: 4/5 🟢
**Effort**: M 🟢

**Implementation sketch**:
1. In `augmentation.py`: comment out all `problems.extend(...)` calls for the 9 augmentation categories.
2. Write `augmenters/mini_rule.py`: for each of 9 test categories, generate 200–300 simplified instances using existing `reasoners/` with narrowed parameter ranges (e.g., bit_manipulation with only 2 operators, gravity with integer g values). Verify each with `compare_answer()`.
3. Rebuild corpus with `uv run python3 corpus.py`. Token count drops from ~50.7M to ~28M (smaller corpus → faster iteration).
4. Retrain; compare eval_slice per-category accuracy vs. baseline to isolate effect.

**Risks**:
- The augmentation tasks may have been providing regularization (diversity anti-overfitting). Removing them could increase overfitting on the small cryptarithm corpus (65 examples). Monitor min logprob on cryptarithm eval_slice.
- Mini-rule traces have the same rigid format issue as solver traces (template duplication). Add paraphrase step (cf. Idea 6) to mini-instances if gradient collapse suspected.

**Falsification test**: After corpus rebuild, run 50-step training and evaluate eval_slice per-category accuracy. If bit_manipulation or cipher accuracy drops > 2 pp relative to baseline (suggesting the augmenters were providing beneficial multi-task regularization), abort and restore augmentation.

---

### Idea 5: OXA offline exploration-aware SFT

- **Pattern**: P1 (Combine — SFT forward pass + logprob-guided reweighting)
- **Tier**: 1 (In-field)
- **Target task**: All categories (especially cryptarithm where high-confidence wrong answers are the main issue)
- **Scope**: enhance-existing — adds `OXALossConfig` to `loss_config.py` and a pre-training logprob computation step; `reasoners/`, `corpus.py`, `train_sft.py` structure unchanged.
- **One-liner**: Weight each training example by its difficulty (low-confidence correct = upweight; high-confidence incorrect = downweight/suppress) to orient the policy toward under-explored correct trajectories.

**Mechanism**:
Before training, run current adapter inference on all corpus entries (forward pass, no sampling). Compute per-example average logprob. Classify corpus into two sets: (A) verified-correct examples with low logprob (≤ threshold, e.g., −0.5 per token) — these are hard, under-explored correct traces; (B) rule_unknown examples where adapter outputs high-confidence incorrect answer (logprob ≥ −0.1 per token) — these are high-confidence errors. OXA objective: add loss term that (i) upweights set-A by factor ×2 in CE (promotes exploration of hard correct modes), (ii) adds a gradient-suppression term for set-B that redistributes probability mass away from incorrect patterns. Implement as `OXALossConfig(name="oxa", promote_weight=2.0, suppress_weight=0.5)` in `loss_config.py`.

**Source inspirations**:
- Primary: "Offline Exploration-Aware Fine-Tuning for Long-Chain Mathematical Reasoning", Mu et al., arXiv Mar 2026 [arXiv:2603.16206] — +6 Pass@1 / +5 Pass@k vs. conventional SFT on Qwen2.5-1.5B-Math across 6 benchmarks; exact offline, off-policy setting.
- Supporting: "How Much Backtracking is Enough?" [arXiv:2505.24273] — confirms that SFT structural coverage is the lever; OXA ensures hard-category traces receive more gradient.

**Why expected to improve**:
OXA's +6 Pass@1 result was on a model with similar characteristics: small (1.5B activated), offline-only setting, verifiable-reward tasks. The mechanism transfers directly: our corpus already has the "low-confidence correct" split (hard cryptarithm traces) and "high-confidence incorrect" split (rule_unknown examples with wrong boxed answers). OXA exploits exactly this structure without requiring new data or RL infrastructure — only a logprob precomputation pass + modified loss weights.

**Expected gain**: +1.0 / +1.75 / +2.5 pp 🟡
**Feasibility**: 3/5 🟡
**Effort**: M 🟡

**Implementation sketch**:
1. Add logprob precomputation step: run `infer_slice.py`-style forward pass on full corpus (batch inference), log average per-token logprob per entry to `corpus/logprobs.jsonl`.
2. In `loss_config.py`, add `OXALossConfig(promote_threshold=-0.5, suppress_threshold=-0.1, promote_weight=2.0, suppress_weight=0.3)`.
3. Modify `train_sft.py` to load `corpus/logprobs.jsonl` and pass entry-level weights to Tinker's loss function (already has `CrossEntropyWithWeightingLossConfig` infrastructure — extend it).
4. Train and evaluate.

**Risks**:
- Logprob threshold selection is sensitive; wrong threshold may upweight noise instead of hard-but-correct traces. Falsify on eval_slice before full run.
- Suppressing rule_unknown examples may remove the only training signal for cryptarithm (65 correct traces vs. 605 suppressed) — risk of cryptarithm accuracy collapsing further.
- Tinker's existing `CrossEntropyWithWeightingLossConfig` supports token-level weighting but may need extension for example-level promote/suppress logic.

**Falsification test**: After logprob precomputation, inspect the top-20 "low-confidence correct" examples (set-A). If they are dominated by one category (e.g., all bit_manipulation), the threshold is miscalibrated — retune before training.

---

### Idea 6: LLM-paraphrase completion traces (Shape-of-Thought)

- **Pattern**: P12 (Self-play — use LLM to generate improved training data for itself)
- **Tier**: 1 (In-field)
- **Target task**: All categories, especially gravity/numeral/unit_conversion where traces have rigid templates causing gradient saturation.
- **Scope**: enhance-existing — replaces trace content in `reasoning/` with paraphrased versions; `corpus.py`, `train_sft.py` unchanged.
- **One-liner**: Use an LLM (Claude/GPT-4o) to paraphrase each solver-generated reasoning trace into varied vocabulary while preserving logical structure — breaks the CoT duplication trap.

**Mechanism**:
The existing `paraphrase_instances.py` already exists in `nemotron-master/`. Extend it to paraphrase `reasoning/<id>.txt` completions: given a trace, instruct Claude/GPT-4o to "rewrite this reasoning trace using different vocabulary and sentence structure but preserving every logical step and the final boxed answer." Verify the paraphrased trace: (a) extract `\boxed{}` answer, (b) confirm `compare_answer(paraphrased_answer, original_answer)` passes, (c) confirm token count ≤ 7,600 (LENGTH_GATE). Replace original traces with paraphrased versions in `reasoning/`. Regenerate corpus.

**Source inspirations**:
- Primary: "Shape of Thought: When Distribution Matters More than Correctness in Reasoning Tasks", Yao et al., arXiv Dec 2025 [arXiv:2512.22255] — LLM paraphrasing brings traces closer to model distribution, improving performance even when content is imperfect; paraphrasing human traces with LLM improves over human originals.
- Supporting: "How Much Backtracking is Enough?" [arXiv:2505.24273] — confirms that structural preservation (not content precision) is what LLMs learn; paraphrasing changes vocabulary (content) while preserving structure — safe.

**Why expected to improve**:
Shape of Thought's key finding: "the distribution of synthetic data closer to the language model's own distribution makes it more amenable to learning." Solver-generated traces use rigid algorithmic templates (e.g., bit_manipulation repeats "Output 0: ... 0 0 ... 1 0 ..." for every column of every bit position with identical formatting). When 1,360 bit_manipulation traces have near-identical templates, structural tokens are 100% predictable → gradient saturation on boilerplate tokens → model doesn't learn the logic. Paraphrasing introduces vocabulary variance while keeping structure — directly fixes gradient saturation. Note: existing exp13 (self-verify traces) regressed because it changed the LOGICAL STRUCTURE; paraphrasing preserves structure (safe) while changing vocabulary (beneficial).

**Expected gain**: +0.5 / +1.0 / +1.5 pp 🟡
**Feasibility**: 3/5 🟡
**Effort**: M 🟡

**Implementation sketch**:
1. Extend `paraphrase_instances.py` to process `reasoning/*.txt` files (not just problem instances).
2. Batch call Claude API (with `enable_prompt_caching` to reduce cost): system prompt = "Rewrite the reasoning trace using varied vocabulary while preserving every logical step and the final `\boxed{answer}` exactly." Rate: ~2,000 traces/day at low cost.
3. Verify each paraphrased trace: extract boxed answer → `compare_answer()` + token count check. If verification fails, keep original.
4. Write verified paraphrases to `reasoning/` (overwrite originals or keep as separate `reasoning_paraphrased/` dir). Rebuild corpus.

**Risks**:
- LLM paraphrase may accidentally change the boxed answer (hallucinate a different answer) — critical failure. Verification gate mitigates.
- High API cost for ~8,600 reasoning traces. Estimate: ~2–3M tokens → ~$6–10 using Claude Haiku.
- Paraphrase quality may be inconsistent (some rewritten, some near-identical) — spot-check 50 random examples before full run.

**Falsification test**: Paraphrase 100 random traces. Compute average per-token logprob of the CURRENT ADAPTER on paraphrased vs. original. If paraphrased logprob is NOT higher than original by >0.1 on average, Shape-of-Thought transfer doesn't hold for this model/corpus — abandon.

---

### Idea 7: REDI offline negative-trace training

- **Pattern**: P1 (Combine — SFT corpus + negative trace signal)
- **Tier**: 1 (In-field)
- **Target task**: All categories (especially cryptarithm where negatives = 92% of training problems)
- **Scope**: enhance-existing — adds a Stage-2 training pass using REDI objective; modifies `train_sft.py` with a new loss config; corpus structure unchanged.
- **One-liner**: After SFT on correct traces, apply REDI objective to push the model away from rule_unknown (incorrect answer) traces while pulling toward correct ones — tapping 1,167 negative examples currently wasted.

**Mechanism**:
Two-stage training: Stage-1 = existing SFT on 18,292 correct corpus entries (as currently). Stage-2 = REDI objective on rule_unknown problems: for each of the 1,167 rule_unknown problems, the "negative trace" is the current model's output (greedy or from solver's best attempt). REDI's REINFORCE-style reference-free loss pushes logprob of the negative trace DOWN and (where a correct trace exists nearby) logprob of correct trace UP. Implement as `REDILossConfig` in `loss_config.py`. Run Stage-2 for 50–100 steps with LR=5e-6, mix 30% original SFT corpus as anchor.

**Source inspirations**:
- Primary: "Harnessing Negative Signals: Reinforcement Distillation from Teacher Data for LLM Reasoning", Xu et al., arXiv May 2025 [arXiv:2505.24850] — REDI achieves 83.1% MATH-500 (matching DeepSeek-R1-Distill-1.5B trained on 800k) with only 131k traces by leveraging negative examples; reference-free, offline, no paired data requirement.
- Supporting: "Iterative DPO" [arXiv:2503.12854] — SFT → preference-optimization recipe strongly recommended; single round sufficient.

**Why expected to improve**:
REDI's data efficiency comes precisely from not discarding negatives. In our corpus, rule_unknown problems are currently included with WRONG boxed answers (the solver's best guess): model learns "for this problem, output this wrong answer." REDI inverts this: push away from that wrong answer. The 605 cryptarithm unknown problems' negative traces become training signal that penalizes incorrect symbolic reasoning patterns — an indirect but real signal. REDI (reference-free) avoids reference model overhead, critical for our LoRA-only infrastructure.

**Expected gain**: +1.0 / +1.5 / +2.0 pp 🟡
**Feasibility**: 3/5 🟡
**Effort**: L 🟡

**Implementation sketch**:
1. Collect negative traces: for rule_unknown problems, run current adapter greedy to collect its (wrong) outputs. Save to `corpus/negatives.jsonl`.
2. Implement `REDILossConfig` in `loss_config.py`: for each negative trace, compute logprob, apply REINFORCE gradient that decreases logprob of wrong trajectory tokens. Mix negatives at ratio 1:3 with SFT positives.
3. In `train_sft.py`, add Stage-2 training loop after Stage-1 SFT completes (100 steps, LR=5e-6).
4. Evaluate eval_slice before submit.

**Risks**:
- REDI on cryptarithm negatives may push the model away from all cryptarithm-like symbolic reasoning (including valid patterns) — monitor eval_slice cryptarithm accuracy separately.
- Batch-2 idea 8 (exp18, offline preference pairs, not yet run) overlaps in mechanism — run exp18 first to see if DPO pairs already provides this signal before implementing REDI.
- LR must be very low (5e-6) to avoid overriding SFT gains on solved categories.

**Falsification test**: After Stage-2, run eval_slice. If accuracy on gravity/cipher/numeral (previously stable categories) drops > 1 pp vs. Stage-1 SFT baseline, Stage-2 is overcorrecting — abort and revert to Stage-1 adapter.

---

### Idea 8: AdaSTaR dynamic per-category adaptive sampling

- **Pattern**: P8 (Specialize — route/weight by category difficulty)
- **Tier**: 2 (Adjacent — math reasoning / curriculum learning)
- **Target task**: All categories (upweight underperforming, downweight saturated)
- **Scope**: enhance-existing — modifies `train_sft.py` `_stratified_batches()` function only.
- **One-liner**: Replace static equal-weight category batching with accuracy-tracked adaptive sampling that upweights cryptarithm/equation_guess and downweights saturated gravity/numeral every N training steps.

**Mechanism**:
Current `_stratified_batches()` distributes examples evenly across categories regardless of per-category model performance. Replace with AdaSTaR-style dynamic sampling: every 200 steps, run quick inference on `eval_slice.jsonl` (170 problems, < 5 min GPU time); compute per-category accuracy; update per-category sampling weight as `w_cat = max(0.1, 1 - accuracy_cat)` (higher weight for lower accuracy); rebuild batch indices using these weights. Categories near 100% (gravity ~95%, numeral ~97%) get minimal weight; cryptarithm (~8%) gets maximum weight.

**Source inspirations**:
- Primary: "AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners", Koh & Oh, NeurIPS 2025 [arXiv:2505.16322] — 58.6% FLOPs reduction and best accuracy in 6/6 benchmarks by balancing trained vs. untrained observations dynamically.
- Supporting: "Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning", arXiv Jun 2025 [arXiv:2506.06632] — dynamic curriculum outperforms static mixture.

**Why expected to improve**:
AdaSTaR's core insight: over-training on already-solved examples wastes gradient steps while under-training on hard categories leaves them uncovered. Our corpus has extreme imbalance: gravity (1,897 examples, 100% solver coverage) is over-represented relative to cryptarithm (54 examples, 8% coverage). Dynamic reweighting ensures every gradient step is spent where the model is still learning — the "boundary-level" problems that are neither already solved nor too hard to improve on. Distinct from batch-3 idea 4 (static mixture weighting) in that AdaSTaR updates weights continuously during training based on live accuracy.

**Expected gain**: +0.5 / +1.0 / +1.5 pp 🟡
**Feasibility**: 3/5 🟡
**Effort**: M 🟡

**Implementation sketch**:
1. In `train_sft.py`, add a `compute_category_accuracy(eval_slice_path, adapter)` function that runs quick inference every 200 steps.
2. Modify `_stratified_batches()` to accept per-category weights and over-sample hard categories proportionally.
3. Store weight history to `training_weights.jsonl` for inspection (mirror `training.html` dashboard format).
4. Tune the update interval (200 steps) based on inference cost: each eval_slice pass should < 10% of training time.

**Risks**:
- Inference overhead every 200 steps adds ~5–10% total training time.
- If cryptarithm gets over-weighted (weight → 1.0) but the model can't improve on 54 examples, it memorizes them → min logprob → 0 on cryptarithm, collapse signal.
- Adaptive weighting may be confused by the eval_slice being too small (only 13 cryptarithm examples → noisy accuracy estimate, high variance per step).

**Falsification test**: Run training for 400 steps with adaptive sampling. Plot per-category weight history. If cryptarithm weight oscillates > 0.5 amplitude across consecutive measurements (noisy eval signal), the per-category eval_slice sample is too small — freeze weights or use exponential moving average.

---

### Idea 9: Iterative offline DPO on eval_slice rollouts

- **Pattern**: P6 (Verify — generate K candidates, verify with compare_answer, select preferred)
- **Tier**: 1 (In-field)
- **Target task**: All categories (focus on medium-difficulty: bit_manipulation, equation_numeric)
- **Scope**: enhance-existing — adds a post-SFT DPO refinement stage; `corpus.py`, `reasoners/` unchanged.
- **One-liner**: Generate 10 rollouts per eval_slice problem, verify with `compare_answer`, form preferred/rejected pairs, and apply 1 round of offline DPO — teaching the model to prefer its own correct completions over incorrect ones.

**Mechanism**:
After SFT training, run temperature=0.5 inference (10 samples) for all 170 eval_slice problems using the trained adapter. Verify each sample with `compare_answer()`. For each problem with ≥1 correct AND ≥1 incorrect sample: construct (chosen=random correct, rejected=random incorrect) pair. Apply offline DPO with reference=SFT adapter and KL penalty β=0.1 on these ~80–120 preference pairs (many eval_slice problems will have mixed results). Run 50–100 steps at LR=1e-5. Output: refined adapter.

**Source inspirations**:
- Primary: "Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation", Tu et al., arXiv Mar 2025 [arXiv:2503.12854] — single round of DPO with coarse filtering significantly enhances reasoning; recommends SFT → DPO two-stage recipe.
- Supporting: "Iterative Reasoning Preference Optimization", Yuan et al., Meta AI arXiv Apr 2024 [arXiv:2404.19733] — DPO + NLL term on winning CoT improves GSM8K/MATH/ARC.

**Why expected to improve**:
Iterative DPO literature consistently shows 1 round of offline DPO after SFT adds +1–2 pp on adjacent benchmarks (GSM8K, MATH). The eval_slice problems span all 9 categories and directly measure the test distribution. DPO on this slice teaches the model "on the exact distribution of test problems, prefer outputs that match the correct answer." The KL anchor prevents forgetting the SFT format. Risk: eval_slice is small (170 problems) → ~80–120 usable pairs if rollout pass rate ~40–60% — minimum viable but sufficient for 1-round DPO.

**Expected gain**: +0.5 / +1.0 / +1.5 pp 🟡
**Feasibility**: 3/5 🟡
**Effort**: M 🟡

**Implementation sketch**:
1. Use `infer_slice.py` (already exists in `nemotron-master/`) with temperature=0.5, n=10 to generate rollouts on `eval_slice.jsonl`.
2. Write `build_dpo_pairs.py`: for each problem, verify rollouts, construct (chosen, rejected) pairs. Write to `dpo_pairs.jsonl`.
3. Run DPO training via new `DPOLossConfig` in `loss_config.py` or standalone HuggingFace TRL script (simpler if Tinker doesn't support DPO natively). 50 steps, LR=1e-5, β=0.1, SFT-mix 30%.
4. Evaluate on eval_slice — this is the SAME slice used to generate pairs, so watch for overfitting signal. Also run on a 50-problem held-out check not in eval_slice.

**Risks**:
- Overfitting to eval_slice distribution (pairs derived from it, then evaluated on it). Mitigate: hold out 30 eval_slice problems from pair construction; evaluate on those.
- Rollout pass rate may be < 20% on cryptarithm → few cryptarithm pairs → DPO doesn't help the hardest category.
- Tinker may not have native DPO support → requires external script that may not match LoRA adapter format.

**Falsification test**: After DPO (50 steps), compute accuracy on the 30 held-out eval_slice problems (not used in pair construction). If accuracy ≤ SFT baseline on these holdouts, the DPO is only memorizing pair constructions, not generalizing — abort.

---

### Idea 10: VCORE key-token loss weighting

- **Pattern**: P4 (Scale — change loss weight distribution across token positions)
- **Tier**: 2 (Adjacent — chain-of-thought supervision optimization)
- **Target task**: All categories (especially bit_manipulation with heavy boilerplate tokens)
- **Scope**: enhance-existing — modifies `corpus.py` binary mask to a float weight mask; updates `CrossEntropyWithWeightingLossConfig` in `loss_config.py`.
- **One-liner**: Replace the binary 0/1 completion mask with a float weight mask that up-weights answer tokens and key decision tokens, down-weights boilerplate — allocates gradient budget to the most informative positions.

**Mechanism**:
Current `corpus.py` creates a binary mask: prompt tokens weight=0, completion tokens weight=1. Replace the completion-side weights with a 3-tier float mask: (A) `\boxed{answer}` tokens and the immediately preceding `</think>` → weight=3.0 (highest); (B) binary decision tokens in the reasoning trace (operator assignment tokens like "NOT", "OR", "AND" in bit_manipulation; operator inferred tokens in equation_numeric; answer hypothesis tokens in cryptarithm) → weight=2.0; (C) all other completion tokens (boilerplate: "Output 0:", "matching output", "Applying to ...") → weight=0.5. Implement via per-token weight array in `corpus/<id>/synthetic.jsonl` and consumed by `CrossEntropyWithWeightingLossConfig`.

**Source inspirations**:
- Primary: "VCORE: Variance-Controlled Optimization-based Reweighting for Chain-of-Thought Supervision", He et al., arXiv Oct 2025, rev. Apr 2026 [arXiv:2510.27462] — principled adaptive allocation of supervision across CoT tokens; standard CE treats all tokens equally, leading to misallocated supervision and weak generalization.
- Supporting: "Rethinking Supervised Fine-Tuning: Emphasizing Key Answer Tokens for Improved LLM Accuracy" [arXiv:2512.21017] — answer-token emphasis improves exact-match accuracy.

**Why expected to improve**:
VCORE's core finding: standard CE misallocates gradient to boilerplate tokens (which are predictable → low gradient signal) while under-allocating to semantically decisive tokens. In bit_manipulation traces (~7,000 tokens), >80% of tokens are boilerplate column/row formatting. The "NOT", "OR", "XOR" operator assignment tokens (< 1% of tokens) determine the final answer. Up-weighting these tokens 6× (3.0 vs. 0.5) relative to boilerplate should improve model attention to the decision logic. The `CrossEntropyWithWeightingLossConfig` in Tinker already supports per-token weights — this requires only a parsing change in `corpus.py`.

**Expected gain**: +0.5 / +1.0 / +1.5 pp 🟡
**Feasibility**: 3/5 🟡
**Effort**: M 🟡

**Implementation sketch**:
1. In `corpus.py`, add a `compute_token_weights(tokens, category)` function: parse trace content to identify operator tokens (e.g., regex for "NOT|OR|XOR|AND" in bit_manipulation; "concatenation|reverse" in cryptarithm; `\boxed{...}` tokens in all categories).
2. Replace binary mask array with float weight array in `corpus/<id>/synthetic.jsonl`.
3. Verify Tinker's `CrossEntropyWithWeightingLossConfig.apply_weights()` accepts float weights (it already does via `branch_weight` mechanism — adapt it to use the new float weight array).
4. Test on 200-step mini-run: compare per-category eval_slice accuracy vs. binary-mask baseline.

**Risks**:
- Key-token identification regex may misfire: a "NOT" in natural language ("not yet confirmed") could be mis-tagged as a decision token.
- Weight ratios (3.0/2.0/0.5) need tuning; wrong ratios could cause gradient explosion on boilerplate underweighting.
- For numeral/cipher categories with short traces (~59–752 words), boilerplate fraction is small → VCORE benefit may be near-zero for those categories.

**Falsification test**: Compute gradient norm per-token-type (decision vs. boilerplate) before and after weight change. If decision-token gradient norm is NOT ≥ 2× boilerplate gradient norm (i.e., weights aren't having the expected effect), check `CrossEntropyWithWeightingLossConfig` weight application — may be normalized away.

---

### Idea 11: GeoRA geometry-aware LoRA init trên adapter 0.86

- **Pattern**: P3 (Replace — swap LoRA init từ random sang geometry-aware)
- **Tier**: 2 (Adjacent — RLVR / LoRA parameter efficiency)
- **Target task**: Cryptarithm_deduce + cryptarithm_guess (758 unsolved, 8% accuracy — category khó nhất)
- **Scope**: enhance-existing — thay LoRA init trong `Continuer_Nemotron_Notebook.py` (hoặc `train_sft.py`); `corpus.py`, `reasoners/` không đổi. `RESET_WEIGHTS=False` để continue từ adapter 0.86.
- **One-liner**: Thay random LoRA init bằng GeoRA init (SVD của activation covariance trong weight space), continue fine-tune adapter 0.86 — lý thuyết là geometry-aware init nắm bắt tốt hơn reasoning manifold cho cryptarithm khó.

**Mechanism**:
GeoRA (arXiv:2601.09361) khởi tạo LoRA matrices A, B dựa trên principal directions của weight geometry — cụ thể là singular vectors của Fisher Information Matrix (FIM) hoặc activation Hessian tại layer đó. Thay vì init A~N(0,σ²), B=0 (random), GeoRA init A với top-r right singular vectors của ∇²L|_{pretrained}, B=0. Điều này đảm bảo LoRA perturbation bắt đầu từ subspace có gradient curvature lớn nhất → nhanh hội tụ hơn trên hard reasoning tasks.

Áp dụng: (1) Forward pass toàn bộ cryptarithm corpus qua adapter 0.86 để collect activations + gradients; (2) SVD trên gradient outer products để lấy top-32 principal directions per layer; (3) Init A_new = top-32 right singular vectors, B_new = 0; (4) Continue SFT từ adapter 0.86 với GeoRA-initialized LoRA.

**Source inspirations**:
- Primary: J. Zhang et al., "GeoRA: Geometry-Aware Low-Rank Adaptation for RLVR," arXiv Jan 2026 [arXiv:2601.09361] — GeoRA init cải thiện RL training stability và convergence speed trên RLVR tasks; claims better sample efficiency vs. random LoRA init.
- Supporting: "How Much Backtracking is Enough?" [arXiv:2505.24273] — confirms that SFT quality is bounded by coverage; GeoRA may help model extract more signal from scarce cryptarithm traces.

**Why expected to improve**:
Adapter 0.86 đã học tốt các category dễ (gravity 95%, cipher 90%), nhưng cryptarithm vẫn 8%. Random LoRA re-init (nếu tiếp tục train từ 0.86) dùng A, B mới → có thể disturb đã-học capacity. GeoRA init từ geometry của cryptarithm-specific gradients → LoRA perturbation align với direction model cần học nhất → ít disturb các category đã tốt hơn, đồng thời tốc độ học cryptarithm nhanh hơn.

**Tuy nhiên — devil's advocate nghiêm túc**:
1. Paper GeoRA gốc thiết kế cho RLVR (online RL), không phải SFT — mechanism "geometry-aware" trong RL context (policy gradient curvature) có thể không transfer sang SFT loss landscape.
2. Nemotron-H là Mamba/MoE hybrid (`modeling_nemotron_h`) — FIM/Hessian computation cho SSM layers (selective state space) **rất khác** với Transformer attention layers; GeoRA chưa được validate trên bất kỳ SSM architecture nào.
3. FIM computation cho model 30B-A3B @ rank-32 cần memory peak > FP16 model itself — có thể OOM ngay cả trên RTX PRO 6000 (48 GB).
4. `lm_head` LoRA được add manually (do Unsloth drop nó cho MoE) — GeoRA cần xử lý special case này.
5. Risk catastrophic forgetting: nếu GeoRA init làm break existing LoRA weights, categories đã tốt (gravity, cipher) sẽ regress.

**Expected gain**: +0.3 / +1.0 / +1.5 pp 🔴 (high uncertainty — architecture mismatch, FIM computation khó)
**Feasibility**: 2/5 🔴
**Effort**: L 🟡

**Implementation sketch**:
1. Implement `compute_geora_init(model, cryptarithm_corpus_subset, rank=32)` trong `Continuer_Nemotron_Notebook.py`:
   - Forward pass 50–100 cryptarithm examples, collect per-layer gradient outer products (G = Σ ∇W·∇Wᵀ).
   - SVD(G) → top-32 right singular vectors → A_init.
   - Set B_init = 0 (standard).
2. Thay thế LoRA init sau `model = FastLanguageModel.get_peft_model(...)`.
3. Load adapter 0.86 (`RESET_WEIGHTS=False`), overwrite LoRA A matrices với GeoRA init.
4. Train 200–500 steps với LR thấp (1e-5) để tránh disturb categories đã tốt.
5. Evaluate eval_slice per-category trước khi submit.

**Risks**:
- OOM khi compute FIM: mitigate bằng gradient checkpointing + only compute FIM trên subset layers (top-8 layers most relevant to answer generation).
- Architecture mismatch (SSM vs Transformer): GeoRA principal directions có thể vô nghĩa với SSM selective scan matrices. Mitigate: apply GeoRA chỉ cho `lm_head` và FFN layers (có Transformer-like structure), skip SSM-specific matrices.
- Catastrophic forgetting nếu A_init too far from current A: mitigate bằng mixing GeoRA init với current A (50% blend).

**Falsification test**: Trước khi full run, chạy 50-step mini-run với GeoRA init. Nếu accuracy trên gravity/cipher eval_slice drops > 2 pp vs. baseline adapter 0.86 → GeoRA init đang disturb existing LoRA capacity → abort. Nếu cryptarithm loss không giảm trong 50 steps hơn baseline → geometry không informative → abandon.

---

### Idea 12: RL→SFT ordering — DPO trước, SFT sau (đảo ngược pipeline)

- **Pattern**: P1 (Combine — reorder hai training stages)
- **Tier**: 2 (Adjacent — training recipe ordering study)
- **Target task**: All categories (distribution-level improvement)
- **Scope**: enhance-existing — giữ nguyên `train_sft.py` và DPO script từ Idea 9; chỉ swap thứ tự chạy. Yêu cầu Idea 9 (DPO pairs) đã được implement.
- **One-liner**: Chạy 1 round offline DPO (50 steps) TRƯỚC khi SFT trên full corpus — lý thuyết của paper là RL squeeze distribution về correct reasoning mode trước, rồi SFT expand coverage.

**Mechanism**:
Paper arXiv:2509.21128v2 phân tích empirically: (1) RL thu hẹp output distribution (squeeze) — model converge về một số specific reasoning patterns; (2) SFT mở rộng distribution (expand) — model học thêm diverse traces. Tác giả argue rằng SFT→RL (hiện tại của chúng ta) có vấn đề: SFT expands first → diverse but suboptimal modes → RL must squeeze a diverse distribution → harder convergence. RL→SFT thì: RL squeezes lên đúng reasoning mode trước → SFT expand từ đúng mode → easier và tốt hơn.

Áp dụng: (1) Khởi đầu từ base adapter 0.86; (2) Chạy offline DPO 50 steps trên eval_slice pairs (correct vs incorrect rollouts); (3) Từ DPO-adapted model, chạy full SFT trên corpus; (4) Submit DPO→SFT adapter.

**Source inspirations**:
- Primary: K. Matsutani et al., "RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs," arXiv Sep 2025 [arXiv:2509.21128v2] — comparative study trên multiple models/benchmarks; RL→SFT consistently outperforms SFT→RL trong several settings.
- Supporting: "Enhancing LLM Reasoning with Iterative DPO" [arXiv:2503.12854] — confirms DPO as valid RL proxy; SFT → DPO recipe recommended, but paper pre-dates arXiv:2509.21128v2.

**Why expected to improve**:
Nếu paper's finding holds, chạy DPO trước tạo "inductive bias" đúng cho model trước khi SFT mở rộng coverage. Trong context của chúng ta: DPO trên eval_slice pairs → model learns "prefer correct reasoning style" → subsequent SFT trên full corpus builds on this correct base → better generalization than current SFT-only.

**Tuy nhiên — devil's advocate nghiêm túc**:
1. Paper nghiên cứu *online* RL (PPO/GRPO) → SFT, còn chúng ta dùng *offline* DPO (rất khác về mechanics). "RL squeezes" trong paper là effect của on-policy reward maximization — offline DPO with only 80–120 pairs có thể không tạo ra đủ squeeze effect.
2. Paper's RL phase thường chạy > 1,000 steps với dense reward signal; DPO 50 steps trên 80–120 pairs eval_slice = "micro-RL" — squeeze effect có thể negligible.
3. **Catastrophic forgetting risk cao**: Nếu DPO step chạy trước SFT và quá aggressive, nó có thể forget categories chưa có trong DPO pairs (cryptarithm, gravity). Subsequent SFT có thể không fully restore này.
4. Với chỉ 9 test categories và 170-problem eval_slice, số pairs DPO đủ đại diện distribution? Gravity chiếm ~20% eval_slice → DPO pairs thiên về gravity → squeeze về gravity reasoning mode → SFT sau đó có thể bias tương tự.
5. Dependency: idea này cần Idea 9 (DPO pairs từ eval_slice) đã done trước — add engineering dependency.
6. **Quan trọng nhất**: Batch-2 exp18 (offline preference pairs) chưa được chạy. Chạy exp18 trước để biết DPO có hoạt động trên architecture này không, trước khi thử đảo thứ tự.

**Expected gain**: +0.3 / +1.0 / +2.0 pp 🔴 (very high variance — strong theory, weak empirical transfer to our offline DPO + small-pair setting)
**Feasibility**: 2/5 🔴
**Effort**: M 🟢 (same code as Idea 9, just different order — thêm re-training SFT sau DPO)

**Implementation sketch**:
1. Prerequisite: Idea 9 DPO pairs đã được generated và `build_dpo_pairs.py` hoạt động.
2. Stage A — DPO 50 steps: từ adapter 0.86, run DPO với `dpo_pairs.jsonl`, LR=1e-5, β=0.1, SFT-mix 10% (thấp để squeeze signal dominant).
3. Stage B — SFT full corpus: từ DPO-adapted model, run full SFT trên corpus (1,000+ steps, standard settings).
4. Evaluate eval_slice sau Stage A và sau Stage B riêng để trace effect của từng stage.

**Risks**:
- Catastrophic forgetting trong Stage A: monitor per-category eval_slice accuracy sau DPO. Set threshold: nếu bất kỳ category nào drops > 3 pp → increase SFT-mix ratio trong Stage A (20–30%) để giảm forgetting.
- Nếu Stage A tạo biased distribution (e.g., over-squeeze về một category), Stage B SFT sẽ chậm recover → final model kém hơn baseline.
- High compute cost: 2 full training stages = ~2× GPU time so với baseline.

**Falsification test**:
1. Sau Stage A (DPO 50 steps), kiểm tra eval_slice: nếu overall accuracy < 0.85 (< baseline) → forgetting quá nặng → abort Stage B, revert về adapter 0.86.
2. Sau Stage B (SFT), nếu final score ≤ 0.86 → ordering không có lợi, submit SFT-only adapter thay thế.
3. Critical pre-check: **chạy exp18 (batch-2) trước** — nếu offline DPO không hoạt động trên Nemotron-H architecture (exp18 fails), Idea 12 không có cơ sở để chạy.

---

## Verification Report

| # | Title | Novelty | Primary source | Provenance | Feasibility | Gain sanity | Falsification | Risk | Compliance | Verdict |
|---|-------|---------|---------------|------------|-------------|-------------|---------------|------|------------|---------|
| 1 | Bit_manip solver extension | EXTENDS (from backtracking paper) | arXiv:2505.24273 ✅ VERIFIED | Resolves, title matches | 4/5 | +2.0 pp within headroom ✅ | SHARP (≥30 newly solved problems threshold) | MED (runtime increase) | PASS | **KEEP** |
| 2 | Procedural crypto + RFT | EXTENDS (SynLogic T2) | arXiv:2505.19641 ✅ VERIFIED NeurIPS'25 | Resolves ✅ | 3/5 | +3.0 pp plausible given 7% gap | SHARP (pass@10 ≥ 2% threshold) | HIGH (sparse reward) | PASS | **KEEP ⚠️ HIGH RISK** |
| 3 | LogicPuzzleRL offline DPO | NOVEL | arXiv:2506.04821 ✅ VERIFIED | Resolves ✅ | 2/5 | +3.5 pp aggressive → 🔴 | SHARP (≥20 problems w/ correct rollout) | HIGH (engineering effort) | PASS | **KEEP ⚠️ HIGH RISK** |
| 4 | Drop augmentation | EXTENDS (data quality lit) | EMNLP 2025 Findings ✅ VERIFIED | ACL Anthology resolves ✅ | 4/5 | +1.25 pp conservative ✅ | SHARP (per-cat eval_slice drop threshold) | MED (regularization loss) | PASS | **KEEP** |
| 5 | OXA exploration-aware | NOVEL | arXiv:2603.16206 ✅ VERIFIED | Resolves ✅ | 3/5 | +1.75 pp (paper reports +6 Pass@1) ✅ | SHARP (logprob threshold inspection) | MED (threshold sensitivity) | PASS | **KEEP** |
| 6 | Paraphrase traces | EXTENDS (Shape-of-Thought) | arXiv:2512.22255 ✅ VERIFIED | Resolves ✅ | 3/5 | +1.0 pp within range ✅ | SHARP (logprob comparison threshold) | MED (API cost, hallucination) | PASS | **KEEP** |
| 7 | REDI negative traces | EXTENDS (batch-2 idea 8 overlap in domain; REDI is distinct mechanism) | arXiv:2505.24850 ✅ VERIFIED | Resolves ✅ | 3/5 | +1.5 pp within range ✅ | SHARP (stable-cat accuracy threshold) | MED (overcorrection risk) | WARN: batch-2 exp18 partially overlaps | **KEEP** |
| 8 | AdaSTaR adaptive sampling | EXTENDS (vs. batch-3 idea 4 static weighting) | arXiv:2505.16322 ✅ VERIFIED NeurIPS'25 | Resolves ✅ | 3/5 | +1.0 pp within range ✅ | SHARP (weight oscillation threshold) | MED (inference overhead) | PASS | **KEEP** |
| 9 | Iterative DPO eval_slice | EXTENDS (iterative DPO lit) | arXiv:2503.12854 ✅ VERIFIED | Resolves ✅ | 3/5 | +1.0 pp within range ✅ | SHARP (holdout accuracy threshold) | MED (eval_slice overfitting) | PASS | **KEEP** |
| 10 | VCORE key-token weighting | EXTENDS (VCORE lit) | arXiv:2510.27462 ✅ VERIFIED | Resolves ✅ | 3/5 | +1.0 pp within range ✅ | SHARP (gradient norm check) | MED (regex misfires) | PASS | **KEEP** |
| 11 | GeoRA geometry-aware LoRA init | NOVEL application (paper = RLVR; we adapt to SFT) | arXiv:2601.09361 — title/abstract verify ✅ | Resolves ✅ | 2/5 | +1.0 pp speculative 🔴; paper reports RLVR gains not directly comparable | SHARP (50-step mini-run: gravity/cipher drop ≤ 2pp + cryptarithm loss decreasing) | HIGH (architecture mismatch: SSM/MoE ≠ Transformer; FIM OOM risk) | WARN: paper targets RLVR, not SFT — mechanism transfer unverified | **KEEP ⚠️ HIGH RISK — run after exp18 confirms DPO works** |
| 12 | RL→SFT ordering (DPO first) | EXTENDS (RL-SFT ordering lit) | arXiv:2509.21128v2 ✅ VERIFIED | Resolves ✅ | 2/5 | +1.0 pp speculative 🔴; paper studies *online* RL, not offline DPO — transfer uncertain | SHARP (eval_slice ≥ 0.85 after DPO stage; final ≥ baseline) | HIGH (catastrophic forgetting; 2× compute; dependency on Idea 9 + exp18) | WARN: batch-2 exp18 prerequisite not yet run | **KEEP ⚠️ HIGH RISK — lowest priority; run only after Idea 9 + exp18 validated** |

**Rejected**: 0 ideas rejected. All 12 passed 7-step verification. Ideas 11–12 carry HIGH RISK flags.

**Cross-idea consistency**:
- Ideas 7 (REDI), 9 (DPO eval_slice) share negative learning mechanism — do NOT run simultaneously; run sequentially with eval_slice check between.
- Ideas 1 (solver extension) + 4 (drop augmentation) are complementary and safe to combine.
- Ideas 2 (procedural crypto) + 3 (LogicPuzzleRL DPO) both target cryptarithm via different mechanisms — run 2 first (cheaper), then 3 if pass rate is sufficient.
- **Idea 12 depends on Idea 9**: DPO pairs từ eval_slice phải được generate trước khi có thể thử RL→SFT ordering. Không chạy Idea 12 độc lập.
- **Ideas 11 + 12 đều phụ thuộc exp18 (batch-2)**: Nếu exp18 confirm DPO/preference learning không hoạt động trên Nemotron-H → cả 2 idea này drop. Chạy exp18 trước.
- **Idea 11 và 12 KHÔNG nên chạy cùng nhau** — cả 2 đều manipulate LoRA initialization/training order; kết hợp sẽ confound results.
- Score distribution: 2/12 ideas at feasibility 4/5, 6/12 at 3/5, 4/12 at 2/5 — Ideas 11–12 đẩy proportion low-feasibility lên cao hơn batch trung bình.

---

## Notes & Warnings

- ⚠️ **Devil's-advocate on Idea 2 (Procedural crypto + RFT)**: Curriculum learning literature (arXiv:2603.27226) shows procedural easy-to-hard yields no consistent gains in some settings. Specific concern: if adapter pass@10 on new procedural problems < 2%, RFT has nothing to bootstrap from. The falsification test (pass@10 ≥ 2% threshold) directly catches this failure mode. Run Idea 2 BEFORE Idea 3 — if Idea 2's rollout pass rate is adequate, Idea 3 (offline DPO) becomes much more viable.
- **T3 tier slightly under quota (10% vs. 15%)**: Only 1 cross-domain idea (Idea 3). Could not surface a second T3 idea that wasn't near-duplicate to T1 ideas under the search budget.
- **Run order recommendation**: 1 (solver) → 4 (drop augmentation) → 9 (DPO eval_slice) → 5 (OXA) → 2 (procedural crypto). Ideas 3, 7 are high-effort — defer until lower-effort ideas are evaluated.
- **Batch-2 overlap warning**: exp18 (offline preference optimization, P12) in batch-2 has not yet been run. Before implementing Idea 7 (REDI) or Idea 9 (DPO), run exp18 first — it may already provide the DPO signal, making Ideas 7/9 redundant.
- **VCORE (Idea 10) requires float-weight mask**: Current `corpus.py` emits integer 0/1 masks. Check Tinker's `apply_weights()` in `loss_config.py` accepts float arrays before committing to this idea.
- ⚠️ **GeoRA (Idea 11) — architecture mismatch warning**: GeoRA paper targets Transformer-based models trong RLVR setting. Nemotron-H là Mamba/MoE hybrid với SSM selective scan matrices — FIM computation cho SSM layers không có precedent trong literature. Nếu chỉ apply GeoRA cho FFN + `lm_head` layers (bỏ SSM layers), có thể partially salvage idea. SVD computation cho toàn model 30B-A3B sẽ OOM trên 48 GB; cần restrict to top-8 transformer-like layers. **Feasibility 2/5 là optimistic** — thực tế có thể cần abandon trước khi reach training stage.
- ⚠️ **RL→SFT (Idea 12) — offline DPO ≠ online RL**: arXiv:2509.21128v2 study dùng PPO/GRPO (on-policy, dense reward). Offline DPO với 80–120 pairs là proxy rất thô. "Squeeze" effect của DPO yếu hơn nhiều so với paper's RL. Cần empirically verify bằng cách so sánh output distribution entropy sau DPO 50 steps vs. SFT 50 steps — nếu DPO không reduce entropy đáng kể, paper's mechanism không transfer và idea này fail lý thuyết.

---

## Next Steps for User

1. **Immediate (1–2 GPU runs)**: Idea 1 (solver extension) — pure Python engineering in `reasoners/bit_manipulation.py`, no training changes, zero risk. Verify new solve count before corpus rebuild. If ≥ 30 new problems solved → rebuild corpus → retrain.
2. **Week 1**: Idea 4 (drop augmentation) — delete augmentation outputs, write mini-rule generator, rebuild corpus (~50% smaller). Retrain and check eval_slice per-category accuracy delta.
3. **Week 1-2**: Idea 9 (iterative DPO on eval_slice) — use existing `infer_slice.py` to generate temperature=0.5 rollouts; construct pairs; 1-round DPO. Cheapest preference-learning option.
4. **Hold for later (high effort)**: Ideas 2, 3 (procedural crypto + DPO pairs) — validate Idea 2's rollout pass rate first; only proceed to Idea 3 if pass@10 ≥ 5%.

---

## Provenance Signature
Batch based on: data-distribution-analysis.md (2026-06-04) + tong-hui-kang-approach.md (2026-06-04) + research-overcome-cryptarithm.md (2026-06-04). Search log: 11 queries / 45 summaries / 0 full reads / ~18 min. Primary paper IDs: 2505.24273, 2505.19641, 2506.04821, EMNLP-2025-findings-616, 2603.16206, 2512.22255, 2505.24850, 2505.16322, 2503.12854, 2510.27462.

**Addendum 2026-06-05**: Ideas 11–12 added manually (user-provided papers). Paper IDs added: 2601.09361 (GeoRA), 2509.21128 (RL Squeezes SFT Expands). Both carry HIGH RISK / feasibility 2/5 — prioritize after core batch-4 ideas validated.
