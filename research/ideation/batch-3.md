# Idea Batch 3 — NVIDIA Nemotron Model Reasoning Challenge / LoRA-adapter reasoning SFT
**Generated**: 2026-06-02
**Skill**: `/benchmark-climb-ideation` (8 ideas, two mandated axes: data-augmentation in `nemotron-master/` + training-algorithm in `Continuer_Nemotron_Notebook.py` / `train_sft.py`; target 0.86 → 0.88+)
**Tier mix (configured)**: `55/30/15` (pipeline-biased, carried from batch 2; bands 45-65 / 20-40 / 5-25)
**Search log**: [batch-3-search-log.md](batch-3-search-log.md) — all primaries traced to live searches this session.

## Inputs
- **Benchmark**: NVIDIA Nemotron Model Reasoning Challenge (Kaggle leaderboard).
- **Task**: Fine-tune ONE rank-32 LoRA adapter for `Nemotron-3-Nano-30B-A3B` (Mamba/MoE hybrid) so that under greedy vLLM inference (`temp=0.0`, `max_tokens=7680`, `max_model_len=8192`) it solves the category problems and emits the final answer in `\boxed{...}`. Grader: string-match OR rel-err ≤ 1e-2; binary strings matched exactly.
- **Existing pipeline**: (A) `Continuer_Nemotron_Notebook.py` — Unsloth single-file trainer, forward monkey-patched to `cut_cross_entropy.linear_cross_entropy`, per-token CE × corpus `weights` mask applied in the loop, manual `lm_head` LoRA, LoRA fp32 / base bf16 except `mixer.gate` fp32, `MOE_TIE_WEIGHTS` (128 expert LoRA slices tied via mean-init + grad-sum), Mamba fast path forced on. (B) `nemotron-master/` — `reasoning.py → augmentation.py → corpus.py → train_sft.py`; `corpus.py` is the single source of truth for token format; `train_sft.py` supports `CE / importance_sampling / ppo / cispo / dro`. **Current score 0.86.**
- **Scope**: all 8 ideas are `enhance-existing` (100% ≥ 50% requirement). No inference-time tricks — single greedy pass; every idea is a data-time or training-time change. (Ideas using sampling do so only for **offline** data generation.)
- **No-overlap mandate**: must not duplicate batch 1–2's 19 ideas (format-verified labels, answer-upweight, concise traces, rsLoRA, module realloc, LIMO, STaR/RFT, hot-expert untying, spaced-repetition, SA-curriculum, DoRA, NEFTune, self-verify traces, PiSSA, arithmetic scratchpad, anchored-KL, LoRA seed-soup, preference-opt/SimPO, Stream-of-Search). Each idea below carries a "Distinct from batch 1–2" note.

## Summary
| Metric | Value |
|--------|-------|
| Batch size | 8 |
| Axis split | Data-augmentation 3 / Training-algorithm 5 |
| Search-tier 1/2/3 (counts) | 4 / 2 / 2 → 50 / 25 / 25 (within 55/30/15 ±10 ✅) |
| Source-trust T1+T2 | 7/8 ≈ 88% (≥ 60% ✅) |
| Patterns | P2×2, P3×1, P4×2, P8×2, P12×1 (5 distinct ✅) |
| Distinct venues | ICML, NeurIPS, ICLR, EMNLP, arXiv (5 ✅) |
| Time windows | <12mo (3), 12-36mo (3), 36-72mo (1: DoReMi); classics HER'17 / GroupDRO'19 |
| P6 (Verify) | Omitted — justified (no inference-trick; self-verify already shipped batch-2 idea-3) |

## Summary table (ranked by composite)
| # | Title | Axis | Pattern | S-Tier | Gain (pp, mid) | Feas | Effort | Score |
|---|-------|------|---------|--------|----------------|------|--------|-------|
| 1 | High-entropy "forking-token" loss weighting | Train | P4 | 1 | +0.7 | 4 | S | 3.70 |
| 2 | LoRA+ — split LR for A vs B matrices | Train | P3 | 1 | +0.5 | 5 | S | 3.55↓ |
| 3 | DoReMi category data-mixture reweighting | Data | P4 | 1 | +0.6 | 4 | M | 3.40 |
| 4 | ESFT — expert-specialized MoE LoRA targeting | Train | P8 | 2 | +0.6 | 3 | M | 3.20 |
| 5 | Deterministic CSP solver unlocks *guess* tasks | Data | P2 | 3 | +0.9 | 3 | L | 3.15 |
| 6 | GroupDRO worst-category robust objective | Train | P8 | 2 | +0.5 | 3 | M | 3.05 |
| 7 | HER-style forward generation of *guess* data | Data | P2 | 3 | +0.7 | 3 | M | 3.00 |
| 8 | GSPO sequence-level MoE-stable online RL | Train | P12 | 1 | +0.8 | 2 | L | 2.80 |

(↓ = LoRA+ had the top raw feasibility but was downgraded one slot by the devil's-advocate pass — see Notes.)

## Top-3 recommendations
- **⚡ Quick win (lowest effort): Idea 2 — LoRA+.** One-line optimizer change (`loraplus_lr_ratio`), zero data work, vLLM-neutral, 5/5 feasibility. Bank it first. (Caveat from devil's-advocate below.)
- **🎯 Big bet (highest ceiling): Idea 5 — deterministic CSP solver for `cryptarithm_guess` + `equation_numeric_guess`.** These two categories (300 problems) are **100% unsolved — zero traces exist today**. A correct solver converts them from dead weight to ~300 high-token verified examples. Largest single data-coverage gap in the corpus.
- **🛡️ Safe bet (highest confidence): Idea 3 — DoReMi category reweighting.** Principled fix for the corpus imbalance (bit_manipulation 26% of tokens); composes with every other idea; downside-bounded (worst case = current uniform mixture).

---

## AXIS 1 — DATA AUGMENTATION (`nemotron-master/`)

### Idea 5: Deterministic CSP backtracking solver to unlock the *guess* categories

- **Pattern**: P2 (Transfer — constraint-satisfaction/backtracking search from symbolic AI → linear CoT training data)
- **Search-tier**: 3 (cross-domain: symbolic AI / constraint programming)
- **Scope**: enhance-existing — add `reasoners/cryptarithm_guess.py` and extend `reasoners/equation_numeric.py`'s guess path; emit a CoT mirroring constraint propagation, ending in the existing `\boxed{answer}` scaffold. `corpus.py` format, mask, trainer unchanged.
- **One-liner**: Write a deterministic constraint-propagation + backtracking solver for the two 100%-unsolved categories and serialize its solving process as verified CoT, so `reasoning.py` finally emits `rule_found` traces for the 300 problems that currently produce none.

**Mechanism**: `cryptarithm_guess` (164) and `equation_numeric_guess` (136) are `rule_unknown` for *every* problem — they contribute **zero** training signal today (CLAUDE.md corpus table). A classic AC-3-style constraint-propagation + DFS-backtracking solver deterministically cracks letter↔digit / variable assignments; rendering its propagate→assign→conflict→backtrack steps as a linear trace (the same way `reasoners/cipher.py` etc. already mirror their solvers) yields exact, verifiable traces. This is a **coverage** unlock (new data where there was none), not a trace-style change.

**Distinct from batch 1–2**: Stream-of-Search (batch-2 idea-9) *re-renders* search for categories that already have solvers; this *builds the missing solver* so the category produces any trace at all. Orthogonal — SoS-style rendering can be reused on top.

**Cross-domain transfer**: constraint satisfaction / backtracking search (symbolic AI, CSP) → deterministic CoT-data synthesis for two reasoning categories.

**Sources** (live-searched): primary "When Do Symbolic Solvers Enhance Reasoning in LLMs?" [arXiv:2512.03272] — solver integration most helps constraint-satisfaction problems "requiring repeated backtracks"; supporting "A Reality Check of LMs as Formalizers on CSPs" [arXiv:2505.13252]; Logic-LM (Pan et al., **EMNLP 2023 Findings**, github-confirmed) — symbolic-solver-augmented faithful reasoning.

**Expected gain**: +0.3 / +0.9 / +2.0 pp 🟡 (ceiling = the ~1.7% of corpus these two categories represent + transfer to related deduce categories). **Feas** 3/5 🟡. **Effort** L.

**Falsification test**: implement the solver, confirm it verifies (`status=rule_found`) on ≥ 90% of the 300 problems; train with the new traces on a slice. If held-out exact-match on the guess categories does not rise from ~0 to ≥ +5 pp (absolute, on those categories), the solver is wrong or the format is off — revert.

**Risks**: solver may not crack all instances (some may be ill-posed → leave `rule_unknown`, don't fabricate); guess-task trace length could approach the 7680 cap on hard instances (cap branch depth, compose with batch-1 concise-trace cap).

---

### Idea 3: DoReMi category data-mixture reweighting

- **Pattern**: P4 (Scale — re-weight the proportion of each category in the training mixture)
- **Search-tier**: 1
- **Scope**: enhance-existing — changes the per-category sampling proportions consumed when assembling `corpus.jsonl` (the mixture feeding `corpus.py`); token format, mask, LoRA config, trainer unchanged.
- **One-liner**: Replace the as-generated category proportions (bit_manipulation alone = 26.4% of unmasked tokens) with DoReMi-optimized weights from a small proxy run, so training compute is spent where it most reduces worst-category loss instead of on the over-represented bucket.

**Mechanism**: DoReMi trains a small proxy with Group-DRO over domains to find mixture proportions that minimize excess loss vs a reference, then resamples the full corpus at those weights. Here the "domains" are the problem categories. The current mixture is an artifact of how many problems each solver happened to emit, not of difficulty/value; reweighting redistributes the fixed step budget toward under-served high-value categories.

**Distinct from batch 1–2**: spaced-repetition (batch-1 idea-9) changes *when* an example is seen across steps; SA-curriculum (batch-1 idea-10) changes *difficulty ordering*. This changes the *stationary mixture proportions per category* — a different axis (what fraction of the corpus each category is), set by an optimization procedure rather than a schedule.

**Sources** (live-searched): primary "DoReMi: Optimizing Data Mixtures Speeds Up LM Pretraining", Xie et al., **NeurIPS 2023 (spotlight)** [arXiv:2305.10429] (+6.5pp avg downstream, 2.6× fewer steps). Links naturally to Idea 6 (GroupDRO is DoReMi's inner loop).

**Expected gain**: +0.2 / +0.6 / +1.4 pp 🟡. **Feas** 4/5 🟢. **Effort** M.

**Falsification test**: run a cheap proxy (or even a one-pass per-category held-out-loss estimate) to derive weights; train a slice at DoReMi weights vs uniform. If macro-averaged (per-category) exact-match doesn't improve by ≥ 0.5 pp, keep current mixture.

**Risks**: DoReMi's proxy run adds cost (mitigate with a tiny proxy or a heuristic loss-proportional reweight as a first cut); down-weighting bit_manipulation too far could regress its 26% share of the leaderboard — constrain weights to a band.

---

### Idea 7: HER-style forward generation of solvable *guess* instances

- **Pattern**: P2 (Transfer — Hindsight Experience Replay goal-relabeling from RL → synthetic data generation)
- **Search-tier**: 3 (cross-domain: reinforcement learning / robotics)
- **Scope**: enhance-existing — adds a forward-construction generator under `augmenters/` (or a `reasoners/` companion) that *manufactures* guess-category problems from a chosen answer; feeds `corpus.py` in the existing format.
- **One-liner**: Instead of only trying to solve the 300 given guess problems, **generate** new ones by sampling a solution first (the "achieved goal") and constructing a puzzle around it — every generated instance is solvable-by-construction and comes with a verified trace, expanding the guess categories far beyond their original count.

**Mechanism**: HER's core trick is to relabel an episode by the goal that was actually achieved, turning sparse-reward failures into successes. Transferred here: for `equation_numeric_guess` / `cryptarithm_guess`, pick a valid assignment/solution, then synthesize a consistent problem statement and emit the forward-derivation CoT. This sidesteps the "can't solve the given puzzle" wall entirely and yields unlimited verified data with controllable difficulty.

**Distinct from batch 1–2**: STaR/RFT (batch-1 idea-7) samples *model* traces on *existing* problems and keeps correct ones; this generates *new problems* deterministically from a known solution (no model sampling, no verifier filtering needed). Complementary to Idea 5: Idea 5 solves given puzzles; Idea 7 fabricates new solvable ones.

**Cross-domain transfer**: hindsight goal-relabeling (RL, Andrychowicz et al.) → answer-first synthetic problem construction for guess categories.

**Sources** (live-searched): primary "Hindsight Experience Replay", Andrychowicz et al., **NeurIPS 2017** [arXiv:1707.01495] (relabel achieved outcomes as goals → learn from sparse/binary reward; "implicit curriculum").

**Expected gain**: +0.2 / +0.7 / +1.8 pp 🟡. **Feas** 3/5 🟡. **Effort** M.

**Falsification test**: generate 500 forward instances per guess category, train a slice. If held-out exact-match on the *original* 300 guess problems doesn't rise by ≥ 3 pp (absolute), the synthetic distribution doesn't match the real puzzles — revert or re-tune the generator's difficulty distribution.

**Risks**: synthetic distribution mismatch (generated puzzles may be easier/structurally different than the leaderboard's) — calibrate generator parameters to the 300 originals; risk of leaking a generation artifact the model memorizes instead of reasoning (gate by held-out-on-originals, not on synthetic).

---

## AXIS 2 — TRAINING ALGORITHM (`Continuer_Nemotron_Notebook.py` / `train_sft.py`)

### Idea 1: High-entropy "forking-token" loss weighting (cut_cross_entropy mask)

- **Pattern**: P4 (Scale — reweight the per-token loss along a new dimension: token entropy)
- **Search-tier**: 1
- **Scope**: enhance-existing — multiplies the existing per-token `weights` mask in the training loop of `Continuer_Nemotron_Notebook.py` (the `model._cached_per_token_ce × weights` step) by an entropy-derived factor; `cut_cross_entropy` path, LoRA config, corpus unchanged.
- **One-liner**: Up-weight the loss on the ~20% high-entropy "forking" tokens (decision points) and down-weight the low-entropy filler, focusing the rank-32 adapter's limited capacity on the tokens that actually steer the reasoning path.

**Mechanism**: "Beyond the 80/20 Rule" (NeurIPS'25) shows that in CoT only ~20% of tokens carry high entropy and act as critical forks; restricting policy-gradient updates to those tokens matches full-token training and *improves* it on larger models, while training only the low-entropy 80% degrades performance. Adapted to SFT: scale each completion token's CE by a monotone function of its (base-model or running) entropy before applying the corpus `weights` mask. The hook point already exists — the loop already multiplies per-token CE by `weights`.

**Distinct from batch 1–2**: answer-token up-weighting (batch-1 idea-2) up-weights a *fixed positional region* (the boxed answer); this up-weights *content-defined* tokens (high-entropy decision points anywhere in the trace), a different and complementary weighting signal.

**Sources** (live-searched): primary "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning", **NeurIPS 2025** [arXiv:2506.01939] (+11.04 AIME'25 on Qwen3-32B using only the 20% forking tokens).

**Expected gain**: +0.2 / +0.7 / +1.6 pp 🟡. **Feas** 4/5 🟢. **Effort** S.

**Falsification test**: compute per-token entropy from a cheap forward (or cache base-model entropy), train a slice with entropy-weighting vs uniform mask. If held-out exact-match doesn't improve by ≥ 0.5 pp at any tested weighting strength, revert to uniform.

**Risks**: the paper's result is for RL (policy gradient), not SFT-CE — the transfer to teacher-forced SFT is the main uncertainty (slice-test gates it); computing entropy needs logits, which conflicts with the no-logits `linear_cross_entropy` path — use a cached/precomputed base-entropy per token rather than live logits.

---

### Idea 2: LoRA+ — separate learning rates for A vs B matrices

- **Pattern**: P3 (Replace — swap the single LoRA LR for a two-LR optimizer grouping)
- **Search-tier**: 1
- **Scope**: enhance-existing — changes only the optimizer param-group construction in `run_training()` (apply `lr_B = ratio × lr_A` to the LoRA `B` matrices, incl. the manual `lm_head` LoRA). LoRA config, corpus, mask, inference unchanged; output is a byte-identical vanilla adapter.
- **One-liner**: Give the LoRA `B` matrices a higher LR than `A` (fixed ratio λ ≈ 4–16) so feature learning is efficient at the model's large width — a free correction to a known LoRA suboptimality, at the same compute.

**Mechanism**: LoRA+ shows the standard single-LR setup under-learns at large width because A and B should scale differently; a fixed ratio η_B/η_A corrects it, giving ~1–2% accuracy and up to 2× speedup at identical cost. Pure optimizer change — no new parameters, no inference change.

**Distinct from batch 1–2**: rsLoRA (batch-1 idea-4) changes the *forward scaling* α/√r; PiSSA (batch-2 idea-4) changes *init*; DoRA (batch-2 idea-1) changes *parameterization*. LoRA+ changes the *optimizer learning-rate geometry* — an orthogonal, composable axis.

**Sources** (live-searched): primary "LoRA+: Efficient Low Rank Adaptation of Large Models", Hayou, Ghosh, Yu, **ICML 2024** [arXiv:2402.12354]. **Contrasting (devil's-advocate)**: "Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning" [arXiv:2602.04998] argues a well-tuned single LR can match LoRA+; "ALLoRA" [arXiv:2410.09692] proposes an adaptive-LR alternative.

**Expected gain**: +0.1 / +0.5 / +1.2 pp 🟡 (contrasting evidence caps the upside — gain may be partly subsumed by simply tuning the base LR). **Feas** 5/5 🟢. **Effort** S.

**Falsification test**: train a slice at ratio λ∈{1(=baseline),4,8,16} with the base LR otherwise matched. If no λ beats a *re-tuned single-LR* baseline by ≥ 0.3 pp, the gain is just LR tuning — keep single LR. (Run the single-LR sweep as the control, per the contrast paper.)

**Risks**: gain may collapse into "you just needed a better base LR" (contrast paper) — the control sweep is mandatory; too-large λ on the fp32 LoRA over `MOE_TIE_WEIGHTS` slices could destabilize the tied-grad-sum step (watch loss spikes).

---

### Idea 4: ESFT — expert-specialized MoE LoRA targeting

- **Pattern**: P8 (Specialize — route adapter capacity to the experts most activated by these tasks)
- **Search-tier**: 2
- **Scope**: enhance-existing — changes *which* MoE experts receive LoRA in `Continuer_Nemotron_Notebook.py`. Profile expert-activation over the corpus, then apply LoRA only to high-affinity experts (and/or relax `MOE_TIE_WEIGHTS` for them) instead of tying all 128 slices uniformly.
- **One-liner**: Measure each expert's routing affinity for the competition's task mix, then concentrate the rank-32 LoRA budget on the top-affinity experts (untie those, keep the rest frozen/tied), so capacity isn't diluted across 128 rarely-relevant experts.

**Mechanism**: ESFT observes that for a given task, MoE routing is highly concentrated on a small expert subset; fine-tuning only those experts matches or beats full PEFT while saving memory/time and preserving specialization. The current code ties all 128 expert LoRA slices identically (`MOE_TIE_WEIGHTS` mean-init + grad-sum) — a uniform prior. ESFT replaces it with a measured, task-concentrated allocation.

**Distinct from batch 1–2**: hot-expert untying (batch-1 idea-8) unties experts heuristically; ESFT *measures* per-expert task affinity from routing statistics over the actual corpus and selects the trained set from data — a principled selection rule, and it can freeze the long tail entirely (capacity reallocation), not just untie.

**Sources** (live-searched): primary "Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural LLMs", **EMNLP 2024 (main)** [arXiv:2407.01906] (ESFT; up to 90% memory / 30% time savings, matches full FT).

**Expected gain**: +0.2 / +0.6 / +1.5 pp 🟡. **Feas** 3/5 🟡. **Effort** M.

**Falsification test**: log gate (`mixer.gate`) activations over a corpus sample → per-expert affinity histogram; train top-k experts only vs the current tie-all setup on a slice. If held-out exact-match doesn't hold or beat tie-all (within noise) at lower cost, revert to `MOE_TIE_WEIGHTS`.

**Risks**: Nemotron-H's expert granularity / routing may be less concentrated than DeepSeek-V2's fine-grained 66 experts → weaker selection signal; interacts with the fp32 router and the tied-grad machinery — verify gradients flow correctly to the untied subset.

---

### Idea 6: GroupDRO worst-category robust objective

- **Pattern**: P8 (Specialize — dynamically up-weight the worst-performing category group in the loss)
- **Search-tier**: 2
- **Scope**: enhance-existing — wraps the per-token CE loss in `Continuer_Nemotron_Notebook.py` with a Group-DRO reweighting over category groups (online up-weighting of the highest-loss category each step); or use `train_sft.py`'s `dro` mode directly.
- **One-liner**: Minimize the *worst-category* loss instead of the average, so the adapter can't coast on the easy/over-represented categories while a hard category (e.g., the guess tasks, cryptarithm) stays weak — directly targeting macro-accuracy on the leaderboard.

**Mechanism**: Group-DRO maintains per-group weights that grow for high-loss groups, optimizing worst-group rather than average loss. With categories as groups, it shifts gradient emphasis toward whichever category is currently failing — the right objective when the grader rewards breadth across categories and the corpus is imbalanced.

**Distinct from batch 1–2**: this is an *objective* change (minimax over groups in the loss), distinct from DoReMi (Idea 3, which changes *data proportions* offline) and from all batch 1–2 weighting ideas (which are static positional/answer weights or schedules). DoReMi sets the static mixture; GroupDRO adapts emphasis *online during training*. They compose (DoReMi = data prior, GroupDRO = dynamic correction).

**Sources** (live-searched): primary "Distributionally Robust Neural Networks for Group Shifts", Sagawa et al., **ICLR 2020** [arXiv:1911.08731] (Group-DRO; needs strong regularization to avoid worst-group overfitting). `train_sft.py` already exposes a `dro` objective.

**Expected gain**: +0.1 / +0.5 / +1.3 pp 🟡. **Feas** 3/5 🟡. **Effort** M.

**Falsification test**: train a slice with GroupDRO over categories vs uniform CE. If the *worst-category* held-out exact-match doesn't rise by ≥ 1 pp (without average dropping > 0.5 pp), revert.

**Risks**: the ICLR'20 paper itself warns Group-DRO overfits worst-group on over-parameterized nets without heavy regularization/early-stopping — pair with a small group-weight step size and the existing LoRA capacity limit; a noisy/tiny worst group can dominate and destabilize — floor group sizes.

---

### Idea 8: GSPO — sequence-level, MoE-stable online RL on verified rewards

- **Pattern**: P12 (Self-play / Self-improve — online RL with the deterministic verifier as reward)
- **Search-tier**: 1
- **Scope**: enhance-existing — adds/selects a sequence-level RL objective in `train_sft.py` (alongside its existing `ppo`/`cispo`/`importance_sampling` modes), with reward = the `reasoners/` verifier's correctness on sampled greedy/temперature rollouts. Final submission stays a single greedy pass.
- **One-liner**: Run online RL where the reward is exact-answer correctness from the deterministic verifier, but compute the importance ratio at the **sequence** level (GSPO) rather than per-token — the only variant shown to keep **MoE** RL training from diverging due to expert-activation volatility.

**Mechanism**: GSPO defines the importance ratio on sequence likelihood and clips/optimizes at the sequence level. Token-level methods (GRPO/PPO) destabilize MoE training because ~10% of activated experts flip between old and new policy; GSPO's sequence-level formulation fixes this and powers Qwen3 (an MoE). On a verifiable-answer task the reward is noise-free, making RL unusually well-posed here.

**Distinct from batch 1–2**: offline preference-opt / SimPO (batch-2 idea-8) is *offline* contrastive learning on pre-built pairs; this is *online* RL with on-policy rollouts and a sequence-level objective explicitly designed for MoE stability — a different objective and training regime, and the MoE-stability angle is unique to this base model.

**Sources** (live-searched): primary "Group Sequence Policy Optimization", Zheng et al. (Alibaba/Qwen), 2025 [arXiv:2507.18071] (sequence-level ratio stabilizes MoE RL; powers Qwen3). Supporting: `train_sft.py` `cispo`/`ppo` modes (CISPO importance-weight clipping as a fallback objective).

**Expected gain**: +0.2 / +0.8 / +2.5 pp 🟡 (highest ceiling; highest variance/effort). **Feas** 2/5 🟡. **Effort** L.

**Falsification test**: stand up a minimal GSPO loop on one category with the verifier reward; train a slice. If (a) MoE training diverges/oscillates even under GSPO, or (b) held-out exact-match doesn't beat SFT by ≥ 1 pp after one RL phase, stop and keep SFT/offline-pref.

**Risks**: full online-RL infra (rollout sampling + verifier in the loop) is the heaviest lift in the batch; the rank-32 LoRA + `MOE_TIE_WEIGHTS` interaction with on-policy updates is untested; reward-hacking the boxed format (mitigate with format checks in the reward). Adopt only after cheaper ideas are banked.

---

## Verification Report — Batch 3
| # | Title (short) | Novelty | Provenance | Feas | Gain | Falsif | Risk | Comply | Final |
|---|---------------|---------|------------|------|------|--------|------|--------|-------|
| 1 | High-entropy token weighting | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.7 🟡 | OK ✅ | MED (SFT-vs-RL transfer; logits-path) | PASS | **KEEP** |
| 2 | LoRA+ split LR | EXTENDS ✅ | VERIFIED ✅ | 5/5 | +0.5 🟡 | OK ✅ | MED (subsumed-by-LR-tuning) | WARN | **KEEP (↓1 slot)** |
| 3 | DoReMi category reweighting | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.6 🟡 | OK ✅ | LOW-MED (proxy cost) | PASS | **KEEP** |
| 4 | ESFT expert-specialized LoRA | EXTENDS ✅ | VERIFIED ✅ | 3/5 | +0.6 🟡 | OK ✅ | MED (routing concentration) | PASS | **KEEP** |
| 5 | CSP solver unlocks guess tasks | NOVEL ✅ | VERIFIED ✅ | 3/5 | +0.9 🟡 | OK ✅ | MED (solver coverage/length) | PASS | **KEEP** |
| 6 | GroupDRO worst-category | EXTENDS ✅ | VERIFIED ✅ | 3/5 | +0.5 🟡 | OK ✅ | MED (worst-group overfit) | PASS | **KEEP** |
| 7 | HER forward guess-data gen | EXTENDS ✅ | VERIFIED ✅ | 3/5 | +0.7 🟡 | OK ✅ | MED (distribution mismatch) | PASS | **KEEP** |
| 8 | GSPO MoE online RL | EXTENDS ✅ | VERIFIED ✅ | 2/5 | +0.8 🟡 | OK ✅ | HIGH (infra + MoE stability) | WARN | **KEEP** |

### Counts
- Verified: 8 / Rejected: 0 / Downgraded: 1 (LoRA+ ↓1 slot, devil's-advocate).
- Re-search cycles used: 0. Final batch size: 8.

### Cross-idea consistency
- **Near-duplicates**: none. Ideas 5 & 7 both target the guess categories but by opposite mechanisms (solve-given vs generate-new) — complementary, not duplicate. Ideas 3 & 6 both address category imbalance but at different stages (offline mixture vs online minimax) — designed to compose.
- **Contradictions (composable, not blocking)**: Idea 1 (entropy weighting) and Idea 6 (GroupDRO group weighting) both reshape the loss weighting — co-tune (apply entropy weighting *within* groups). Idea 4 (untie/select experts) vs the base `MOE_TIE_WEIGHTS` prior — mutually exclusive on the chosen expert subset by design.
- **Score distribution**: healthy spread (feas 2–5; all gains 🟡 with honest ceilings); no all-5 / all-🟢 over-confidence.

## Notes & warnings
- **Single greedy-pass honored**: every idea is data-time or training-time. Idea 8 (GSPO) and Idea 7 (HER gen) use sampling/rollouts **only offline**; the submission remains one greedy vLLM pass, rank ≤ 32.
- **P6 (Verify) intentionally omitted**: the no-inference-trick constraint rules out test-time verifiers, and the training-time self-verify pattern is already shipped (batch-2 idea-3). Forcing a P6 here would duplicate or violate constraints. 5 distinct patterns still met.
- **Devil's-advocate (top-1)**: LoRA+ was the highest-feasibility idea but [arXiv:2602.04998] shows a well-tuned single LR can match it — so its gain may reduce to LR tuning (a prerequisite, not a contribution). Downgraded #1→#2 and the single-LR control sweep is made mandatory in its falsification test.
- **Prerequisites / measurements** (NOT ideas — surfaced per skill rule):
  - The base LR sweep is a *control* for Idea 2, not a standalone idea.
  - A per-category held-out-loss / error-type bucketing (format-zero / truncation / arithmetic-slip / method-wrong / unsolved-category) is needed to choose between Ideas 1/3/6 (weighting) and Ideas 5/7 (coverage). This is the single most useful measurement before spending a full run.
  - Logits-availability check: Ideas 1 (entropy) and any KL-style term need token logits, which conflict with the `cut_cross_entropy` no-logits fast path — precompute/cache base statistics rather than materializing logits live.
- **Composition map**: Ideas 1+2+4 are independent training-axis levers (loss weighting / optimizer LR / expert selection) and stack. Ideas 3+6 stack (data prior + online correction). Ideas 5+7 stack (solve + generate) and are the highest-leverage *coverage* pair. Idea 8 should be attempted last, on top of the best SFT adapter.
- **vLLM note**: unlike batch-2's DoRA/PiSSA, none of these change the adapter *format* — all 8 yield a standard vanilla rank-32 adapter (Idea 4 changes which modules carry LoRA but the saved form is standard). No load-test gate needed.

## Run-order recommendation
1. **Bank cheap training wins**: Idea 2 (LoRA+, with the control sweep) + Idea 1 (high-entropy weighting). Both are S-effort, vLLM-neutral.
2. **Fix the data**: Idea 3 (DoReMi reweighting) — then the big coverage bet, Idea 5 (CSP solver) and its complement Idea 7 (HER generation) for the 300 unsolved guess problems.
3. **MoE-specific**: Idea 4 (ESFT expert targeting), then Idea 6 (GroupDRO) once the mixture is set.
4. **Hold for the 0.87→0.88+ push**: Idea 8 (GSPO online RL) — highest ceiling, highest cost; only after the above are banked and measured.
