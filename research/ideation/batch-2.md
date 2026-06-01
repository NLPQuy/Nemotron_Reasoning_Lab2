# Idea Batch 2 — NVIDIA Nemotron Model Reasoning Challenge / LoRA-adapter reasoning SFT
**Generated**: 2026-06-02T00:00:00Z
**Time-to-batch**: ~14 min
**Skill version**: 0.1.0
**Skill invocation**: `/benchmark-climb-ideation` (8–10 training ideas, enhance Continuer_Nemotron_Notebook.py + nemotron-master/, target 0.86 → 0.88+, `--tier-mix 55/30/15`)

## Inputs
- Benchmark: NVIDIA Nemotron Model Reasoning Challenge (Kaggle public leaderboard).
- Task / problem: Fine-tune ONE rank-32 LoRA adapter for `Nemotron-3-Nano-30B-A3B` (Mamba/MoE hybrid, `modeling_nemotron_h`) so that, under greedy vLLM inference (`temperature=0.0`, `max_tokens=7680`, `max_model_len=8192`), it solves reasoning problems and emits the final answer inside `\boxed{...}`. Grader: string match OR relative error ≤ 1e-2; binary strings matched **exactly**. Deliverable: `submission.zip` with `adapter_config.json`, vLLM-loadable.
- Existing pipeline: two non-shared codebases. **(A)** `Continuer_Nemotron_Notebook.py` (root, Unsloth single-file trainer): forward monkey-patched → `cut_cross_entropy.linear_cross_entropy`; per-token CE × corpus `weights` mask applied in the loop; `lm_head` LoRA added manually (renamed `backbone.lm_head` on save); LoRA fp32, base bf16 except MoE router `mixer.gate` fp32; `MOE_TIE_WEIGHTS` keeps all 128 expert LoRA slices identical (mean-init + grad-sum); Mamba CUDA fast path forced on; knobs at top: `LORA_RANK`, `NUM_STEPS`, `BATCH_SIZE`, `RESET_WEIGHTS`. **(B)** `nemotron-master/` (data-gen + Tinker SFT): `reasoning.py → augmentation.py → corpus.py → train_sft.py`; `corpus.py` is the single source of truth for the token format — completion `"{reasoning}\n</think>\n\\boxed{{answer}}<|im_end|>"`, prompt mask 0, completion mask 1; `reasoners/<category>.py` deterministic solvers; `train_sft.py` supports `CE / importance_sampling / ppo / cispo / dro`. **Current score: 0.86.**
- Batch scope: **all 9 enhance-existing** (≥ 50% requirement satisfied — 100%).
- Tier mix (configured): `55/30/15` (pipeline-biased, user override).
- Baseline: `Nemotron-3-Nano-30B-A3B` + current rank-32 LoRA @ **0.86**.
- Compute budget: single RTX-PRO-6000 (Modal) or Kaggle GPU per run; each full train+submit cycle is the expensive unit, so ideas are validated on a small held-out problem slice first, in < 1 short training run.
- Time budget: implied — favor changes validatable in < 1 short training run.
- Constraints: `max_lora_rank=32`; greedy `temperature=0.0`; vLLM-loadable adapter; deliverable `submission.zip` with `adapter_config.json`. **No inference-time tricks (single greedy pass)** — every idea is a training-time / data-time change.
- **No-overlap mandate**: must not duplicate batch-1's 10 ideas (format-verified labels, answer-token loss up-weighting, concise anti-truncation traces, rsLoRA √r scaling, target-module/rank reallocation, LIMO/s1 curation, STaR/RFT, hot-expert untying, spaced-repetition scheduling, simulated-annealing curriculum). Every idea below is mechanism-level distinct — see the per-idea "Distinct from batch-1" note.

## Summary
| Metric | Value |
|--------|-------|
| Batch size | 9 |
| Tier 1 / 2 / 3 (counts) | 5 / 3 / 1 |
| Tier mix vs configured | 56/33/11 vs 55/30/15 (deviation ≤ 10pp per tier ✅) |
| Scope mix | 9 enhance-existing / 0 greenfield (≥ 50% enhance ✅) |
| Patterns used | P3×2, P4×2, P1, P2, P5, P6, P12 (7 distinct) |
| Distinct venues | 6 (ICML, NeurIPS, ICLR, COLM, + arXiv preprints, + workshop) |
| Time windows | <12mo (2), 12-36mo (5), 36-72mo (2) |
| Avg feasibility | 4.0/5 |
| Avg confidence | 🟢 22%, 🟡 67%, 🔴 11% |

## Summary table
| # | Title | Pattern | Tier | Gain (pp) | Feas | Effort | Score |
|---|-------|---------|------|-----------|------|--------|-------|
| 1 | DoRA: weight-decomposed LoRA | P3 | 1 | +0.7 | 5 | M | 3.82 |
| 2 | NEFTune noisy-embedding regularization | P4 | 1 | +0.5 | 5 | S | 3.87↓ |
| 3 | Self-verification "check-then-box" traces | P6 | 2 | +0.7 | 4 | M | 3.57 |
| 4 | PiSSA / LoRA-GA principled LoRA init | P3 | 1 | +0.6 | 4 | M | 3.40 |
| 5 | Exact-arithmetic scratchpad for numeric categories | P5 | 2 | +0.6 | 4 | M | 3.35 |
| 6 | Anchored-SFT KL regularization vs drift | P4 | 1 | +0.5 | 4 | M | 3.32 |
| 7 | LoRA seed-soup (weight averaging) | P1 | 1 | +0.6 | 3 | M | 3.05 |
| 8 | Offline preference optimization on verified pairs | P12 | 2 | +0.8 | 3 | L | 3.00 |
| 9 | Stream-of-Search backtracking traces | P2 | 3 | +0.6 | 3 | L | 2.90 |

(↓ = NEFTune had the top raw composite 3.87 but was downgraded one slot by the devil's-advocate pass — see Notes.)

## Top-3 recommendations

### 🏆 Top-1 by composite score
**Idea 1: DoRA — weight-decomposed LoRA** — Score: 3.82
The strongest provenance/upside drop-in: decompose each adapted weight into magnitude + direction, learn the direction with the existing rank-32 LoRA and a small magnitude vector. ICML'24 Oral, **no inference overhead** (merges back), consistently beats LoRA on commonsense/reasoning at equal rank — exactly the "get more out of a fixed rank-32 budget" lever the competition forces. Promoted to #1 after the devil's-advocate pass downgraded NEFTune. (One flag: confirm the DoRA adapter is vLLM-loadable — see risks.)

### ⚡ Quick win (lowest effort)
**Idea 2: NEFTune noisy-embedding regularization** — Effort: S
A few-line change: add scaled uniform noise to the input embeddings during training only (inference unchanged). Zero data work, vLLM-neutral, near-free. **Caveat from the devil's-advocate pass**: NEFTune's documented gains concentrate on conversational quality; on reasoning/factual leaderboards the original paper reports scores merely *stable*. Treat as a cheap regularizer worth one slice-test, not a guaranteed win.

### 🛡️ Safe bet (highest confidence)
**Idea 4: PiSSA / LoRA-GA principled LoRA init** — Confidence: 🟢
Replace LoRA's random A / zero B init with a principled init (PiSSA: principal singular vectors of the weight; LoRA-GA: first-step full-FT gradient alignment). NeurIPS'24 Spotlight, +5.16 pp on GSM8K for Mistral-7B at identical setup. Pure init change, composes with everything else, and PiSSA can be exported as a vanilla vLLM-loadable LoRA via the residual-conversion save path.

## Ranked ideas

### Idea 1: DoRA — weight-decomposed low-rank adaptation

- **Pattern**: P3 (Replace — swap the LoRA parameterization for DoRA)
- **Tier**: 1
- **Target task**: Same as batch — extract more learning capacity from the fixed rank-32 budget by changing *how* the adapter is parameterized (not which modules / what scaling).
- **Scope**: enhance-existing — changes the adapter form in the `LoraConfig` of `Continuer_Nemotron_Notebook.py` (`use_dora=True`) and mirrors it for the manual `lm_head` LoRA. Forward monkey-patch, per-token weight mask, `MOE_TIE_WEIGHTS`, tokenizer, and corpus stay unchanged.
- **One-liner**: Decompose each adapted weight into a learnable magnitude vector + a LoRA-updated direction so the rank-32 adapter recovers full-FT-like update geometry that vanilla LoRA cannot express.

**Mechanism**:
DoRA writes `W' = m · (W0 + BA) / ‖W0 + BA‖_c`, learning the column-wise magnitude `m` separately from the LoRA direction `BA`. Vanilla LoRA couples magnitude and direction (it can only scale them together); DoRA decouples them, which the paper shows recovers the negative magnitude-vs-direction correlation seen in full fine-tuning. At submission the magnitude folds into the merged weight, so there is **no inference overhead** and the result is still a single adapter.

**Distinct from batch-1**: batch-1 idea-4 (rsLoRA) changes the *scaling factor* (α/√r); idea-5 changes *which modules* get rank. DoRA changes the *parameterization* (adds a magnitude DOF) — orthogonal and composable with both.

**Source inspirations**:
- Primary: "DoRA: Weight-Decomposed Low-Rank Adaptation", Liu et al., **ICML 2024 (Oral)** [arXiv:2402.09353](https://arxiv.org/abs/2402.09353).
- Supporting: NVlabs official PyTorch impl (https://github.com/NVlabs/DoRA); PEFT `use_dora` flag.

**Why expected to improve**:
DoRA consistently beats LoRA on commonsense-reasoning suites at equal rank/params (its core benchmark family), and the competition pins you at max rank 32 where squeezing more expressivity from the same budget is the whole game. The magnitude DOF is nearly free in params and merges away at inference, so it cannot add latency or break the single-greedy-pass constraint.

**Expected gain**: +0.2 / +0.7 / +1.5 pp 🟡
**Feasibility**: 5/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. Set `use_dora=True` in the `LoraConfig`; replicate for the manually-added `lm_head` LoRA so it isn't left as plain LoRA.
2. DoRA changes effective step size slightly — run a short LR check on the slice.
3. **Confirm the saved adapter is vLLM-loadable** (vLLM LoRA support for the DoRA magnitude vector is the gating risk — see below); if not, fall back to merge-then-extract-vanilla-LoRA.

**Risks**:
- **vLLM compatibility (MED)**: vLLM's LoRA loader may not accept DoRA's magnitude tensor in `adapter_config.json`/weights; must verify load + a few greedy generations before committing a run.
- Mamba `mixer` + DoRA magnitude on non-standard Nemotron-H modules could interact with the forced CUDA fast path — test load.
- Slightly higher train memory/time than LoRA (extra magnitude vector + norm).

**Falsification test**: Train DoRA vs vanilla LoRA at matched (re-tuned) LR on the 200-problem slice. If (a) the adapter fails to vLLM-load, OR (b) held-out exact-match is not ≥ baseline within noise, revert to vanilla LoRA.

---

### Idea 2: NEFTune noisy-embedding regularization

- **Pattern**: P4 (Scale — add a single training-time noise-magnitude knob)
- **Tier**: 1
- **Target task**: Same as batch — regularize the rank-32 SFT on a small synthetic corpus to reduce overfitting and improve generalization to held-out reasoning problems.
- **Scope**: enhance-existing — adds a forward hook on the input-embedding output inside `run_training()` of `Continuer_Nemotron_Notebook.py` (active during training only). Corpus, mask, LoRA config, and inference path unchanged.
- **One-liner**: During training, add scaled uniform noise `α/√(Ld)` to the token-embedding activations so the model learns a smoother manifold and overfits the small synthetic corpus less.

**Mechanism**:
On each training forward, perturb the embedding output `x ← x + Uniform(−1,1)·α/√(Ld)` (L = seq len, d = hidden), with `α` a single new knob (NEFTune uses 5–15). No change at inference (hook disabled), so the deliverable is byte-identical in format and vLLM-loadable. This is a denoising/adversarial-style regularizer, not a data or decoding change.

**Distinct from batch-1**: no batch-1 idea touches activations/regularization; all batch-1 levers are data, loss-weight, scaling, module-routing, or scheduling.

**Source inspirations**:
- Primary: "NEFTune: Noisy Embeddings Improve Instruction Finetuning", Jain et al., **ICLR 2024** [arXiv:2310.05914](https://arxiv.org/abs/2310.05914).
- Supporting: official repo (https://github.com/neelsjain/NEFTune); integrated in HF TRL (`neftune_noise_alpha`).
- Contrasting: NEFTune's own OpenLLM-leaderboard ablation — reasoning/factual scores stay *stable* (gains concentrate on AlpacaEval conversational quality), so the reasoning upside is uncertain.

**Why expected to improve**:
A rank-32 adapter on a finite synthetic corpus is prone to memorizing trace surface form; embedding noise is a cheap regularizer shown to lift instruction-tuned quality with no data and no inference cost. If even a fraction of current losses are overfit/format-brittle generalization gaps, a small `α` may recover them at near-zero cost.

**Expected gain**: +0.0 / +0.5 / +1.2 pp 🟡 (honest: contrasting evidence caps the reasoning upside)
**Feasibility**: 5/5 🟢
**Effort**: S 🟢

**Implementation sketch**:
1. Register a forward hook on the embedding module; add the scaled noise only when `model.training`.
2. Add `NEFTUNE_ALPHA` knob; sweep `α ∈ {0(=baseline), 5, 10, 15}` on the slice.
3. Confirm inference path has the hook disabled (no noise at eval/submit).

**Risks**:
- Reasoning/exact-match upside is uncertain (contrasting evidence) — may be flat on this grader.
- Too-large `α` degrades exact arithmetic (noise hurts precise token sequences).

**Falsification test**: Train `α=10` vs `α=0` on the slice. If held-out exact-match does not improve by ≥ 0.5 pp at any tested `α`, drop (it's free to test, so low cost to falsify).

---

### Idea 3: Self-verification "check-then-box" reasoning traces

- **Pattern**: P6 (Verify — bake a verification step into the training trace)
- **Tier**: 2
- **Target task**: Same as batch — reduce *reasoning-wrong* (not format, not truncation) losses by teaching the model to re-derive/check its answer before emitting `\boxed{}`.
- **Scope**: enhance-existing — modifies the `{reasoning}` content produced in `nemotron-master/reasoning.py` / `augmentation.py` and the trace assembled in `corpus.py`; the completion scaffold (`\n</think>\n\\boxed{{answer}}`), mask scheme, and trainer are unchanged.
- **One-liner**: Append a short, deterministic self-check segment ("verify: substitute the answer back / recompute the key quantity → matches") just before `</think>`, so the model learns to validate then commit rather than commit blindly.

**Mechanism**:
For each verified trace, the deterministic solver in `reasoners/` already knows the intermediate quantities; emit a compact verification tail ("Check: plug `answer` into the original constraint → LHS=RHS ✓") inside `{reasoning}`, then the unchanged `</think>\n\boxed{answer}` tail. The model is trained on traces that always *verify-before-boxing*, internalizing a behavior that catches its own arithmetic/logic slips at greedy inference.

**Distinct from batch-1**: idea-1 cleans *labels* (format hygiene); this changes *trace content* to include a reasoning-level self-check. It is NOT STaR (idea-7): no model sampling, no new problems — it rewrites existing verified traces with a verify step.

**Source inspirations**:
- Primary: "VeriCoT: Neuro-symbolic Chain-of-Thought Validation via Logical Consistency Checks", 2025 [arXiv:2511.04662](https://arxiv.org/abs/2511.04662) — SFT on verification-distilled traces improves CoT validity.
- Supporting: "SkillFactory: Self-Distillation For Learning Cognitive Behaviors", 2025 [arXiv:2512.04072](https://arxiv.org/abs/2512.04072) (teach verification as an SFT-stage cognitive skill); "SCOTT: Self-Consistent CoT Distillation", ACL 2023 [arXiv:2305.01879](https://arxiv.org/abs/2305.01879).

**Why expected to improve**:
A verify-then-commit trace teaches an explicit consistency check that, under greedy decoding, catches the off-by-one / sign / carry errors that turn a correct method into a wrong boxed value. Because the solvers make the check exact, the added segment is always truthful (no hallucinated verification), and it composes with batch-1's concise-trace cap to stay under the 7680 budget.

**Expected gain**: +0.2 / +0.7 / +1.8 pp 🟡
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. In each `reasoners/<category>.py`, emit a 1–3 line verification tail using quantities the solver already computed.
2. In `corpus.py`, place the tail inside `{reasoning}` (mask 1), before the scaffold; re-tokenize and confirm length budget.
3. Slice-compare exact-match and average completion length vs baseline traces.

**Risks**:
- The verify segment lengthens traces → risks truncation; cap its length and compose with batch-1 idea-3.
- For binary-string answers a "substitute-back" check may be trivial/unhelpful; target numeric & equation categories first.

**Falsification test**: Train verify-traces vs plain traces on the slice. If held-out exact-match on the numeric/equation categories does not improve by ≥ 0.5 pp (and total truncation-rate doesn't worsen), drop.

---

### Idea 4: PiSSA / LoRA-GA principled LoRA initialization

- **Pattern**: P3 (Replace — swap random LoRA init for a principled init)
- **Tier**: 1
- **Target task**: Same as batch — converge to a better rank-32 solution within the fixed `NUM_STEPS` by starting the adapter from an informed subspace instead of random A / zero B.
- **Scope**: enhance-existing — changes only the LoRA initialization in the `LoraConfig` (`init_lora_weights="pissa"` or LoRA-GA pre-pass); rank, target modules, scaling, mask, and trainer unchanged. Requires `RESET_WEIGHTS=True` (fresh adapter) to apply the init.
- **One-liner**: Initialize the LoRA factors from the principal singular vectors of the base weight (PiSSA) or from the first-step full-FT gradient (LoRA-GA), so the adapter spends its 32 ranks on high-signal directions from step 0.

**Mechanism**:
PiSSA runs SVD on each target weight `W`, puts the top-r singular components into the trainable A/B and freezes the residual; LoRA-GA instead initializes A/B from the eigenvectors of the first-batch full-FT gradient. Both replace LoRA's random/zero init, aligning early updates with full fine-tuning and reaching a better basin in the same step budget. For deployment, PiSSA is exported back to a standard LoRA via PEFT's residual-conversion save so the adapter is vanilla and vLLM-loadable.

**Distinct from batch-1**: rsLoRA (idea-4) changes the *scaling factor* at fixed init; this changes the *initialization* at fixed scaling. Composable.

**Source inspirations**:
- Primary: "PiSSA: Principal Singular Values and Singular Vectors Adaptation", Meng et al., **NeurIPS 2024 (Spotlight)** [arXiv:2404.02948](https://arxiv.org/abs/2404.02948) — +5.16 pp on GSM8K (Mistral-7B) vs LoRA, same setup.
- Supporting: "LoRA-GA: Low-Rank Adaptation with Gradient Approximation", Wang et al., **NeurIPS 2024** [arXiv:2407.05000](https://arxiv.org/abs/2407.05000) (+11.5% GSM8K on Llama-2-7B).

**Why expected to improve**:
Both papers show large reasoning-benchmark gains from initialization alone at identical rank and step count — directly relevant since your budget (steps, rank 32) is fixed. A better starting subspace converts the same compute into a higher-quality adapter.

**Expected gain**: +0.2 / +0.6 / +1.5 pp 🟡
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. Set `init_lora_weights="pissa"` (or `"pissa_niter_k"` for fast approx SVD); ensure `RESET_WEIGHTS=True`.
2. Handle the manual `lm_head` LoRA init consistently; for export use PiSSA→residual conversion so the saved adapter is vanilla LoRA.
3. Confirm SVD on Nemotron-H Mamba/MoE module shapes succeeds; slice-compare convergence + exact-match.

**Risks**:
- PiSSA modifies the effective base (residual) during training; the conversion-to-vanilla-LoRA export step must be correct or the adapter won't vLLM-load as expected.
- SVD over very large MoE expert weights adds one-time init cost; LoRA-GA's gradient pre-pass needs one full-FT-grad batch.
- Only applies to fresh training (`RESET_WEIGHTS=True`); cannot be bolted onto a continued adapter.

**Falsification test**: Train PiSSA-init vs default-init at matched steps/LR on the slice. If held-out exact-match is not ≥ +0.5 pp AND converged-loss is not lower, revert to default init.

---

### Idea 5: Exact-arithmetic scratchpad for numeric categories

- **Pattern**: P5 (Decompose — break arithmetic into explicit digit-level scratchpad steps)
- **Tier**: 2
- **Target task**: Same as batch — rescue numeric/`equation_numeric`/`unit_conversion` losses caused by silent multi-digit arithmetic slips, where the *method* is right but the computed number is off.
- **Scope**: enhance-existing — enriches the `{reasoning}` arithmetic rendering in `nemotron-master/reasoners/` (using the long-multiplication/long-division helpers already in `reasoners/store_types.py`) and `corpus.py`. Scaffold, mask, and trainer unchanged.
- **One-liner**: Render every nontrivial arithmetic step as an explicit per-digit scratchpad (carry-by-carry, with the reverse-digit ordering shown to help length generalization) so greedy decoding reproduces exact digits instead of guessing the result.

**Mechanism**:
Where the solver currently states `1234 × 5678 = …`, instead emit the worked scratchpad (partial products / running carries) that `store_types.py` can already generate, optionally writing operands least-significant-digit-first. The model learns to *compute* digit-by-digit rather than recall a result, which the scratchpad literature shows is what makes multi-step arithmetic reliable and length-generalizing.

**Distinct from batch-1**: idea-3 *shortens* traces; this *lengthens* the arithmetic portion deliberately for precision. They trade off and must be co-tuned (apply scratchpad only to arithmetic-heavy steps, keep narration concise elsewhere).

**Source inspirations**:
- Primary: "Show Your Work: Scratchpads for Intermediate Computation with Language Models", Nye et al., 2021 [arXiv:2112.00114](https://arxiv.org/abs/2112.00114) — scratchpads make long addition/eval generalize where direct prediction fails.
- Supporting: "Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count", 2024 [arXiv:2410.15787](https://arxiv.org/abs/2410.15787) (reverse-digit / index-hint formatting).

**Why expected to improve**:
The grader matches binary strings exactly and numbers within 1e-2; a single mis-carried digit is a hard zero. Explicit scratchpad computation is the canonical fix for exactly that failure mode, and your deterministic solvers can emit *correct* scratchpads for free (no model sampling).

**Expected gain**: +0.2 / +0.6 / +1.6 pp 🟡 (upside = fraction of losses that are arithmetic slips, not method errors)
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. Add a `scratchpad=True` rendering path in the arithmetic helpers of `store_types.py`; wire it into the numeric `reasoners/`.
2. Gate scratchpad to operands above a digit-count threshold; keep small arithmetic inline to control length.
3. Slice-compare exact-match on numeric categories + cap-hit rate (length guardrail).

**Risks**:
- Scratchpads inflate length → truncation risk on hard problems; gate by digit count and compose with idea-3's cap.
- Over-verbose scratchpad on easy ops wastes budget for no gain.

**Falsification test**: Train scratchpad vs baseline arithmetic rendering on the slice. If held-out exact-match on numeric categories doesn't rise by ≥ 0.5 pp OR the 7680-cap-hit rate rises > 2 pp, revert.

---

### Idea 6: Anchored-SFT KL regularization against distributional drift

- **Pattern**: P4 (Scale — add a KL-anchoring regularization-strength knob to the loss)
- **Tier**: 1
- **Target task**: Same as batch — keep the rank-32 SFT from drifting off the base model's strong general reasoning prior while still fitting the synthetic corpus.
- **Scope**: enhance-existing — adds a KL-to-base regularizer term in the loss computed in the training loop of `Continuer_Nemotron_Notebook.py` (alongside the existing per-token CE × weights). LoRA config, corpus, mask, and inference unchanged.
- **One-liner**: Add a small KL(base ‖ adapted) anchor on the completion tokens so SFT tightens the answer distribution without the catastrophic-forgetting drift that plain/up-weighted SFT can cause.

**Mechanism**:
During training, compute the frozen base model's next-token distribution on the completion (LoRA disabled / adapters-off forward) and add `β·KL` to the CE loss. This bounds how far the adapter moves from the base's general reasoning competence, preserving capabilities the synthetic corpus doesn't cover while still learning the target format. `β` is a single new knob.

**Distinct from batch-1**: idea-2 *up-weights* answer tokens (pushes harder); this *anchors* against drift (pulls back). They are opposing-but-composable forces — anchoring is exactly the guardrail the batch-1 idea-2 warning called for, but realized as an explicit KL rather than a clamp.

**Source inspirations**:
- Primary: "Anchored Supervised Fine-Tuning", 2025 [arXiv:2509.23753](https://arxiv.org/abs/2509.23753) — KL anchoring beats DFT/SFT on math reasoning (+17.89 vs base) by preventing drift.
- Supporting: "SFT Doesn't Always Hurt General Capabilities", 2025 [arXiv:2509.20758](https://arxiv.org/abs/2509.20758); "Learning from the Undesirable: Robust Adaptation without Forgetting", 2025 [arXiv:2511.13052](https://arxiv.org/abs/2511.13052).

**Why expected to improve**:
The base `Nemotron-3-Nano` already reasons well; aggressive SFT on a narrow synthetic corpus can erode that prior on the (unknown) leaderboard mix. A KL anchor retains base competence on out-of-corpus problem types while still learning the boxed format — directly targeting generalization to unseen leaderboard categories.

**Expected gain**: +0.1 / +0.5 / +1.3 pp 🟡
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. Add an adapters-disabled forward pass (or cache base logits) for the completion tokens; compute token-KL.
2. Add `KL_ANCHOR_BETA` knob; loss = CE×weights + β·KL. Sweep `β ∈ {0, 0.05, 0.1, 0.2}` on the slice.
3. Watch that the `cut_cross_entropy` monkey-patch still works — KL needs base probabilities, which may require a logits path for the base pass (manage memory).

**Risks**:
- The extra base forward + KL conflicts with the no-logits `linear_cross_entropy` path; may need a cheap top-k or cached-base-logits approximation (compute/memory).
- Too-large `β` under-fits the target format → boxed-format regressions.

**Falsification test**: Train with `β=0.1` vs `β=0` on the slice. If held-out exact-match (esp. on held-out *categories*) does not improve by ≥ 0.5 pp, set `β=0`.

---

### Idea 7: LoRA seed-soup (weight averaging of multiple runs)

- **Pattern**: P1 (Combine — average several independently-trained adapters into one)
- **Tier**: 1
- **Target task**: Same as batch — reduce variance and land in a flatter, better-generalizing basin of the rank-32 adapter without any inference-time ensembling.
- **Scope**: enhance-existing — a post-training step that averages the LoRA tensors of N runs of `Continuer_Nemotron_Notebook.py` (varied seed / LR / data order) into a single adapter. Training loop and inference unchanged; output is one vanilla adapter.
- **One-liner**: Train N cheap rank-32 adapters with different seeds/LRs and average their weights ("model soup") into one adapter that generalizes better than any single run — at zero extra inference cost.

**Mechanism**:
Run N short trainings differing only in seed/LR/data-order; for each LoRA tensor (and the manual `lm_head` LoRA), average across runs (uniform "uniform soup" or greedy-soup by held-out exact-match). Because fine-tuned models lie in a shared low-loss basin, the averaged adapter often beats the best single member, and the result is still one rank-32 adapter — fully vLLM-loadable, single greedy pass.

**Distinct from batch-1**: no batch-1 idea touches multi-run aggregation; all are single-run data/loss/scaling/scheduling changes.

**Source inspirations**:
- Primary: "Model soups: averaging weights of multiple fine-tuned models…", Wortsman et al., **ICML 2022** [arXiv:2203.05482](https://arxiv.org/abs/2203.05482).
- Supporting: model-soups official repo (https://github.com/mlfoundations/model-soups); LoRA averaging is linear in adapter space (composes cleanly at fixed rank).

**Why expected to improve**:
Greedy/uniform soups reliably add robustness and a small accuracy bump over the best single run by flattening the loss surface. On a fixed-rank adapter where single-run variance can be ±noise on the leaderboard, averaging banks the variance reduction as a stable gain with no inference change.

**Expected gain**: +0.1 / +0.6 / +1.2 pp 🟡
**Feasibility**: 3/5 🟡 (cost driver: N× training runs)
**Effort**: M 🟡

**Implementation sketch**:
1. Launch N=3–5 runs varying seed/LR/data-order; keep rank, modules, format identical.
2. Average matching LoRA tensors (try uniform soup first, then greedy-soup ordered by slice exact-match).
3. Verify the soup adapter vLLM-loads and beats the best single member on the slice.

**Risks**:
- N× compute — the main feasibility hit; mitigate by reusing cheap short runs.
- Averaging only valid if runs share the *same* rank/module layout and init regime (don't mix PiSSA-init with random-init members).

**Falsification test**: Compare uniform/greedy soup vs the best single member on the slice. If the soup does not beat the best member by ≥ 0.3 pp held-out exact-match, ship the best single member instead.

---

### Idea 8: Offline preference optimization on verified correct-vs-incorrect pairs

- **Pattern**: P12 (Self-improve — use the model's own outputs, labeled by the verifier, as preference data)
- **Tier**: 2
- **Target task**: Same as batch — push probability mass toward correct boxed answers and *away* from the specific wrong answers the current adapter produces, beyond what SFT-on-correct (STaR) achieves.
- **Scope**: enhance-existing — reuses `train_sft.py`'s preference-capable objectives (`dro` / `cispo` / `ppo`) in `nemotron-master/`, fed by pairs built from `reasoners/` verifier labels; the corpus format and tokenizer are reused. Final inference is still greedy single-pass.
- **One-liner**: For problems where the adapter can produce both a correct and an incorrect trace, form (chosen=correct, rejected=incorrect) pairs and run a reference-free preference loss (SimPO/DPO-style) so the model contrastively learns to avoid its own failure modes.

**Mechanism**:
Offline (data-gen sampling allowed; submission stays greedy), sample K traces per problem with the current adapter, label each via the deterministic `reasoners/` verifier, and form chosen/rejected pairs. Train with SimPO's reference-free length-normalized margin loss (or DPO via the existing `dro`/`cispo` path), which the existing `train_sft.py` objectives support. This adds a *negative* signal that pure SFT-on-correct lacks.

**Distinct from batch-1**: idea-7 (STaR) is SFT on *correct-only* self-generated traces; this is *contrastive* (correct vs incorrect) preference optimization — a different objective using the same verifier.

**Source inspirations**:
- Primary: "SimPO: Simple Preference Optimization with a Reference-Free Reward", Meng et al., **NeurIPS 2024** [arXiv:2405.14734](https://arxiv.org/abs/2405.14734).
- Supporting: "Direct Preference Optimization", Rafailov et al., **NeurIPS 2023** [arXiv:2305.18290](https://arxiv.org/abs/2305.18290); `train_sft.py` `dro`/`cispo`/`ppo` modes.

**Why expected to improve**:
On a verifiable-answer task, the verifier gives a perfect preference label, so contrastive optimization sharply separates correct from the model's *actual* near-miss wrong answers — addressing the exact errors SFT leaves on the table. SimPO's reference-free, length-normalized form avoids a second reference model (compute-friendly) and curbs length exploitation.

**Expected gain**: +0.2 / +0.8 / +2.2 pp 🟡 (high headroom, higher variance)
**Feasibility**: 3/5 🟡
**Effort**: L 🟡

**Implementation sketch**:
1. Sample K traces/problem with the current adapter; verifier-label; build chosen/rejected pairs (dedupe, balance categories).
2. Run SimPO/DPO via `train_sft.py`'s preference objective; tune `β`/margin on the slice; keep the SFT-on-correct anchor to prevent collapse.
3. Compare greedy held-out exact-match; cap to ≤ 2 rounds to limit drift.

**Risks**:
- Preference training can shorten/degrade traces or reward-hack length — mitigate with SimPO length-norm + an SFT anchor term.
- Needs enough problems with *both* correct and incorrect samples; very-easy or very-hard problems yield no usable pair.
- Generation compute for K samples × problems is the cost driver.

**Falsification test**: After one preference round, held-out greedy exact-match on previously-failed problems. If it does not rise by ≥ 1 pp (or overall exact-match drops), stop and keep SFT-only.

---

### Idea 9: Stream-of-Search backtracking traces

- **Pattern**: P2 (Transfer — search/planning "serialized search" → reasoning SFT data)
- **Tier**: 3
- **Target task**: Same as batch — improve the hardest search-like categories (e.g., `cryptarithm`, `cipher`) by teaching the model to explore, hit dead-ends, backtrack, and recover within a single linear trace.
- **Scope**: enhance-existing — adds a backtracking-style trace variant in `nemotron-master/reasoners/` / `augmentation.py` for search-structured categories; the scaffold/mask/trainer are unchanged. Final inference remains one greedy pass.
- **One-liner**: Serialize the solver's *search process* (proposals, failed branches, "backtrack", corrected branch) into the training trace so the model learns to self-correct mid-derivation instead of committing to a first wrong path.

**Mechanism**:
For search-structured categories, the deterministic solver's explored-and-pruned branches are rendered as a linear "stream of search" (try → check → fail → backtrack → try) ending in the verified `\boxed{answer}`. Training on traces that *contain* recoverable mistakes teaches the model to backtrack at greedy inference, which outcome-only traces never demonstrate.

**Distinct from batch-1**: opposite of idea-3 (concise traces) — this deliberately includes search/backtrack content for categories where exploration is the bottleneck; gated to those categories only.

**Source inspirations**:
- Primary: "Stream of Search (SoS): Learning to Search in Language", Gandhi et al., **COLM 2024** [arXiv:2404.03683](https://arxiv.org/abs/2404.03683) — training on serialized search (incl. backtracking) beats optimal-only trajectories by 25%.
- Supporting: "Step Back to Leap Forward: Self-Backtracking for Boosting Reasoning", 2025 [arXiv:2502.04404](https://arxiv.org/abs/2502.04404).

**Why expected to improve**:
SoS shows models trained on search-with-backtracking solve more problems than models trained on the optimal path only. For combinatorial categories (cryptarithm/cipher), the right answer often requires trying and rejecting candidates; teaching that behavior directly attacks the categories most likely to be unsolved.

**Expected gain**: +0.0 / +0.6 / +1.8 pp 🟡 (high variance; category-gated)
**Feasibility**: 3/5 🟡
**Effort**: L 🟡

**Adjacent / Cross-domain notes** (Tier 3):
- Original domain: heuristic search / planning (serialized search streams from solvers).
- Target domain: SFT trace content for combinatorial reasoning categories.
- Adaptation needed: render solver search trees as linear streams; cap branch count to fit the 7680 budget; gate to search-structured categories only.

**Implementation sketch**:
1. In the search-structured `reasoners/`, log explored/pruned branches; serialize a bounded stream-of-search trace.
2. Cap branches so traces stay well under 7680 tokens; compose with idea-3's length guard.
3. Slice-compare exact-match on the targeted categories vs concise-only traces.

**Risks**:
- Backtracking traces are long → truncation risk; strict branch cap required.
- Could teach the model to *generate* unnecessary failed branches on easy problems (wasting budget) — gate strictly by category/difficulty.

**Falsification test**: Train SoS-traces (cryptarithm/cipher only) vs concise traces on the slice. If exact-match on those categories doesn't rise by ≥ 1 pp OR the cap-hit rate rises > 3 pp, revert.

---

## Verification Report — Batch 2

| # | Title (short) | Novelty | Provenance | Feas | Gain (pp) | Falsif | Risk | Comply | Final |
|---|---------------|---------|------------|------|-----------|--------|------|--------|-------|
| 1 | DoRA weight-decomposed LoRA | EXTENDS ✅ | VERIFIED ✅ | 5/5 | +0.7 🟡 | OK ✅ | MED ⚠️ (vLLM load) | WARN | **KEEP (warn)** |
| 2 | NEFTune noisy-embedding reg | EXTENDS ✅ | VERIFIED ✅ | 5/5 | +0.5 🟡 | OK ✅ | MED ⚠️ (reasoning upside) | PASS | **KEEP (↓1 slot)** |
| 3 | Self-verify check-then-box traces | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.7 🟡 | OK ✅ | MED | PASS | **KEEP** |
| 4 | PiSSA / LoRA-GA init | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.6 🟡 | OK ✅ | MED (export→vanilla) | WARN | **KEEP** |
| 5 | Exact-arithmetic scratchpad | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.6 🟡 | OK ✅ | MED (length) | PASS | **KEEP** |
| 6 | Anchored-SFT KL regularization | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.5 🟡 | OK ✅ | MED (CCE-path conflict) | PASS | **KEEP** |
| 7 | LoRA seed-soup | EXTENDS ✅ | VERIFIED ✅ | 3/5 | +0.6 🟡 | OK ✅ | MED (N× compute) | PASS | **KEEP** |
| 8 | Preference opt on verified pairs | EXTENDS ✅ | VERIFIED ✅ | 3/5 | +0.8 🟡 | OK ✅ | MED (drift/compute) | PASS | **KEEP** |
| 9 | Stream-of-Search backtracking | EXTENDS ✅ | VERIFIED ✅ | 3/5 | +0.6 🟡 | OK ✅ | MED (length/variance) | PASS | **KEEP** |

## Counts
- Verified: 9
- Rejected: 0 (Novelty 0, Provenance 0, Falsification 0, Compliance 0, Other 0)
- Downgraded: 1 (Idea 2 NEFTune dropped one rank slot by devil's-advocate; gain held within range)
- Re-search cycles used: 0
- Final batch size: 9

## Warnings (per idea)
- **Idea 1 (DoRA)**: MED — vLLM may not load the DoRA magnitude vector; gate every run on a load-test, fall back to merge→vanilla-LoRA extract if needed.
- **Idea 2 (NEFTune)**: MED — devil's-advocate evidence shows gains concentrate on conversational quality; reasoning/exact-match upside uncertain. Downgraded rank 1→2. Free to slice-test, so kept as a cheap probe.
- **Idea 4 (PiSSA)**: MED — must export via PiSSA→residual conversion to a vanilla LoRA for vLLM; SVD over MoE expert weights adds one-time cost; only works with `RESET_WEIGHTS=True`.
- **Idea 6 (Anchored-SFT)**: MED — KL term needs base probabilities, conflicting with the no-logits `cut_cross_entropy` fast path; may need cached/top-k base logits.
- **Ideas 7/8/9**: MED — compute (N× runs / K samples) and, for 8/9, distribution-drift & trace-length risks; all gated behind a slice test before a full run.
- Recency/trust: primaries for ideas 3 (2511.04662) and 6 (2509.23753) are 2025 arXiv preprints (T2/T3 trust); ideas 1/4/8 are ICML/NeurIPS (T1), idea 7 is ICML (T1), idea 9 is COLM (T1/T2). Batch T1+T2 ≈ 78% ≥ 60% ✅.

## Cross-idea consistency
- **Near-duplicates**: none. Ideas 1/4 both touch the adapter but on orthogonal axes (parameterization vs initialization); 2/6 both regularize but by different mechanisms (input noise vs output KL).
- **Contradictions flagged (composable, not blocking)**: Idea-3 (batch-1, concise) vs Idea-5 (scratchpad, longer) and Idea-9 (backtracking, longer) pull opposite on trace length — resolved by category-gating the lengthening ideas and keeping the length cap elsewhere. Idea-2 (batch-1, up-weight) vs Idea-6 (anchor/pull-back) are opposing forces meant to be co-tuned. Surfaced for user awareness; not a defect.
- **Score-distribution**: healthy spread (🟢 22% / 🟡 67% / 🔴 11%); no all-5/all-🟢 over-confidence.

## Notes & warnings
- **Single greedy-pass honored**: every idea is training-time or data-time. No idea adds self-consistency, sampling, beam, or inference-time verifiers; submission stays one greedy vLLM pass, rank ≤ 32. (Ideas 8/9 use sampling **only for offline data generation**, never at submission.)
- **vLLM format is the recurring gate**: DoRA (idea 1) and PiSSA (idea 4) need an explicit load-test / conversion step before they can be trusted as deliverables. The other seven produce a standard vanilla rank-32 adapter.
- **Composition map** (with batch-1): idea-1/idea-4 compose with batch-1 idea-4 (rsLoRA) and idea-5 (module realloc) — all are independent LoRA-config axes. idea-5/idea-9 must be co-tuned with batch-1 idea-3 (concise traces) on the length budget. idea-6 is the principled version of batch-1 idea-2's "keep λ small" guardrail.
- **Prerequisite measurement (carried from batch-1)**: bucket held-out losses into {format-zero, truncation-zero, arithmetic-slip, method-wrong, search/explore}. This tells you whether ideas 3/5 (arithmetic/verify), idea-9 (search), or ideas 1/4/7 (capacity) are the right lever for the missing 0.02.
- **Tier mix**: observed 56/33/11 vs configured 55/30/15 — within ±10pp on every tier ✅.

## Next steps for user
1. **Bank the cheap, independent LoRA-capacity wins first**: ship **Idea 1 (DoRA) + Idea 4 (PiSSA init)** together (both gate on a vLLM load-test) and slice-test **Idea 2 (NEFTune)** as a near-free probe. These are the lowest-risk path toward 0.87.
2. **Then attack the dominant loss bucket** the prerequisite measurement reveals: Idea 5 (arithmetic) / Idea 3 (self-verify) for numeric slips, or Idea 9 (Stream-of-Search) for combinatorial categories.
3. **Hold Idea 7 (soup), Idea 8 (preference), Idea 6 (anchored-KL)** for the push from 0.87→0.88+: highest headroom but each carries compute or implementation-conflict risk — adopt only after the cheap wins are banked and measured.

## Provenance signature
SHA256 of (inputs + paper IDs + timestamp): 985a259bdfd33b89ff713a7176722bd0e7368f076b3dd9f21196c0c87744ec2c
