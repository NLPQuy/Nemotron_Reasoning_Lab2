# Idea Batch 1 — NVIDIA Nemotron Model Reasoning Challenge / LoRA-adapter reasoning SFT
**Generated**: 2026-06-01T10:15:00Z
**Time-to-batch**: ~12 min
**Skill version**: 0.1.0
**Skill invocation**: `/benchmark-climb-ideation` (10 training ideas, enhance Continuer_Nemotron_Notebook.py, target +0.01)

## Inputs
- Benchmark: NVIDIA Nemotron Model Reasoning Challenge (Kaggle public leaderboard)
- Task / problem: Fine-tune ONE rank-32 LoRA adapter for `Nemotron-3-Nano-30B-A3B` (Mamba/MoE hybrid) so that, under greedy vLLM inference (`max_tokens=7680`, `max_model_len=8192`), it solves reasoning problems and emits the final answer inside `\boxed{...}`. Grader: string match OR relative error ≤ 1e-2; binary strings matched **exactly**.
- Existing pipeline: `Continuer_Nemotron_Notebook.py` — Unsloth single-file trainer. Monkey-patched forward → `cut_cross_entropy.linear_cross_entropy`; per-token CE × corpus `weights` mask applied in the loop. Manual `lm_head` LoRA (renamed `backbone.lm_head` on save). LoRA fp32, base bf16 except MoE router `mixer.gate` fp32. `MOE_TIE_WEIGHTS` keeps all 128 expert LoRA slices identical (mean-init + grad-sum across expert dim). Mamba CUDA fast path forced on. Corpus is pre-tokenized; completion = `"{reasoning}\n</think>\n\\boxed{{answer}}<|im_end|>"`, prompt masked out (mask 0), completion mask 1. **Current score: 0.86.**
- Batch scope: **mixed — ≥ 50% enhance-existing** (8 of 10 enhance-existing; 2 greenfield-leaning scheduling transfers that still plug into the existing training loop's sampler)
- Tier mix (configured): `45/35/20` (default; pipeline supplied but user did not override — kept default to preserve cross-domain diversity)
- Baseline: `Nemotron-3-Nano-30B-A3B` + current rank-32 LoRA @ **0.86**
- Compute budget: single RTX-PRO-6000 (Modal) or Kaggle GPU per run; assume each full train+submit cycle is the expensive unit, so ideas are validated on a small held-out problem slice first.
- Time budget: implied — favor changes validatable in < 1 short training run.
- Constraints: `max_lora_rank=32`; greedy `temperature=0.0`; vLLM-loadable adapter; deliverable `submission.zip` with `adapter_config.json`. No inference-time tricks (single greedy pass), so **every idea is a training-time / data-time change.**

## Summary
| Metric | Value |
|--------|-------|
| Batch size | 10 |
| Tier 1 / 2 / 3 (counts) | 4 / 4 / 2 |
| Tier mix vs configured | 40/40/20 vs 45/35/20 (deviation ≤ 10pp per tier ✅) |
| Scope mix | 8 enhance-existing / 2 greenfield-leaning (≥ 50% enhance ✅) |
| Patterns used | P6, P3×2, P4×2, P8×2, P12, P2×2 (6 distinct) |
| Distinct venues | 5+ (NeurIPS, EMNLP, COLM, Science/AAAS, ICML + arXiv preprints) |
| Time windows | <12mo (3), 12-36mo (4), 36-72mo (1), 72+mo (2) |
| Avg feasibility | 4.1/5 |
| Avg confidence | 🟢 30%, 🟡 60%, 🔴 10% |

## Summary table
| # | Title | Pattern | Tier | Gain (pp) | Feas | Effort | Score |
|---|-------|---------|------|-----------|------|--------|-------|
| 1 | Format-verified clean labels + truncation-robust completions | P6 | 2 | +0.7 | 5 | S | 3.9 |
| 2 | Up-weight `\boxed{}`/critical answer tokens in the loss | P3 | 1 | +0.8 | 5 | S | 3.8 |
| 3 | Difficulty-aware concise traces (anti-truncation) | P3 | 2 | +0.9 | 4 | M | 3.6 |
| 4 | rsLoRA √r scaling for the rank-32 budget | P4 | 1 | +0.6 | 5 | S | 3.6 |
| 5 | Reasoning-critical target-module & rank reallocation | P8 | 1 | +0.8 | 4 | M | 3.4 |
| 6 | LIMO/s1 difficulty+diversity corpus curation | P4 | 1 | +0.6 | 4 | M | 3.2 |
| 7 | STaR/RFT self-generated verified-correct traces | P12 | 2 | +1.0 | 3 | L | 3.1 |
| 8 | Hot-expert untying of `MOE_TIE_WEIGHTS` | P8 | 2 | +0.5 | 3 | M | 2.7 |
| 9 | Spaced-repetition (forgetting-curve) data scheduling | P2 | 3 | +0.4 | 4 | M | 2.6 |
| 10 | Simulated-annealing difficulty curriculum | P2 | 3 | +0.4 | 4 | M | 2.5 |

## Top-3 recommendations

### 🏆 Top-1 by composite score
**Idea 1: Format-verified clean labels + truncation-robust completions** — Score: 3.9
The cheapest reliable +0.01: every training completion is verified to end in the exact `\n</think>\n\\boxed{answer}<|im_end|>` scaffold and every `answer` is verified to round-trip through the grader's `compare_answer`. This eliminates the long tail of *hard zeros* caused by malformed/missing boxed spans — a pure-upside data hygiene change with no modeling risk. (Promoted to #1 after the devil's-advocate pass downgraded the token-weighting idea by one slot.)

### ⚡ Quick win (lowest effort)
**Idea 4: rsLoRA √r scaling** — Effort: S
A one-line config change (`use_rslora=True`). At your fixed max rank (32), vanilla LoRA's α/r scaling under-drives gradients; rsLoRA's α/√r is specifically the regime where higher ranks start paying off. Zero data work, vLLM-neutral.

### 🛡️ Safe bet (highest confidence)
**Idea 2: Up-weight `\boxed{}`/critical answer tokens in the loss** — Confidence: 🟢
Reuses the existing per-token `weights` codepath — you already multiply per-token CE by the mask. Mildly raising the weight on the answer span tokens directly aligns the loss with the exact-match grader. Guardrailed (small λ, no token dropped) per the contrasting evidence below.

## Ranked ideas

### Idea 1: Format-verified clean labels + truncation-robust completions

- **Pattern**: P6 (Verify — add a verifier/filter over the training labels)
- **Tier**: 2
- **Target task**: Same as batch — emit a grader-parseable `\boxed{...}` under greedy decoding; here we attack the subset of problems lost to *format/parse* failures rather than reasoning errors.
- **Scope**: enhance-existing — modifies `corpus.py` (completion construction) and adds a validation gate before a row enters `corpus.jsonl`. The tokenizer, mask scheme, monkey-patched forward, and LoRA config stay unchanged.
- **One-liner**: Guarantee every training label is byte-exact in the canonical boxed scaffold and grader-verifiable, so the model never learns a malformed/missing-box pattern.

**Mechanism**:
For each corpus row, after building `"{reasoning}\n</think>\n\\boxed{{answer}}<|im_end|>"`, run the grader's own extractor (`compare_answer` / boxed-regex from `nemotron-master/reasoning.py`) on the completion string; drop or repair any row where (a) the box doesn't parse, (b) `answer` round-trips to a different normalized value, or (c) a stray earlier `\boxed{}` appears in `reasoning`. Additionally synthesize a small fraction (~3–5%) of *truncation-robust* variants where `reasoning` is hard-truncated mid-sentence but the completion still appends the exact `\n</think>\n\\boxed{answer}<|im_end|>` tail — teaching the model that the boxed emission is unconditional even when the budget is nearly spent.

**Source inspirations**:
- Primary: "Decoupling Task-Solving and Output Formatting in LLM Generation", 2025 [arXiv:2510.03595] — format failures are mostly structural and separable from task competence; explicitly handling format raises reliability.
- Supporting: "LLMs Are Biased Towards Output Formats!", 2024 [arXiv:2408.08656]; competition grader `compare_answer` in `nemotron-master/reasoning.py`.

**Why expected to improve**:
On an exact-match/boxed grader, a single missing or malformed box is a full zero regardless of reasoning quality. 2510.03595 shows format compliance is a separable failure mode; cleaning labels and adding scaffold-invariant examples removes that failure mode at the data source. Because the change only *tightens* label quality, it cannot teach a worse format.

**Expected gain**: +0.2 / +0.7 / +2.0 pp 🟡 (upside scales with however many of your current losses are format-zeros, not reasoning-zeros)
**Feasibility**: 5/5 🟢
**Effort**: S 🟢

**Implementation sketch**:
1. In `corpus.py`, import the grader's boxed extractor + `compare_answer`; assert each built completion parses and round-trips; log+drop violations.
2. Add a `truncation_robust_frac` knob that emits scaffold-tail-preserving truncated variants.
3. Re-tokenize; confirm the `</think>\n\boxed{` scaffold tokens are inside the mask=1 (completion) region.

**Risks**:
- If essentially all current losses are *reasoning* errors (not format), gain ≈ 0 — but still no downside.
- Over-aggressive truncation variants could bias toward premature `</think>`; cap the fraction ≤ 5%.

**Falsification test**: On a 200-problem held-out slice, log the count of outputs with an unparseable/missing `\boxed{}`. If the baseline adapter's format-zero count is < 3 (i.e., near-zero format failures already), this idea cannot deliver and is abandoned.

---

### Idea 2: Up-weight `\boxed{}` / critical answer tokens in the loss

- **Pattern**: P3 (Replace — swap the uniform completion mask for a real-valued answer-aware weighting)
- **Tier**: 1
- **Target task**: Same as batch — exact-match boxed answer; bias optimization toward the tokens that actually decide grading.
- **Scope**: enhance-existing — changes only the values written into the corpus per-token `weights` (the loop already multiplies per-token CE by `weights`). No change to forward, LoRA, or tokenizer.
- **One-liner**: Give the answer-bearing span (`\boxed{ ... }` and the `</think>\n\boxed{` scaffold) a mildly higher loss weight than free-form reasoning tokens, aligning the objective with the exact-match grader.

**Mechanism**:
During corpus building, after computing the 0/1 completion mask, multiply the weights of tokens inside `\boxed{...}` (and the literal `\boxed{` opener + `</think>` marker) by a factor λ∈[1.5, 3.0], leaving all other completion tokens at 1.0 (never below 1.0, never dropped). The training loop's existing `per_token_ce * weights` then emphasizes getting the final answer characters exactly right — which is what the binary-exact / 1e-2 grader rewards — without suppressing reasoning tokens.

**Source inspirations**:
- Primary: "Enhancing Large Language Model Reasoning via Selective Critical Token Fine-Tuning", 2025 [arXiv:2510.10974] — a small fraction of *critical* tokens drives correctness; weighting them can beat uniform SFT.
- Supporting: "Dynamic Fine-Tuning (DFT): probability-reweighted SFT loss", 2025 [arXiv:2508.05629] (+15.66 on math benchmarks); "Instruction Fine-Tuning: Does Prompt Loss Matter?", EMNLP 2024 [arXiv:2401.13586] (PLW = real-valued generalization of masking).
- Contrasting: "SFT Doesn't Always Hurt… / Anchored SFT", 2025 [arXiv:2509.20758] — *unconstrained* reweighting causes distributional drift; keep λ small and bounded.

**Why expected to improve**:
The grader scores only the boxed value; reasoning tokens are unscored at inference. 2510.10974 and 2508.05629 show that concentrating loss on outcome-critical tokens improves math reasoning over uniform SFT. Mildly up-weighting the answer span pushes capacity toward the exactly-scored characters while the λ cap and "never drop a token" rule avoid the drift failure mode in 2509.20758.

**Expected gain**: +0.3 / +0.8 / +1.5 pp 🟡
**Feasibility**: 5/5 🟢
**Effort**: S 🟢

**Implementation sketch**:
1. In `corpus.py`, locate answer-span token indices (search for the `\boxed{`…`}` token ids within the completion).
2. Add `ANSWER_TOKEN_WEIGHT = λ` knob; multiply those weights, clamp ≥ 1.0.
3. Sweep λ ∈ {1.0(=baseline), 1.5, 2.0, 3.0} on the small slice; pick by held-out exact-match, not loss.

**Risks**:
- Too-large λ over-smooths reasoning-token gradients / encourages answer shortcutting (see contrasting paper + non-monotonic-CoT evidence [arXiv:2505.17827]).
- For relative-error numeric answers, character-level weighting helps less than for binary-exact answers.

**Falsification test**: Train with λ=2.0 vs λ=1.0 on the slice. If held-out exact-match does not improve by ≥ 0.5 pp at any tested λ, reject.

---

### Idea 3: Difficulty-aware concise reasoning traces (anti-truncation)

- **Pattern**: P3 (Replace — substitute long traces with length-controlled traces)
- **Tier**: 2
- **Target task**: Same as batch — but specifically rescue problems whose correct boxed answer currently falls *after* the 7680-token budget and is never emitted.
- **Scope**: enhance-existing — modifies trace generation in `reasoning.py` / `augmentation.py` (the `{reasoning}` field) and a length cap in `corpus.py`. Mask scheme and training loop unchanged.
- **One-liner**: Shorten/compress training reasoning traces (proportional to problem difficulty) so the `\boxed{}` answer reliably fits inside the 7680-token greedy budget.

**Mechanism**:
Measure the token length distribution of completions in the corpus; for traces exceeding a target budget (e.g., 4–5k tokens, leaving headroom under 7680), regenerate or compress the reasoning to a concise-but-complete form (drop restatements, keep the verified arithmetic), keeping length roughly proportional to difficulty. Train on the shortened traces so the model's learned generation length stays under budget and always reaches the boxed tail.

**Source inspirations**:
- Primary: "Concise Reasoning, Big Gains: Pruning Long Reasoning Trace with Difficulty-Aware Prompting", 2025 [arXiv:2505.19716] — shorter difficulty-aware CoT matches or beats long chains across 11 benchmarks.
- Supporting: "Less is More Tokens: Efficient Math Reasoning via Difficulty-Aware CoT Distillation", 2025 [arXiv:2509.05226]; "Don't Overthink it: Preferring Shorter Thinking Chains", 2025 [arXiv:2505.17813].

**Why expected to improve**:
Under greedy decoding with a hard 7680 cap, an over-long trace truncates before `\boxed{}` → guaranteed zero. 2505.19716/2509.05226 show concise traces preserve accuracy while cutting length 35–57%. Shortening the *training* distribution shifts the model's generation length below budget, converting truncation-zeros into scored answers.

**Expected gain**: +0.3 / +0.9 / +2.5 pp 🟡 (upside = fraction of current losses that are truncation, not logic)
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. Instrument `corpus.py` to histogram completion token lengths; flag rows > target budget.
2. Regenerate/compress flagged traces (the deterministic solvers in `reasoners/` already produce exact arithmetic — just trim narration).
3. Validate that compressed traces still pass the grader round-trip (composes with Idea 1).

**Risks**:
- Over-compression can remove steps the model needs to *reach* the right answer → reasoning accuracy drops.
- Difficulty estimate may be noisy for some categories.

**Falsification test**: On the slice, measure (a) fraction of generations hitting the 7680 cap and (b) exact-match. If baseline cap-hit rate < 2% AND exact-match doesn't improve, reject (no truncation problem to fix).

---

### Idea 4: rsLoRA √r scaling for the rank-32 budget

- **Pattern**: P4 (Scale — change the rank-scaling dimension of the adapter)
- **Tier**: 1
- **Target task**: Same as batch — extract more learning capacity from the fixed, maxed-out rank-32 budget.
- **Scope**: enhance-existing — flips the LoRA scaling factor in the adapter config; all target modules, the manual `lm_head` LoRA, fp32 casting, and the training loop stay identical.
- **One-liner**: Replace vanilla α/r LoRA scaling with rank-stabilized α/√r so gradients don't collapse at the high (32) rank you're forced to use.

**Mechanism**:
Set `use_rslora=True` (Unsloth/PEFT `LoraConfig`). This changes the adapter forward scaling from α/r to α/√r. Because you are pinned at the maximum allowed rank (32) — exactly where vanilla LoRA's gradient magnitude collapses — the √r scaling restores effective gradient signal so the rank-32 adapter actually realizes its capacity.

**Source inspirations**:
- Primary: "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA (rsLoRA)", 2023 [arXiv:2312.03732] — α/r stunts higher ranks; α/√r unlocks monotone gains with rank.
- Supporting: "Learning Rate Scaling across LoRA Ranks", 2026 [arXiv:2602.06204] (LR/rank/scaling interaction).

**Why expected to improve**:
The competition forces max rank 32; rsLoRA's whole thesis is that vanilla scaling makes high ranks underperform their potential, and √r fixes it precisely in the high-rank regime. Since you can't increase rank, getting full value from rank 32 is the lever rsLoRA targets.

**Expected gain**: +0.2 / +0.6 / +1.2 pp 🟡
**Feasibility**: 5/5 🟢
**Effort**: S 🟢

**Implementation sketch**:
1. Set `use_rslora=True` in the `LoraConfig`; mirror the √r scaling for the manually-added `lm_head` LoRA so it's consistent.
2. Re-tune base LR slightly (rsLoRA changes effective step size); a short LR check on the slice.
3. Confirm the saved adapter still vLLM-loads (rsLoRA writes a flag in `adapter_config.json`, supported by vLLM).

**Risks**:
- Effective LR shifts → may need a small LR re-tune to see the gain (or it can look flat/worse if LR unchanged).
- Manual `lm_head` LoRA must use the same scaling or it becomes mismatched.

**Falsification test**: Train rsLoRA vs baseline at matched (re-tuned) LR on the slice. If held-out exact-match is not ≥ baseline within noise after the LR check, reject.

---

### Idea 5: Reasoning-critical target-module & rank reallocation

- **Pattern**: P8 (Specialize — route the limited rank budget to reasoning-critical modules)
- **Tier**: 1
- **Target task**: Same as batch — spend the fixed parameter budget where reasoning capacity actually lives in this Mamba/MoE hybrid.
- **Scope**: enhance-existing — changes `target_modules` (and which modules get the rank-32 budget) in the LoRA config; keeps `MOE_TIE_WEIGHTS`, the manual `lm_head` LoRA, and the loop unchanged.
- **One-liner**: Ensure the rank-32 adapters land on the MLP/expert and `o_proj`-equivalent + Mamba `mixer` projections (where reasoning is shown to concentrate) rather than being spread thinly across low-value modules.

**Mechanism**:
Audit the current `target_modules` list against the Nemotron-H module names; ensure LoRA covers the expert FFN/`gate_up`/`down` projections and the Mamba `mixer.in_proj`/`out_proj`, since reasoning capability is empirically concentrated in MLP/`o_proj`-type layers and lives in a low-rank subspace. Drop or down-prioritize modules that add little (pure attention-only adapters add no benefit over MLP), keeping the total within the rank-32 ceiling.

**Source inspirations**:
- Primary: "Planning vs Reasoning: Ablations to Test Capabilities of LoRA layers", 2024 [arXiv:2412.00029] — reasoning lives in low-rank subspaces; MLP layers dominate, 2–3× lower rank suffices for reasoning.
- Supporting: "LoRALib: A Standardized Benchmark for Evaluating LoRA-MoE Methods", 2025 [arXiv:2509.18137] (expert-targeting in MoE); "LoRA Without Regret", Thinking Machines Lab 2025 [blog — concept reference only, T4] (all-linear/MLP ≈ full-FT).

**Why expected to improve**:
2412.00029 shows reasoning ability is captured in low-rank MLP-side updates; with only rank 32 to spend, concentrating it on those modules (and the Mamba mixer that carries sequence mixing) should out-perform an even spread that wastes budget on attention-only slices.

**Expected gain**: +0.3 / +0.8 / +1.8 pp 🟡
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. Print the model's named modules; map Nemotron-H expert FFN + `mixer` projections.
2. Define two `target_modules` variants (current vs MLP/expert/mixer-focused); keep rank 32.
3. Compare on the slice; verify vLLM still loads the adapter for the new module set.

**Risks**:
- Mamba `mixer` LoRA may interact badly with the forced CUDA fast path — test load + a few greedy generations.
- Some Nemotron-H module names are non-standard; a wrong target string silently no-ops.

**Falsification test**: Train MLP/expert/mixer-focused vs current target set on the slice. If held-out exact-match doesn't improve by ≥ 0.5 pp, revert to the current module set.

---

### Idea 6: LIMO/s1 difficulty + diversity corpus curation

- **Pattern**: P4 (Scale — scale the corpus *down* to a higher-quality, harder, more diverse subset)
- **Tier**: 1
- **Target task**: Same as batch — improve generalization per training step by curating which verified traces are trained on.
- **Scope**: enhance-existing — a selection filter over `corpus.jsonl` (which rows survive), plus the matching `NUM_STEPS`. Trace format, masking, and model config unchanged.
- **One-liner**: Down-select the corpus to fewer, harder, category-diverse verified traces (LIMO/s1 "less is more") instead of training on many near-trivial repeats.

**Mechanism**:
Score each verified trace by (a) difficulty (solver steps / answer magnitude / category rarity) and (b) diversity (per-category and per-knowledge-point caps). Remove trivially-easy, near-duplicate problems that dominate step count; keep a balanced, difficulty-skewed subset. Train the same rank-32 adapter on this curated set so each gradient step carries more reasoning signal.

**Source inspirations**:
- Primary: "LIMO: Less is More for Reasoning", COLM 2025 [arXiv:2502.03387] — 800 curated samples beat far larger SFT sets.
- Supporting: "s1: Simple test-time scaling", EMNLP 2025 [arXiv:2501.19393] (1k curated, diversity+difficulty+quality selection).

**Why expected to improve**:
LIMO/s1 show curated difficulty+diversity beats raw volume for reasoning SFT. Your corpus is synthetic and category-skewed; trimming easy/duplicated rows and balancing categories raises the average informativeness per step within the same `NUM_STEPS`.

**Expected gain**: +0.2 / +0.6 / +1.5 pp 🟡
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Implementation sketch**:
1. Add per-row difficulty + category features in `corpus.py`.
2. Apply coarse easy-problem filtering + per-category caps (LIMO recipe); record kept/dropped counts.
3. Retrain with adjusted `NUM_STEPS`; compare held-out exact-match per category to catch coverage loss.

**Risks**:
- Over-pruning a category the leaderboard actually tests → coverage regression (the leaderboard mix is unknown).
- Synthetic "difficulty" proxy may misrank.

**Falsification test**: Train on curated vs full corpus, equal steps. If macro-average held-out exact-match across categories does not improve (and no category drops > 1 pp), reject.

---

### Idea 7: STaR/RFT self-generated verified-correct traces

- **Pattern**: P12 (Self-play / self-improve)
- **Tier**: 2
- **Target task**: Same as batch — expand coverage and reasoning-path diversity on problems the current adapter gets wrong.
- **Scope**: enhance-existing (justified) — adds a self-generation stage *upstream* of the existing corpus, reusing `reasoners/`' verifiers and the `corpus.jsonl` format; the trainer itself is untouched. It is not greenfield because it augments the existing data pipeline rather than replacing it.
- **One-liner**: Use the current adapter to sample multiple traces per problem, keep only those whose `\boxed{}` answer is verified correct, and add the deduped correct traces to the corpus before retraining.

**Mechanism**:
For problems where the current adapter fails (or for held-out categories), sample K traces (offline, can be temperature>0 *for data generation only* — inference at submission stays greedy), filter to traces whose boxed answer passes the deterministic verifier in `reasoners/`, dedupe, and append to `corpus.jsonl`. Retrain the rank-32 adapter on the union. This is classic STaR/RFT bootstrapping restricted to verifiable answers.

**Source inspirations**:
- Primary: "STaR: Bootstrapping Reasoning With Reasoning", Zelikman et al., NeurIPS 2022 [arXiv:2203.14465].
- Supporting: "AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners", 2025 [arXiv:2505.16322]; RFT augments human data with model-generated correct chains.

**Why expected to improve**:
STaR/RFT reliably lift reasoning accuracy by adding self-generated *verified-correct* and diverse chains, improving coverage on hard problems without new human data. Your verifiers make the "keep only correct" filter exact, mitigating the usual reward-hacking risk.

**Expected gain**: +0.3 / +1.0 / +3.0 pp 🟡 (largest headroom of the batch, but higher variance)
**Feasibility**: 3/5 🟡
**Effort**: L 🟡

**Implementation sketch**:
1. Batch-generate K traces/problem with the current adapter (offline, data-gen sampling allowed).
2. Filter via `reasoners/` verifiers + grader round-trip (composes with Idea 1); dedupe.
3. Append, retrain, compare held-out exact-match; iterate ≤ 2 rounds to limit drift.

**Risks**:
- Distribution drift / mode collapse over rounds (cap rounds; keep original corpus anchored).
- Self-generated traces may be verbose → compose with Idea 3's length cap.
- Generation compute for K samples × many problems is the cost driver.

**Falsification test**: After one STaR round, held-out exact-match on previously-failed problems. If it does not rise by ≥ 1 pp (or overall drops), stop the loop.

---

### Idea 8: Hot-expert untying of `MOE_TIE_WEIGHTS`

- **Pattern**: P8 (Specialize — give capacity to the experts that actually carry tokens)
- **Tier**: 2
- **Target task**: Same as batch — let reasoning-relevant experts specialize instead of forcing all 128 expert LoRA slices identical.
- **Scope**: enhance-existing — relaxes the `MOE_TIE_WEIGHTS` grad-sum/mean-init logic for a subset of experts; everything else (rank 32, router fp32, loop) unchanged.
- **One-liner**: Keep cold experts tied (regularized, data-efficient) but untie the top-k "hot" experts so they can specialize, recovering MoE capacity the full-tie currently suppresses.

**Mechanism**:
Profile expert routing frequency on the corpus; identify the small set of hot experts that handle most tokens. Replace the global grad-sum-across-all-128 with a *grouped* scheme: untie (independent LoRA) for the hot experts, keep the cold-expert tail tied (mean-init + grad-sum) for regularization. This concentrates the trainable diversity where tokens actually flow.

**Source inspirations**:
- Primary: "DR-LoRA: Dynamic Rank LoRA for Fine-Tuning Mixture-of-Experts Models", 2026 [arXiv:2601.04823] — heterogeneous per-expert budget; concentrate capacity on task-critical experts. *(T3-trust: recent preprint.)*
- Supporting: "LoRALib: A Standardized Benchmark for Evaluating LoRA-MoE Methods", 2025 [arXiv:2509.18137] (routing concentrated; per-expert adaptation matters).

**Why expected to improve**:
Full tying maximizes data efficiency but caps specialization; MoE-LoRA work shows routing is concentrated and hot experts deserve dedicated capacity. Untying only the hot tail buys specialization where it counts while the tied cold tail preserves the regularization that helps at your data scale.

**Expected gain**: +0.0 / +0.5 / +1.5 pp 🟡 (could be ≤ 0 if your corpus is small — flagged)
**Feasibility**: 3/5 🟡
**Effort**: M 🟡

**Implementation sketch**:
1. Log per-expert routing counts during a forward pass over the corpus; pick top-k (e.g., k=8).
2. In the grad-tying code, exclude top-k experts from the grad-sum (independent slices); keep the rest tied.
3. Watch total trainable param count stays within budget; verify vLLM load.

**Risks**:
- Untying raises effective adapter params for hot experts → may overfit a small corpus (the very reason tying was chosen).
- Routing profile may be unstable across problems.

**Falsification test**: Train grouped-untie vs full-tie on the slice. If held-out exact-match does not improve by ≥ 0.5 pp, keep full tie (the current default).

---

### Idea 9: Spaced-repetition (forgetting-curve) data scheduling

- **Pattern**: P2 (Transfer — cognitive psychology → SFT data ordering)
- **Tier**: 3
- **Target task**: Same as batch — improve retention of harder categories within the fixed `NUM_STEPS`, no data change, only ordering.
- **Scope**: enhance-existing — replaces the uniform shuffle in the training loop's sampler with a spaced-review schedule; corpus content, masking, and model untouched.
- **One-liner**: Re-present problems the model still gets wrong at expanding intervals (spaced repetition) instead of i.i.d. shuffling, so limited steps buy more durable mastery of hard categories.

**Mechanism**:
Periodically (every N steps) evaluate a small probe set; for items/categories answered wrong, schedule them to reappear soon and at growing intervals as they become correct (Ebbinghaus spacing). Easy, consistently-correct items are spaced out. This concentrates gradient steps on not-yet-learned reasoning patterns within the same budget.

**Source inspirations**:
- Primary: "Repeat before Forgetting: Spaced Repetition for Efficient and Effective Training of Neural Networks", Amiri et al., EMNLP 2017 [ACL D17-1255] — uses 34–50% of data per epoch, 2.9–4.8× faster, no accuracy loss.
- Supporting: Ebbinghaus forgetting curve / spacing effect (cognitive psychology, foundational concept).

**Why expected to improve**:
Spacing-effect scheduling lets a fixed step budget allocate more reviews to material not yet retained, flattening the forgetting curve. Transferred to SFT, harder reasoning categories get re-exposed exactly when the model is about to "forget" them, improving end-of-training mastery vs uniform shuffle.

**Expected gain**: +0.0 / +0.4 / +1.0 pp 🟡 (scheduling effects under LoRA can be small — honest)
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Adjacent / Cross-domain notes** (Tier 3):
- Original domain: cognitive psychology (human memory / spacing effect, Ebbinghaus).
- Target domain: SFT data sampling schedule for a reasoning LoRA.
- Adaptation needed: define a cheap "recall probe" per category; map review intervals to training steps; integrate with the existing batch sampler.

**Implementation sketch**:
1. Build a tiny per-category probe (re-uses held-out slice).
2. Implement an interval scheduler in the sampler (wrong → short interval; correct → ×2 interval).
3. Compare to uniform shuffle at equal `NUM_STEPS`.

**Risks**:
- Probe evaluation adds overhead each interval.
- Under LoRA with few steps, ordering may not matter → flat result.

**Falsification test**: Spaced vs shuffled at identical steps/seed. If held-out exact-match is within ±0.3 pp, conclude no scheduling effect and drop.

---

### Idea 10: Simulated-annealing difficulty curriculum

- **Pattern**: P2 (Transfer — statistical physics / simulated annealing → difficulty sampling schedule)
- **Tier**: 3
- **Target task**: Same as batch — order problems easy→hard with an annealing schedule to improve optimization within fixed steps.
- **Scope**: enhance-existing — modifies the sampler's difficulty-sampling temperature over training steps; corpus and model unchanged. (Composes with, but is distinct from, Idea 9: this orders by *difficulty*, Idea 9 by *forgetting*.)
- **One-liner**: Anneal the sampling distribution from easy to hard over training — a high "temperature" early (broad, easy-weighted) cooling to hard problems late — mirroring simulated annealing's coarse-to-fine optimization.

**Mechanism**:
Assign each problem a difficulty score; define a sampling temperature T(step) that starts high (samples broadly, easy-skewed smoothed objective) and decays so late training concentrates on hard problems. This is the simulated-annealing / continuation-method view of curriculum: optimize a smoothed objective first, then sharpen — yielding a better basin before tackling hard cases.

**Source inspirations**:
- Primary: "Optimization by Simulated Annealing", Kirkpatrick, Gelatt & Vecchi, **Science** 220:671-680, 1983 [doi:10.1126/science.220.4598.671] — coarse-to-fine via temperature annealing escapes poor local optima.
- Supporting: "Curriculum Learning", Bengio et al., ICML 2009 [dl.acm.org/10.1145/1553374.1553380] — explicitly frames curriculum as a continuation/annealing method (easy→hard improves convergence + generalization).

**Why expected to improve**:
Bengio et al. show curriculum = continuation/annealing improves both convergence and generalization, with the annealing principle drawn directly from Kirkpatrick's statistical-physics method. Annealing difficulty within your fixed `NUM_STEPS` can find a better solution basin than uniform sampling, especially for the hardest categories that currently fail.

**Expected gain**: +0.0 / +0.4 / +1.0 pp 🟡
**Feasibility**: 4/5 🟢
**Effort**: M 🟡

**Adjacent / Cross-domain notes** (Tier 3):
- Original domain: statistical physics / combinatorial optimization (simulated annealing).
- Target domain: difficulty-sampling temperature schedule for reasoning SFT.
- Adaptation needed: a difficulty score per category; a T(step) annealing schedule; a sampler that draws with probability ∝ softmax(−difficulty/T).

**Implementation sketch**:
1. Reuse the difficulty score from Idea 6.
2. Add a `difficulty_temperature_schedule` to the sampler (cosine/linear decay).
3. Compare to uniform sampling at equal steps; also test the easy→hard vs hard→easy direction as a sanity control.

**Risks**:
- Wrong direction (hard-first) or too-fast cooling can hurt — include the reversed-schedule control.
- Difficulty proxy noise (shared with Idea 6).

**Falsification test**: Annealed easy→hard vs uniform at equal steps. If held-out exact-match does not improve by ≥ 0.3 pp AND the reversed schedule isn't worse (no signal of a real curriculum effect), drop.

---

## Verification Report — Batch 1

| # | Title (short) | Novelty | Provenance | Feas | Gain (pp) | Falsif | Risk | Comply | Final |
|---|---------------|---------|------------|------|-----------|--------|------|--------|-------|
| 1 | Format-verified clean labels | EXTENDS ✅ | VERIFIED ✅ | 5/5 | +0.7 🟡 | OK ✅ | LOW | PASS | **KEEP** |
| 2 | Up-weight boxed/critical tokens | EXTENDS ✅ | VERIFIED ✅ | 5/5 | +0.8 🟡 | OK ✅ | MED ⚠️ | PASS | **KEEP (warn)** |
| 3 | Concise anti-truncation traces | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.9 🟡 | OK ✅ | MED | PASS | **KEEP** |
| 4 | rsLoRA √r scaling | EXTENDS ✅ | VERIFIED ✅ | 5/5 | +0.6 🟡 | OK ✅ | LOW | PASS | **KEEP** |
| 5 | Target-module/rank realloc | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.8 🟡 | OK ✅ | MED | PASS | **KEEP** |
| 6 | LIMO/s1 curation | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.6 🟡 | OK ✅ | MED | PASS | **KEEP** |
| 7 | STaR/RFT self-gen traces | EXTENDS ✅ | VERIFIED ✅ | 3/5 | +1.0 🟡 | OK ✅ | MED | PASS | **KEEP** |
| 8 | Hot-expert untying | EXTENDS ✅ | VERIFIED ✅ | 3/5 | +0.5 🟡 | OK ✅ | MED ⚠️ | PASS | **KEEP (warn)** |
| 9 | Spaced-repetition scheduling | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.4 🟡 | OK ✅ | LOW | PASS | **KEEP** |
| 10 | Simulated-annealing curriculum | EXTENDS ✅ | VERIFIED ✅ | 4/5 | +0.4 🟡 | OK ✅ | LOW | PASS | **KEEP** |

## Counts
- Verified: 10
- Rejected: 0 (Novelty: 0, Provenance: 0, Falsification: 0, Compliance: 0, Other: 0)
- Downgraded: 1 (Idea 2 dropped one rank slot by devil's-advocate; gain held within range)
- Re-search cycles used: 0
- Final batch size: 10

## Warnings (per idea)
- Idea 2: MED risk — unconstrained answer-token up-weighting can cause distributional drift / answer-shortcutting (contrasting [arXiv:2509.20758], [arXiv:2505.17827]); guardrail = small bounded λ, never drop a token. Devil's-advocate downgrade applied (rank 1 → 2).
- Idea 8: MED risk — untying raises hot-expert params and can overfit a small synthetic corpus (the original motivation for full tying). Gate on the slice before adopting.
- Ideas 9 & 10: scheduling effects under LoRA with few steps may be small (honest 🟡, low gain_low). They compose (forgetting-based vs difficulty-based) and are the batch's exploratory T3 slots.
- Recency/trust: primaries for Ideas 5, 8 (2412.00029, 2601.04823) and Ideas 1, 3 (2510.03595, 2505.19716) are arXiv preprints; treated as T2/T3 trust. Batch (T1+T2) ≈ 70% ≥ 60% ✅.

## Cross-idea consistency
- Near-duplicates collapsed: none. Ideas 1/2/3 all touch the boxed answer but via distinct mechanisms (data-clean/verify vs loss-weighting vs trace-length) on distinct components.
- Contradictions flagged: none. Ideas 9 & 10 both reorder data but on different axes (forgetting vs difficulty); compatible/composable.
- Score-distribution: healthy (mix of 🟢/🟡; no all-5/all-🟢 over-confidence).

## Notes & warnings
- **Tier-3 honesty**: only 2 genuine cross-domain primaries were found (cognitive psychology — Amiri/Ebbinghaus; statistical physics — Kirkpatrick). The coreset/gradient-matching line (CRAIG/GRAD-MATCH/TAGCOS) was **deliberately not tagged T3** because its primaries are ML-venue; it was folded into Idea 6 (data curation) instead of mis-labeled. This is the honest call per the skill's Tier-3 audit.
- **Single greedy-pass constraint honored**: every idea is a training-time or data-time change. No idea adds self-consistency, sampling, beam search, verifiers-at-inference, or any decoding change — inference stays one greedy pass, vLLM-loadable, rank ≤ 32.
- **Prerequisites / measurements** (NOT ideas — run these first to target the +0.01):
  - Bucket your current held-out losses into {format-zero, truncation-zero, reasoning-wrong}. This single measurement tells you whether Idea 1, Idea 3, or Ideas 2/5/6/7 is the right lever. Without it you're guessing which bucket holds your missing 0.01.
  - Record the baseline LR/`NUM_STEPS`/`BATCH_SIZE` so rsLoRA (Idea 4) and reweighting (Idea 2) are compared at matched, re-tuned settings (avoids false negatives).
- **Tier mix**: kept default 45/35/20 (observed 40/40/20, within bands). If you'd rather bias toward drop-in in-field swaps only, a re-run with `--tier-mix 55/30/15` would drop the two T3 scheduling ideas for more in-field options.

## Next steps for user
1. **Measure the loss buckets** (prerequisite above), then ship **Idea 1 + Idea 2 + Idea 4 together** — all S-effort, all reuse existing codepaths (`corpus.py` weights/format + one LoRA flag), mutually independent, jointly the most likely reliable +0.01.
2. If buckets show truncation/long-trace losses: add **Idea 3** (and compose Idea 7's outputs through it).
3. Hold **Idea 7 (STaR)** and **Idea 8 (MoE untie)** for a second iteration — highest headroom but highest variance; only after the cheap wins are banked.

## Provenance signature
SHA256 of (inputs + paper IDs + timestamp): a222a58d99cdf58cb014d76b041dee6bbd60f2cfaa433927c4b35f6bd8ec9d2f
