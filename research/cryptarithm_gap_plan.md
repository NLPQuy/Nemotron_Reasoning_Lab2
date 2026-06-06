# Cryptarithm gap — deep analysis & implementation plan

**Goal:** raise the number of *verified-correct* cryptarithm reasoning traces in the
corpus (currently `cryptarithm_deduce` rule_found **85/659**, `cryptarithm_guess`
**14/164**), so the model learns the symbol→digit / operator→operation deduction
procedure from more examples and generalises better on the hidden cryptarithm test split.

This plan is **evidence-based**: every claim below is backed by a measurement run on the
local `nemotron-master/problems/*.jsonl` (823 cryptarithm problems). Scripts live in
`/tmp/crypto_*.py` (reproduce by re-running them from inside `nemotron-master/`).

---

## 0. TL;DR — what to actually do

1. **Primary (do this):** make the arithmetic solver in
   [reasoners/cryptarithm.py](../nemotron-master/reasoners/cryptarithm.py) **robust to one
   unexplainable example** — allow ≤1 *non-query* example to be skipped during the
   backtracking search. **Base 5 ops only, no new operations.** Measured ceiling:
   arithmetic-correct **61 → 78 (+17)**, still verify-gated so **zero regression / zero
   poisoning of wrong answers**. Effort: ~1 hour. This is the single best lever.
2. **Secondary (optional, lower value):** add a small honest-op set
   (`floordiv, mod, min, max`) behind the same verify gate. Measured marginal gain **+1**.
   Cheap but almost nothing; do it only while you're in the file.
3. **Do NOT** add flexible algebraic ops (`a*b+1`, `2a+b`, `avg`, `a²`, …). They inflate
   "consistent mapping" by overfitting 1–2-example operators and create the **poisoned-trace
   risk** (right answer via a bogus operation the trace then "explains").
3b. **The hypothesis space is exhausted** (§3b): leading-zeros, digit-wise, string/positional,
   and reduction operation families were all measured and recover **nothing** (each ≤ +1, some
   negative). CP-SAT/Z3 and MDL synthesis only speed up the *same* model, so they add no
   coverage. The **only** method with upside for the residual is a **verifier-gated LLM rule
   proposer** (offline STaR/RFT, keep traces that reproduce all examples + the answer) — run it
   as a separate budgeted experiment only if a category eval says cryptarithm is worth it.
4. **Dead ends — do not invest:**
   - `cryptarithm_guess` where the query operator never appears in the examples
     (the majority): **deterministically unsolvable** — see §4.3 (glyph prior is too noisy).
   - `brace_in_answer` (~17 problems): blocked by the grader's own `\boxed{}` regex; even a
     perfect answer can't be scored. Confirmed.

Honest expectation on the leaderboard: this adds ~15–25 correct traces out of a 17,963-row
corpus (≈0.1%). The *direct* score impact is likely within noise; the *real* bet is that
denser, correct deduction traces improve generalisation on the test split's cryptarithm
category. **Measure with an eval, don't assume.** (Prior cryptarithm wiring added +34
rule_found and did not visibly move the 0.86 baseline — see
`memory/cryptarithm-solver-wired.md`.)

---

## 1. Problem structure (verified from real data)

Each problem gives 4–5 examples and one query, all of the form `s0 s1 ⟨op⟩ s3 s4` (a 5-char
string), e.g. `` `!*[{ ``:

- `s0,s1` = left operand (2-digit number `10·s0 + s1`), `s3,s4` = right operand,
  index-2 char = **operator glyph**.
- Every glyph is a **digit symbol** under a **global bijection** (symbol↔digit, distinct
  digits). The output is the result number rendered back through the same bijection.
- The current solver's hypothesis space: operator ∈ {`add`, `abs_diff`, `mul`, `concat`,
  `rev_concat`}, operands are 2-digit, bijection is unique. (`OPS`/`OP_NAMES`,
  [reasoners/cryptarithm.py:56](../nemotron-master/reasoners/cryptarithm.py#L56).)

**Confirmed facts from the data:**
- Operator glyph is **not fixed**: `+ - *` are the common ones (~69% of slots) but *any*
  glyph (`) ' } \ { #` …) can be the operator. operator→operation is a free per-problem map.
- The same glyph can be both the operator *and* a digit symbol elsewhere in the same problem
  (handled fine — op is keyed by position-2 char, digits by `mapping`).
- Output length distribution: 1 (149), 2 (932), 3 (1134), 4 (1127). Single-digit results exist.
- 1 distinct op (43 problems), 2 ops (406), 3 ops (374) → most operators are pinned by only
  **1–2 examples** → heavily under-determined per operator.
- Pipeline: `reasoning_cryptarithm` emits a trace → `reasoning.py` runs `compare_answer`; if it
  matches ground truth → `status = rule_found` + trace written to `reasoning/<id>.txt` →
  `corpus.py` tokenises it. **Every arithmetic trace is already verify-gated**
  ([reasoners/cryptarithm.py:604](../nemotron-master/reasoners/cryptarithm.py#L604)), so any
  solver change can only *add* correct traces — it can never poison the corpus with a wrong
  *answer* (wrong answers fall back to the concat path).

---

## 2. Empirical investigation — what each fix is worth

Measured over **all 823 cryptarithm problems**, `correct` = solver answer passes
`compare_answer` against ground truth (the production gate). `any` = a self-consistent mapping
was found (whether or not the query answer is right — the gap between `any` and `correct` is
the wrong_mapping rate).

| Configuration | any-consistent | **correct** | Δ correct vs base |
|---|---|---|---|
| **BASE 5-op, unique bijection** (≈ production arithmetic path) | 136 | **61** | — |
| + honest ops (`floordiv, mod, min, max, sumdig, l+rev(r)`) | 152 | **62** | **+1** |
| + flexible ops (`a·b+1, avg, 2a+b, a+2b, a², b²`) | 188 | 79 | +18 *(overfit, see §3)* |
| **Robust: skip ≤1 non-query example, base ops** | 174 | **78** | **+17** ✅ |
| Robust: skip ≤2 | 240 | 82 | +21 *(any explodes → under-determined)* |

Production's actual cryptarithm rule_found is **85 (deduce) + 14 (guess) = 99**; the
arithmetic path contributes ~61 of those, the concat path the rest. So the robust solver lifts
the *arithmetic* contribution from ~61 to ~78. **Validate the exact end-to-end delta by
re-running `reasoning.py` after the change** (§6).

### Why "robust skip" works (the core finding)
Hand-tracing the unsolved bucket showed the dominant failure mode: a query whose operator is
**fully determined** (e.g. `*` = concat, answer trivially derivable) is dragged down because
*one other* example uses an operator whose operation lies **outside the 5-op set** (or is itself
under-determined). The current solver demands a **single globally consistent assignment across
all examples**, so that one bad example makes the *entire* search return "no consistent
mapping" — and the query, which was answerable, is lost. Allowing the search to *skip* that one
example recovers the answer. `skip=1` captures almost all the gain; `skip=2` mostly adds
under-determined noise (`any` jumps 174→240 but `correct` only 78→82).

---

## 3. Why NOT to add flexible operations
The "+18" from flexible ops is mostly **spurious**. With operators pinned by only 1–2 examples,
a flexible family like `a·b+1` finds a self-consistent bijection by **coincidence**
(`a*b+1` alone "unlocked" 27 problems, but the honest ops unlocked only +1 total). Two harms:
- **Poisoned traces:** a coincidentally-correct answer produced via a bogus operation makes the
  rendered trace *explain* "multiply then add one" — teaching the model wrong reasoning. The
  verify gate checks the *answer*, not the *operation*, so these slip through.
- **Higher wrong_mapping rate:** more degrees of freedom → more self-consistent-but-wrong query
  inferences.

Keep the operation set to the 5 the dataset actually uses.

---

## 3b. Deep research — the full approach space (tested) and what is actually left

Two orthogonal questions: **(A) WHAT rule** could the residual follow, and **(B) by what
METHOD** do we find it. Both were investigated; the headline is that **(A) is exhausted by hand
and the only remaining upside is (B) a verifier-gated LLM proposer.**

### 3b.1 Axis A — hypothesis-space families (all measured, robust skip≤1, verify-gated)

Every family below was added on top of the robust base solver and scored on all 823 problems.
Probe: `investigators/crypto_families_probe.py`.

| Family added | correct | vs base 78 | Verdict |
|---|---|---|---|
| base 5-op (robust skip≤1) | **78** | — | the lever |
| + leading-zero results (`add_pad{2,3}`, `mul_pad{3,4}`, …) | 79 | +1 | generator rarely zero-pads |
| + digit-wise ops (`(a±c, b±d)`, max/min/×mod10) | 77 | −1 | not the rule |
| + string/positional (`sort`, `interleave`, `reverse_all`, `swap_pairs`) | 77 | −1 | not the rule |
| + reductions (`sum_all`, `prod_all`) | 76 | −2 | not the rule |
| + non-unique bijection (digits may repeat) | _pending_ | _tbd_ | classic cryptarithms forbid repeats; expected ≤0 |

**Conclusion: the hypothesis space is not the bottleneck.** No structural family recovers
problems; several *reduce* `correct` because extra operations manufacture
consistent-but-wrong mappings that win the most-frequent-answer vote (harmless in production —
they fail the verify gate — but proof that there is no cheap structural win). The decade-old
result that classic cryptarithmetic = a distinct-digit, base-arithmetic CSP (see Sources)
matches what we see: the solvable problems already fit that model; the residual does not fit
*any* simple extension of it.

### 3b.2 Axis B — solving methods, ranked by fit to this problem

1. **Hand-coded backtracking (current).** Fixed (bijection × 5-op) search. *Keep, plus the
   robust-skip fix.* Ceiling reached at ~78 correct.
2. **Robust / soft-constraint search (the §5 fix).** Allow ≤1 unexplained example. *The one
   deterministic win, +17.* Already the plan's primary action.
3. **Constraint programming — CP-SAT / Z3** (cf. Google OR-Tools' cryptarithmetic model).
   Encodes all-different digits + per-operator op choice + per-example equations declaratively.
   *Verdict: do not adopt for coverage.* It only makes the *same* search faster/cleaner — it
   cannot expand the model, and §3b.1 shows expanding the model yields nothing. Worth it only if
   the hand solver becomes a maintenance burden; not a score lever.
4. **MDL / Occam program synthesis over a primitive DSL** (digit ops, carry, string moves,
   composition). The MDL prior (prefer the *shortest* consistent program) is the principled
   antidote to the §3 overfit/poison problem — `a·b+1` loses to `concat` on description length.
   *Verdict: low expected yield here.* The obvious compositions (`sort`, `interleave`,
   `reverse`, padded arithmetic) are already in §3b.1 and paid nothing, so the residual rules are
   not short compositions of obvious primitives. Only pursue if you first find, by hand, a
   recurring exotic primitive in the residual worth adding.
5. **Verifier-gated LLM rule proposal (offline STaR / RFT) — the only method with real upside
   for the residual.** For each unsolved problem, sample the base Nemotron model (or any strong
   model) to *induce and apply* the rule; keep the trace **only if the proposed rule reproduces
   ALL examples AND the boxed query answer matches ground truth**. This can capture rules outside
   any hand DSL, and the all-examples+answer gate is a far stronger anti-poison filter than the
   answer-only gate. It is **offline** (sample once, filter, add to corpus) — explicitly *not*
   the on-policy RL that is a documented dead end in CLAUDE.md (no vLLM-in-the-loop, no rollout
   regeneration). Costs: generation compute; yield is uncertain and bounded below by the
   under-determined problems (see next), which *no* method can solve.

### 3b.3 The hard ceiling — what no method can recover
- **Under-determined (`wrong_mapping`, ~95–120).** A consistent base-op mapping exists but the
  examples do not *uniquely* pin it, so the query is genuinely ambiguous. This is the gap between
  `any` (173) and `correct` (78) in the robust run. **Information-theoretically unsolvable** —
  an LLM would also be guessing. Out of reach.
- **`cryptarithm_guess` with an unseen query operator (~150).** §4.3 — no glyph prior. Out of reach.
- **`brace_in_answer` (~17).** Grader regex. Out of reach.

So of the ~580 unsolved: ~17 robustly recoverable by the deterministic skip-fix; a further
uncertain slice (rule-outside-DSL but inducible) only by the gated-LLM route; and a large core
(under-determined + guess + brace, ~280+) that is **provably or practically unsolvable** and
should not absorb effort.

### 3b.4 Recommended escalation ladder
1. **Ship the robust skip≤1 fix** (§5–6). Banked +17, ~1h, zero risk.
2. **Stop on the deterministic axis** — §3b.1 proves further op/structure work is wasted.
3. *Only if a category-level eval shows cryptarithm is a high-value test slice worth more:*
   run the **gated-LLM proposer** on the still-unsolved set as an offline data-gen pass,
   `chosen`-only (keep correct traces). Treat as a separate, budgeted experiment with its own
   round entry; measure marginal correct traces before trusting it.
4. **Never** chase under-determined / guess / brace problems. If you ever extend it, restrict to
operations that are (a) common in the data and (b) *not* a free additive/affine knob — i.e. at
most `floordiv`/`mod` (clean, integer, low-coincidence), and only behind the verify gate.

---

## 4. Bucket-by-bucket diagnosis (mapped to the original triage)

| Original bucket | Count (deduce+guess) | True cause | Lever | Recoverable? |
|---|---|---|---|---|
| `no_consistent_mapping` | 465 + 117 = **582** | mostly: query is answerable but **1 example's op is out-of-set / under-determined** → global-consistency requirement fails. Some: query op genuinely outside 5-op set; some: guess (op unseen). | **Robust skip ≤1** (§2) | **Partly** — ~+17 verified |
| `wrong_mapping` | 91 + 29 = **120** | example set does not uniquely pin the bijection; solver picks a consistent-but-wrong branch | better tie-break / abstain | mostly **no** (under-determined; gate already drops them) |
| `brace_in_answer` | 13 + 4 = **17** | correct answer contains `}`, breaks `\boxed{...}` extraction — **the competition grader uses the same regex** | none | **no** (blocked by scoring rules) |
| `concat_path_wrong` | **5** | routed to concat but concat is wrong for that query | minor routing fix | yes, tiny |

### 4.1 The big bucket (582) — split by recoverability
The robust-skip experiment shows ~+17 of these are recoverable with **base ops + skip≤1**, no
new operations. The remainder are genuinely out-of-model (query op outside the 5 ops, or
unseen) or under-determined.

### 4.2 `wrong_mapping` (120) — mostly irrecoverable
The gap between `any` (174–240) and `correct` (78–82) in §2 *is* this bucket: self-consistent
mappings that infer the query wrongly because the examples don't uniquely determine the
bijection. These are already gated out (they fail verify → fall back to concat), so they don't
poison the corpus. Better tie-breaking can't manufacture information that isn't in the examples.
Low priority.

### 4.3 `cryptarithm_guess` — deterministically unsolvable, deprioritise
In `guess` problems the **query operator does not appear in any example** (e.g. examples use
`-,*` and the query uses `+`). To answer you must *guess* the operation for an unseen glyph.
A global glyph→operation prior was measured and is **too noisy to use**:

```
'+'  purity 0.37  [concat 39, add 39, abs_diff 13, mul 10, rev_concat 5]
'*'  purity 0.43  [mul 41, concat 38, rev_concat 9, ...]
'-'  purity 0.70  [abs_diff 67, add 16, mul 13]      ← only mildly consistent
(most other glyphs ~0.33, i.e. random)
```

Only `-`→`abs_diff` is even mildly predictive (0.70, and only within the solvable subset).
There is no reliable convention to exploit, so guessing produces wrong answers that fail
verify → no gain. **Treat `guess`-with-unseen-op as a designed-impossible bucket** (it likely
tests abstention). Do not build a guesser.

---

## 5. Recommended design — "robust query solver"

Replace the all-or-nothing global search with one that **maximises explained examples while
guaranteeing the query operator is explained**, allowing ≤1 non-query example to be skipped.

Key invariants (preserve correctness & the no-regression guarantee):
- **Base 5 ops only.** No exotic operations.
- **Verify gate stays** — emit the arithmetic trace only if its boxed answer matches ground
  truth; otherwise fall back to concat reasoning. Unchanged behaviour
  ([reasoners/cryptarithm.py:599-608](../nemotron-master/reasoners/cryptarithm.py#L599-L608)).
- The query operator **must** be determined by at least one example (`qop ∈ op_assign`);
  never guess an unseen operator (avoids the §4.3 trap).
- Skip budget = **1** (skip=2 adds noise, not correctness).
- When an example is skipped, the **rendered trace must not pretend to explain it** — render
  only the examples the mapping explains, plus the query. (Pedagogically clean: "these
  examples fix the mapping; one example is ambiguous and I set it aside.")

This is search over `examples × {fit with op | skip}` with `skip ≤ 1` and a hard requirement
that all query-operator examples are fit. Order examples so query-operator ones are visited
first (prunes hard).

---

## 6. Implementation plan (line-level)

**File:** [nemotron-master/reasoners/cryptarithm.py](../nemotron-master/reasoners/cryptarithm.py)

### Step 1 — add skip support to `_Solver._process`
- Thread a `skips` counter and `max_skip` (default 1) through `_process(idx, skips)`
  ([:116](../nemotron-master/reasoners/cryptarithm.py#L116)).
- Compute `qop` once and pass example order so query-operator examples are processed first
  (add an `order` list in `solve()`/`_solve_arith`).
- After the normal fit loop for an example, add the alternative branch:
  ```python
  # skip this example's operation constraint (digits still unknown for it,
  # so we simply don't use it) — only for non-query operators, within budget
  if op_sym != self.qop and skips < self.max_skip:
      self._process(idx + 1, skips + 1)
  ```
- In `_compute_query` ([:237](../nemotron-master/reasoners/cryptarithm.py#L237)), **require**
  `qop in self.op_assign` (return early otherwise) — never fall back to `range(len(OP_NAMES))`
  for an unseen query operator. This both fixes guess-trap behaviour and tightens correctness.

### Step 2 — keep the budgets; skip≤1 only
- `max_skip = 1`. Leave `_UNIQUE_BUDGET` / `_NONUNIQUE_BUDGET` as-is
  ([:67-68](../nemotron-master/reasoners/cryptarithm.py#L67-L68)); the extra branch is bounded.

### Step 3 — trace renderer must not explain skipped examples
- `_reasoning_arith` ([:339](../nemotron-master/reasoners/cryptarithm.py#L339)) loops over all
  `examples` in the "Examples"/"Verifying" sections. Pass the **set of explained example
  indices** (those whose operator is in `op_info` *and* whose operands+result are fully in
  `mapping` and verify). For the rest, either omit them or print them under a short
  "ambiguous, set aside" note — do **not** emit a fake `_op_calc`. The gate already guarantees
  the query line is correct.

### Step 4 — (optional) honest ops
- Only if desired: append `floordiv`/`mod` to `OPS`/`OP_NAMES`
  ([:56-63](../nemotron-master/reasoners/cryptarithm.py#L56-L63)) and to `_OP_VERB` /
  `_op_calc`. Measured marginal gain +1; skip unless trivial.

### Validation
```
cd nemotron-master
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run python3 reasoning.py     # regenerate traces
python3 - <<'PY'   # count rule_found delta
import json,collections
c=collections.Counter()
for l in open('problems.jsonl'):
    d=json.loads(l)
    if d['category'].startswith('cryptarithm'): c[(d['category'],d['status'])]+=1
print(sorted(c.items()))
PY
uv run --frozen ruff format reasoners/cryptarithm.py
uv run --frozen ruff check reasoners/cryptarithm.py
uv run --frozen mypy reasoners/cryptarithm.py
uv run pytest        # ensure no reasoner test regressions
```
**Acceptance:** `cryptarithm_deduce` rule_found rises from 85 toward ~100 (target: at least
+12), no other category regresses, and **spot-check 5 newly-`rule_found` traces** to confirm
the rendered reasoning is honest (no fabricated explanation of a skipped example).
See `memory/corpus-rebuild-hf-token.md` for the HF-token gotcha when rebuilding the corpus.

### Rollback
Single-file change; `git checkout nemotron-master/reasoners/cryptarithm.py` reverts. Because the
verify gate is untouched, the worst case is "no new traces", never corpus corruption.

---

## 7. After the solver: does it move the score?
Adding ~15–25 correct cryptarithm traces is ~0.1% of the 17,963-row corpus. **Do not assume a
leaderboard gain.** Two ways the bet can pay off, both requiring an eval to confirm:
1. **Generalisation:** more correct deduction traces teach the symbol/operation-deduction skill
   the test's cryptarithm split needs. Plausible, unproven.
2. **Augmentation multiplier:** check whether any augmenter consumes cryptarithm
   `reasoning/*.txt` (the `matching` augmenter consumes bit_manipulation traces, *not*
   cryptarithm — confirmed — so no multiplier here today). If a cryptarithm-derived augmenter
   were added, more solved problems would compound.

**Measurement protocol:** rebuild corpus → train one adapter with the new traces → eval on the
held-out cryptarithm subset specifically (not just the aggregate 0.86), so a small category-
level gain isn't washed out by the other categories. Log the result in
`tracker/rounds/round_<N>.md` and `tracker/leaderboard.md`.

---

## 8. Priority / run order
1. **Robust skip≤1 solver** (§5–6) — best ROI, safe, ~1h. Validate rule_found delta.
2. Spot-check trace honesty on skipped-example problems.
3. (Optional) honest `floordiv`/`mod` ops — +1, only if free.
4. Eval the rebuilt corpus on the cryptarithm subset; record in tracker.
5. **Stop.** `guess`-unseen-op and `brace_in_answer` are out of reach by design — do not
   spend further compute there.

## 9. Reproduce the measurements
Scripts are saved under `nemotron-master/investigators/` (run from `nemotron-master/`,
plain `python3 -u investigators/<script>`):
- `crypto_extended_op_probe.py` — base vs extended-op `any`/`correct` + which extra ops "unlock".
- `crypto_opset_probe.py` — base vs honest-op, unique vs non-unique bijection.
- `crypto_robust_solver_probe.py` — robust skip∈{0,1,2} `any`/`correct` (the §2 headline numbers).
- `crypto_glyph_prior_probe.py` — global glyph→operation purity (the §4.3 prior).
- `crypto_families_probe.py` — hypothesis-family coverage: leadzero / digitwise / stringops /
  reductions on top of robust skip≤1 (the §3b.1 table).
- `crypto_nonuniq_probe.py` — non-unique-bijection (digits may repeat) coverage (slow).

(Each globs `problems/*.jsonl`; the non-unique-bijection branch in `crypto_opset_probe.py` is
slow — expect minutes — the others finish in well under a minute.)
