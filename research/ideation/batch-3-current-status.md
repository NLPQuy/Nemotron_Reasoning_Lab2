# Batch 3 Current Status

Date: 2026-06-03 (updated)

## Summary

Batch 3 = exp20–exp26, **data-time augmentation only**. Main work is upstream in `nemotron-master/`; each `exp<N>.py` mostly points the trainer at a regenerated corpus snapshot. **exp files live in `experiments/`** (moved from repo root); base trainer is `Continuer_Nemotron_Notebook.py` at root.

**Blocker from the previous status is RESOLVED.** User cloned full upstream as `nemotron/`, which includes `problems/` (9500 structured `Problem` payloads). Verified `nemotron/` and `nemotron-master/` are the **same baseline** (byte-identical `train.csv`, `problems.jsonl`, `reasoning.py`, `store_types.py`, all 7 reasoners), so `nemotron/problems/` was copied into `nemotron-master/problems/`. Solver pipeline now runs end-to-end.

Current state:
- `experiments/exp20.py … exp26.py` created from base notebook.
- **exp21 IMPLEMENTED + VERIFIED** (verify gate + global anti-truncation length gate in `nemotron-master/corpus.py`; `experiments/exp21.py` corpus-pointer knob).
- exp20, exp22, exp23, exp24, exp25, exp26 still base copies (0 diff vs base) — not implemented.

## Pipeline now runs (unblocked)

```
cd nemotron-master
uv run --frozen python3 reasoning.py   # 8333/9500 rule_found (87.7%)
uv run --frozen python3 corpus.py      # 8329 entries after exp21 gate
```

`reasoning.py` per-category rule_found: cipher/gravity/numeral/unit_conversion 100%, bit_manipulation 85.1%, equation_numeric_deduce 90.6%, cryptarithm_deduce 8.2%, cryptarithm_guess 6.7%, equation_numeric_guess 15.4%.

## exp21 — DONE

Files: `nemotron-master/corpus.py`, `experiments/exp21.py`.

**Verify gate** (`VERIFY_GATE`): reuses upstream `compare_answer`/`extract_answer`; drops any trace whose boxed answer ≠ `train.csv` answer. This is real: `corpus.py` previously included **any** pid with a `reasoning/<pid>.txt` file and took the boxed value from the trace, never re-checking the stored answer. Empirically drops the low-accuracy categories' wrong traces:
- cryptarithm_deduce 91.81%, cryptarithm_guess 93.29%, equation_numeric_guess 84.56%, bit_manipulation 14.86%, equation_numeric_deduce 9.40%; cipher/gravity/numeral/unit_conversion 0%.

**Length gate** (`LENGTH_GATE` + `GLOBAL_LENGTH_CAP`): **recalibrated.** Original `mean+1σ` per-category cap dropped ~16% of *every* category (statistical artifact, violates the ≤5% plan threshold, narrows distribution — e.g. numeral capped at 141 tokens). Replaced with a **global cap = 7600** (just under the 7680 output budget): a pure anti-truncation safety wrapper. Now drops ≈0% of current correct traces (bit_manipulation 0.25%, all others 0%) and only fires on genuinely over-budget augmented traces. Removed `infer_cap_by_category` / `LENGTH_CAP_SIGMA` / `CAP_BY_CATEGORY`; kept `build_reasoning_completion`.

Result: **8329 corpus entries**, 27.1M unmasked tokens, max seq 7817 (≤8192).

Verification:
- `ruff format/check corpus.py`: pass (`All checks passed!`)
- `python3 -m py_compile experiments/exp21.py`: pass
- `git diff --check`: clean
- corpus.py dry run: real, non-zero, gate counts sane

## Known harness issues (pre-existing, not introduced by batch-3)

- **mypy not installed** in the frozen env (`pyproject.toml` dev deps have ruff+pytest, no mypy). Plan §12.1 `uv run --frozen mypy` cannot run until mypy is added (`uv add --dev mypy`) — defer/decide.
- **doctest fails**: `compare_answer` docstring examples call `verify(...)` (function is `compare_answer`). Pre-existing typo in `reasoning.py`. Doc-only fix (no behavior change) would unblock plan §12.2, but left untouched to respect the "don't touch verifier" constraint until explicitly approved.
- **pytest 2 failures** in `.claude/hooks` (`test_main_stop_notification`, `test_stop_validator_with_edits_no_confirmation`) — unrelated to corpus/data work, pre-existing.

## Status by experiment

| Exp | Status | Next action |
| --- | --- | --- |
| exp21 | **DONE + verified** | (gate is the dependency for 20/22/24/25/26) |
| exp22 | **DONE + verified** | 4 augmenters + registration + `experiments/exp22.py` pointer |
| exp20 | **DONE + verified (gravity)** | generator + driver + `experiments/exp20.py`; extend to more safe categories if gravity moves LB |
| exp23 | Slice infra DONE; waiting on GPU eval | Run infer_slice.py → eval_slice.py → send per-category acc; then build inverse-acc weights. |

### Eval-slice infrastructure (prerequisite for exp23 + all A/B) — DONE locally

- `build_eval_slice.py` → `eval_slice.jsonl` (170 held-out: 25/category for big cats, capped ≤25% for tiny ones) + `eval_slice_ids.txt`.
- `eval_slice.py` → per-category + micro/macro exact-match from a `preds.jsonl`, reusing grader `compare_answer`/`extract_answer`; flags missing / no-box (format-zero). Smoke-tested.
- `pack_kaggle_snapshot.py --exclude eval_slice_ids.txt` → holds the slice OUT of training (verified: 18392→18222). Leak-free eval.
- `infer_slice.py` → vLLM greedy (temp 0, max_tokens 7680, same chat-template+suffix+enable_thinking as corpus.py). **GPU/Kaggle only** (30B), lint-clean but untested locally.
| exp24 | **DONE + verified (gravity)** | paraphrase_instances.py + exp24.py; extend swaps/categories later |
| exp25 | **DONE + verified (gravity)** | surface_instances.py + exp25.py; extend invariant knobs to more categories later |
| exp23 | Mechanism pending; waiting on GPU eval | needs per-category slice acc → inverse-acc weights |
| exp26 | **DROPPED** (high risk) | removed per decision 2026-06-03; model-written traces regressed in round-2 |

### Built snapshots (ready to upload + run)
- `kaggle_snapshot/` (17992) — exp21+exp22 baseline. **Uploaded.** Knob: `EXP21_GATED_CORPUS`.
- `kaggle_snapshot_exp20/` (18392) — +400 gravity gen. Knob: `EXP20_SCALED_CORPUS`.
- `kaggle_snapshot_exp24/` (18292) — +300 gravity paraphrase. Knob: `EXP24_PARA_CORPUS`.
- `kaggle_snapshot_exp25/` (18292) — +300 gravity surface-reorder. Knob: `EXP25_SURFACE_CORPUS`.
- All validated (0 missing, trainer-loadable). Each isolates one experiment over the 21+22 baseline → clean A/B vs the uploaded `kaggle_snapshot`. Working tree currently holds rand_* (exp25 state).

### exp24 — DONE (gravity) · prompt paraphrase
Files: `nemotron-master/instance_io.py` (shared append/clear/load), `nemotron-master/paraphrase_instances.py` (para_*), `experiments/exp24.py` (`EXP24_PARA_CORPUS` root knob).
Mechanism: deterministic meaning-preserving phrase swaps on the prompt; same examples/question/answer → **solver trace identical**, prompt masked → only conditioning changes. Verify: ruff clean; 300 para_* all rule_found; sample confirms **prompt differs + trace byte-identical to source**; corpus built; compile OK. Caveat: low expected gain (test uses same templates); safe (prompt masked).

### exp25 — DONE (gravity) · surface randomization
Files: `nemotron-master/surface_instances.py` (rand_*), `experiments/exp25.py` (`EXP25_SURFACE_CORPUS` root knob).
Mechanism: reorder gravity example lines (median k is order-invariant); rebuild prompt; **solver re-verifies the same answer** before keeping. Verify: ruff clean; 300 rand_* all rule_found (answer invariant under reorder); traces genuinely differ; corpus built; compile OK. Restricted to gravity (confirmed invariant); extend per category only after confirming the knob.

### Snapshot recipes (working tree now has gen_400 + para_300 + rand_300 gravity)
Each augmentation uses an independent prefix with its own `--clear`. To build an isolated snapshot, clear the prefixes you don't want, then `reasoning.py && corpus.py && pack_kaggle_snapshot.py --out <name>`:
- exp24-only: `surface_instances.py --clear && generate_instances.py --clear` (keep para_)
- exp25-only: `paraphrase_instances.py --clear && generate_instances.py --clear` (keep rand_)
- combo 20+24+25: keep all three (current state).
- baseline 21+22: clear all three.

## exp22 — DONE

Files: `nemotron-master/augmenters/{reverse,count_substring,char_index,digit_extract}.py` (new, mirror `splitting.py`/`spelling.py`; masked, no `\boxed`, 300 problems × 100 rows each), `nemotron-master/augmentation.py` (registered, bracketed `# >>> EXP22`), `experiments/exp22.py` (banner + `EXP22_AUG_CORPUS` corpus-pointer knob).

Verification:
- ruff format/check on all 4 + augmentation.py + corpus.py: **All checks passed** (the only ruff errors in the tree are pre-existing in `matching.py`, untouched).
- No `\boxed` in any new completion/prompt: **0** (asserted).
- `augmentation.py` run: wrote 9663 problems incl. reverse/count_substring/char_index/digit_extract = 300 each; corpus rebuilt to 17992 entries.
- Aux token share: NEW-4 add **+3.36pp**. Finding: **all** augmenters already total 30.6% (baseline was ~27% aux), so the plan's ≤20% guideline sits below the existing baseline — not a hard constraint; the A/B slice is the real arbiter. NEW-4 addition is modest.
- `python3 -m py_compile experiments/exp22.py`: pass.

## exp20 — DONE (gravity)

Files: `nemotron-master/generators/gravity_gen.py` (generator), `nemotron-master/generate_instances.py` (idempotent append driver, `gen_*` ids, `--clear` to restore baseline), `experiments/exp20.py` (banner + `EXP20_SCALED_CORPUS` root knob).

Mechanism: generator samples gravity params within measured support (t∈[1.00,5.00] 2-dp, 3–5 examples, k=d/t²∈[2.45,9.79]), then **runs `reasoning_gravity` and takes its boxed output as the answer** → every instance `rule_found` by construction. Driver appends to `problems/`, `problems.jsonl`, `train.csv`.

Verification:
- ruff clean on generators/ + generate_instances.py.
- In-memory dry-run: 50/50 rule_found, params in support.
- End-to-end: generated 400 → `reasoning.py` gravity 1597→1997 (100%), **all 400 gen_\* rule_found (100%)**; `corpus.py` 17992→**18392** entries (gravity dropped_wrong=0, dropped_long=0); packed `kaggle_snapshot_exp20/` (18392 files, max seq 7817, 0 missing).
- `python3 -m py_compile experiments/exp20.py`: pass.

Caveat: gravity is solver-100% (likely already well-learned) → scaling it may give flat LB gain. The *mechanism* is proven (zero-noise procedural scaling); whether gravity specifically helps is the empirical A/B. If flat, extend the generator to other safe categories.

## Snapshots / experiment isolation

- `nemotron-master/kaggle_snapshot/` (473M, 17992) = exp21 gate + exp22 augmenters. **Already uploaded by user.**
- `nemotron-master/kaggle_snapshot_exp20/` (488M, 18392) = exp21 + exp22 + exp20 gravity (combo). Upload as a 2nd dataset.
- Clean A/B for exp20 = `kaggle_snapshot` (21+22) vs `kaggle_snapshot_exp20` (21+22+20); the only delta is +400 gravity.
- ⚠️ Working tree now has the 400 `gen_*` baked into `train.csv`/`problems.jsonl`/`problems/`. To rebuild a 21+22-only corpus: `uv run python3 generate_instances.py --clear && uv run python3 reasoning.py && uv run python3 corpus.py`.

## Recommended next steps

1. **exp20** (big bet, now next): design `generators/<cat>_gen.py` for 1–2 weakest categories; take the solver's boxed output as ground truth (guaranteed `rule_found`); append to `train.csv`+`problems.jsonl`. **Decide parameter ranges first** (measure support from existing data) — this is the open design fork.
2. Build the per-category held-out eval slice (prerequisite for exp23 and all A/B falsification).
3. exp24 → exp25 → exp23 → exp26 per plan order, each behind the exp21 gate.
4. Resolve harness: `uv add --dev mypy`; decide on the doctest docstring fix.
5. **Snapshot build** (blocking before any real training): both exp21/exp22 regenerate `corpus/<pid>/synthetic.jsonl` (segments), but the trainer reads `tokens/<pid>/synthetic.json` `{tokens,mask}`. The snapshot-packaging step (huikang `04-08-16-14` layout) must convert + upload before `EXP21_GATED_CORPUS`/`EXP22_AUG_CORPUS` can point at real data.

## Constraint reminder

Data-time only. Do not modify inference, decoding, `compare_answer`, or `extract_answer` behavior.

---

# SESSION HANDOFF — 2026-06-04

> State cho session mới. Phần "confirmed" là việc mình (session này) làm + verify; phần
> "parallel" là việc session khác làm (exp27-38, exp29 DPO) — ĐỌC plan/file riêng, đừng tin trí nhớ.

## Leaderboard (confirmed)
- baseline **0.86** = adapter huikang (submission.zip), KHÔNG phải retrain của mình.
- **exp20 (21+22+gravity, fresh-retrain RESET_WEIGHTS=True) = 0.85** (−0.01). Chưa idea nào vượt 0.86.
- Chi tiết + insight: [tracker/rounds/round_3.md](../../tracker/rounds/round_3.md).

## Insight chốt (đọc kỹ trước khi chạy tiếp)
1. **Đừng quy 0.85 cho augment** — confound: corpus đổi + recipe khác huikang + fresh-init.
2. **Nghi phạm chính −0.01 = verify gate (exp21)**: corpus gốc 0.86 VỐN chứa trace boxed-sai (rule_unknown); gate xóa ~600 cryptarithm + ~115 eq_guess → mất coverage. Gate "chất lượng" có thể NET-ÂM.
3. **Gravity bão hòa** (solver 100%) → scale phẳng. Ngừng scale category đã bão hòa.
4. **Continue-train né được vấn đề gate** (0.86 đã thuộc data bị gate xóa) → hướng hứa hẹn hơn fresh-retrain.
5. **Offline Kaggle → KHÔNG vLLM**; inference phải backend "hf" (transformers.generate) + **merge_and_unload() + use_cache=True** (nếu không → hybrid-cache hang). BASE phải là path LOCAL, không HF id.

## Snapshots đã build (trong nemotron-master/, chỉ kaggle_snapshot đã upload)
- `kaggle_snapshot/` (17992) = 21 gate + 22 augmenters. **ĐÃ UPLOAD.** Knob `EXP21_GATED_CORPUS`.
- `kaggle_snapshot_exp20/` (18392) = +400 gravity. Knob `EXP20_SCALED_CORPUS`. (đã chạy=0.85)
- `kaggle_snapshot_exp24/` (18292) = +300 paraphrase. Knob `EXP24_PARA_CORPUS`.
- `kaggle_snapshot_exp25/` (18292) = +300 surface-reorder. Knob `EXP25_SURFACE_CORPUS`.
- ⚠️ Working tree có thể đang ở state lẫn gen_/para_/rand_ — chạy `git status` + check `nemotron-master/problems/{gen_,para_,rand_}*` trước khi rebuild. Mỗi script có `--clear`.

## Việc session này làm (confirmed)
- exp24 (paraphrase) + exp25 (surface) build+verify; snapshots tách biệt đã build.
- `experiments/exp_continue.py` MỚI: continue-train 0.86 (RESET_WEIGHTS=False, LR 2e-5, NUM_STEPS=300, SHUFFLE_DATASET=True bắt buộc vì augment sort cuối corpus). Knobs CONTINUE_CORPUS + CONTINUE_ADAPTER_DIR. Chưa chạy.
- Vá `nemotron-master/infer_slice.py` nhánh hf: thêm merge_and_unload + use_cache (chống hybrid-cache hang). Chưa test trên GPU.
- exp_continue: loop = 1 epoch tối đa (clamp NUM_STEPS về len//batch, KHÔNG wrap).

## Parallel work (KHÔNG verify bởi session này — đọc file gốc)
- `experiments/exp27.py … exp38.py` (batch-4) + `exp29.py` (iterative DPO via TRL) — xem [research/ideation/plan-exp29.md](plan-exp29.md) + các plan batch-4 nếu có.
- `generators/` đã đủ 7 category (session khác mở rộng exp20 sang all categories) + `generate_instances.py --category all`.
- exp29 bug đã chẩn: chạy bản CŨ + BASE=HF id offline. Fix: re-sync nemotron-master + BASE local. infer_slice hf-cache đã vá phiên này.

## Task queue đề xuất (ưu tiên)
1. **Submit `kaggle_snapshot` (21+22)** — câu hỏi quan trọng nhất: gate hại hay không (vs 0.86).
2. **Chạy exp_continue control** (CONTINUE_CORPUS=None, CONTINUE_ADAPTER_DIR=adapter-0-86) → phải ≈0.86 (xác nhận continue-train không phá).
3. **exp_continue test** trên kaggle_snapshot_exp25 (surface) → đo augment có cộng giá trị lên nền 0.86 không.
4. Submit exp24 / exp25 (đã build) nếu còn quota.
5. exp29 DPO: re-sync + BASE local + chạy (infer hf-cache đã vá).
6. eval-slice per-category (infer hf-cache đã vá → giờ chạy được) để biết gate hại category nào → mới làm exp23 mixture.
