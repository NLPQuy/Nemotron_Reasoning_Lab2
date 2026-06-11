# Runbook — chạy exp29–38 (Batch-4)

Hướng dẫn thực thi, neo vào **trạng thái code thật** (verified 2026-06-05). Bổ sung cho
[plan-batch-4.md](plan-batch-4.md): plan mô tả *thiết kế*, runbook này mô tả *thứ tự bấm nút* +
các blocker phải build trước.

> Quy tắc bất biến: KHÔNG tự suy đáp án — solver/`compare_answer()` là source of truth.
> Mọi loss mới phải qua **50-step eval_slice mini-run** trước full run. eval_slice = 170 bài / 9 cat.

---

## 0. Sự thật hạ tầng đã verify (đọc trước khi bắt đầu)

| Thành phần | Trạng thái thật | Hệ quả |
|---|---|---|
| `infer_slice.py` | **GREEDY-ONLY** (temperature=0.0 hardcoded dòng 78; không có `--temperature/--n_samples`) | Phải patch để sampling → blocker chung của **exp29, exp31, exp36** |
| Tinker `LossFnType` | `('cross_entropy','importance_sampling','ppo','cispo','dro')` — **KHÔNG có `dpo`** | exp29/35/38 không có DPO native → map sang `dro`/`importance_sampling` HOẶC đi TRL |
| `CrossEntropyWithWeightingLossConfig.apply_weights()` | Đã tồn tại (loss_config.py:117), nhận per-token weights | **exp30 (OXA), exp33 (VCORE)** chỉ cần subclass + đổi corpus mask — KHÔNG cần engine mới |
| `eval_slice.jsonl` / `eval_slice.py` / `build_eval_slice.py` | Đã có. 170 bài, 9 cat | Harness chấm mọi falsification gate |
| `paraphrase_instances.py` | Đã có (4.1KB) — cho instances, chưa chắc cho `reasoning/*.txt` | exp34 cần đọc/extend, không viết mới từ đầu |
| Trained adapter 0.86 | **CHƯA có path trong repo** | Mọi exp cần GPU đều cần điền `--adapter <path>` — đây là prerequisite #0 |

---

## 1. Prerequisite #0 — adapter + eval harness (làm 1 lần, dùng cho TẤT CẢ)

```bash
cd nemotron-master
# (a) Phải có adapter baseline 0.86 (hoặc output exp27/exp28). Ghi lại path:
ADAPTER=<path-to-0.86-or-exp28-adapter>

# (b) Sanity check eval harness chạy được (greedy baseline trên eval_slice):
uv run python3 infer_slice.py --base nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --adapter $ADAPTER --slice eval_slice.jsonl --out preds_baseline.jsonl
uv run python3 eval_slice.py --preds preds_baseline.jsonl --slice eval_slice.jsonl
# → ghi lại per-category accuracy LÀM MỐC SO SÁNH cho mọi falsification gate bên dưới.
```

---

## 2. Shared infra build — làm theo nhu cầu, KHÔNG build hết một lúc

### A. Sampling patch cho infer_slice.py  → mở khoá exp29, exp31, exp36
Thêm `--temperature` (default 0.0) + `--n_samples` (default 1) vào argparse, truyền vào
`SamplingParams(temperature=..., n=...)`. Giữ default = hành vi greedy cũ (không vỡ exp27/28).
```bash
# Sau patch, smoke test:
uv run python3 infer_slice.py --base <base> --adapter $ADAPTER \
  --slice eval_slice.jsonl --out smoke.jsonl --temperature 0.5 --n_samples 4
python3 -c "import json; print(len([json.loads(l) for l in open('smoke.jsonl')]))"
```

### B. Quyết định DPO engine  → mở khoá exp29, exp35, exp36, exp38
Tinker không có `dpo`. Hai đường:
- **Đường Tinker (ưu tiên):** thử map preference → `dro` hoặc `importance_sampling` trong `loss_config.py`
  (đọc `ImportanceSamplingLossConfig` dòng 257 + `DRO` nếu có). Ưu điểm: dùng nguyên pipeline train_sft.py.
- **Đường TRL (fallback):** dùng `enhance_cot/redi/experiments_trl/open_r1_dpo.py` làm reference,
  adapter 0.86 = reference model. Tốn công dựng môi trường TRL riêng.

👉 **exp18 (batch-2 offline preference) là cổng chặn:** chạy exp18 TRƯỚC để xác nhận preference learning
hoạt động trên Nemotron-H. exp35/exp38 ghi rõ prerequisite cứng này; nếu exp18 fail → bỏ exp35/38.

### C. Weighting subclass  → mở khoá exp30 (OXA), exp33 (VCORE)
Cả hai extend `CrossEntropyWithWeightingLossConfig`. Đọc `apply_weights()` (loss_config.py:117–139)
để biết weights là per-token `list[float]`. Không cần engine mới — chỉ subclass + đổi nguồn weight.

---

## 3. Thứ tự chạy khuyến nghị (theo ROI tăng dần độ rủi ro)

```
[Tier 1 — tractable, dùng infra có sẵn]
  exp33 (VCORE float-mask)   ── chỉ sửa corpus.py + 1 subclass. Rủi ro thấp nhất.
  exp30 (OXA logprob-weight) ── cần logprob precompute (GPU forward) + 1 subclass.
  exp34 (paraphrase)         ── Claude API, KHÔNG đổi loss. Verify gate bắt buộc.

[Tier 2 — cần sampling patch (2.A)]
  exp31 (procedural crypto + RFT) ── falsify pass@10 ≥ 2% TRƯỚC khi build corpus.
  exp32 (AdaSTaR adaptive sampling) ── port utils_adastar.py vào _stratified_batches().

[Tier 3 — cần DPO engine (2.B) + exp18]
  exp29 (DPO on eval_slice)  ── nền tảng DPO; build dpo_pairs trước.
  exp35 (REDI negative)      ── requires exp18.
  exp38 (RL→SFT ordering)    ── requires exp18 + exp29 pairs.
  exp36 (LogicPuzzleRL DPO)  ── requires exp29 + pass@20 ≥ 2%. XL effort.

[Tier 4 — high risk, chạy cuối]
  exp37 (GeoRA init)         ── architecture mismatch Mamba/MoE; feasibility check trước.
```

---

## 4. Per-exp — lệnh + cổng falsification

Mỗi exp theo cùng data-flow batch-3 nếu có đổi corpus:
`edit upstream → reasoning.py → corpus.py → pack_kaggle_snapshot.py --out kaggle_snapshot_expN → upload → set EXPN_CORPUS → modal run experiments/expN.py`.
Các exp chỉ đổi loss (không đổi corpus) thì sửa `train_sft.py` Cfg().loss_config rồi mini-run.

### exp33 — VCORE key-token weighting  *(Tier 1, bắt đầu ở đây)*
1. Đọc `enhance_cot/vcore/` README để lấy công thức weight; quyết định 3-tier:
   prompt 0.0 / boilerplate 0.5 / operator-decision 2.0 / `\boxed{}`+`</think>` 3.0.
2. Sửa `corpus.py` reasoning branch: mask int → float (đọc trace từng category để viết regex token).
3. Thêm `VCORELossConfig(CrossEntropyWithWeightingLossConfig)` (hoặc dùng thẳng nếu weights vào qua mask).
4. **Gate:** 200-step mini-run; gradient-norm token decision ≥ 2× boilerplate. Nếu bị normalize away → sai.

### exp30 — OXA exploration-aware SFT  *(Tier 1)*
1. Build `compute_corpus_logprobs.py` (đọc `train_sft.py` load corpus format) → forward pass GPU →
   `corpus_logprobs.jsonl` (avg per-example logprob của adapter 0.86).
2. `OXALossConfig(CrossEntropyWithWeightingLossConfig)`: promote low-logprob-correct (w=2.0),
   suppress high-conf (w=0.3). Map entry logprob → per-token weight trong `apply_weights()`.
3. **Gate:** 50-step eval_slice; cryptarithm KHÔNG giảm. Paper kỳ vọng +1–2.5pp.

### exp34 — Paraphrase traces  *(Tier 1, không đổi loss)*
1. `head -50 paraphrase_instances.py` → hiểu interface; thêm mode xử lý `reasoning/*.txt`.
2. Paraphrase → `reasoning_paraphrased/` (KHÔNG overwrite `reasoning/`).
3. **Verify gate CỨNG:** extract `\boxed{}` → `compare_answer(original, paraphrased)` pass; token ≤ 7600.
   Fail → giữ original. Pre-check: paraphrase 100 trace, nếu logprob không tăng > 0.1 → abandon.
4. Rebuild corpus từ `reasoning_paraphrased/` → pack → run. Cost ~$6–10 Claude Haiku.

### exp31 — Procedural cryptarithm + RFT  *(Tier 2, cần 2.A)*
1. Viết `generators/cryptarithm_procedural.py` (giống pattern exp28 generators — solver verify).
   Lưu ý: cryptarithm solver chỉ làm concat 5-char; "procedural cryptarithm" thật khó hơn → đọc kỹ.
2. **Falsify TRƯỚC khi build corpus:** rollout 200 bài, `--temperature 0.5 --n_samples 10`;
   pass@10 < 2% → ABORT (không đủ signal).
3. Giữ correct completions (RFT) → add corpus → pack → run.

### exp32 — AdaSTaR adaptive sampling  *(Tier 2, cần 2.A cho eval)*
1. Đọc `enhance_cot/adastar/utils_adastar.py` (weight update formula, framework-agnostic).
2. Sửa `train_sft.py` `_stratified_batches()` nhận per-cat weights + `compute_category_accuracy()`
   gọi eval_slice mỗi `ADASTAR_EVAL_INTERVAL=200` steps → upweight cryptarithm, downweight gravity/numeral.
3. **Gate:** macro eval_slice cuối ≥ baseline; cryptarithm tăng.

### exp29 — Iterative DPO on eval_slice  *(Tier 3, cần 2.A + 2.B + exp18)*
1. `infer_slice.py ... --temperature 0.5 --n_samples 10` → `preds_exp29.jsonl`.
2. Build `build_dpo_pairs.py` (template `build_eval_slice.py`): mỗi bài cần ≥1 correct + ≥1 incorrect
   (verify `compare_answer`) → `{prompt, chosen, rejected}`. < 20 pairs → ABORT.
3. DPO 50 steps qua engine ở 2.B (LR=1e-5, beta=0.1, SFT-mix 30%).

### exp35 — REDI negative-trace  *(Tier 3, requires exp18)*
1. Greedy infer trên 1,167 rule_unknown → `negatives_exp35.jsonl`.
2. `REDILossConfig` REINFORCE-style đẩy logprob âm token sai (đọc `enhance_cot/redi/rllm/`, KHÔNG port TRL thẳng).
3. **Gate:** sau 50-step, gravity/cipher KHÔNG giảm > 1pp (overcorrection → abort).

### exp38 — RL→SFT ordering  *(Tier 3, requires exp18 + exp29)*
1. Stage-A: DPO 50 steps (pairs từ exp29). **Gate cứng:** eval_slice ≥ 0.85 (không catastrophic forget);
   < 0.85 → tăng SFT-mix 20–30% retry.
2. Stage-B: SFT full corpus 1000 steps. Không vượt 0.86 → ordering hypothesis fails.

### exp36 — LogicPuzzleRL offline DPO  *(Tier 3, XL, requires exp29 + pass@20 ≥ 2%)*
1. vLLM `n=20 temperature=0.8` trên ~873 bài cryptarithm/equation_guess.
2. `build_dpo_pairs.py` threshold ≥1 correct + ≥1 incorrect; < 20 bài có pair → ABORT.
3. DPO 150 steps, SFT anchor 20%.

### exp37 — GeoRA LoRA init  *(Tier 4, chạy cuối)*
1. Feasibility check (plan §12): FIM memory budget + layer names Nemotron-H.
2. Chỉ apply GeoRA cho FFN + lm_head (SSM geometry không interpretable). Reset=False, continue 200–500 steps.

---

## 5. Verify trước khi chốt mỗi exp
```bash
cd nemotron-master
uv run --frozen ruff format *.py && uv run --frozen ruff check *.py
# mypy: hiện CHƯA cài trong uv env (uv run mypy → not found). Bỏ qua hoặc `uv add --dev mypy` nếu cần.
uv run pytest          # bỏ qua 2 fail có sẵn ở .claude/hooks (stop-validator, không liên quan pipeline)
```
Sau khi có score: `tracker/rounds/round_<N>.md` + append `tracker/leaderboard.md`.

---

## 6. Đường tới submit nhanh nhất (nếu chỉ muốn 1–2 exp)
exp33 (VCORE) hoặc exp30 (OXA) — rủi ro thấp, dùng `CrossEntropyWithWeightingLossConfig` có sẵn,
KHÔNG cần DPO engine / sampling patch / exp18. Kết hợp với corpus exp28 (đã pack) là lựa chọn ROI cao nhất.
```
corpus exp28 (9625, in-distribution) + VCORE float-mask  →  candidate submit
```
