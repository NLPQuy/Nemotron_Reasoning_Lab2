# exp35 (REDI negative-trace) trên Kaggle — runbook

REDI = Reinforcement Distillation: học từ **cả trace đúng (chosen) lẫn trace sai (rejected)** —
preference loss DPO giữa (đúng, sai) **CỘNG** một SFT term trên trace đúng. Trong TRL điều này =
`DPOTrainer` với `rpo_alpha > 0` (SFT-mix trên chosen). Đã thêm knob `--rpo_alpha` vào
`nemotron-master/train_dpo_trl.py` (mặc định 0.0 → DPO thường, KHÔNG đụng exp29).

> ⛔ **CỔNG CHẶN (đọc trước):** exp35 cưỡi trên đường DPO/TRL. **PHẢI chạy exp29 trước và G2 smoke
> PASS** (TRL chạy được trên Nemotron-H Mamba/MoE). Nếu G2 FAIL → DPO không hoạt động ở đây →
> **bỏ exp35** (xem fallback Tinker `dro` trong plan-exp29 §7-R1). exp18 (preference viable) cũng là
> tiền đề mềm.

> ⚠️ **Khác stub gốc:** stub nói "negatives cho 1,167 rule_unknown". Nhưng rule_unknown = solver
> KHÔNG verify được đáp án → KHÔNG có chosen đúng → không lập được cặp. REDI đúng cần bài **CÓ đáp án**:
> rollout nhiều mẫu → verify `compare_answer` → đúng=chosen, sai=rejected. Tập giàu negative nhất =
> category yếu (cryptarithm, equation_numeric_guess) nơi model sai nhiều.

---

## 0. Dataset attach (giống exp29)
1. `nemotron-master/` đã cập nhật (có `train_dpo_trl.py` đã thêm `--rpo_alpha`, `build_dpo_pairs.py`,
   `build_eval_slice.py`, `infer_slice.py` vá hf-cache, `reasoning.py`).
2. Adapter 0.86 (dir có `adapter_config.json`).
3. Base model đã mount (path LOCAL, KHÔNG HF id khi offline).
4. Wheels `trl`/`peft`/`datasets` (offline: add dataset wheel rồi `pip --no-index`).

---

## 1. Bước chạy (mỗi block = 1 cell, theo mẫu exp29_kaggle.md)

### Ô 1 — Setup: như exp29 (điền SRC_NM / ADAPTER_PATH / BASE, `HF_HUB_OFFLINE=1`, copy ra /kaggle/working/nm).

### Ô 2 — Build slice category-yếu (CPU)
```python
# Slice tập trung bài hay sai (nhiều negative). build_eval_slice lấy per-category;
# tăng --n để có nhiều bài cryptarithm/equation_guess.
get_ipython().system(f"cd {WORK} && python build_eval_slice.py --n 80 --seed 11")
# → eval_slice.jsonl (đổi tên nếu muốn tách khỏi slice exp29). Lọc 2 category yếu nếu cần:
import json
rows = [json.loads(l) for l in open(f"{WORK}/eval_slice.jsonl") if l.strip()]
hard = [r for r in rows if r["category"] in ("cryptarithm_guess","cryptarithm_deduce","equation_numeric_guess")]
open(f"{WORK}/redi_slice.jsonl","w").write("".join(json.dumps(r)+"\n" for r in hard))
print("redi_slice:", len(hard))
```

### Ô 3 — Rollout sinh negatives (GPU): nhiều mẫu, temperature > 0
```python
get_ipython().system(
    f"cd {WORK} && python infer_slice.py --base {BASE} --adapter {ADAPTER_PATH} "
    f"--slice redi_slice.jsonl --out preds_exp35.jsonl --temperature 0.8 --n_samples 16 --seed 0"
)
# (infer_slice cần patch sampling --temperature/--n_samples — chung blocker exp29; nếu chưa có thì
#  thêm như plan §2.A trước.)
```

### Ô 4 — Lập cặp REDI (CPU). ABORT nếu < 20 cặp
```python
import subprocess
rc = subprocess.run(["python","build_dpo_pairs.py","--slice","redi_slice.jsonl",
    "--preds","preds_exp35.jsonl","--out","redi_pairs_exp35.jsonl",
    "--holdout_n","20","--min_pairs","20","--seed","7"], cwd=WORK)
assert rc.returncode == 0, "G1 FAIL: < 20 cặp (đúng+sai) → model sai chưa đủ đa dạng. DỪNG."
```

### Ô 5 — REDI = DPO + SFT-mix. Smoke (G2) rồi train thật
```python
# G2 smoke — BẮT BUỘC (Nemotron-H × TRL). Nếu fail → bỏ exp35.
rc = subprocess.run(["python","train_dpo_trl.py","--base",BASE,"--adapter",ADAPTER_PATH,
    "--pairs","redi_pairs_exp35.jsonl","--out","adapter_exp35_smoke","--rpo_alpha","1.0","--smoke"], cwd=WORK)
assert rc.returncode == 0, "G2 FAIL: TRL/REDI không chạy trên Nemotron-H → bỏ exp35."

# REDI thật: 100 bước, LR 5e-6, beta 0.1, rpo_alpha 1.0 (SFT-mix ~ 'SFT 30%' của paper).
get_ipython().system(
    f"cd {WORK} && python train_dpo_trl.py --base {BASE} --adapter {ADAPTER_PATH} "
    f"--pairs redi_pairs_exp35.jsonl --out adapter_exp35 "
    f"--beta 0.1 --lr 5e-6 --max_steps 100 --rpo_alpha 1.0"
)
```

### Ô 6 — Chấm holdout + falsification
```python
# Greedy cả 2 adapter trên redi_slice → acc trên eval_holdout_ids.txt (leak-free) bằng grader thật.
# (Tái dùng đúng code Ô5 của exp29_kaggle.md, đổi tên file.)
```
**Cổng falsification (theo stub):**
- **HARD:** gravity/cipher accuracy KHÔNG giảm > 1pp vs baseline 0.86 → nếu giảm = REDI
  overcorrect → giảm `--lr` hoặc tăng `--rpo_alpha`.
- cryptarithm KHÔNG tăng > 0pp → REDI signal quá yếu → abandon.

---

## 2. Ghi chú
- `rpo_alpha` = trọng số SFT term trên chosen (TRL native). 0.0 = DPO thuần (exp29). 1.0 ≈ REDI cân
  bằng preference + SFT. Sweep {0.5, 1.0, 2.0} nếu G2 ok nhưng LB phẳng.
- Reference model = base adapter-disabled (PEFT default, `ref_model=None`) → KL anchor về chính nó.
- Chỉ train; KHÔNG đụng inference/`compare_answer`/`extract_answer`.
- Có điểm → `tracker/rounds/round_4.md` + `tracker/leaderboard.md`. DỪNG trước submit, hỏi user.
- Phụ thuộc: sampling patch cho `infer_slice.py` (chung exp29/31/36) + exp29 G2 pass.
```
