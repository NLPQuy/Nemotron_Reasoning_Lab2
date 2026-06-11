# exp29 trên Kaggle — 5 ô dán chạy thẳng

DPO tinh chỉnh adapter 0.86 (TRL). Mỗi block dưới = **một cell** trong Kaggle GPU notebook
(RTX PRO 6000). Chạy lần lượt. Chỉ cần điền `ADAPTER_PATH` (và `BASE` nếu base mount khác).

**Trước khi chạy:** Add 2 Kaggle dataset vào notebook:
1. Thư mục `nemotron-master/` đã cập nhật (có `build_dpo_pairs.py`, `train_dpo_trl.py`, `infer_slice.py` đã vá, `reasoners/`, `eval_slice.jsonl`).
2. Adapter 0.86 (dir có `adapter_config.json`).
Bật **GPU** + (nếu được) **Internet** cho cell cài `trl`. Nếu internet off → add thêm dataset chứa wheel `trl`/`peft` rồi `pip install` từ path.

---

### Ô 1 — Setup (điền ADAPTER_PATH ở đây)
```python
import os, shutil, subprocess

# ── ĐIỀN 3 đường dẫn này ──────────────────────────────────────────────
SRC_NM      = "/kaggle/input/<nemotron-master-dataset>/nemotron-master"  # ← dataset (1)
ADAPTER_PATH = "/kaggle/input/<adapter-086-dataset>"                     # ← dataset (2), dir có adapter_config.json
# BASE PHẢI là path model ĐÃ MOUNT (internet OFF → add model làm dataset/Kaggle Models).
# KHÔNG để HF id "nvidia/..." khi offline — sẽ lỗi resolve huggingface.co.
BASE = "/kaggle/input/<nemotron-base-dataset>"  # ← dataset (3), dir có config.json + tokenizer + modeling_*.py
# ──────────────────────────────────────────────────────────────────────

# Offline: chặn gọi HuggingFace Hub (khỏi retry resolve huggingface.co).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
assert os.path.isdir(BASE) and os.path.isfile(os.path.join(BASE, "config.json")), \
    f"BASE phải là dir model đã mount (có config.json): {BASE}"

WORK = "/kaggle/working/nm"        # bản ghi-được (input là read-only)
if os.path.exists(WORK):
    shutil.rmtree(WORK)
shutil.copytree(SRC_NM, WORK)
os.chdir(WORK)
print("cwd:", os.getcwd())

# TRL + peft (vllm/transformers/torch đã có trong image cuộc thi).
# Nếu internet off: thay bằng `!pip install --no-index --find-links /kaggle/input/<wheels> trl peft`
get_ipython().system("pip install -q trl peft datasets")

assert os.path.isfile("eval_slice.jsonl"), "Thiếu eval_slice.jsonl trong dataset nemotron-master"
assert os.path.isdir(ADAPTER_PATH) and os.path.isfile(os.path.join(ADAPTER_PATH, "adapter_config.json")), \
    "ADAPTER_PATH phải là dir có adapter_config.json"
print("Setup OK. ADAPTER_PATH =", ADAPTER_PATH)
```

---

### Ô 2 — [Bước 1] Sinh rollouts (GPU): 170 bài × 10 mẫu, temperature 0.5
```python
get_ipython().system(
    f"cd {WORK} && python infer_slice.py "
    f"--base {BASE} --adapter {ADAPTER_PATH} "
    f"--slice eval_slice.jsonl --out preds_exp29.jsonl "
    f"--temperature 0.5 --n_samples 10 --seed 0"
)
```

---

### Ô 3 — [Bước 2] Ghép cặp DPO (CPU). G1: tự ABORT nếu < 20 cặp
```python
rc = subprocess.run(
    ["python", "build_dpo_pairs.py",
     "--slice", "eval_slice.jsonl", "--preds", "preds_exp29.jsonl",
     "--out", "dpo_pairs_exp29.jsonl", "--holdout_n", "30", "--seed", "7"],
    cwd=WORK,
)
assert rc.returncode == 0, "G1 FAIL: < 20 cặp → rollout pass-rate không hợp, DỪNG exp29."
print("G1 PASS. Đã tạo dpo_pairs_exp29.jsonl + eval_holdout_ids.txt")
```

---

### Ô 4 — [Bước 3] DPO: smoke 1 bước (G2) rồi train thật 50 bước
```python
# G2: smoke 1 bước / 2 cặp — kiểm Nemotron-H (Mamba/MoE) × TRL trước khi đốt GPU full.
rc = subprocess.run(
    ["python", "train_dpo_trl.py", "--base", BASE, "--adapter", ADAPTER_PATH,
     "--pairs", "dpo_pairs_exp29.jsonl", "--out", "adapter_exp29_smoke", "--smoke"],
    cwd=WORK,
)
assert rc.returncode == 0, "G2 FAIL: TRL không chạy được trên Nemotron-H → xem fallback plan §7-R1."
print("G2 PASS. Bắt đầu DPO thật...")

# DPO thật: 50 bước, LR 5e-7, beta 0.1 → lưu adapter_exp29/
get_ipython().system(
    f"cd {WORK} && python train_dpo_trl.py "
    f"--base {BASE} --adapter {ADAPTER_PATH} "
    f"--pairs dpo_pairs_exp29.jsonl --out adapter_exp29 "
    f"--beta 0.1 --lr 5e-7 --max_steps 50"
)
```

---

### Ô 5 — [Bước 4+5] Chấm holdout: adapter MỚI vs baseline 0.86 → verdict G3/G4
```python
import json

# Greedy trên eval_slice cho cả 2 adapter (n_samples mặc định = 1 → schema {id, output}).
for tag, adp in [("baseline", ADAPTER_PATH), ("exp29", f"{WORK}/adapter_exp29")]:
    get_ipython().system(
        f"cd {WORK} && python infer_slice.py --base {BASE} --adapter {adp} "
        f"--slice eval_slice.jsonl --out preds_{tag}.jsonl"
    )

# Tính accuracy CHỈ trên 30 id holdout (leak-free) bằng grader thật.
import sys
sys.path.insert(0, WORK)
from reasoning import compare_answer, extract_answer  # noqa: E402

hold = set(open(f"{WORK}/eval_holdout_ids.txt").read().split())
recs = {json.loads(l)["id"]: json.loads(l)
        for l in open(f"{WORK}/eval_slice.jsonl") if l.strip()}

def holdout_acc(preds_file):
    preds = {json.loads(l)["id"]: json.loads(l)
             for l in open(preds_file) if l.strip()}
    c = t = 0
    for pid in hold:
        t += 1
        p = preds.get(pid)
        if not p:
            continue
        if compare_answer(recs[pid]["answer"], extract_answer(p.get("output", ""))):
            c += 1
    return c, t

cb, tb = holdout_acc(f"{WORK}/preds_baseline.jsonl")
ce, te = holdout_acc(f"{WORK}/preds_exp29.jsonl")
ab, ae = cb / tb, ce / te
print(f"Holdout baseline(0.86): {cb}/{tb} = {ab*100:.1f}%")
print(f"Holdout exp29(DPO):     {ce}/{te} = {ae*100:.1f}%")
print("G3 (không thoái hoá, ae>=ab):", "PASS" if ae >= ab else "FAIL → ship 0.86")
print("G4 (cải thiện, ae>ab):     ", "PASS → dùng adapter_exp29" if ae > ab else "neutral → ship 0.86")
```

---

### Ô 6 — Đóng gói submission.zip (chạy nếu G4 PASS)
```python
import os, json, zipfile

SAVE_DIR = f"{WORK}/adapter_exp29"   # G4 FAIL → dùng submission.zip của 0.86 thay vì cái này
SUBMISSION = "/kaggle/working/submission.zip"

adapter_files = [f for f in os.listdir(SAVE_DIR) if f.startswith("adapter")]
assert "adapter_config.json" in adapter_files
with open(os.path.join(SAVE_DIR, "adapter_config.json")) as f:
    cfg = json.load(f)
assert cfg.get("r") == 32, f"LoRA rank phải = 32, đang là {cfg.get('r')}"

# adapter* nén ở GỐC zip (arcname = tên file, không thư mục con).
with zipfile.ZipFile(SUBMISSION, "w", zipfile.ZIP_DEFLATED) as zf:
    for fn in adapter_files:
        zf.write(os.path.join(SAVE_DIR, fn), fn)
print(f"Wrote {SUBMISSION} (r={cfg['r']}): {adapter_files}")
print("→ Nộp file này lên Kaggle. G4 FAIL thì nộp lại submission.zip của 0.86.")
```

---

**Ghi chú:**
- Thứ tự cổng: G1 (ô3) → G2 (ô4 smoke) → G3/G4 (ô5). Cổng nào FAIL thì dừng, không chạy ô sau.
- Nếu ô4 smoke lỗi (rủi ro #1: Nemotron-H × TRL) → xem fallback ở `research/ideation/plan-exp29.md` §7-R1 (Tinker `dro` hoặc Unsloth DPO).
- Output cần lấy về: `adapter_exp29/` (nếu G4 PASS) — đây là adapter nộp.
```
