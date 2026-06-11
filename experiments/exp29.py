# %% [markdown]
# # EXP29 — Iterative DPO on eval_slice rollouts  (Batch-4 Idea 9)
# ============================================================
# Notebook Kaggle 5 ô — dán/chạy thẳng trong Kaggle GPU notebook (RTX PRO 6000).
# Plan chi tiết: research/ideation/plan-exp29.md
#
# Engine: HuggingFace TRL DPOTrainer. Continue-train CHÍNH adapter 0.86 → artifact vẫn 1
# LoRA rank-32 (hợp lệ submission). Reference = base (LoRA tắt); kiểm soát trôi bằng
# LR nhỏ (5e-7) + 50 bước + cổng holdout — KHÔNG dùng ref=0.86 (nạp 2 model 30B vượt 96GB).
#
# TRƯỚC KHI CHẠY — add 2 Kaggle dataset:
#   (1) thư mục nemotron-master/ đã cập nhật (build_dpo_pairs.py, train_dpo_trl.py,
#       infer_slice.py đã vá, reasoners/, eval_slice.jsonl)
#   (2) adapter 0.86 (dir có adapter_config.json)
# Bật GPU + (nếu được) Internet cho ô cài trl. Internet off → add dataset wheel rồi pip --no-index.
#
# Cổng dừng: G1 (ô3, <20 cặp) → G2 (ô4 smoke) → G3/G4 (ô5). FAIL ở đâu thì dừng ở đó.
# ============================================================

# %% [code]
# ───────────────────────── Ô 1 — Setup (ĐIỀN ADAPTER_PATH) ─────────────────────────
import json
import os
import shutil
import subprocess
import sys

# ── ĐIỀN 3 đường dẫn này ──────────────────────────────────────────────
SRC_NM = "/kaggle/input/<nemotron-master-dataset>/nemotron-master"  # ← dataset (1)
ADAPTER_PATH = "/kaggle/input/<adapter-086-dataset>"  # ← dataset (2), dir có adapter_config.json
# BASE PHẢI là path model ĐÃ MOUNT (Kaggle competition internet OFF → add model làm
# dataset/Kaggle Models). KHÔNG để HF id "nvidia/..." khi offline — sẽ lỗi resolve huggingface.co.
BASE = "/kaggle/input/<nemotron-base-dataset>"  # ← dataset (3), dir có config.json + tokenizer
# ──────────────────────────────────────────────────────────────────────

# Offline: chặn mọi lượt gọi HuggingFace Hub (khỏi retry resolve huggingface.co).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# BASE phải là thư mục cục bộ khi offline.
assert os.path.isdir(BASE) and os.path.isfile(os.path.join(BASE, "config.json")), (
    f"BASE phải là dir model đã mount (có config.json), đang là: {BASE}"
)

WORK = "/kaggle/working/nm"  # bản ghi-được (input là read-only)
if os.path.exists(WORK):
    shutil.rmtree(WORK)
shutil.copytree(SRC_NM, WORK)
os.chdir(WORK)
print("cwd:", os.getcwd())

# Backend sinh text: "hf" = transformers.generate (KHÔNG cần vllm — chạy được ngay trên
# image này), "vllm" = nhanh hơn nhưng cần vllm wheel tương thích (sm_120/torch). Image
# Kaggle hiện KHÔNG có vllm → để "hf". Nếu add được vllm wheel thì đổi "vllm".
BACKEND = "hf"

# Cài deps. Image đã có peft/transformers/torch/accelerate/datasets; chỉ thiếu trl.
# Offline → cài từ wheels dataset:
WHEELS = "/kaggle/input/datasets/llkh0a/rtx-wheels/wheels"  # ← đổi thành dataset wheels của bạn
get_ipython().system(  # noqa: F821
    f"pip install -q --no-index --find-links {WHEELS} trl"
    + (f" vllm" if BACKEND == "vllm" else "")
)

assert os.path.isfile("eval_slice.jsonl"), "Thiếu eval_slice.jsonl trong dataset nemotron-master"
assert os.path.isdir(ADAPTER_PATH) and os.path.isfile(
    os.path.join(ADAPTER_PATH, "adapter_config.json")
), "ADAPTER_PATH phải là dir có adapter_config.json"
print(f"Setup OK. BACKEND={BACKEND}  ADAPTER_PATH={ADAPTER_PATH}")

# %% [code]
# ──────────── Ô 2 — [Bước 1] Sinh rollouts (GPU): 170 bài × N_SAMPLES mẫu, temp 0.5 ────────────
# ⚠️ THỜI GIAN: backend "hf" rất chậm. 170 × N_SAMPLES × MAX_NEW token có thể vượt giới hạn
# notebook Kaggle (9–12h). Hai cần gạt để vừa thời gian:
#   - N_SAMPLES nhỏ hơn → ít cặp hơn nhưng nhanh hơn.
#   - MAX_NEW thấp (rollout KHÔNG cần đủ 7680; trace dài bị cắt → tính là "sai" → vẫn dùng
#     làm rejected). Eval ở Ô 5 vẫn dùng 7680 đầy đủ (đúng contract).
# Nếu chạy được "vllm" thì để N_SAMPLES=10, MAX_NEW=7680 (nhanh, khỏi lo).
N_SAMPLES = 10 if BACKEND == "vllm" else 6
MAX_NEW = 7680 if BACKEND == "vllm" else 3072
get_ipython().system(  # noqa: F821
    f"cd {WORK} && python infer_slice.py "
    f"--base {BASE} --adapter {ADAPTER_PATH} --backend {BACKEND} "
    f"--slice eval_slice.jsonl --out preds_exp29.jsonl "
    f"--temperature 0.5 --n_samples {N_SAMPLES} --max_new_tokens {MAX_NEW} --seed 0"
)

# %% [code]
# ──────────── Ô 3 — [Bước 2] Ghép cặp DPO (CPU). G1: ABORT nếu < 20 cặp ────────────
rc = subprocess.run(
    [
        "python",
        "build_dpo_pairs.py",
        "--slice",
        "eval_slice.jsonl",
        "--preds",
        "preds_exp29.jsonl",
        "--out",
        "dpo_pairs_exp29.jsonl",
        "--holdout_n",
        "30",
        "--seed",
        "7",
    ],
    cwd=WORK,
)
assert rc.returncode == 0, "G1 FAIL: < 20 cặp → rollout pass-rate không hợp, DỪNG exp29."
print("G1 PASS. Đã tạo dpo_pairs_exp29.jsonl + eval_holdout_ids.txt")

# %% [code]
# ──────────── Ô 4 — [Bước 3] DPO: smoke 1 bước (G2) rồi train thật 50 bước ────────────
# G2: smoke 1 bước / 2 cặp — kiểm Nemotron-H (Mamba/MoE) × TRL trước khi đốt GPU full.
rc = subprocess.run(
    [
        "python",
        "train_dpo_trl.py",
        "--base",
        BASE,
        "--adapter",
        ADAPTER_PATH,
        "--pairs",
        "dpo_pairs_exp29.jsonl",
        "--out",
        "adapter_exp29_smoke",
        "--smoke",
    ],
    cwd=WORK,
)
assert rc.returncode == 0, "G2 FAIL: TRL không chạy được trên Nemotron-H → xem plan §7-R1 (fallback)."
print("G2 PASS. Bắt đầu DPO thật...")

# DPO thật: 50 bước, LR 5e-7, beta 0.1 → lưu adapter_exp29/
get_ipython().system(  # noqa: F821
    f"cd {WORK} && python train_dpo_trl.py "
    f"--base {BASE} --adapter {ADAPTER_PATH} "
    f"--pairs dpo_pairs_exp29.jsonl --out adapter_exp29 "
    f"--beta 0.1 --lr 5e-7 --max_steps 50"
)

# %% [code]
# ──────────── Ô 5 — [Bước 4+5] Chấm holdout: adapter MỚI vs baseline 0.86 → G3/G4 ────────────
# Chỉ infer trên 30 bài HOLDOUT (leak-free + nhanh hơn ~5.6× so với cả 170). Greedy, đủ
# 7680 token (đúng contract). n_samples=1 → schema {id, output}.
hold = set(open(f"{WORK}/eval_holdout_ids.txt").read().split())
recs = {
    json.loads(line)["id"]: json.loads(line)
    for line in open(f"{WORK}/eval_slice.jsonl")
    if line.strip()
}
with open(f"{WORK}/eval_holdout.jsonl", "w") as f:
    for pid in hold:
        f.write(json.dumps(recs[pid]) + "\n")

for tag, adp in [("baseline", ADAPTER_PATH), ("exp29", f"{WORK}/adapter_exp29")]:
    get_ipython().system(  # noqa: F821
        f"cd {WORK} && python infer_slice.py --base {BASE} --adapter {adp} "
        f"--backend {BACKEND} --slice eval_holdout.jsonl --out preds_{tag}.jsonl"
    )

# Tính accuracy trên holdout bằng grader thật.
sys.path.insert(0, WORK)
from reasoning import compare_answer, extract_answer  # noqa: E402


def holdout_acc(preds_file: str) -> tuple[int, int]:
    preds = {
        json.loads(line)["id"]: json.loads(line)
        for line in open(preds_file)
        if line.strip()
    }
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
print(f"Holdout baseline(0.86): {cb}/{tb} = {ab * 100:.1f}%")
print(f"Holdout exp29(DPO):     {ce}/{te} = {ae * 100:.1f}%")
G3 = ae >= ab
G4 = ae > ab
print("G3 (không thoái hoá, ae>=ab):", "PASS" if G3 else "FAIL → ship 0.86")
print("G4 (cải thiện, ae>ab):     ", "PASS → dùng adapter_exp29" if G4 else "neutral → ship 0.86")

# %% [code]
# ──────────── Ô 6 — Đóng gói submission.zip (chỉ khi G4 PASS) ────────────
# submission.zip = các file adapter* nén ở GỐC zip (không thư mục con), phải có
# adapter_config.json. train_dpo_trl.py đã đổi tên lm_head -> backbone.lm_head khi save.
import zipfile

if not G4:
    print("G4 không PASS → KHÔNG đóng gói adapter_exp29. Nộp lại submission.zip của 0.86.")
else:
    SAVE_DIR = f"{WORK}/adapter_exp29"
    SUBMISSION = "/kaggle/working/submission.zip"

    adapter_files = [f for f in os.listdir(SAVE_DIR) if f.startswith("adapter")]
    assert "adapter_config.json" in adapter_files, "Thiếu adapter_config.json trong adapter_exp29/"

    # Kiểm tra rank-32 (ràng buộc cuộc thi).
    with open(os.path.join(SAVE_DIR, "adapter_config.json")) as f:
        _cfg = json.load(f)
    assert _cfg.get("r") == 32, f"LoRA rank phải = 32, đang là {_cfg.get('r')}"

    with zipfile.ZipFile(SUBMISSION, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in adapter_files:
            zf.write(os.path.join(SAVE_DIR, fn), fn)  # arcname = fn → ở gốc zip
    print(f"Wrote {SUBMISSION} (r={_cfg['r']}) với: {adapter_files}")
    print("→ Nộp file này lên Kaggle competition.")
