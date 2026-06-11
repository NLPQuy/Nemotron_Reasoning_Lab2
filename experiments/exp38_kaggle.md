# exp38 (RL→SFT ordering) trên Kaggle — runbook

Giả thuyết: **DPO trước (squeeze mode đúng) → SFT sau (mở rộng coverage)** tốt hơn SFT-only.
exp38 KHÔNG có thuật toán mới — nó là **orchestration** chuỗi 2 stage bằng công cụ đã có:
- **Stage-A** = DPO 50 bước từ adapter 0.86 (`train_dpo_trl.py` — y hệt exp29).
- **Stage-B** = SFT tiếp tục từ adapter Stage-A (`experiments/exp_continue.py`, RESET_WEIGHTS=False).

> ⛔ **CỔNG CHẶN:** giống exp35 — cần **exp29 G2 smoke PASS** (DPO chạy trên Nemotron-H) **và**
> `dpo_pairs_exp29.jsonl` đã build. exp18 (preference viable) là tiền đề mềm. Chưa có → defer exp38.
> ⚠️ Paper nghiên cứu RL **online**; ta dùng DPO **offline** → hiệu ứng squeeze yếu hơn → kỳ vọng dè dặt.

---

## 0. Prerequisite
- exp29 đã chạy tới Ô4 (có `dpo_pairs_exp29.jsonl`) và G2 PASS.
- Adapter 0.86 (dir), base mount LOCAL, wheels trl/peft (xem exp29_kaggle.md Ô1).

---

## 1. Stage-A — DPO squeeze (từ 0.86, SFT-mix THẤP để tối đa squeeze)
```python
# Bắt đầu từ 0.86 (KHÔNG từ adapter SFT khác). SFT-mix thấp = rpo_alpha nhỏ.
get_ipython().system(
    f"cd {WORK} && python train_dpo_trl.py --base {BASE} --adapter {ADAPTER_PATH} "
    f"--pairs dpo_pairs_exp29.jsonl --out adapter_stageA "
    f"--beta 0.1 --lr 1e-5 --max_steps 50 --rpo_alpha 0.1"
)
```

### Cổng Stage-A (BẮT BUỘC trước khi sang B)
```python
# Greedy eval_slice cho adapter_stageA; tính overall acc bằng grader thật (reuse Ô5 exp29).
# G-A: overall >= 0.85 (KHÔNG catastrophic forgetting).
print("G-A: nếu overall < 0.85 → Stage-A quá mạnh → tăng --rpo_alpha (0.1→0.2) hoặc giảm --lr, retry.")
print("     >= 0.85 → tiếp Stage-B.")
```

---

## 2. Stage-B — SFT expand (continue từ Stage-A, KHÔNG từ 0.86)

Stage-B = `experiments/exp_continue.py` (notebook Kaggle riêng, hoặc cùng kernel nếu đủ VRAM).
`adapter_stageA/` đã có `adapter_config.json` + `adapter_model.safetensors` (lm_head keys đã rename
sang `backbone.lm_head` bởi train_dpo_trl.py → load được trong trainer Unsloth).

Config đầu `exp_continue.py`:
```python
RESET_WEIGHTS = False
CONTINUE_ADAPTER_DIR = "/kaggle/working/nm/adapter_stageA"   # ← output Stage-A
CONTINUE_CORPUS = None              # corpus SFT chuẩn (21+22 hoặc exp28); None = default huikang
SHUFFLE_DATASET = True
NUM_STEPS = 1000                    # expand phase đầy đủ
LEARNING_RATE = 2e-4               # SFT LR chuẩn
```
> Lưu ý hand-off: nếu `exp_continue.py` (nhánh Kaggle) chỉ nhận adapter từ `submission.zip`, thì
> nén `adapter_stageA/` thành zip và trỏ path đó; còn `CONTINUE_ADAPTER_DIR` trỏ thẳng dir là gọn
> nhất — kiểm code load trong file trước khi chạy.

Chạy → `submission.zip` (adapter Stage-B, r=32).

---

## 3. Falsification cuối
- Submit adapter Stage-B. **score ≤ 0.86 → ordering KHÔNG có lợi (offline DPO)** → nộp lại best of
  {exp33, exp30, exp29, 0.86}.
- So 3 mốc: 0.86 (baseline) · SFT-only · DPO→SFT (exp38). exp38 chỉ thắng nếu finding của paper
  transfer sang offline.
- Tùy chọn entropy check (stub): sau Stage-A, entropy token trên eval_slice giảm > 5% vs SFT
  baseline → xác nhận DPO thật sự "squeeze". Không giảm → squeeze yếu → kỳ vọng exp38 phẳng.

---

## 4. Ghi chú
- exp38 = exp29 (Stage-A) + exp_continue (Stage-B) nối tiếp; KHÔNG file trainer mới.
- Hai stage có thể 2 kernel Kaggle (lấy `adapter_stageA/` làm dataset cho kernel Stage-B).
- Chỉ train; KHÔNG đụng inference/grader. DỪNG trước submit, hỏi user.
- Có điểm → `tracker/rounds/round_4.md` + `tracker/leaderboard.md` (ghi rõ Stage-A acc + Stage-B score).
```
