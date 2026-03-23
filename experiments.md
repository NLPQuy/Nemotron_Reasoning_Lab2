# 15 Thí nghiệm Cải thiện — NVIDIA Nemotron Reasoning Challenge

> **Baseline hiện tại**: 0.61/1 (SFT với template CoT, LoRA r=32, 1 epoch, seq_len=1024)  
> **Mục tiêu**: 0.75/1  
> **Ràng buộc cứng**: Không internet, không API, không external data, không distillation từ model mạnh hơn.  
> Tất cả chạy offline trên Kaggle. Data synthesis dùng chính Nemotron-3-Nano-30B.

---

## Phân tích Điểm yếu Hiện tại

| Vấn đề | Chi tiết |
|---|---|
| **MAX_SEQ_LEN = 1024** | Inference cho phép **7680 tokens**, nhưng training chỉ 1024 → model không học suy luận dài |
| **CoT traces giả/template** | Chỉ 4-5 dòng template cứng, không phải reasoning thực sự — model học format, không học *tư duy* |
| **1 epoch duy nhất** | Chưa đủ để model hội tụ trên 9.5k samples |
| **Greedy decode không khớp training** | Inference dùng temperature=0 (greedy), training dùng teacher forcing — distribution mismatch |
| **Không có validation** | Không đo lường overfitting, không chọn checkpoint tốt nhất |
| **Không packing** | Lãng phí compute khi sequences ngắn hơn max_seq_len |

---

## Nhóm B: Training Configuration (Impact trung bình–cao)

### Thí nghiệm 5: Tăng MAX_SEQ_LEN lên 4096 🔴 CRITICAL

- **Vấn đề**: Training ở 1024 tokens, inference ở 7680 → distribution mismatch nghiêm trọng
- **Phương pháp**:
  ```python
  MAX_SEQ_LEN = 4096
  packing = True  # giữ throughput
  ```
  CoT traces cũng cần dài hơn (kết hợp với TN1 để có real CoT dài)
- **Tradeoff**: Chậm hơn ~3-4x, cần giảm gradient_accumulation hoặc dùng gradient checkpointing
- **Expected gain**: +0.03–0.06

### Thí nghiệm 6: Multi-Epoch Training + Best Checkpoint Selection 🟡 HIGH

- **Phương pháp**:
  ```python
  NUM_EPOCHS = 3
  save_strategy = "steps"
  save_steps = 100
  eval_strategy = "steps"
  eval_steps = 100
  load_best_model_at_end = True
  ```
  Tách 5–10% làm validation set, chọn checkpoint có val loss thấp nhất
- **Expected gain**: +0.02–0.04

---

## Nhóm C: Post-SFT Alignment (Impact trung bình–cao)

### Thí nghiệm 9: GRPO (Group Relative Policy Optimization) 🔴 CRITICAL

- **Phương pháp**: Sau SFT, apply GRPO:
  1. Với mỗi prompt, sample K=4–8 responses từ SFT model
  2. Reward function:
     - `+1.0` nếu extracted answer == ground truth
     - `+0.1` nếu có `\boxed{}` (format reward)
     - `0.0` otherwise
  3. Train 1–2 epochs GRPO với TRL's `GRPOTrainer`
- **Chi tiết triển khai**:
  ```python
  from trl import GRPOConfig, GRPOTrainer

  grpo_config = GRPOConfig(
      output_dir="./grpo_output",
      num_generations=4,
      per_device_train_batch_size=1,
      gradient_accumulation_steps=8,
      learning_rate=5e-6,  # much lower than SFT
      num_train_epochs=1,
      bf16=True,
  )
  ```
- **Expected gain**: +0.03–0.07

### Thí nghiệm 10: DPO (Direct Preference Optimization) 🟡 HIGH

- **Phương pháp**:
  1. Từ SFT model, sample 4 responses/prompt (temperature=0.7)
  2. Tạo preference pairs: (correct_response → chosen, wrong_response → rejected)
  3. Train DPO 1 epoch
- **Chi tiết triển khai**:
  ```python
  from trl import DPOConfig, DPOTrainer

  dpo_config = DPOConfig(
      output_dir="./dpo_output",
      beta=0.1,
      learning_rate=5e-6,
      num_train_epochs=1,
      bf16=True,
  )
  ```
- So sánh DPO vs GRPO, chọn phương pháp tốt hơn
- **Expected gain**: +0.02–0.05

---

## Nhóm D: Inference-Aligned Training (Impact trung bình)

### Thí nghiệm 11: System Prompt Engineering 🟢 MEDIUM

- **Phương pháp**: Thử nhiều system prompt variants:
  - Detailed: "Think carefully about EACH example pair. Find the EXACT transformation rule. Verify your rule on ALL examples before applying..."
  - Domain-specific: prompt khác nhau cho math vs logic vs pattern
  - Minimal: tiết kiệm token budget cho reasoning
  - **Quan trọng**: Match chính xác prompt mà inference server sẽ dùng
- **Expected gain**: +0.01–0.02

### Thí nghiệm 12: Answer Format Reinforcement 🟢 MEDIUM

- **Phương pháp**:
  - Thêm diverse format examples: `\boxed{42}`, `\boxed{hello world}`, `\boxed{0101}`
  - Đảm bảo model **luôn** output `\boxed{}` → evaluator extract chính xác thay vì dùng heuristic
  - Thêm 200-500 synthetic samples chỉ để drill format compliance
- **Expected gain**: +0.01–0.02

### Thí nghiệm 13: Test-Time Prompt Format Matching 🟡 HIGH

- **Phương pháp**: Reverse-engineer inference pipeline. Đảm bảo training format **y hệt** inference format:
  - Cùng system prompt (hoặc không có system prompt nếu inference không dùng)
  - Cùng chat template tokens (`<|im_start|>`, `<|im_end|>`, etc.)
  - Cùng instruction suffix ("Put your final answer inside \boxed{}")
  - Kiểm tra: inference server có dùng `add_generation_prompt=True` không?
- **Expected gain**: +0.01–0.03

---

## Nhóm E: Advanced Techniques

### Thí nghiệm 14: NEFTune (Noisy Embedding Fine-Tuning) 🟢 MEDIUM

- **Phương pháp**: Thêm noise vào embedding layer — đã chứng minh improve generalization:
  ```python
  training_args = SFTConfig(
      ...
      neftune_noise_alpha=5.0,  # SFTConfig supports this natively
  )
  ```
- Rất dễ implement (chỉ 1 dòng code), không có downside
- **Expected gain**: +0.01–0.02

### Thí nghiệm 15: Ensemble of LoRA Adapters (Multi-Task Merge) 🟢 MEDIUM

- **Phương pháp**:
  1. Train 3 LoRA adapters riêng biệt:
     - (a) Math-focused data subset
     - (b) Logic/pattern-focused data subset
     - (c) Full mixed data
  2. Merge bằng TIES-Merging hoặc DARE weight averaging
  3. Export merged adapter (vẫn giữ rank ≤ 32)
- **Chi tiết triển khai**:
  ```python
  from peft import PeftModel
  # Load base + adapter_a, merge, then add adapter_b with weight
  model = PeftModel.from_pretrained(base_model, "adapter_a")
  model.load_adapter("adapter_b", adapter_name="b")
  model.add_weighted_adapter(["default", "b"], weights=[0.6, 0.4], combination_type="ties")
  ```
- **Expected gain**: +0.01–0.03

---

