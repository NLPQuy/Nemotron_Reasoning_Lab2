# 15 Thí nghiệm Cải thiện — NVIDIA Nemotron Reasoning Challenge (v2)

> **Baseline hiện tại**: 0.61/1 (SFT với template CoT, LoRA r=32, 1 epoch, seq_len=1024)  
> **Mục tiêu**: 0.75/1  
> **Ràng buộc**: Không internet, không API, không distillation từ model mạnh.  
> Tất cả phải chạy offline trên Kaggle với GPU Blackwell duy nhất.

---

## Bối cảnh Nghiên cứu — Xu hướng Top-Tier Conferences (2024–2026)

Các hướng nghiên cứu chi phối tại ICLR, NeurIPS, ICML, ACL giai đoạn 2024–2026 cho reasoning improvement **không cần teacher model**:

| Hướng | Paper tiêu biểu | Venue | Ý tưởng cốt lõi |
|---|---|---|---|
| **Self-Taught Reasoning (STaR)** | Zelikman et al. "STaR: Bootstrapping Reasoning With Reasoning" | NeurIPS 2022 | Model tự sinh rationale → filter chỉ giữ rationale dẫn đến đáp án đúng → SFT trên đó → lặp lại |
| **ReST / ReST-EM** | Gulcehre et al. "Reinforced Self-Training" | NeurIPS 2023 | Iterative: sample từ model → reward filter → fine-tune lặp lại, giống EM algorithm |
| **Rejection Sampling FT (RFT)** | Yuan et al. "Scaling Relationship on Learning Math Reasoning with LLMs" | ICML 2023 (workshop) | Sample K solutions/prompt → giữ correct → SFT. Đơn giản nhưng cực hiệu quả |
| **GRPO** | Shao et al. (DeepSeek) "DeepSeekMath" → DeepSeek-R1 | 2024–2025 | RL không cần reward model, chỉ cần verifiable outcome. Group-relative advantage estimation |
| **Process Reward Models (PRM)** | Lightman et al. "Let's Verify Step by Step" / Math-Shepherd (Wang et al.) | ICLR 2024 / NeurIPS 2024 | Đánh giá từng bước reasoning, không chỉ kết quả cuối. Auto-annotate bằng Monte Carlo rollouts |
| **Self-Rewarding LMs** | Yuan et al. | ICML 2024 | Model tự đánh giá output của chính nó → dùng làm preference data → DPO |
| **Quiet-STaR** | Zelikman et al. | 2024 | Model học "nghĩ thầm" trước mỗi token, cải thiện reasoning implicitly |
| **Iterative DPO / Self-Play** | Rosset et al. "Direct Nash Optimization" / SPIN (Chen et al.) | NeurIPS 2024 | Model tạo negative samples từ chính nó → DPO/KTO iterations |
| **NEFTune** | Jain et al. | ICLR 2024 | Noise embedding cải thiện generalization, trivial to implement |
| **Long-Context Reasoning** | Multiple papers on extended CoT | 2025–2026 | Training với longer sequences + explicit reasoning tokens budget |

**Insight quan trọng**: Trend lớn nhất 2025–2026 là **self-improvement loop** — model tự sinh data, tự đánh giá, tự cải thiện — hoàn toàn không cần teacher bên ngoài.

---

## Phân tích Điểm yếu Hiện tại (giữ nguyên)

| Vấn đề | Chi tiết |
|---|---|
| **MAX_SEQ_LEN = 1024** | Inference cho phép **7680 tokens**, training chỉ 1024 → model không học suy luận dài |
| **CoT traces giả/template** | Template cứng 4-5 dòng, không phải reasoning thực sự |
| **1 epoch duy nhất** | Chưa đủ hội tụ |
| **Không validation** | Không biết checkpoint nào tốt nhất |
| **Không packing** | Lãng phí compute |

---

## 15 Thí nghiệm (Revised — Offline Only)

---

### Nhóm A: Self-Generated Data & Bootstrapped Reasoning

#### TN1: STaR — Self-Taught Reasoner (Bootstrapped CoT) 🔴 CRITICAL

> *Ref: Zelikman et al., NeurIPS 2022; Hosseini et al., "V-STaR", 2024*

- **Core idea**: Thay vì dùng template CoT cứng nhắc, để chính Nemotron-3-Nano-30B tự sinh reasoning traces, rồi chỉ giữ những traces nào dẫn đến đáp án đúng.
- **Pipeline**:
  1. Load base model (hoặc SFT checkpoint hiện tại đạt 0.61)
  2. Với mỗi prompt trong train set, sample K=8 responses (temperature=0.7)
  3. Parse `\boxed{}` từ mỗi response, so sánh với ground truth
  4. **Filter**: chỉ giữ responses correct → đây là "gold" CoT
  5. Nếu tất cả K responses đều sai → dùng **rationalization**: nhét answer vào prompt hint, generate lại ("Given the answer is X, explain step by step why")
  6. SFT trên bộ data mới này
  7. **Lặp lại** 2-3 iterations (mỗi iteration model tốt hơn → sinh CoT tốt hơn)
- **Tại sao hiệu quả**: CoT traces đến từ chính distribution của model → không bị distribution mismatch như khi distill từ model khác. Model học reasoning patterns mà nó *có khả năng* tái tạo.
- **Code sketch**:
  ```python
  for iteration in range(3):
      traces = []
      for prompt, answer in train_data:
          responses = model.generate(prompt, num_return_sequences=8, temperature=0.7)
          correct = [r for r in responses if extract_boxed(r) == answer]
          if correct:
              traces.append((prompt, correct[0]))  # best correct trace
          else:
              # Rationalization: hint the answer
              hint_prompt = f"{prompt}\nThe answer is {answer}. Explain step by step why."
              rationalized = model.generate(hint_prompt, num_return_sequences=4, temperature=0.5)
              correct_r = [r for r in rationalized if extract_boxed(r) == answer]
              if correct_r:
                  traces.append((prompt, correct_r[0]))
      # SFT on new traces
      model = sft_train(model, traces)
  ```
- **Expected gain**: +0.05–0.10
- **Difficulty**: Medium-Hard (cần generation loop trước training)

#### TN2: Rejection Sampling Fine-Tuning (RFT) 🔴 CRITICAL

> *Ref: Yuan et al., ICML 2023 workshop; DeepSeek-Math, 2024*

- **Core idea**: Đơn giản hóa STaR — chỉ 1 round: sample nhiều, giữ correct, SFT.
- **Pipeline**:
  1. Từ SFT model hiện tại (0.61), sample K=16 responses/prompt
  2. Filter: giữ tất cả correct responses (không chỉ 1)
  3. Ưu tiên responses có CoT trace dài hơn, chi tiết hơn
  4. SFT trên collected data
- **Key insight**: Nhiều correct solutions cho 1 bài → model học diverse reasoning paths
- **So sánh với TN1**: Đơn giản hơn (1 round), nhưng TN1 (iterative) thường tốt hơn
- **Expected gain**: +0.03–0.06
- **Difficulty**: Medium

#### TN3: Self-Play Preference Optimization (SPIN) 🟡 HIGH

> *Ref: Chen et al. "Self-Play Fine-Tuning Converts Weak LMs to Strong LMs", ICML 2024*

- **Core idea**: Tạo preference data từ chính model, không cần model mạnh hơn.
- **Pipeline**:
  1. SFT model hiện tại sinh responses cho train set
  2. **Chosen** = ground truth response (từ training data)
  3. **Rejected** = model's own response (nếu sai)
  4. Train DPO trên preference pairs này
  5. Lặp lại: model mới sinh responses mới → preference data mới → DPO
- **Tại sao hay**: Model liên tục học phân biệt giữa output đúng vs output sai của *chính nó*
- **Expected gain**: +0.02–0.05
- **Difficulty**: Medium

#### TN4: Self-Rewarding Iterative DPO 🟡 HIGH

> *Ref: Yuan et al. "Self-Rewarding Language Models", ICML 2024*

- **Core idea**: Model tự chấm điểm output của mình → tạo preference pairs → DPO
- **Pipeline**:
  1. Sample K=4 responses/prompt
  2. Model tự evaluate mỗi response: "Is this solution correct? Rate 1-5"
  3. Dùng self-evaluation scores + ground truth answer check để rank responses
  4. Tạo pairs (high-score response, low-score response) → DPO
- **Ưu điểm so với TN3**: Không chỉ dựa vào final answer, còn dùng model's judgment về quality
- **Expected gain**: +0.02–0.04
- **Difficulty**: Medium-Hard

---

### Nhóm B: Reinforcement Learning với Verifiable Rewards

#### TN5: GRPO — Group Relative Policy Optimization 🔴 CRITICAL

> *Ref: Shao et al. "DeepSeekMath", 2024; DeepSeek-R1, 2025*

- **Core idea**: RL thuần không cần reward model, chỉ cần kiểm tra đáp án đúng/sai. Đây là phương pháp đã tạo ra DeepSeek-R1 — breakthrough lớn nhất 2025.
- **Pipeline**:
  1. Bắt đầu từ SFT checkpoint (TN1 hoặc TN2 output)
  2. Với mỗi prompt, sample G=4–8 responses
  3. Reward function (không cần neural network):
     - `r = 1.0` nếu `extract_boxed(response) == ground_truth`
     - `r += 0.1` nếu response chứa `\boxed{}` (format reward)
     - `r += 0.05` nếu response chứa `<think>` (reasoning format)
     - `r = 0.0` otherwise
  4. Compute group-relative advantage: `A_i = (r_i - mean(r_group)) / std(r_group)`
  5. Policy gradient update với KL penalty to SFT reference
- **Tại sao đây là game-changer**: Model learns to *search* for correct answers, không chỉ *imitate* training data
- **Code sketch**:
  ```python
  from trl import GRPOConfig, GRPOTrainer
  
  def reward_fn(completions, prompts, ground_truths):
      rewards = []
      for comp, gt in zip(completions, ground_truths):
          r = 0.0
          extracted = extract_boxed(comp)
          if extracted == gt:
              r = 1.0
          if "\\boxed{" in comp:
              r += 0.1
          if "<think>" in comp:
              r += 0.05
          rewards.append(r)
      return rewards
  
  grpo_config = GRPOConfig(
      num_generations=4,
      learning_rate=5e-6,    # Much lower than SFT
      per_device_train_batch_size=1,
      gradient_accumulation_steps=8,
      num_train_epochs=2,
      max_completion_length=4096,
      bf16=True,
      kl_coef=0.05,
  )
  ```
- **Expected gain**: +0.04–0.08
- **Difficulty**: Hard (cần generation trong training loop, memory-intensive)

#### TN6: Outcome-based RL với Mixed Reward 🟡 HIGH

> *Ref: Lightman et al., ICLR 2024; Uesato et al., 2022*

- **Core idea**: Reward function phong phú hơn TN5, vẫn không cần neural reward model
- **Reward design**:
  ```
  r_correctness = 1.0 if answer correct else 0.0
  r_format      = 0.2 if \boxed{} present else -0.1
  r_reasoning   = 0.1 if <think>...</think> present else 0.0
  r_length      = -0.001 * max(0, num_tokens - 2048)  # penalize unnecessarily long
  r_consistency = 0.1 if answer appears ≥2 times in trace (self-check)
  
  r_total = r_correctness + r_format + r_reasoning + r_length + r_consistency
  ```
- **Key insight**: Reward shaping giúp model học nhanh hơn, tránh reward hacking
- **Expected gain**: +0.01–0.03 (on top of TN5)
- **Difficulty**: Medium

---

### Nhóm C: Training Configuration Optimization

#### TN7: Tăng MAX_SEQ_LEN + Packing 🔴 CRITICAL

- **Vấn đề**: Train ở 1024 tokens nhưng inference dùng 7680. Model chưa bao giờ thấy sequence dài.
- **Phương pháp**:
  ```python
  MAX_SEQ_LEN = 4096   # hoặc 2048 nếu OOM
  packing = True        # combine short samples → tiết kiệm compute
  ```
- **Kết hợp**: Với STaR/RFT traces (dài tự nhiên hơn template), seq_len dài giúp giữ nguyên reasoning chain
- **Tradeoff**: Chậm hơn, nhưng packing bù lại đáng kể
- **Expected gain**: +0.03–0.06
- **Difficulty**: Easy (chỉ thay hyperparameters)

#### TN8: Multi-Epoch + Validation + Best Checkpoint 🟡 HIGH

- **Phương pháp**:
  ```python
  NUM_EPOCHS = 3
  # Tách 5% data làm validation
  train_test = hf_dataset.train_test_split(test_size=0.05, seed=42)
  
  training_args = SFTConfig(
      num_train_epochs=3,
      eval_strategy="steps",
      eval_steps=50,
      save_strategy="steps",
      save_steps=50,
      load_best_model_at_end=True,
      metric_for_best_model="eval_loss",
  )
  ```
- **Expected gain**: +0.02–0.04
- **Difficulty**: Easy

#### TN9: NEFTune + Gradient Noise 🟢 MEDIUM

> *Ref: Jain et al. "NEFTune: Noisy Embeddings Improve Instruction Finetuning", ICLR 2024*

- **Core idea**: Thêm noise vào embedding → regularization → better generalization
- **Phương pháp**:
  ```python
  training_args = SFTConfig(
      neftune_noise_alpha=5.0,   # Paper recommends 5-15
  )
  ```
- **Trivial to implement** — 1 dòng code, đã built-in trong TRL
- **Expected gain**: +0.01–0.02
- **Difficulty**: Trivial

---

### Nhóm D: Data Engineering (Offline)

#### TN10: Offline Pre-Downloaded Reasoning Datasets 🟡 HIGH

- **Core idea**: Upload reasoning datasets lên Kaggle Datasets TRƯỚC khi submit (hoàn toàn hợp lệ — nhiều top solutions dùng cách này)
- **Datasets đề xuất** (download trước, upload lên Kaggle):
  - `NovaSky-AI/Sky-T1_data_17k` — diverse reasoning traces
  - `TIGER-Lab/MathInstruct` — 260k math problems
  - `open-r1/OpenR1-Math-220k` — subset filtered
  - `di-dimitrov/mathwell` — word problems
  - Hoặc: Pre-generate STaR traces offline trên máy local, upload traces lên Kaggle
- **Pipeline**:
  1. Download datasets trên máy local (có internet)
  2. Filter & format theo competition chat template
  3. Upload lên Kaggle Dataset
  4. Training notebook reference offline dataset
- **Expected gain**: +0.03–0.06
- **Difficulty**: Easy-Medium

#### TN11: Data Augmentation — Rephrasing & Permutation 🟢 MEDIUM

- **Core idea**: Tăng diversity của training data mà KHÔNG cần thêm data ngoài
- **Phương pháp**:
  1. **Example permutation**: Mỗi puzzle có nhiều examples → shuffle thứ tự examples → tạo N variants
  2. **Input perturbation**: Thay đổi nhỏ formatting (spaces, newlines) → model robust hơn
  3. **Answer format diversity**: Cùng 1 answer, format nhiều kiểu: `\boxed{42}`, `\boxed{42.0}`, etc.
- **Không cần model generate** — chỉ string manipulation
- **Expected gain**: +0.01–0.03
- **Difficulty**: Easy

#### TN12: Curriculum Learning — Easy → Hard 🟢 MEDIUM

> *Ref: Xu et al. "A Survey on Curriculum Learning", 2020; recent work on difficulty-aware training*

- **Core idea**: Train model trên dữ liệu dễ trước, khó sau
- **Pipeline**:
  1. Chạy base model inference trên train set (greedy, temperature=0)
  2. Phân loại: Easy (model đúng), Medium (gần đúng), Hard (sai hoàn toàn)
  3. Epoch 1: chỉ train Easy + Medium
  4. Epoch 2: train Medium + Hard
  5. Epoch 3: train tất cả (hoặc chỉ Hard)
- **Variant**: Anti-curriculum — train Hard trước (có paper cho thấy works tốt hơn cho reasoning)
- **Expected gain**: +0.01–0.03
- **Difficulty**: Medium

---

### Nhóm E: Inference-Aligned & Advanced

#### TN13: Test-Time Prompt Format Exact Match 🟡 HIGH

- **Core idea**: Distribution mismatch giữa training prompt và inference prompt làm giảm accuracy
- **Phương pháp**:
  1. Nghiên cứu chính xác inference pipeline dùng prompt gì (system prompt? user format?)
  2. Đảm bảo training format **Y HỆT** inference format
  3. Kiểm tra: inference dùng `add_generation_prompt=True`? Có system prompt không?
  4. Nếu inference KHÔNG dùng system prompt → bỏ system prompt khỏi training
  5. Nếu inference dùng template khác → match template đó
- **Expected gain**: +0.01–0.03
- **Difficulty**: Easy (nhưng cần reverse-engineer inference code)

#### TN14: LoRA Config Optimization 🟢 MEDIUM

- **Hyperparameter sweep**:
  ```
  lora_alpha  ∈ {32, 64, 128}       # alpha/r = 1, 2, 4
  lora_dropout ∈ {0.0, 0.05}
  target_modules: "all-linear" vs {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
  LR ∈ {5e-5, 1e-4, 2e-4}
  ```
- Dùng validation set (TN8) để evaluate
- **Expected gain**: +0.01–0.02
- **Difficulty**: Easy

#### TN15: LoRA Adapter Merging / Soup 🟢 MEDIUM

> *Ref: Yadav et al. "TIES-Merging", NeurIPS 2023; Yu et al. "DARE", 2024*

- **Core idea**: Train nhiều adapters khác nhau, merge chúng lại
- **Pipeline**:
  1. Train Adapter A: SFT trên full data, LR=2e-4
  2. Train Adapter B: SFT trên full data, LR=1e-4 (or different seed)
  3. Train Adapter C: RFT data (từ TN2), LR=2e-4
  4. Merge:
     ```python
     from peft import PeftModel
     # Average weights hoặc TIES-Merge
     merged_weights = 0.4 * A + 0.3 * B + 0.3 * C
     ```
  5. Export merged adapter (rank vẫn ≤ 32)
- **Expected gain**: +0.01–0.03
- **Difficulty**: Medium

---

---

## Nhóm F: Decoding & Inference-Time Optimization (NEW — 2025-2026 Research)

### Ràng buộc quan trọng

Inference pipeline của cuộc thi là **CỐ ĐỊNH**:
```
temperature = 0.0      → Greedy decoding (không sampling)
top_p = 1.0
max_tokens = 7680      → Budget token tối đa
max_model_len = 8192
engine = vLLM
```

**Không thể thay đổi** decoding strategy lúc test. Do đó, mục tiêu của nhóm này là: **train model sao cho greedy decode cho kết quả tốt nhất có thể**, và **tận dụng tối đa 7680 token budget**.

---

### Bối cảnh Nghiên cứu — Inference/Decoding Optimization (Top-Tier 2025-2026)

| Hướng | Paper | Venue | Ý tưởng cốt lõi |
|---|---|---|---|
| **Test-Time Compute Scaling** | Snell et al. "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters" | NeurIPS 2024 / ICLR 2025 | Optimal allocation of compute budget: search (parallel sampling) vs revision (sequential refinement). Với greedy decode, revision path là khả thi duy nhất. |
| **Budget Forcing / s1** | Muennighoff et al. "s1: Simple Test-Time Scaling" | 2025 | Force model sử dụng hết thinking budget bằng cách thay `</think>` thành `"Wait"` khi model kết thúc sớm. Train model với explicit budget control. |
| **Journey Learning** | Qin et al. "O1 Replication Journey: A Strategic Progress Report" | 2024-2025 | Train trên cả failed reasoning paths cùng corrections, không chỉ correct-only data. Model học cách self-correct trong single pass. |
| **Distilling System 2 → System 1** | Yu et al. "Distilling System 2 into System 1" | NeurIPS 2024 | Train model sao cho single greedy pass = quality of multi-sample search+vote. Distill ensemble behavior INTO the model weights. |
| **Coconut (Continuous CoT)** | Hao et al. "Training LLMs to Reason in a Continuous Latent Space" | ICLR 2025 | Reasoning trong latent space thay vì token space. Mỗi "thought" là 1 hidden state, không cần decode ra text → nhiều computation per token hơn. |
| **Pause Tokens** | Goyal et al. "Think Before You Speak: Training LMs With Pause Tokens" | ICLR 2024 | Chèn `<pause>` tokens để model có thêm compute trước mỗi answer token. Cải thiện accuracy trên reasoning tasks. |
| **Quiet-STaR** | Zelikman et al. "Quiet-STaR: LMs Can Teach Themselves to Think Before Speaking" | 2024 | Model tự học chèn implicit reasoning tokens tại mỗi position. Tương tự Pause Tokens nhưng end-to-end learned. |
| **Thinking Tokens / Extended Thinking** | DeepSeek-R1, QwQ, Claude thinking | 2025-2026 | Model được train để generate `<think>...</think>` dài — deliberate reasoning trước khi trả lời. Budget 7680 tokens cho phép reasoning chains rất dài. |
| **Self-Consistency Distillation** | Huang et al. "Large LMs Can Self-Improve" | NeurIPS 2023; follow-ups 2024-2025 | Sample K responses, majority vote → train trên majority answer. "Distill" ensemble behavior vào single greedy pass. |
| **Beam Search Distillation** | Multiple works 2024-2025 | Various | Run beam search offline → train model output = beam search output. Greedy ≈ beam search after distillation. |
| **Adaptive Computation** | Raposo et al. "Mixture-of-Depths" / early exit works | ICML 2024 | Model allocates different compute per token/layer. Hard problems get more processing. |
| **Verify-then-Generate** | Weng et al. "Large LMs are Better Reasoners with Self-Verification" | 2024-2025 | Train model để verify own reasoning inline, tự detect errors trong single pass, tự correct. |

---

### TN16: Budget Forcing — Maximize Thinking Token Usage 🔴 CRITICAL

> *Ref: Muennighoff et al. "s1: Simple Test-Time Scaling", 2025*

- **Insight**: Model hiện tại (train ở 1024 tokens) có xu hướng trả lời ngắn. Nhưng inference cho phép 7680 tokens. Model đang "lãng phí" ~6600 tokens compute budget.
- **Core idea**: Train model LUÔN nghĩ dài trước khi trả lời. Sử dụng gần hết thinking budget.
- **Pipeline**:
  1. Tạo training data với CoT traces DÀI (từ STaR/RFT — TN1/TN2)
  2. Thêm explicit thinking budget vào prompt: `"Think step by step. Use at least 500 words of reasoning before answering."`
  3. **Budget forcing trick**: Trong training data, nếu trace quá ngắn → pad thêm `"Wait, let me reconsider..."` + additional analysis
  4. Train với `MAX_SEQ_LEN = 4096` hoặc cao hơn
  5. **Key**: Model học pattern "longer thinking = better answers"
- **Tại sao quan trọng**: Paper s1 cho thấy chỉ riêng budget forcing (force model think longer) đã improve accuracy 10-30% trên reasoning benchmarks.
- **Training data format**:
  ```
  <think>
  Let me analyze this problem carefully.
  
  Step 1: [detailed analysis of examples...]
  Step 2: [identify pattern...]
  Step 3: [verify pattern on all examples...]
  
  Wait, let me double-check my reasoning.
  [re-verification...]
  
  Step 4: [apply pattern to target...]
  </think>
  \boxed{answer}
  ```
- **Expected gain**: +0.04–0.08
- **Difficulty**: Medium

### TN17: Self-Consistency Distillation (Offline) 🔴 CRITICAL

> *Ref: Wang et al. "Self-Consistency Improves CoT Reasoning", ICLR 2023; Huang et al. NeurIPS 2023*

- **Insight**: Self-consistency (majority voting over K samples) outperforms greedy decoding by large margins. Nhưng test time CHỈCHO greedy. → **Distill majority-vote behavior vào greedy decode.**
- **Pipeline**:
  1. Từ SFT model, sample K=16 responses/prompt (temperature=0.7)
  2. **Majority vote**: Đếm frequency của mỗi answer → chọn answer phổ biến nhất
  3. **Chọn best trace**: Trong các responses có answer = majority answer, chọn trace chi tiết nhất
  4. SFT trên (prompt, best_trace_with_majority_answer) pairs
  5. → Model học greedy output = majority-voted answer
- **Tại sao mạnh**: Biến "K parallel samples → 1 greedy" mà KHÔNG mất accuracy. Paper gốc cho thấy majority voting improves 5-15% → distilling this back vào model recaptures most of that gain.
- **So sánh TN2 (RFT)**: RFT chỉ filter correct answers. TN17 chọn answer mà *model tin tưởng nhất* (highest consistency) → robust hơn khi ground truth label có noise.
- **Expected gain**: +0.03–0.06
- **Difficulty**: Medium

### TN18: Journey Learning — Train on Failures + Corrections 🟡 HIGH

> *Ref: Qin et al. "O1 Replication Journey", 2024-2025; follow-up works at ICLR 2025*

- **Insight**: Chỉ train trên correct solutions → model chưa bao giờ thấy errors → không biết self-correct. Journey learning train trên cả "wrong path → realization → correction → right answer".
- **Pipeline**:
  1. Sample responses từ SFT model
  2. Với mỗi response SAI: nối thêm correction
     ```
     <think>
     [model's original wrong reasoning...]
     
     Wait, I made an error in step 3. Let me reconsider.
     [corrected reasoning...]
     </think>
     \boxed{correct_answer}
     ```
  3. Training data = mix of:
     - 60% direct correct solutions (RFT traces)
     - 40% journey paths (wrong → correct)
- **Tại sao quan trọng cho greedy decode**: Tại inference, greedy decode có thể đi vào wrong path. Nếu model đã train trên journey paths, nó biết cách *phát hiện lỗi và tự sửa* trong cùng 1 generation pass.
- **Expected gain**: +0.02–0.05
- **Difficulty**: Medium

### TN19: Verify-then-Commit — Inline Self-Verification 🟡 HIGH

> *Ref: Weng et al. "Large LMs are Better Reasoners with Self-Verification", 2024; Shinn et al. "Reflexion", NeurIPS 2023*

- **Insight**: Thay vì generate 1 answer rồi submit, train model để **verify inline** trước khi commit.
- **Pipeline — Training data format**:
  ```
  <think>
  [Step-by-step reasoning...]
  My tentative answer is X.
  
  Let me verify: if the answer is X, does it satisfy all the examples?
  - Example 1: input → apply_rule(X) → expected_output ✓
  - Example 2: input → apply_rule(X) → expected_output ✓
  - Example 3: input → apply_rule(X) → expected_output ✗ ← MISMATCH
  
  The rule I found doesn't work for Example 3. Let me reconsider.
  [Revised reasoning...]
  
  New answer: Y.
  Verification: 
  - Example 1: ✓
  - Example 2: ✓  
  - Example 3: ✓
  All examples verified.
  </think>
  \boxed{Y}
  ```
- **Tạo training data**: 
  1. Lấy RFT correct traces
  2. Programmatically thêm verification block (check answer against examples trong prompt)
  3. Cho cả traces failed-then-corrected (nếu first attempt sai, verify catches it)
- **Tại sao quan trọng**: Model tự verify → giảm errors mà KHÔNG cần multiple samples → perfect cho greedy decode
- **Expected gain**: +0.02–0.04
- **Difficulty**: Medium

### TN20: Greedy-Aware Training — DPO on Greedy vs Sampled 🟡 HIGH

> *Ref: Yu et al. "Distilling System 2 into System 1", NeurIPS 2024*

- **Insight**: Model tốt khi sampling nhưng kém khi greedy. Vì training dùng teacher forcing (next token prediction) — không optimize cho autoregressive greedy behavior.
- **Pipeline**:
  1. Với mỗi prompt, generate:
     - Response A: **greedy decode** (temperature=0) — đây là response sẽ xảy ra lúc test
     - Response B: **best-of-K sampling** (temperature=0.7, K=8, pick correct one)
  2. Nếu greedy ĐÚNG: skip (đã tốt rồi)
  3. Nếu greedy SAI nhưng best-of-K ĐÚNG:
     - **Chosen** = best-of-K response (correct)
     - **Rejected** = greedy response (incorrect)
     - → DPO training pair
  4. Train DPO → model's greedy behavior dịch chuyển về phía best-of-K behavior
- **Tại sao đây là key insight**: Directly optimize for THE EXACT decoding strategy used at test time (greedy)
- **Expected gain**: +0.03–0.06
- **Difficulty**: Medium-Hard

### TN21: Thinking Token Insertion (Pause Tokens) 🟢 MEDIUM

> *Ref: Goyal et al. "Think Before You Speak", ICLR 2024; Quiet-STaR, 2024*

- **Insight**: Transformer reasoning power ∝ number of tokens generated. Thêm "pause" tokens → thêm computation layers trước mỗi answer.
- **Pipeline**:
  1. Thêm special tokens: `<|think|>` vào tokenizer
  2. Training data: chèn N=5-10 `<|think|>` tokens trước mỗi reasoning step
     ```
     <think>
     <|think|><|think|><|think|> Step 1: Analyze the pattern...
     <|think|><|think|><|think|> Step 2: The rule is...
     </think>
     \boxed{answer}
     ```
  3. Train model → at inference, model sinh `<|think|>` tokens (extra compute) rồi mới sinh reasoning
- **Tradeoff**: Tốn thêm tokens (trừ vào 7680 budget), nhưng mỗi token "chất" hơn
- **Expected gain**: +0.01–0.03
- **Difficulty**: Medium (cần modify tokenizer + training data)

### TN22: Structured Output — Multi-Pass Reasoning in Single Generation 🟢 MEDIUM

> *Ref: Yao et al. "Tree of Thoughts", NeurIPS 2024; Besta et al. "Graph of Thoughts", 2024*

- **Insight**: Thay vì 1 linear reasoning chain, train model sinh structured reasoning: multiple candidate answers → self-evaluate → pick best.
- **Training data format**:
  ```
  <think>
  ## Approach 1
  [Reasoning path A...]
  → Candidate answer: X
  
  ## Approach 2  
  [Different reasoning path B...]
  → Candidate answer: Y
  
  ## Evaluation
  Approach 1 gives X. Let me verify: [check against examples...] → 2/3 match
  Approach 2 gives Y. Let me verify: [check against examples...] → 3/3 match
  
  Approach 2 is more consistent. Going with Y.
  </think>
  \boxed{Y}
  ```
- **Core idea**: **Simulate majority voting WITHIN a single generation pass**. Model explores 2-3 approaches, picks the best one — all in 1 greedy decode.
- **Tại sao powerful**: Effectively "internal ensemble" mà không cần multiple samples. Tận dụng 7680 token budget.
- **Expected gain**: +0.02–0.05
- **Difficulty**: Medium-Hard (cần craft training data cẩn thận)

### TN23: Reward-Weighted Likelihood — Implicit Best-of-N 🟢 MEDIUM

> *Ref: Pang et al. "Iterative Reasoning Preference Optimization (IRPO)", NeurIPS 2024; Gulcehre et al. "ReST", 2023*

- **Insight**: Thay vì coi tất cả correct solutions equal, weight loss theo quality.
- **Pipeline**:
  1. Sample K=8 responses/prompt
  2. Score mỗi response:
     - `score = correctness (0/1) × reasoning_quality`
     - `reasoning_quality` = length of trace + number of verification steps + consistency
  3. Weighted SFT: `loss_i = -weight_i × log P(response_i | prompt)`
     - High-quality correct traces → high weight
     - Low-quality / barely-correct → low weight
     - Wrong → weight = 0 (excluded)
  4. Model learns to reproduce the *best* style of reasoning, not just any correct one
- **Expected gain**: +0.01–0.03
- **Difficulty**: Medium

---

## Execution Roadmap (Final — Offline Only, with Decoding Optimization)

```
Phase 1 — Quick Config Wins (1-2 ngày):
  ├── TN7:  MAX_SEQ_LEN 1024 → 4096 + packing
  ├── TN9:  NEFTune (1 dòng code)
  ├── TN8:  Multi-epoch (3 epochs) + validation split
  └── TN13: Prompt format matching
  → Expected: 0.61 → ~0.65-0.67

Phase 2 — Self-Generated Data + Budget Forcing (3-5 ngày):
  ├── TN2:  RFT — Rejection Sampling (đơn giản nhất, làm trước)
  ├── TN1:  STaR — Iterative bootstrapping (mạnh nhất)
  ├── TN16: Budget forcing — train model to think longer ⭐ NEW
  └── TN17: Self-consistency distillation ⭐ NEW
  → Expected: 0.67 → ~0.73

Phase 3 — Greedy-Aware RL (3-5 ngày):
  ├── TN5:  GRPO (game-changer)
  ├── TN20: Greedy-aware DPO (chosen=best-of-K, rejected=greedy) ⭐ NEW
  └── TN6:  Mixed reward tuning
  → Expected: 0.73 → ~0.77+

Phase 4 — Self-Correction & Polish (2-3 ngày):
  ├── TN18: Journey learning (wrong→correct paths) ⭐ NEW
  ├── TN19: Inline self-verification ⭐ NEW
  ├── TN22: Multi-approach within single generation ⭐ NEW
  └── TN15: Adapter merging
  → Expected: polish → 0.75+
```

---

## Bảng Tổng hợp

| # | Thí nghiệm | Hướng nghiên cứu | Priority | Expected Gain | Difficulty | Cần model generate? |
|---|---|---|---|---|---|---|
| **Nhóm A: Self-Improvement** ||||||
| 1 | STaR (Self-Taught Reasoner) | Self-improvement | 🔴 Critical | +0.05–0.10 | Medium-Hard | ✅ |
| 2 | Rejection Sampling FT (RFT) | Self-improvement | 🔴 Critical | +0.03–0.06 | Medium | ✅ |
| 3 | SPIN (Self-Play DPO) | Preference learning | 🟡 High | +0.02–0.05 | Medium | ✅ |
| 4 | Self-Rewarding DPO | Preference learning | 🟡 High | +0.02–0.04 | Medium-Hard | ✅ |
| **Nhóm B: RL** ||||||
| 5 | GRPO | Reinforcement learning | 🔴 Critical | +0.04–0.08 | Hard | ✅ (in-loop) |
| 6 | Mixed Reward RL | Reinforcement learning | 🟡 High | +0.01–0.03 | Medium | ✅ (in-loop) |
| **Nhóm C: Config** ||||||
| 7 | Tăng seq_len + packing | Training config | 🔴 Critical | +0.03–0.06 | Easy | ❌ |
| 8 | Multi-epoch + checkpoint | Training config | 🟡 High | +0.02–0.04 | Easy | ❌ |
| 9 | NEFTune | Regularization | 🟢 Medium | +0.01–0.02 | Trivial | ❌ |
| **Nhóm D: Data** ||||||
| 10 | Offline datasets upload | Data engineering | 🟡 High | +0.03–0.06 | Easy-Medium | ❌ |
| 11 | Data augmentation | Data engineering | 🟢 Medium | +0.01–0.03 | Easy | ❌ |
| 12 | Curriculum learning | Training strategy | 🟢 Medium | +0.01–0.03 | Medium | ❌ |
| **Nhóm E: Inference-Aligned** ||||||
| 13 | Prompt format matching | Inference alignment | 🟡 High | +0.01–0.03 | Easy | ❌ |
| 14 | LoRA config sweep | Hyperparameter | 🟢 Medium | +0.01–0.02 | Easy | ❌ |
| 15 | Adapter merging | Ensembling | 🟢 Medium | +0.01–0.03 | Medium | ❌ |
| **Nhóm F: Decoding/Inference Optimization** ⭐ NEW ||||||
| 16 | Budget Forcing (s1) | Test-time compute | 🔴 Critical | +0.04–0.08 | Medium | ✅ |
| 17 | Self-Consistency Distillation | Ensemble → greedy | 🔴 Critical | +0.03–0.06 | Medium | ✅ |
| 18 | Journey Learning | Self-correction | 🟡 High | +0.02–0.05 | Medium | ✅ |
| 19 | Inline Self-Verification | Verify-then-commit | 🟡 High | +0.02–0.04 | Medium | Partial |
| 20 | Greedy-Aware DPO | Greedy optimization | 🟡 High | +0.03–0.06 | Medium-Hard | ✅ |
| 21 | Pause/Thinking Tokens | Adaptive computation | 🟢 Medium | +0.01–0.03 | Medium | ❌ |
| 22 | Multi-Approach Single Gen | Internal ensemble | 🟢 Medium | +0.02–0.05 | Medium-Hard | Partial |
| 23 | Reward-Weighted SFT | Implicit best-of-N | 🟢 Medium | +0.01–0.03 | Medium | ✅ |

---

## Key Takeaway (Updated)

> **Xu hướng lớn nhất 2025–2026**: Self-improvement + Inference-aware training.
> 
> **Insight mới — Decoding Optimization**:
> Cuộc thi dùng **greedy decode** (temp=0). Hầu hết fine-tuning methods optimize cho next-token prediction (teacher forcing) — nhưng KHÔNG trực tiếp optimize cho autoregressive greedy generation. Nhóm F khắc phục gap này:
> - **TN16 (Budget forcing)**: Model sử dụng hết 7680 token budget → think dài hơn → đúng hơn
> - **TN17 (SC Distillation)**: Greedy output = quality of majority-voted output
> - **TN20 (Greedy-aware DPO)**: Trực tiếp optimize cho greedy behavior
> - **TN18/19 (Journey + Verification)**: Model tự sửa lỗi trong single pass
> - **TN22 (Multi-approach)**: "Internal ensemble" trong 1 generation
>
> **Top 5 thí nghiệm nếu chỉ chọn 5**:
> 1. **TN1 (STaR)** — self-bootstrapped reasoning traces
> 2. **TN16 (Budget Forcing)** — maximize thinking token usage  
> 3. **TN5 (GRPO)** — RL with verifiable reward
> 4. **TN17 (SC Distillation)** — distill ensemble → greedy
> 5. **TN20 (Greedy-Aware DPO)** — directly optimize greedy decode
>
> Kết hợp top 5 → estimated: **0.61 → 0.78+**
