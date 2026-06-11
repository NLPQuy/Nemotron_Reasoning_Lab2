# Round 2 — Batch-2 sweep (exp11–exp19)

**Date**: 2026-06-01
**Score**: best = **0.83** (exp12) — **không exp nào vượt baseline 0.86**
**Δ vs previous best**: −0.03 đến −0.18 (toàn bộ hồi quy) + 2 ERROR

---

## Hypothesis

Batch-2 ([batch-2.md](../../research/ideation/batch-2.md)) đề xuất 9 idea mới (không trùng batch-1) để leo 0.86 → 0.88+: đổi parameterization/init LoRA (DoRA, PiSSA), regularizer (NEFTune, anchored-KL), data-time (self-verify traces, arithmetic scratchpad, stream-of-search), và aggregation (soup, preference). Mỗi idea là một `exp<N>.py` standalone.

## Config changes

```python
# exp11: USE_DORA = True                      → ERROR (adapter ~500MB)
# exp12: NEFTUNE_ALPHA = 10.0                  → 0.83
# exp13: VERIFY_TRACES_CORPUS = <regenerated>  → 0.68
# exp14: LORA_INIT = "pissa_niter_4"           → ERROR (Unsloth validate_loftq_config)
# exp15: SCRATCHPAD_CORPUS = <regenerated>     → pending
# exp16: KL_ANCHOR_BETA = 0.1                  → chưa chạy
# exp17: SOUP_SEED sweep + soup_adapters.py    → chưa chạy
# exp18: PAIRS_PATH + SimPO                    → chưa chạy
# exp19: SOS_CORPUS = <regenerated>            → 0.79
```

## Training run

| Field | Value |
|-------|-------|
| Platform | Kaggle |
| GPU | RTX PRO 6000 Blackwell (sm_120, 102 GB) |
| Base | Nemotron-3-Nano-30B-A3B bf16 |
| Corpus | 7830 examples, 27.8M tokens (default) hoặc regenerated (exp13/15/19) |
| Steps / LR | mặc định (NUM_STEPS=1000, LR=2e-4) |

## Result

| exp | Idea | Score | Δ vs 0.86 | Trạng thái |
|-----|------|-------|-----------|-----------|
| exp11 | DoRA | ERROR | — | Adapter 500MB, thiếu tensor |
| exp12 | NEFTune | 0.83 | −0.03 | Hồi quy |
| exp13 | self-verify traces | 0.68 | −0.18 | Hồi quy nặng |
| exp14 | PiSSA init | ERROR | — | Unsloth không hỗ trợ |
| exp15 | arithmetic scratchpad | 0.84 | — | — |
| exp16 | anchored-KL | chưa chạy | — | — |
| exp17 | LoRA soup | chưa chạy | — | — |
| exp18 | preference SimPO | chưa chạy | — | — |
| exp19 | stream-of-search | 0.79 | −0.07 | Hồi quy |

## Insights

### Bức tranh tổng: baseline 0.86 đang ở "đỉnh nhọn"
Mọi exp hoàn tất đều **dưới** baseline, nhiều cái rớt sâu. Đây không phải nhiễu — nó nói rằng corpus/format hiện tại đã được tinh chỉnh sát, và **bất kỳ thay đổi nào làm lệch phân phối trace hoặc thêm nhiễu đều bị grader exact-match phạt nặng**. Hệ quả quan trọng cho quy trình: phải có **eval slice held-out cục bộ** trước khi submit (xem mục cuối). Round này đốt 5 lần submit toàn âm vì bỏ qua bước "validate trên slice" mà chính batch-1/batch-2 đã dặn.

### Phân tích từng idea

**exp14 PiSSA — ERROR (chẩn đoán chắc chắn):** Unsloth `get_peft_model` → `validate_loftq_config` chỉ chấp nhận `init_lora_weights ∈ {True, False, "gaussian", "loftq", "corda"}`. `"pissa"`/`"pissa_niter_4"` bị reject thẳng. → **Cải tiến rõ ràng & rẻ nhất batch:** đổi sang `init_lora_weights="corda"` — CorDA (NeurIPS 2024) cũng là init dựa SVD/data-aware cùng họ với PiSSA và **được Unsloth hỗ trợ native**. Đây là salvage ưu tiên #1. (Phương án 2: bỏ Unsloth `get_peft_model`, dựng `peft.LoraConfig(init_lora_weights="pissa")` thủ công + `peft.get_peft_model` — nặng hơn, dễ vỡ với MoE.)

**exp11 DoRA — ERROR (adapter ~500MB, thiếu tensor):** kích thước nhẹ bất thường ⇒ adapter lưu ra **thiếu các slice expert** (bình thường phải emit 128 bản copy per-expert do `MOE_TIE_WEIGHTS`). Nguyên nhân nhiều khả năng: DoRA thêm `lora_magnitude_vector` đổi cấu trúc param → bước **emit 128 expert copies + rename `backbone.lm_head`** lúc save không bắt được các key DoRA, nên adapter bị cắt cụt. DoRA + manual lm_head LoRA + MoE-tie-emit là tổ hợp quá giòn. → **Cải tiến:** hoặc (a) chỉ bật DoRA cho module **không phải expert** (attention/mlp dày + lm_head), tắt cho `mixer`/expert; hoặc (b) bỏ DoRA cho kiến trúc này. Trước khi chạy lại **bắt buộc kiểm tra số tensor & tổng size adapter so với baseline**, và vLLM load-test (cổng đã ghi trong plan nhưng bị bỏ qua).

**exp12 NEFTune — 0.83 (−0.03):** đúng y cảnh báo devil's-advocate trong batch-2.md. Noise trên embedding cải thiện chất lượng hội thoại nhưng **phá việc tái tạo token chính xác** (số học, binary string) — chính là thứ grader chấm. → Kết luận: **drop**. Nếu muốn thử lại, chỉ `alpha≤5` và gần như chắc vẫn net-âm trên task này; ưu tiên thấp.

**exp19 stream-of-search — 0.79 (−0.07):** dạy backtracking làm model sinh trace **dài, lan man, tự "thử-sai"** lúc greedy → hoặc vượt budget 7680 (truncate trước `\boxed`) hoặc tự dẫn mình tới đáp án sai. Có thể SoS đã áp rộng hơn cryptarithm/cipher hoặc cap nhánh chưa đủ chặt. → **Cải tiến nếu quay lại:** gate cực chặt theo category, cap số nhánh, và A/B trên slice trước. Hiện deprioritize.

**exp13 self-verify — 0.68 (−0.18, tệ nhất):** dạy "verify rồi mới box" phản tác dụng mạnh. Hai cơ chế hại cộng dồn: (1) **answer-flipping** — model bắt chước *hình thức* verify trên bài lạ, sinh check sai rồi **đổi đáp án đúng thành sai**; (2) **length inflation** — verify tail đẩy `\boxed` ra xa, tăng truncate. SFT thuần (không RL) học hành vi self-correct rất dễ lệch. → **Cải tiến nếu quay lại:** verify **không bao giờ được phép đổi giá trị đã chốt** (chốt box trước, verify chỉ xác nhận), cap độ dài cứng, gate category, và đo flip-rate trên slice. Hiện deprioritize.

### Mẫu hình xuyên suốt (rút ra cho batch sau)
1. **Data-time đổi nội dung trace = rủi ro cao, dễ âm** (exp13, exp19): đỉnh nhọn không chịu được lệch phân phối. Idea batch-2 #3/#5/#9 cần A/B slice bắt buộc, hoặc gác lại.
2. **Regularizer thêm nhiễu (exp12) hại exact-match** — khác hẳn trực giác từ benchmark hội thoại.
3. **Idea parameterization/init (exp11/14) đụng tường tooling** Unsloth + MoE-save, không phải sai về ý tưởng. exp14→`corda` là cửa sống sạch nhất.
4. **Các lever "an toàn" chưa chạy mới là hy vọng:** exp17 (soup — thuần giảm variance, **không** đổi phân phối, gần như chỉ có lợi), exp16 (anchored-KL — regularizer *kéo về* base, bảo thủ). Nên ưu tiên 2 cái này + exp14-corda.

## Next actions (ưu tiên)
1. **[Bắt buộc, làm trước mọi submit] Dựng eval slice held-out cục bộ** (vLLM greedy, ~200 bài đa category) để chấm trước khi submit. Round này âm 5 lần vì thiếu nó.
2. **exp14 → `init_lora_weights="corda"`** (salvage rẻ nhất, init họ-SVD được hỗ trợ). Chạy lại.
3. **exp17 LoRA soup**: train 3–5 member khác seed, average; thuần giảm variance, ít rủi ro đổi phân phối. (Nhớ cast bf16 + đóng submission.zip — `soup_adapters.py` hiện chưa làm 2 việc này.)
4. **exp16 anchored-KL** với β nhỏ — nhưng **trước hết khử rủi ro OOM** (đang 2 forward full-logits/micro-batch): dùng top-k logits hoặc giảm `MICRO_BATCH_SIZE`.
5. **exp11 DoRA**: chỉ thử lại nếu giới hạn DoRA ngoài expert/mixer + thêm assert kiểm tra size/tensor-count adapter; nếu không, bỏ.
6. **Deprioritize exp12/exp13/exp19** trừ khi có slice-eval và guard (no answer-flip, hard length cap, category gate).
7. **exp15 (pending)** + **exp18 (preference)**: chỉ chạy sau khi có slice-eval; exp18 cần `pairs.jsonl` sinh offline (xem [plan-batch-2.md](../../research/ideation/plan-batch-2.md)).

## Status

- [x] Submitted (exp12, exp13, exp19; exp11/exp14 lỗi)
- [x] Result recorded in leaderboard.md
- [ ] exp15 result
- [ ] exp16/17/18 chạy
