# exp31 redesign — cryptarithm rollout-free (bỏ RL/exp29)

Date: 2026-06-10. Mục tiêu: biến exp31 (procedural cryptarithm + RFT) thành **thuần data-time**,
KHÔNG model rollout / sampling / RFT / exp29. Mọi số dưới đây ĐO THẬT trên repo, không đoán.

## 0. Task thật của "cryptarithm" (đọc kỹ — không phải số học cổ điển)
Mỗi bài: vài ví dụ `XYopZW = OUT` (input 5 ký-tự = 2 toán hạng + 1 ký-tự-toán-tử + 2 toán hạng,
output là chuỗi) + 1 query 5 ký-tự. Symbol = **chữ số 0-9** mã hóa bí mật (per-problem); toán tử =
phép số học thật. Phải suy mapping + nghĩa toán tử từ ví dụ, áp cho query.
- `cryptarithm_deduce`: 959 bài, rule_found hiện **354 (36.9%)**.
- `cryptarithm_guess`: 164 bài, rule_found hiện **11 (6.7%)**.
- Tổng 1123, đang giải được 365, **chưa giải 758**. Alphabet thật: 26 ký-tự printable ASCII.

## 1. Phát hiện hạ tầng: ĐÃ CÓ 2 solver, production dùng cái yếu
- **Production** `reasoners/cryptarithm.py`: **CHỈ concat fwd/rev**. Giải được mọi bài có query
  op=concat (~37% bài có query concat → đó là nguồn 36.9% rule_found, KHÔNG phải số học).
- **Investigator** `investigators/cryptarithm_deduce.py`: **solver backtracking thật** — symbol→digit
  (CSP, unique/non-unique) + 5 phép `{add, abs_diff, mul, concat, rev_concat}`, suy mapping+op từ
  ví dụ, áp query. Đã emit luôn lời giải số học đầy đủ (lý tưởng làm CoT). **CHƯA wire vào production.**

## 2. Đo coverage thật của solver số học (số quyết định)
| Tập đo (150 mẫu, timeout 5s) | correct | wrong | failed | rate |
|---|---|---|---|---|
| Ngẫu nhiên toàn cryptarithm | 45 | 16 | 89 | **30%** |
| **CHỈ rule_unknown (758 bài khó)** | 9 | 33 | 108 | **6%** |

→ **30% bị thổi phồng** do trùng bài concat đã giải. **Incremental thật trên bài KHÓ chỉ ~6%**
(~45 bài mới / 758) và **22% ra đáp án SAI** → 758 bài khó **dùng luật NGOÀI họ 5 phép** (nghi: base≠10,
modular, hoán vị/thế ký tự, phép trên từng vị trí). Solver số học KHÔNG phải viên đạn bạc cho phần khó.

## 3. Kết luận research
1. **Cơ chế rollout-free TỒN TẠI và sạch** — gỡ hẳn RL/exp29/infer_slice (và OOM rollout). Xác nhận.
2. **Trần coverage bị chặn bởi HỌ PHÉP TOÁN**, không phải bởi việc thiếu generation. Thêm rollout cũng
   không giúp (model rollout trên bài khó cũng sai — RFT pass@10 sẽ rất thấp, đúng cảnh báo abort < 2%).
3. Đòn bẩy thật cho cryptarithm = **khám phá thêm họ luật** rồi vừa giải vừa generate chúng.

## 4. Redesign đề xuất (3 nhánh, tất cả rollout-free)

### A. Forward-construction generator — THAY RFT (rủi ro thấp, làm trước)
Không solve ngược; **dựng xuôi**: chọn ngẫu nhiên injective symbol→digit (10 ký-tự từ alphabet thật) +
gán mỗi op-symbol một phép trong họ; sinh toán hạng 2 chữ số ngẫu nhiên; tính kết quả; mã hóa lại →
**bài + đáp án đúng 100% by construction** (như exp20 gravity). Emit CoT số học xác định (tái dùng
format investigator). Unlimited, 0 model-call, in-distribution (cùng 5 phép + alphabet test dùng).
- File: `generators/cryptarithm_gen.py` + driver như `generate_instances.py` (đã có pattern exp20).
- Knob exp31: `EXP31_CORPUS` trỏ snapshot (trainer exp31.py ĐÃ sẵn).
- **Rủi ro: SATURATION.** Concat đã 37% rule_found; model có thể đã thạo 5 phép → thêm = phẳng
  (như gravity). → phải đo accuracy model TRƯỚC (xem §5).

### B. Promote solver số học vào production (rẻ, +~45 trace, làm kèm A)
Wire `solve_problem` vào nhánh cryptarithm của `reasoning.py` (sau khi concat fail), emit CoT từ
mapping/op_info. Verify gate (exp21) đã lọc 22% sai. Lợi: +~45 trace thật trên bài khó + làm solver
cho generator A. Chi phí: ~30s/bài backtracking → cần timeout + cache.

### C. Op-family discovery — đòn bẩy THẬT cho 758 bài khó (research-heavy, làm sau)
Phân tích 108 failed: thống kê (độ dài output vs phép, có carry không, output có ký-tự ngoài input
không → dấu hiệu base≠10 / thế ký-tự). Thêm phép vào `OPS[]` (vd: add mod 100, subtract có dấu,
digit-wise op, base-N). Mỗi phép mới mở thêm cả solve (B) lẫn generate (A). Đây mới là nơi coverage
758→ cao. Bắt đầu bằng `investigators/cryptarithm_deduce.py` + thêm op, đo lại rate trên rule_unknown.

## 5. Falsification (BẮT BUỘC trước khi build corpus)
**Câu hỏi gốc chưa trả lời: model HIỆN giỏi cryptarithm tới đâu?** rule_found-rate của solver ≠
accuracy model. Nếu model đã ~tốt trên 5-phép → nhánh A bão hòa.
1. Dựng eval-slice cryptarithm (đã có `build_eval_slice.py`) → `infer_slice.py` greedy n=1 (RẺ, đã vá
   `.ids` + `--gen_chunk`; n=1 không OOM) → đo accuracy per sub-rule.
2. **Gate A:** nếu model accuracy trên bài 5-phép đã ≥ ~80% → nhánh A bão hòa → BỎ, đi thẳng C.
   Nếu < 60% → A có dư địa → build generator.
3. **Gate C:** mỗi op mới phải nâng rate trên 150-mẫu rule_unknown > +3pp mới giữ.
4. Off-dist guard: generator A chỉ dùng 5 phép + alphabet ĐÃ thấy trong test (đừng bịa phép) →
   tránh lặp lại regress kiểu exp26/exp19.

## 6. So với exp31 gốc (RFT)
| | exp31 gốc (RFT) | Redesign |
|---|---|---|
| Sinh data | model rollout temp=0.5 n=10 (OOM, chậm, cần exp29) | solver/forward-construct (xác định, 0 GPU-gen) |
| Đáp án đúng | lọc pass@10 (bài khó pass<2% → abort) | đúng 100% by construction |
| Phụ thuộc | infer_slice + exp29 path | KHÔNG |
| Trần | giới hạn bởi model tự giải được | giới hạn bởi họ phép ta express được (mở rộng ở C) |

## 7. Bước làm (đề xuất)
1. **§5 falsification trước** (đo model accuracy cryptarithm — rẻ, n=1 greedy). Quyết A có đáng không.
2. Nếu A đáng: `generators/cryptarithm_gen.py` (forward-construct 5 phép) + B (promote solver) →
   `reasoning.py`+`corpus.py`+`pack_kaggle_snapshot.py` → snapshot → exp31 EXP31_CORPUS → train.
3. C (op-discovery) là round sau, ROI cao nhất nhưng cần phân tích failed cases.
- Liên quan: [[project_corpus_two_paths]] (in-dist phải qua reasoning/generators), exp20 gravity gen
  (cùng pattern forward-construct), [[feedback_experiment_lessons]] (off-dist regress).
