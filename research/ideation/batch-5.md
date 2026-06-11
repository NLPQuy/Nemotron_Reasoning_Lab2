# Batch 5 — Observation-driven (rút từ kết quả exp1–exp39, KHÔNG search ngoài)

> Nguồn: [tracker/leaderboard.md](../../tracker/leaderboard.md) + memory. Mọi hướng dưới đây
> suy ra TỪ các thí nghiệm đã chạy, không phải từ literature mới.
> Quyết định khung: quay về **corpus gốc full + cải thiện ở nguồn + retrain-from-scratch**
> (xem [[next-step-original-corpus]]). KHÔNG continue-train-on-mix, KHÔNG negatives, KHÔNG nén post-hoc.

## A. Các quy luật quan sát được (cross-experiment laws)

**L1 — Full corpus 0.86 là local-optimum bão-hoà-coverage: trừ / nén / reweight / negatives đều TỤT.**
Bằng chứng: exp6 (cắt hard-subset) 0.79; exp3 (concise) 0.58; exp33 (TokenSkip nén) 0.62;
exp29 (negatives sign=−1) 0.37. Mọi thao tác *lấy bớt hoặc bóp méo* corpus đều regress.

**L2 — Chỉ from-scratch trên corpus gốc tái tạo 0.86; mọi continue-train đã thử đều tụt (nhưng confound).**
Sự thật (đã verify config): baseline = from-scratch (`RESET_WEIGHTS=True`) trên corpus gốc = 0.86;
**exp21 CŨNG là from-scratch** (`RESET_WEIGHTS=True`, [exp21.py:20](../../exp21.py#L20)) + LoRA+ = 0.86 —
KHÔNG phải continue-train. Batch-4 (exp29–39, `RESET_WEIGHTS=False`, continue từ 0.86) tụt 0.37–0.74,
monotone theo độ-đổi-shape: exp29 (negatives) 0.37 < exp33 (thay trace bit) 0.62 < exp30/32/34 0.70 <
exp38/39 (RFT self-gen + shuffle) 0.72 < exp35 0.74.
⚠️ **Confound chưa gỡ:** mọi continue-train ĐỀU đồng thời đổi shape corpus ⇒ chưa tách bạch được
"continue-train có hại" vs "đổi shape có hại". **Không có datapoint dương nào cho continue-train.**
→ **Hệ quả cho batch-5:** user chọn continue-train từ 0.86 — đây là đường **chưa kiểm chứng, có rủi
ro**. Giả thuyết để nó sống: (a) delta = **trace SOLVER xác định** (cùng nguồn corpus 0.86, KHÔNG
self-gen/negatives), (b) giữ **order gốc** (không shuffle-mix), (c) **liều nhẹ** (LR ≤1e-5, ≤1 epoch).
Giả thuyết này HỢP LÝ nhưng CHƯA có bằng chứng — phải validate bằng control run (xem Regime).

**L3 — Đòn bẩy = (tỉ trọng category) × (tỉ lệ đang fail). bit_manip ≫ guess.**
Theo corpus stats: bit_manip 26.4%, gravity 13.9%, cipher 12% vs cryptarithm_deduce 1.0%,
equation_numeric_guess 2.0%, cryptarithm_guess 0.2%. bit có **219 zero** (p95 completion 7243/7680 —
truncation) trên 1602 bài ⇒ ~13.7% bit fail ⇒ fix được ≈ **0.137 × 0.26 ≈ 3.6pp tiềm năng**.
Giải sạch cryptarithm_guess (0.2%) chỉ ≈ 0.2pp. **Guess categories là bẫy low-leverage và đã fail 1
lần** ([[cryptarithm-unsolved-levers]]: "guess+brace dead ends"). Tiền nằm ở **bit truncation**.

**L4 — Tweak duy nhất giữ 0.86 là exp21 (LoRA+), tức optimization-side, KHÔNG đụng data.**
⇒ Lever an toàn = data-preserving / optimization-side. Lever nguy hiểm = data-subtractive.

**L5 — Coverage trên category hiếm CHƯA chứng minh dời được leaderboard.** Solver cryptarithm cho
+34 *internal solve-count* (54→85) nhưng leaderboard vẫn 0.86 ([[cryptarithm-solver-wired]]). Hoặc
gain đó chưa retrain đúng regime, hoặc category 1% không đủ dời điểm. ⇒ batch-5 phải có **control
run** để biết "coverage → score" có thật không trước khi đổ công vào category hiếm.

---

## B. Hướng batch-5 (ranked theo leverage × evidence)

### Regime (thay D0) — continue-train từ 0.86 + CONTROL bắt buộc
- **Quyết định user:** train tiếp từ adapter 0.86 (`RESET_WEIGHTS=False`), KHÔNG retrain-from-scratch.
- **CONTROL trước (rẻ, không bỏ được):** vì L2 cho thấy continue-train *chưa từng* được test sạch,
  chạy 1 run continue-train từ 0.86 trên **corpus GỐC full + order gốc, KHÔNG delta**, liều nhẹ
  (LR ≤1e-5, ≤1 epoch, `SHUFFLE_DATASET=False`). **Nếu giữ ~0.86 ⇒ continue-train per se an toàn**,
  gỡ confound, tự tin thêm delta. **Nếu đã tụt ⇒ chính continue-train là vấn đề** (không phải đổi
  shape) → buộc quay lại from-scratch hoặc giảm liều mạnh. Khác D0 cũ: đây là continue, không from-scratch.
- **Kỷ luật khi thêm delta (từ L2):** delta = **trace solver xác định** (không self-gen RFT/negatives);
  giữ **order gốc** (không shuffle-mix); liều nhẹ (LR ≤1e-5, ≤1 epoch, cân nhắc 5e-6).
- **Corpus mỗi run:** corpus GỐC full + delta — delta **override id cũ** (cùng problem_id) hoặc cộng
  id mới, phần còn lại nguyên vẹn.
- **Gate:** so 5-nhóm-mạnh sau mỗi run; tụt >0.5pp ⇒ giảm liều hoặc rollback delta.

### D1 — ⭐ Token-efficient bit_manip solver (headline, leverage cao nhất ~3.6pp)
- **Hypothesis (từ L3):** 219 bit-zero là do trace dài vượt 7680 ⇒ `\boxed` không xuất kịp. Compression
  post-hoc đã fail (exp3 0.58, exp33 0.62) NHƯNG cả hai bị nhiễu: exp3 nén-toàn-cục, exp33 nén-trên-mix
  (dính luôn L2-erosion). **Chưa ai test sạch: rút ngắn CoT bit Ở NGUỒN + full regenerate + retrain-scratch.**
- **Cơ chế:** viết lại CoT trong [nemotron-master/reasoners/bit_manipulation.py](../../nemotron-master/reasoners/bit_manipulation.py)
  cho ngắn **cấu trúc** (ký hiệu cột gọn, bỏ câu văn lặp lại) — giữ NGUYÊN mọi bước tính, chỉ bỏ prose →
  khác hẳn token-pruning (exp33) vốn xoá cả bước. Mục tiêu p95 completion < ~6000.
- **Expected:** tới ~3.6pp (cận trên) nếu phần lớn 219 zero là truncation thuần.
- **Risk:** dính lại lời nguyền exp3/33 (chất lượng reasoning giảm). Mitigate: chỉ bỏ prose, giữ đủ
  bước cột-bit; verify-by-solver mọi trace mới.
- **Falsify:** trên held-out bit dài, p95 completion ↓ **VÀ** solve-rate ↑. Nếu solve-rate bit ↓ → revert.

### D2 — Training-time keep-boxed-tail / skip-overflow (rẻ, ghép với D1)
- **Hypothesis:** truncation còn xảy ra LÚC TRAIN: [Continuer:175-176](../../Continuer_Nemotron_Notebook.py#L175-L176),
  [:219-221](../../Continuer_Nemotron_Notebook.py#L219-L221), [:243-245](../../Continuer_Nemotron_Notebook.py#L243-L245)
  cắt phải về 8192 ⇒ trace >8192 mất `\boxed` nhưng vẫn mask=1 ⇒ dạy model "reasoning dài rồi KHÔNG kết luận".
- **Cơ chế:** ở corpus build, hoặc **skip** trace tràn, hoặc **keep-boxed-tail** (cắt giữa reasoning, luôn chừa đuôi boxed).
- **Falsify:** đếm trước số example >8192. ~0 → bỏ. >0 → fix dạy model luôn xuất boxed.

### D3 — Đẩy tiếp cryptarithm_deduce — ❌ BỎ (2026-06-10, no-op ở mức corpus)
> Phase 3 đã chạy: solver skip-1 sinh trace cryptarithm GIỐNG HỆT cũ (per-id fresh==bitshort 300/300,
> bitshort==S_solver 823/823) → corpus_crypto==bitshort, exp46 ≡ exp44. Đã xoá exp46. Skip-1 không giải
> thêm trong corpus này (memory "+17" không hiện thực hoá). Ưu tiên thấp; điều tra sau nếu cần.
- **Hypothesis (từ memory):** robust skip-1 solver cho +17 internal — chỉ là cái win thật duy nhất ở
  cryptarithm. cryptarithm_deduce còn **169 chưa giải** (58 hyp + 111 unknown).
- **Code path:** mở rộng solver trong [nemotron-master/reasoners/](../../nemotron-master/reasoners/) đóng nốt 169 → regenerate → retrain-scratch.
- **Expected:** ≤ ~1pp (category 1%). Giá trị chính: **đo L5** — nếu leaderboard vẫn phẳng sau retrain
  ⇒ xác nhận coverage category-hiếm không dời điểm (kết luận quý dù âm).
- **Falsify:** rule_found 490→>600 sau regenerate; nếu leaderboard không nhích, deprioritize mọi category <2%.

### D4 — Quality gate: bỏ trace SAI khỏi corpus (khác exp6)
- **Hypothesis (từ [research/data_status.md](../data_status.md)):** corpus.py hiện train cả
  `rule_unknown` / `hypothesis_formed` (chưa verify, có thể SAI) như thể đúng ⇒ dạy reasoning sai.
  Đây KHÁC exp6 (cắt bài dễ-đúng → mất coverage → 0.79); ở đây chỉ bỏ bài **đã verify là SAI**, giữ
  TOÀN BỘ `rule_found`.
- **Falsify:** nếu số bị bỏ nhỏ hoặc leaderboard không đổi → neutral. Risk: nếu các trace "unverified"
  thực ra đôi khi đúng đáp số và đang giúp như soft-coverage → tụt nhẹ (theo dõi 5-nhóm-mạnh).

### D5 — In-run EMA + warmup/clip (free, optimization-side, regime-safe như exp21)
- **Hypothesis (từ L4):** lever data-preserving/optimization-side là loại an toàn duy nhất. Hiện
  [Continuer:711](../../Continuer_Nemotron_Notebook.py#L711) luôn ship state CUỐI; thêm EMA bóng của
  LoRA params trong vòng train-from-scratch → ship EMA ("soup" miễn phí trong 1 run). Kèm warmup +
  grad-clip thật ([Continuer:683](../../Continuer_Nemotron_Notebook.py#L683),
  [:687-689](../../Continuer_Nemotron_Notebook.py#L687-L689)).
- **Falsify:** adapter-EMA ≥ adapter-final-step trên leaderboard. Độc lập hoàn toàn với D0–D4 → ghép được lên bất kỳ run nào.

### D6 — (LAST / hedge) guess categories bằng constraint-search
- Đã bị flag dead-end, leverage thấp nhất (0.2–2%). CHỈ làm nếu D0–D5 cạn. Không ưu tiên.

---

## C. Thứ tự chạy đề xuất
1. **D1 + D2** (bit truncation) — đòn bẩy lớn nhất; delta = bit-trace ngắn (solver), override id bit cũ.
2. **D5** (EMA + warmup/clip) — layer free lên chính run D1.
3. **D3** (cryptarithm_deduce) — thêm coverage solver + đo L5 (coverage category-hiếm có dời điểm?).
4. **D4** (quality gate) — nếu D1–D3 chưa đủ.
5. **D6** — chỉ khi hết đường.

**Regime mọi run = continue-train từ 0.86** (`RESET_WEIGHTS=False`, LR ≤1e-5, order gốc, delta solver
xác định). **Gate chung:** pass@1 của 5 nhóm mạnh (numeral/unit/gravity/cipher/eq_deduce) không tụt
> 0.5pp; tụt ⇒ giảm liều hoặc rollback delta. KHÔNG self-gen RFT, KHÔNG negatives, KHÔNG shuffle-mix.

---

## D. Thí nghiệm bổ sung — khai thác WEIGHT-SPACE của fleet đã có (deep-reasoned, gần như FREE)

> **Insight lõi (chưa khai thác):** mọi đề xuất trước đều là *train run mới*. Nhưng ta đã có ~10
> adapter trả tiền rồi: baseline 0.86, exp21 0.86, batch-4 (0.70–0.74). Các adapter batch-4 KHÔNG phải
> rác — chúng là **endpoint α=1 của một đường nội suy** mà nội-thất (0<α<1) **chưa ai dò**. Và ta có
> **hai checkpoint 0.86 độc lập** (baseline vanilla vs exp21 LoRA+) **chưa từng soup**. Đây là hướng
> ROI cao nhất, gần như 0 GPU: chỉ average safetensors (CPU) + 1 lần inference/candidate.

### D7 — ⭐ WiSE-FT: nội suy adapter 0.86 ↔ adapter batch-4, dò α (FREE, không train)
- **Hypothesis (từ L2 monotone):** batch-4 tụt monotone theo độ-đổi-shape ⇒ adapter lever có thể đã
  *học coverage mới NHƯNG over-train/quên phần khác*. Trộn weight `θ = (1−α)·θ_0.86 + α·θ_lever` ở α
  nhỏ có thể **giữ robustness 0.86 mà nhặt được coverage mới** — đúng tinh thần WiSE-FT (nội suy
  zero-shot↔fine-tuned cho điểm > cả hai endpoint). Batch-4 chỉ chạy α=1; nội-thất chưa dò.
- **Code path:** mở rộng [soup_adapters.py](../../soup_adapters.py) (hiện average đều) thành weighted:
  `avg = (1-α)*t0 + α*t1`. Lấy θ_0.86 (adapter baseline) + θ_lever (vd exp35=0.74 hoặc exp38=0.72).
- **CLI:** scan α ∈ {0.1, 0.2, 0.3, 0.5} → 4 submission, 0 GPU-train.
- **Falsify:** nếu mọi α ≤ 0.86 ⇒ nội suy chết, adapter lever không có coverage thật (chỉ quên).
  Nếu có α cho >0.86 ⇒ **chứng minh lever có giá trị, chỉ over-train** → justify đầu tư D1 bản sạch.

### D8 — ⭐ Model-soup hai checkpoint 0.86 (baseline + exp21) (FREE, không train)
- **Hypothesis:** ta có **hai adapter ≈0.86 KHÁC nhau** (vanilla vs LoRA+ split-LR) — đa dạng nhưng
  cùng chất lượng. Average chúng (model-soup) thường vượt cả hai endpoint khi chúng đa dạng. exp17
  (seed-soup) bị hoãn vì *chưa có* hai checkpoint để soup; **giờ có rồi**.
- **Code path:** [soup_adapters.py](../../soup_adapters.py) nguyên trạng (α=0.5): `python soup_adapters.py
  <baseline_dir> <exp21_dir> <out_dir>`.
- **Falsify:** soup ≤ 0.86 ⇒ chết. Rủi ro thấp nhất, chi phí gần 0 → **làm TRƯỚC TIÊN**.

### D9 — Anchored continue-train về 0.86 (sửa anchor của exp16) — chống drift sang phân bố delta
- **Cơ chế ĐÚNG (sửa lại — KHÔNG phải "quên"):** continue-train trên *đúng data cũ* gần như no-op;
  tụt chỉ xảy ra khi delta kéo weight về **phân bố khác/xấu hơn** (self-gen/negatives/nén) HOẶC khi
  "cú đá" LR-tươi (optimizer-state không được load lại, schedule mới) đẩy weight rời minimum 0.86. D9
  chặn cả hai: phạt **kéo LoRA params về θ_0.86** (L2) HOẶC KL logits về model-0.86 → cho phép học delta
  nhưng **giới hạn độ trôi khỏi 0.86**. exp16 đã có khung KL ([exp16.py:615-643](../../exp16.py#L615-L643))
  **nhưng anchor về BASE model** (sai đích — kéo về model *chưa* finetune, ngược hướng 0.86). Đổi anchor → reference 0.86.
- **Code path:** trong exp16, thay target KL/L2 từ base-logits sang **adapter-0.86** (load θ_0.86 làm
  reference, phạt `‖θ − θ_0.86‖²` hoặc KL về logits-0.86).
- **Distinct:** đây là continue-train-COMPATIBLE (khớp lựa chọn user) + nhắm thẳng cơ chế erosion.
- **Falsify:** anchored continue-train trên delta D1 giữ 5-nhóm-mạnh trong 0.5pp **VÀ** bit-zero ↓;
  nếu vẫn xói như batch-4 ⇒ anchor không cứu được, vấn đề sâu hơn drift.

### D10 — Continue-train khu trú module (freeze tất cả trừ module liên quan delta)
- **Hypothesis:** quên ít hơn nếu continue-train chỉ được phép động một subspace nhỏ. Cờ
  `IN_PROJ_ONLY` ([Continuer:20](../../Continuer_Nemotron_Notebook.py#L20)) đã có sẵn cơ chế freeze.
- **Code path:** continue-train delta nhưng freeze mọi LoRA trừ module then chốt cho category đích.
- **Falsify:** so với D9, nếu khu-trú giữ strong-5 tốt hơn mà vẫn cải thiện target ⇒ giữ; nếu không, bỏ.

### D11 — ⭐ Đổi optimizer (Muon cho LoRA 2D) — class được L4 ủng hộ, mới khai thác 1 nước
- **Hypothesis (từ L4):** tweak phi-baseline DUY NHẤT giữ 0.86 là exp21 = **optimizer-side** (split-LR,
  vẫn AdamW). Data-side đã chết sạch. Optimizer *thuật toán* (AdamW vs Muon/Lion) **chưa exp nào động**
  ([Continuer:676](../../Continuer_Nemotron_Notebook.py#L676) AdamW cố định ở MỌI exp). Muon
  (orthogonalized momentum qua Newton–Schulz) thiết kế đúng cho **ma trận 2D** — LoRA A/B chính là 2D
  ⇒ fit tự nhiên; thường vượt AdamW trên hidden-2D ở cùng compute.
- **Code path:** thay AdamW ở [Continuer:676-682](../../Continuer_Nemotron_Notebook.py#L676-L682) bằng
  **Muon cho LoRA 2D (q/k/v/o/in/out/lm_head) + AdamW cho phần còn lại** (router/bias) — recipe Muon
  chuẩn. Impl Newton–Schulz ~50 dòng, không cần dep nặng.
- **Continue-train compatible:** chỉ đổi optimizer trong loop sẵn có, khớp regime continue-từ-0.86.
- **Risk:** (a) **LR-scale khác AdamW** → phải chỉnh LR riêng (nhớ exp4 rsLoRA nổ vì LR sai) → bắt đầu
  LR thấp; (b) tương tác với `_tie_grads` (MoE sum-grad ở [Continuer:686](../../Continuer_Nemotron_Notebook.py#L686)):
  Muon orthogonalize PHẢI chạy sau tie_grads, trên grad đã-tie; expert tensor batched [128,m,n] cần
  orthogonalize per-slice (sau tie thì 128 slice giống nhau → làm 1 lần).
- **Distinct với exp21:** exp21 giữ AdamW chỉ split-LR; D11 đổi **chính update-rule**.
- **Falsify:** Muon continue-train giữ 5-nhóm-mạnh trong 0.5pp **VÀ** ≥0.86; nếu bất ổn kiểu exp4 ⇒
  scale lại Muon-LR hoặc revert. (Lion = phương án rẻ-hơn-rủi-ro-hơn nếu Muon-impl ngại: sign-momentum,
  LR ~1/3 AdamW.)

### Thứ tự (bổ sung): **D8 → D7 trước cả D1** — vì gần-0-chi-phí và nếu D7 cho >0.86 thì xác nhận
toàn bộ thesis "lever có thật, chỉ over-train" trước khi đổ công vào solver/data của D1. D9/D10 là
regularizer ghép lên các run continue-train (D1/D3) để chống erosion. **D11 (Muon) chạy song song
nhánh optimizer-side** — độc lập với data, là nước đi tự nhiên nối tiếp exp21 (lever an toàn nhất theo L4).
