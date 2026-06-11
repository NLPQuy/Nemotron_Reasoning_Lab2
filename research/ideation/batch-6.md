# Batch 6 — Tấn công KIẾN TRÚC LoRA (paper-grounded, posterior từ exp1–47)

> Mục tiêu: 0.86 → 0.88+. Khác batch 1-5: đây là tầng **kiến trúc adapter** (đổi cấu trúc/vị trí
> tham số train), KHÔNG phải data, KHÔNG phải hyperparameter. Mọi paper **search live 2026-06-10**
> (arXiv id verify trong phiên), KHÔNG trích từ trí nhớ.
>
> Ràng buộc thi: `max_lora_rank=32`, greedy inference, `\boxed{}`, vLLM-loadable. Model =
> **Nemotron-H (Mamba-MoE hybrid)**, `modeling_nemotron_h`.

## Reframe then chốt (lý do batch này tồn tại)

**Bài học hậu nghiệm (xem [batch-5.md] L2 + thảo luận no-op):** continue-train từ 0.86 trên **cùng
corpus** ≈ **no-op** — 0.86 đã ở đáy của data đó, gradient ≈ 0. Mọi tweak *re-adapt các module 0.86
đã chỉnh* (q/k/v/o/up/down/in/out/lm_head) đều không có signal → tái tạo ~0.86.

→ **Luật batch-6:** trên same-corpus, signal CHỈ tồn tại ở **directions 0.86 CHƯA TỪNG adapt**. Search
SOTA chỉ ra đúng 2 chỗ đó trong Nemotron-H:
1. **Động lực SSM (Mamba)** — [PEFT of State Space Models, NeurIPS 2024, arXiv:2410.09016](https://arxiv.org/abs/2410.09016)
   chứng minh **"LoRA hiệu quả cho linear-projection NHƯNG THẤT BẠI trên SSM modules"**. LoRA trên
   `in_proj/out_proj` của 0.86 KHÔNG chạm selective-scan (A, dt, state). Mamba = 26%+ reasoning (bit).
2. **Router MoE** (`mixer.gate`) — fp32 base, **chưa exp nào adapt** (kể cả ESFT exp23 chỉ động expert).

Mạch hậu nghiệm khác: **exp21 (LoRA+ A/B-asymmetry) là thứ DUY NHẤT giữ 0.86** → A/B-asymmetry là seam sống.

---

## B6-1 — SSM-specific tuning (state-offset / additional-scan) ⭐⭐ TOP
- **Paper:** [State-offset Tuning, arXiv:2503.03499](https://arxiv.org/pdf/2503.03499) (inject trainable
  state-offset mỗi timestep, freeze phần khác) + [MambaPEFT, arXiv:2411.03855](https://arxiv.org/html/2411.03855v3)
  + Additional-scan (thêm state-dim mới). Nền: [arXiv:2410.09016](https://arxiv.org/abs/2410.09016).
- **Hypothesis:** 0.86 LoRA chưa chạm động lực SSM ⇒ thêm state-offset = **capacity HOÀN TOÀN mới** ⇒
  CÓ signal kể cả trên same-corpus. Nhắm đúng bit_manipulation (Mamba-heavy, 26%).
- **Code path:** trong `modeling_nemotron_h` (Mamba mixer forward) thêm trainable **state-offset** (bias
  cộng vào hidden-state SSM mỗi step) HOẶC **additional-scan** (state-dim mới) — ngoài LoRA hiện có.
  Train kèm regime continue-từ-0.86. Cần đọc Mamba mixer forward + chèn param mới + đảm bảo vLLM-loadable.
- **Vì sao thắng no-op:** đụng tham số 0.86 chưa từng có → gradient thật. Đúng chỗ paper nói LoRA fail.
- **Expected:** +0.5–2pp nếu SSM-tuning vá được bit dài; 🟡.
- **Risk:** (a) effort cao (sửa SSM forward); (b) **vLLM-loadable** — param mới phải nằm trong adapter
  format vLLM nạp được (KHÔNG chỉ PEFT-LoRA chuẩn) → rủi ro deploy lớn nhất, phải verify sớm.
- **Falsify:** trên held-out bit dài, pass@1 ↑ so exp44 (bitshort) mà 5-nhóm-mạnh giữ; nếu adapter
  không vLLM-load được → KILL (chuyển hướng).

## B6-2 — Router-LoRA / learnable routing ⭐
- **Paper:** [LD-MoLE, arXiv:2509.25684](https://arxiv.org/abs/2509.25684) (learnable dynamic routing) +
  [AdaMoE null-experts, arXiv:2406.13233](https://arxiv.org/pdf/2406.13233) (token "opt-out").
- **Hypothesis:** router `mixer.gate` chưa bao giờ adapt → new signal. Re-route token reasoning tới
  expert tốt hơn **mà không đụng expert weights** (tránh ESFT-exp23 fail).
- **Code path:** gắn LoRA rank 4–8 vào `mixer.gate` (giữ fp32 cho phần LoRA của gate, như cách lm_head
  được thêm thủ công ở [Continuer:323-339](../../Continuer_Nemotron_Notebook.py#L323-L339)); LR cực thấp;
  ghép **anchor D9** (exp42) để router không trôi hỗn loạn.
- **Vì sao thắng no-op:** thay đổi *đường đi* token, không phải trọng số đã hội tụ.
- **Expected:** +0.3–1.5pp; 🟡. Capacity rẻ.
- **Risk:** router rất nhạy (Nemotron giữ fp32 có lý do) → đổi routing có thể phá cân bằng expert. Bắt
  buộc LR thấp + anchor + gate 5-nhóm-mạnh.
- **Falsify:** macro pass@1 ↑ mà không nhóm mạnh nào tụt >0.5pp; nếu routing collapse (load imbalance) → giảm rank/LR hoặc KILL.

## B6-3 — MiLoRA / OPLoRA init: adapt subspace PHỤ, anti-drift
- **Paper:** [MiLoRA, arXiv:2406.09044](https://arxiv.org/pdf/2406.09044) (adapt MINOR singular
  components, freeze principal) + [OPLoRA, arXiv:2510.13003](https://arxiv.org/pdf/2510.13003)
  (orthogonal-projection chống catastrophic forgetting) + [Controlled-LoRA continued-training,
  arXiv:2410.16801](https://arxiv.org/pdf/2410.16801).
- **Hypothesis:** đúng bài toán continue-từ-0.86. Merge 0.86 vào base, SVD, train adapter mới trên
  **minor singular subspace** (orthogonal với directions 0.86 đã học) ⇒ preserve 0.86 + explore capacity
  mới ⇒ có signal mà không drift.
- **Code path:** merge adapter 0.86 vào base weight → SVD per-module → init LoRA A/B từ minor components
  (vs PiSSA principal). Một script offline + đổi init trong `get_peft_model`. Effort trung-cao.
- **Vì sao (một phần) thắng no-op:** train hướng orthogonal với cái 0.86 đã fit → còn signal ở đó.
- **Expected:** +0.3–1pp; 🟡.
- **Risk:** minor-component có thể là noise (capacity yếu); SVD per-module tốn RAM.
- **Falsify:** vượt 0.86 với 5-nhóm-mạnh không tụt; nếu = 0.86 → minor subspace cũng cạn.

## B6-4 — Freeze-A, train-B (+ better A-init) — rẻ nhất, neo exp21
- **Paper:** [Impact of Initialization on LoRA Finetuning Dynamics, arXiv:2406.08447](https://arxiv.org/pdf/2406.08447)
  (Hayou — chứng minh **B chi phối update, frozen-A ≈ full-LoRA**); cùng dòng nghiên cứu với LoRA+ (exp21 thắng).
- **Hypothesis:** đẩy thẳng mạch A/B-asymmetry của exp21. Freeze A (re-init bằng orthogonal/spectral
  basis tốt hơn) → chỉ train B tìm điểm tốt hơn B-của-0.86. Cắt nửa param train ⇒ giảm bề mặt drift.
- **Code path:** sau nạp 0.86, set `requires_grad=False` cho mọi `.lora_A.` (như cơ chế freeze
  [Continuer:473-481](../../Continuer_Nemotron_Notebook.py#L473-L481)); optional re-init A orthogonal. ~5–10 dòng.
- **Vì sao no-op-limited:** vẫn re-adapt module cũ → signal hạn chế trên same-corpus; giá trị chính là
  **ổn định/anti-drift** khi ghép với data mới (exp44).
- **Expected:** ~0–0.5pp đứng riêng; tốt hơn khi layer lên exp44; 🟡.
- **Risk:** B-only thiếu capacity nếu A-init kém. Rẻ nên thử trước.
- **Falsify:** ≥ exp21 trên cùng setup; nếu < → A-init là thủ phạm, thử orthogonal.

## B6-5 — Hierarchical MoE LoRA (shared base + per-expert residual rank-2)
- **Paper:** [HDMoLE, arXiv:2409.19878](https://arxiv.org/abs/2409.19878) (hierarchical routing) + MoBE
  (factorize expert = shared bases + expert-specific, từ search MoE-LoRA).
- **Hypothesis:** trung gian giữa full-tie (hiện tại, an toàn) và untie (exp23 fail): 1 LoRA **shared**
  (như `_tie_grads` hiện có) **+** residual **rank-2 per-expert** không-tie ⇒ "chung + chuyên biệt".
- **Code path:** giữ `_tie_grads` ([Continuer:510-540](../../Continuer_Nemotron_Notebook.py#L510-L540))
  cho phần shared; thêm slice rank-2 per-expert KHÔNG tie. Effort cao (va logic tie sẵn có).
- **Vì sao no-op-limited:** re-adapt expert đã-adapt → signal hạn chế same-corpus.
- **Expected:** +0–1pp; 🔴 (exp23 untie đã fail, rủi ro lặp lại).
- **Risk:** cao; để cuối.
- **Falsify:** vượt full-tie baseline; nếu ≤ → middle-ground cũng không cứu được MoE-LoRA.

---

## Đợt 2 — 5 idea bổ sung (search live 2026-06-10)

### B6-6 — HiRA: Hadamard high-rank adaptation ⭐⭐ (lách trần rank≤32)
- **Paper:** [HiRA, ICLR 2025 **Oral**, OpenReview TwJrTz9cRS](https://openreview.net/pdf?id=TwJrTz9cRS) —
  code [github.com/hqsiswiliam/hira](https://github.com/hqsiswiliam/hira), đã có [PEFT PR #2668](https://github.com/huggingface/peft/pull/2668).
- **Hypothesis (neo ràng buộc CỨNG):** thi giới hạn `max_lora_rank=32` ⇒ update LoRA bị **trần biểu
  đạt rank-32**. HiRA viết `ΔW = W_0 ⊙ (B·A)` (Hadamard với base weight đông cứng) ⇒ **update high-rank
  từ tham số rank-r** — nâng trần biểu đạt MÀ KHÔNG phá ràng buộc rank (vẫn rank-32 trainable). Không
  idea nào khác trong batch chạm tới giới hạn này.
- **Code path:** thay công thức LoRA `ΔW = scaling·B@A` thành `ΔW = W_0 ⊙ (B@A)` ở các target module.
  Port từ repo HiRA / PEFT PR. Forward CCE patch ([Continuer:397-428](../../Continuer_Nemotron_Notebook.py#L397-L428))
  cần đổi cho lm_head; các module khác qua PEFT.
- **Vì sao mạnh:** (a) tấn công đúng bottleneck cứng (rank); (b) **merge vào base, KHÔNG thêm inference
  overhead → vLLM-safe** (paper khẳng định); (c) ICLR oral + code sẵn → port nhanh.
- **Expected:** +0.5–2pp 🟡 (HiRA report vượt LoRA trên math/commonsense reasoning).
- **Risk:** `W_0 ⊙ (BA)` đổi semantics scaling — phải dò lại LR/alpha; tương tác với fp32-LoRA + MoE-tie.
- **Falsify:** vượt 0.86 với 5-nhóm-mạnh giữ; nếu adapter HiRA không vLLM-load/merge đúng → KILL.

### B6-7 — Flat-LoRA: flat minimum cho greedy decode ⭐
- **Paper:** [Flat-LoRA, arXiv:2409.14396](https://arxiv.org/pdf/2409.14396) + [EFlat-LoRA, arXiv:2508.00522](https://arxiv.org/abs/2508.00522).
- **Hypothesis (neo: eval GREEDY + continue-từ-0.86):** điểm phẳng trong LoRA-space có thể **sắc** trong
  full-param-space → greedy decode (không sample) cực nhạy với minima sắc. Flat-LoRA tối ưu sharpness ở
  **full-param-space** (perturbation ngẫu nhiên, KHÔNG double-cost như SAM) → tìm basin 0.86 **phẳng
  hơn = robust hơn** cho greedy + chịu được train/test shift.
- **Code path:** thêm random weight-perturbation vào forward khi train (Bayesian expectation loss) — 1
  hook quanh `model(...)` trong loop ([Continuer:643-660](../../Continuer_Nemotron_Notebook.py#L643-L660)).
- **Vì sao mới:** không phải đổi vị trí param, mà đổi **hình học minimum** — trực giao mọi idea khác.
- **Expected:** +0.3–1pp 🟡 (robustness, không phải capacity).
- **Risk:** perturbation scale phải dò; thêm chút compute/forward.
- **Falsify:** ≥ baseline cùng setup + 5-nhóm-mạnh ổn định hơn (variance thấp hơn) qua seed.

### B6-8 — LoRA-Pro: gradient bám full-fine-tuning (thoát no-op một phần)
- **Paper:** [LoRA-Pro, arXiv:2407.18242, ICLR 2025](https://arxiv.org/abs/2407.18242).
- **Hypothesis (neo no-op):** trên same-corpus, **full-FT vẫn tìm được minimum tốt hơn** LoRA đã hội
  tụ. LoRA-Pro chỉnh gradient của A,B sao cho low-rank update **xấp xỉ gradient full-FT** → LoRA "đuổi
  theo" quỹ đạo full-FT → rút thêm signal từ cùng data. Khác exp21 (chỉ split LR) và exp41 (Muon
  orthogonalize) — đây derive **gradient-adjustment tối ưu** để khớp full-FT.
- **Code path:** thay update gradient A/B bằng công thức LoRA-Pro (đóng quanh `_tie_grads`/trước
  `optimizer.step`, [Continuer:686-690](../../Continuer_Nemotron_Notebook.py#L686-L690)). Port từ paper.
- **Vì sao mạnh:** lý thuyết chặt, đúng dòng A/B-asymmetry (exp21 thắng). Ghép được với continue-train.
- **Expected:** +0.3–1.5pp 🟡.
- **Risk:** công thức gradient-adjust phức tạp; tương tác MoE-tie (sum-grad) phải xử đúng thứ tự.
- **Falsify:** vượt exp21 cùng setup; nếu ≤ → gap full-FT không phải bottleneck.

### B6-9 — ❌ BỎ (slot trống — batch-6 còn 9 idea)
> Mọi ứng viên thay AdaLoRA đều bị loại sau **fetch-verify** (không tin search-summary):
> - **AdaLoRA** — đụng exp5 (=0.84, FAIL).
> - **ReFT** — paper báo yếu CoT + sinh-dài (đúng điểm yếu task).
> - **O-LoRA** — trùng OPLoRA (B6-3).
> - **NoRA** (2408.10280) — fetch: eval nặng vision/multimodal, CHƯA published (đang review).
> - **LoRA-XS** (2405.17604) — fetch OpenReview: **Withdrawn, ICLR 2025**.
> - **ALoRA** (noNpK9Vt8l) — fetch OpenReview: **Withdrawn, ICLR 2026** + multi-task/federated (fit đơn-task kém).
>
> **Kết luận:** vùng SVD-init/minimal-param đã phủ bởi B6-3 + B6-4 — thêm idea ở đây là thừa, và mọi
> ứng viên ICLR-2025/2026 đều withdrawn. Để slot trống. **Honest under-quota: 9 idea chắc > 10 có 1 lung lay.**

---

## ⚠️ PREREQUISITE toàn batch — "Learning Rate Matters" (fetch-verify 2602.04998, 02/2026)

Finding (đã fetch-verify): khi **tune LR đúng cho từng biến thể**, mọi LoRA-variant đạt peak **chênh
1–2%** — "cải thiện trước đây phần lớn là khác hyperparameter, không phải khác phương pháp"; khác-LR-tối-ưu
do **largest Hessian eigenvalue**. Khớp lịch sử ta: **exp4 (rsLoRA) sập 0.50 = LR sai**; **exp21 (LoRA+)
= LR-scaling**; "single-knob LoRA cạn" = wash-out dưới LR đúng.

**Hệ quả bắt buộc cho batch-6:**
1. **Mỗi exp B6-* PHẢI dò LR riêng** (đừng dùng cứng 1e-5/0.5e-3) — nếu không, so sánh vô nghĩa (đúng kiểu exp4).
2. Idea **re-parameterize cùng module** (B6-3/B6-4/B6-6/B6-7/B6-8/B6-10) **có thể wash-out** dưới LR-tuning
   → kỳ vọng thấp hơn. Idea **thêm capacity hướng MỚI** (B6-1 SSM, B6-2 router) ít bị finding này phủ định
   → **ưu tiên đúng top-3 đã chọn** (B6-6 phá-trần-rank, B6-1, B6-2).

### B6-10 — LoRA-Dropout / BiLoRA: chống overfit continue-train
- **Paper:** [LoRA-Dropout, arXiv:2404.09610](https://arxiv.org/abs/2404.09610) (sparsity regularizer) +
  [BiLoRA, arXiv:2403.13037](https://arxiv.org/pdf/2403.13037) (bi-level: train singular-vectors vs
  -values trên subset khác nhau).
- **Hypothesis (neo no-op + small-data):** continue-train từ 0.86 trên cùng corpus dễ **overfit/memorize
  thêm** (adapter đã hội tụ). LoRA-Dropout (hiện `LORA_DROPOUT=0.0`, [Continuer:8](../../Continuer_Nemotron_Notebook.py#L8))
  hoặc BiLoRA tách data → regularize, đẩy về biểu diễn sparse generalize hơn.
- **Code path:** LoRA-Dropout = bật `LORA_DROPOUT` + refined dropout trên A/B; BiLoRA = tách corpus, luân
  phiên train vectors/values. LoRA-Dropout rẻ nhất.
- **Expected:** +0–0.5pp 🟡 (regularization).
- **Risk:** nếu overfit KHÔNG phải vấn đề (data đã fit) → vô hiệu; BiLoRA effort cao.
- **Falsify:** vượt baseline mà 5-nhóm-mạnh không tụt; nếu = baseline → overfit không phải bottleneck.

---

## Đợt 3 — THÊM-CAPACITY (chạm param 0.86 chưa adapt; sống sót LR-caveat) — search 2026-06-10

> Theo prereq LR-Matters: idea re-param cùng module dễ wash-out; chỉ **thêm capacity hướng mới** mới
> chắc cửa. 3 idea dưới đều đụng tham số 0.86 CHƯA touch. Nguồn **MambaPEFT = ICLR 2025 (fetch-verified venue)**.

### B6-11 — Partial-LoRA trên SSM-slices của in_proj ⭐ (vLLM-safe)
- **Paper:** [MambaPEFT, ICLR 2025, arXiv:2411.03855](https://arxiv.org/abs/2411.03855) — báo **"Partial LoRA
  shows superior performance"** (theo abstract/trang dự án; chưa đọc bảng đầy đủ).
- **Hypothesis:** 0.86 đặt LoRA generic lên TOÀN BỘ `in_proj` (coi như 1 linear). Mamba `in_proj` project
  ra [x, B, C, dt] — Partial-LoRA đặt LoRA **chỉ lên slice SSM then chốt** (dt/B/C) thay vì cả khối →
  capacity tập trung đúng động lực SSM. Khác B6-1 (additional-scan = thêm state-dim mới) và khác generic-LoRA.
- **Vì sao thoát no-op + sống LR-caveat:** không phải re-param cùng thứ — đây là **đặt LoRA đúng chỗ SSM**
  mà generic-LoRA pha loãng. **vLLM-safe** (vẫn là LoRA chuẩn).
- **Code path:** cần biết split-dims của `in_proj.out` trong `modeling_nemotron_h`; áp LoRA per-slice.
- **Expected:** +0.3–1pp 🟡. **Risk:** phải xác định đúng dim split; thấp hơn B6-1 về effort.
- **Falsify:** vượt generic-in_proj-LoRA (exp43-style) trên bit; nếu = → slice không quan trọng.

### B6-12 — Affix-tuning: token học-được kiểu Mamba (thêm token = capacity mới)
- **Paper:** [MambaPEFT, ICLR 2025](https://arxiv.org/abs/2411.03855) — đề xuất **Affix-tuning** (prefix-tuning
  **thiết kế lại cho Mamba**: bỏ prefix sau SSM, chèn affix token ở vị trí bất kỳ; prefix-tuning thường KHÔNG chạy trên Mamba).
- **Hypothesis:** thêm **token học-được** = capacity 0.86 hoàn toàn chưa có, trực giao mọi weight-LoRA.
- **Vì sao mạnh:** capacity mới thuần tuý (không đụng weight đã hội tụ) → thoát no-op rõ nhất.
- **Risk lớn = DEPLOY:** affix token phải áp lúc inference; **KHÔNG phải LoRA-adapter chuẩn → vLLM-load rủi ro cao**.
  Verify vLLM TRƯỚC khi train (như B6-1). Nếu vLLM không nạp affix → KILL.
- **Expected:** +0.3–1.5pp 🟡. **Falsify:** vLLM-loadable + vượt baseline; không nạp được → KILL.

### B6-13 — Norm-tuning (RMSNorm gains) + BitFit (bias) — capacity rẻ, untouched
- **Paper:** [BitFit, arXiv:2106.10199](https://arxiv.org/abs/2106.10199) (train chỉ bias) +
  [LayerNorm-tuning, arXiv:2403.20284](https://arxiv.org/abs/2403.20284) ("LayerNorm **đổi NHIỀU NHẤT**
  sau fine-tune"). (Venue 2 paper chưa fetch-verify.)
- **Hypothesis:** 0.86 đặt `bias="none"` + KHÔNG đụng RMSNorm. Mở **RMSNorm gains + bias** = capacity
  ở tham số chưa-touch; norm là chỗ đổi-nhiều-nhất → leverage cao. Stack thêm lên LoRA hiện có. Rẻ nhất.
- **Risk = DEPLOY:** norm/bias là full-param (`modules_to_save`), KHÔNG phải LoRA → **vLLM-load rủi ro**
  (format adapter vLLM mong lora_A/B). Verify trước.
- **Expected:** +0–0.8pp 🟡. **Falsify:** vLLM-loadable + vượt baseline.

---

## Thứ tự chạy & refs cần clone (12 idea; B6-9 trống)
| | Idea | Tấn công gì | Thoát no-op | Effort | Refs/code |
|---|---|---|---|---|---|
| 1 | **B6-6 HiRA** | trần rank≤32 (ràng buộc CỨNG) | — (capacity mới) | trung | github hqsiswiliam/hira + PEFT PR#2668 |
| 2 | **B6-11 Partial-LoRA SSM-slice** | đặt LoRA đúng dt/B/C (vLLM-safe) | CÓ | trung | MambaPEFT (ICLR25) |
| 3 | **B6-1 SSM-tuning** | SSM 0.86 chưa adapt | CÓ | cao | state-offset-tuning, MambaPEFT |
| 4 | **B6-2 Router-LoRA** | router chưa adapt | CÓ | trung | LD-MoLE, AdaMoE |
| 5 | **B6-8 LoRA-Pro** | gap full-FT (same-data signal) | một phần | trung | paper 2407.18242 |
| 6 | **B6-13 Norm+BitFit** | RMSNorm/bias chưa adapt (rẻ) | CÓ | thấp | BitFit, LayerNorm-tuning |
| 7 | B6-12 Affix-tuning | token mới (capacity thuần) | CÓ | trung | MambaPEFT (ICLR25) |
| 8 | B6-7 Flat-LoRA | minima sắc / greedy-fragile | — (robustness) | thấp-trung | paper 2409.14396 |
| 9 | B6-4 Freeze-A | A/B-asymmetry (exp21) | hạn chế | thấp | paper 2406.08447 |
| 10 | B6-3 MiLoRA/OPLoRA | drift / forgetting | một phần | trung | MiLoRA, OPLoRA |
| — | ~~B6-9~~ **BỎ** | — | — | — | — |
| 11 | B6-10 LoRA-Dropout | overfit continue-train | hạn chế | thấp | paper 2404.09610 |
| 12 | B6-5 Hier-MoE | tie↔untie middle | hạn chế | cao | HDMoLE |

**Ưu tiên cao nhất (group "thêm-capacity, sống LR-caveat"):**
- **B6-6 HiRA** — bottleneck CỨNG (rank≤32), code sẵn, **vLLM-safe**. Cú đáng giá nhất.
- **B6-11 Partial-LoRA SSM-slice** — thêm-capacity đúng SSM **VÀ vLLM-safe** (vẫn LoRA) → rủi-ro-thấp nhất trong nhóm thoát-no-op.
- **B6-1 SSM-tuning + B6-2 Router-LoRA** — thoát no-op mạnh nhất (param chưa chạm), nhưng **deploy-risk** (verify vLLM trước).
- **B6-13 Norm+BitFit** — capacity rẻ nhất ở param untouched (deploy-risk modules_to_save).

**Cảnh báo deploy xuyên suốt nhóm thêm-capacity:** B6-1/B6-12/B6-13 đụng param NGOÀI LoRA-chuẩn → **verify
vLLM-loadable TRƯỚC khi train**. B6-6/B6-11/B6-2 là LoRA-chuẩn → vLLM-safe, ưu tiên làm trước.

B6-7/B6-4 rẻ → thử song song layer lên exp44. B6-3/B6-9/B6-10 đợt sau. B6-5 cuối (rủi ro lặp exp23).

**Gate chung (như batch-5):** continue-train từ 0.86, LR ≤1e-5, order gốc; 5-nhóm-mạnh không tụt >0.5pp.
**Risk lớn nhất xuyên suốt = vLLM-loadable:** B6-1 (param SSM mới) và B6-2 (LoRA trên router) phải verify
adapter nạp được dưới vLLM TRƯỚC khi train tốn GPU — đây là điều kiện sống/chết của cả batch.

**Plan cho Codex:** xem `plan-batch-6.md` (line-level), đã triage theo ràng buộc deploy.

---

## Đợt 4 — THAY THẾ idea bị BLOCK bằng capacity-idea CÓ THỂ DEPLOY (search top-tier 2026-06-10)

> **Tại sao phải thay:** đọc code thật cho thấy 4 idea "vượt trần" (HiRA, SSM-offset, MiLoRA, BitFit)
> **không deploy được** dưới ràng buộc submission = **rank-32 additive LoRA trên base gốc, nạp bằng vLLM**.
> Search thêm các phương pháp tăng-capacity ở **top-tier** và **fetch-verify khả năng merge**:

### ⚠️ Định lý chặn (cốt lõi batch này): "vượt trần per-module = bất khả thi nếu phải deploy LoRA rank-32"
Mọi `ΔW` deploy = `scaling·B@A` (r≤32) cộng lên **base gốc** ⇒ **rank ≤ 32 cứng**. Mọi phương pháp
"phá trần rank" **đều** đạt high-rank bằng cách **merge vào base** (→ full-weight, không submit được) hoặc
**phi tuyến/không-merge** (→ vLLM không áp được). Fetch-verify từng cái:

| Method (top-tier) | Cách tăng capacity | Deploy rank-32 LoRA? | Verdict |
|---|---|---|---|
| **MoRA** (2405.12130, square high-rank) | square matrix + compress/decompress, **merge vào base** | ❌ merged ΔW high-rank trên base | BLOCK (như HiRA) |
| **ReLoRA / COLA / Chain-of-LoRA** (ICLR'24) | merge-restart, tích lũy high-rank **vào base** | ❌ base cuối = high-rank | BLOCK |
| **LoRA-GA / CorDA / KaSA / PiSSA** (NeurIPS'24 init) | init từ SVD, **trừ delta khỏi base** | ❌ convert→LoRA rank 2r=64>32 | BLOCK (như MiLoRA) |
| **AuroRA / NEAT / DenseLoRA / AFA-LoRA / SineLoRA / LoDA** (ACL/ICLR'25 phi tuyến) | nonlinear mapping phá low-rank bottleneck | ❌ phi tuyến → vLLM không merge | BLOCK (như HiRA) |
| **MoSLoRA** (EMNLP 2024) | mixer học-được `ΔW=B·M·A` | ✅ rank(BMA)≤r; **fold B'=B·M** → LoRA chuẩn | **REPLACE → B6-14** |
| **target nhiều module hơn** (rank-32 mỗi cái) | thêm chỗ đặt LoRA (gate, SSM-slice…) | ✅ vẫn LoRA chuẩn | **REPLACE → B6-15** |
| **SBoRA** (2407.05413, regional) | standard-basis A → "double rank cùng param" | ✅ linear, mergeable (cap 32) | phụ → B6-16 |

**Kết luận thẳng:** dưới ràng buộc thi, **không có cách nào vượt rank-32 *trong một module***. Hai đòn
bẩy capacity hợp lệ DUY NHẤT: (1) **dùng rank-32 hiệu quả hơn** (MoSLoRA mixer), (2) **đặt LoRA ở NHIỀU
module hơn** (mỗi cái rank-32). Đây là thay thế trung thực cho 4 idea bị block.

### B6-14 — MoSLoRA: learnable subspace-mixer (THAY HiRA/MoRA) ⭐⭐ TOP deployable
- **Paper:** [Mixture-of-Subspaces, EMNLP 2024, arXiv:2406.11909](https://arxiv.org/abs/2406.11909) —
  code [refs/moslora](../../refs/moslora) (wutaiqiang/MoSLoRA) + [PEFT PR #2294](https://github.com/huggingface/peft/pull/2294).
- **Code thật (đã đọc):** mixer `lora_AB = nn.Linear(r, r, bias=False)`
  ([layer.py:113](../../refs/moslora/subject_driven_generation/peft/tuners/lora/layer.py#L113)),
  forward `lora_B(lora_AB(lora_A(x)))·scaling` ([layer.py:347](../../refs/moslora/subject_driven_generation/peft/tuners/lora/layer.py#L347)),
  init mixer kaiming (hoặc orthogonal, [layer.py:166](../../refs/moslora/subject_driven_generation/peft/tuners/lora/layer.py#L166)).
- **Hypothesis:** LoRA chuẩn cố định cách trộn r subspace (mixer = identity ẩn). MoSLoRA học **mixer r×r**
  ⇒ khai thác `r²` cách-trộn subspace ⇒ biểu đạt mạnh hơn **trong cùng rank-32** — đúng *tinh thần* HiRA
  (more expressivity) nhưng **giữ tuyến tính rank≤32**.
- **Vì sao DEPLOY được (khác HiRA):** `rank(B·M·A) ≤ min(r)=r≤32`; lúc save **fold** `B' = B·M` (out×r),
  giữ `A` ⇒ adapter lưu = `lora_A, lora_B'` **chuẩn** ⇒ **vLLM nạp như rank-32 LoRA, 0 latency** (paper
  khẳng định "merged into original weights, no inference latency"). Mixer chỉ thêm `r²=1024` param/module
  (không đáng kể), biến mất sau fold.
- **Expected:** +0.3–1.5pp 🟡 (MoSLoRA report vượt LoRA trên commonsense reasoning cùng rank). **Risk:**
  scaling tương tác mixer (paper dùng `alpha/√r` rslora-style) → dò LR/alpha; MoE-tie phải tie cả mixer.
- **Falsify:** vượt baseline rank-32 với 5-nhóm-mạnh giữ; nếu adapter sau-fold không vLLM-load đúng → KILL.

### B6-15 — Mở rộng độ phủ target-module (THAY SSM-offset) — đòn bẩy capacity THẬT duy nhất
- **Cơ sở:** vì rank/module cố định 32, **capacity tổng = 32 × số module được LoRA**. Đòn bẩy còn lại =
  **thêm module**. Hiện target q/k/v/o/up/down/in_proj/out_proj/lm_head. Chưa chạm: **router gate**
  (=B6-2/exp51) và **SSM-slice của in_proj** (=B6-11/exp50, đặt lại đúng dt/B/C). Cả hai **vẫn LoRA chuẩn**.
- **Hypothesis:** mỗi module-mới = +rank-32 capacity ở hướng 0.86 chưa adapt, **không phá deploy**.
- **Đây không phải idea mới về thuật toán** — nó là **khung hợp nhất** cho exp50+exp51: capacity hợp lệ =
  coverage, không phải high-rank. Ưu tiên đo: in_proj-SSM-slice (exp50) trước (Mamba 26%).
- **Falsify:** macro pass@1 ↑ theo từng module thêm; nếu phẳng → coverage đã bão hòa.

### B6-16 — SBoRA (standard-basis, regional) — phụ, biến thể Freeze-A
- **Paper:** [SBoRA, arXiv:2407.05413](https://arxiv.org/abs/2407.05413) — A (hoặc B) = standard-basis
  one-hot ⇒ "double rank với cùng param" / hoặc nửa param. **Linear, mergeable** ⇒ deploy OK (nhưng vẫn
  cap 32). Hiệu quả gần **Freeze-A (exp48)** — gom vào đó, chỉ thử nếu exp48 có tín hiệu.
- **Expected:** +0–0.5pp 🟡. **Falsify:** vượt exp48 cùng setup.

**Cập nhật thứ tự ưu tiên (deployable-capacity):** **B6-14 MoSLoRA** lên #1 nhóm "thêm-capacity vLLM-safe"
(thay HiRA — cùng spirit, deploy được), giữ **B6-11 (exp50)** + **B6-2 (exp51)** làm đòn bẩy coverage
(B6-15). Bỏ hẳn HiRA/SSM-offset/MiLoRA/BitFit khỏi danh sách code (xem plan-batch-6 PHẦN 5).

---

## Đợt 5 — Idea từ DEEP-RESEARCH (Trục 2 MoSLoRA + Trục 3 coverage, 2026-06-10)

> Hai deep-research (cited trong [memory] + plan-batch-6) đảo hai kết luận và bật ra coverage-idea **mới,
> grounded vào `supported_lora_modules` THẬT của vLLM cho class Nemotron-H** + version grader đã chốt.
>
> **2 fact nền (đã verify):** (a) **MoSLoRA NO-GO** — PEFT-maintainer reproduce 20-run = ngang LoRA (p≈1.0),
> paper chưa từng test text-math → B6-14 hạ xuống near-free. (b) **Grader chạy vLLM 0.11.2–0.12.0** (recipes
> page chính thức cho Nemotron-3-Nano-30B-A3B; Blackwell sm_120 ép cao) ⇒ **FusedMoE expert-LoRA (PR #21229,
> ship đúng 0.11.2) ĐƯỢC hỗ trợ** — expert-coverage không bị version chặn.
>
> **`supported_lora_modules` thật của class này** (vLLM #38085): `[conv1d, down_proj, gate_up_proj,
> in_proj_ba, in_proj_qkv, in_proj_z, linear_fc1, linear_fc2, o_proj, out_proj, proj, qkv, qkv_proj]`.
> So với target của adapter 0.86 (q/k/v/o, in_proj, out_proj, up/down experts, lm_head) → lộ ra **coverage
> chưa dùng**.

### B6-17 — Module-map TĨNH từ source vLLM 0.12.0 (NỀN cho Đợt-5; KHÔNG cần GPU)
> Thay cho "audit runtime" (bỏ — không có máy chạy vLLM). Đọc thẳng source `v0.12.0` cho
> `NemotronHForCausalLM` → biết **tĩnh** module nào áp LoRA, không cần probe.
- **Nguồn (đã đọc 2026-06-10):** `vllm/model_executor/models/nemotron_h.py@v0.12.0`
  (`packed_modules_mapping: qkv_proj→[q,k,v]`; `embedding_modules: embed_tokens, lm_head`; experts =
  `SharedFusedMoE`; inherits `SupportsLoRA, MixtureOfExperts`) + `vllm/lora/layers/fused_moe.py@v0.12.0`
  (`FusedMoEWithLoRA.can_replace_layer`: `isinstance(source_layer, FusedMoE) and len(packed)==2`, assert
  `not use_ep`).

| Module | 0.12.0 áp LoRA? | Ghi chú |
|---|---|---|
| q/k/v/o_proj | ✅ | qkv packed + o_proj |
| in_proj / out_proj (Mamba2) | ✅ | nn.Linear trong `MambaMixer2`; in_proj tách `in_proj_z/qkv/ba` |
| lm_head | ✅ | embedding_modules (0.86 đã dùng) |
| **embed_tokens** | ✅ | embedding_modules — **0.86 CHƯA dùng → free coverage** |
| **MoE experts gate_up/down** | ✅ (1-GPU) | `SharedFusedMoE ⊂ FusedMoE`; cần `use_ep=False` (đúng trên 1 GPU) |
| router `mixer.gate` | ❌ | ngoài supported set → KILL B6-2 |
| conv1d / A_log / D / dt_bias | ❌ | Conv1d & raw nn.Parameter, không có LoRA path |

- **Kết luận quan trọng:** **expert-LoRA của 0.86 ĐANG được áp** (không có "expert chết" để thu hồi → ý
  redistribute gần như vô hiệu). Module applicable **duy nhất** 0.86 chưa chạm = **embed_tokens** ⇒ đẩy
  vào B6-19. Router/conv1d/SSM-param chốt KHÔNG áp. **Đây là nền tĩnh, đủ để code Đợt-5 mà không cần probe.**

### B6-18 — in_proj 3-slice fan-out (coverage SSM, deployable) ⭐
- **Nguồn:** vLLM tách `in_proj` thành **3 target độc lập** `in_proj_z / in_proj_qkv / in_proj_ba`
  (#38085). Nemotron-H Mamba2 **fuse dt/B/C vào in_proj** (không x_proj riêng — transformers
  `NemotronHMamba2Mixer`, deep-research Trục 3).
- **Hypothesis:** thay vì 1 LoRA rank-32 lên in_proj gộp, đặt **rank-32 riêng cho mỗi slice** ⇒ ~3× capacity
  đúng động lực SSM (z-gate, x/conv-qkv, B/C/dt), **vẫn LoRA chuẩn rank-32/slice** ⇒ vLLM-safe. Bản mạnh hơn
  của B6-11/exp50 (không chỉ mask 1 slice mà phủ cả 3).
- **Code path:** target_modules = `[in_proj_z, in_proj_qkv, in_proj_ba, ...]` (tên vLLM-side); cần map sang
  tên PEFT lúc train (PROBE-0 xác nhận tên).
- **Expected:** +0.3–1pp (coverage thật) 🟡. **Risk:** in_proj_ba/qkv chia capacity — slice nào ít leverage
  thì phí. **Falsify:** vượt in_proj-gộp trên bit; nếu = → slice không quan trọng.

### B6-19 — Dense-FFN + embed_tokens coverage (FREE, module untouched) ⭐
- **Nguồn:** supported set chứa `linear_fc1, linear_fc2` (= MLP các block **dense non-MoE** của hybrid) và
  `embed_tokens` là loại LoRA-supported (VocabParallelEmbedding) — adapter 0.86 **không target** (chỉ chạm
  expert up/down + lm_head).
- **Hypothesis:** thêm rank-32 LoRA lên **dense-FFN (linear_fc1/fc2)** + **embed_tokens** = capacity ở module
  **được apply chắc** (không vướng FusedMoE fragility) mà 0.86 chưa dùng ⇒ coverage gần-miễn-phí.
- **Code path:** thêm `linear_fc1, linear_fc2, embed_tokens` vào `TARGET_MODULES`. **VERIFY**
  `modeling_nemotron_h` có dense-FFN tách khỏi experts (PROBE-0). embed_tokens dùng `lora_A/B` (KHÔNG
  `modules_to_save` — sẽ hard-ValueError).
- **Expected:** +0.3–1.5pp 🟡 (module mới, robust-apply). **Risk:** dense-FFN có thể không tồn tại tách
  riêng (toàn MoE). **Falsify:** macro pass@1 ↑ khi thêm; nếu phẳng → các block đó ít trọng số.

### B6-20 — Expert-LoRA layout-correct (đã xác nhận LIVE; chỉ còn lo layout)
- **Cập nhật từ B6-17 tĩnh:** expert-LoRA **được áp** trên 0.12.0 1-GPU (`SharedFusedMoE ⊂ FusedMoE`,
  `use_ep=False`). Nên đây **không phải** "recover expert chết" nữa — chỉ còn rủi ro **layout 2D-Megatron vs
  3D-PEFT** (declare sai → garbage thầm, #42008).
- **Hypothesis:** đảm bảo adapter save đúng format mà `vllm/vllm-openai:v0.12.0` mong (PEFT-3D mặc định) để
  expert-LoRA không thành garbage. Đây là **de-risk** cho adapter hiện có, không phải capacity mới.
- **Code path (tĩnh, không probe):** đối chiếu format `adapter_model.safetensors` của 0.86 với khóa expert
  mà `FusedMoEWithLoRA` mong (`experts.gate_up_proj/down_proj.lora_A/B` 3D-stacked); nếu lệch → sửa lúc save.
- **Expected:** +0–0.5pp (tránh garbage) 🟡. **Falsify:** không có cách verify không-GPU tuyệt đối → giữ ở
  mức "đảm bảo format chuẩn PEFT", coi như low-risk hygiene.

### B6-21 — HP-sweep plain rank-32 LoRA (idea trung thực nhất, từ Trục 2)
- **Nguồn:** [LR-Matters arXiv:2602.04998], [Batch-Size-Matters arXiv:2602.09492], [unified 2601.22708] —
  variants hội tụ 1–2% khi tune LR/batch; **batch-size một mình swing >10% trên GSM8K**; **plain LoRA tune kỹ
  vượt PiSSA/MiLoRA**.
- **Hypothesis:** đòn bẩy thật KHÔNG phải method mới mà là **sweep LR × batch-size** cho plain rank-32
  continue-train — chỗ literature nói 1–2% thật sự ở. Ta chưa làm sweep có hệ thống (exp21 LoRA+ chỉ split LR).
- **Code path:** grid nhỏ `LEARNING_RATE ∈ {5e-6,1e-5,2e-5,5e-5}` × `BATCH_SIZE ∈ {16,32,64}` trên plain
  recipe (không thêm method). Rẻ, đo trước khi tin bất kỳ method-delta nào.
- **Expected:** +0.5–2pp 🟡 (đúng theo 3 paper). **Risk:** GPU cho grid. **Falsify:** nếu best-of-grid = 0.86
  → recipe đã ở optimum HP, method mới mới đáng.

### B6-22 — Robust-coverage redesign (de-risk; hạ ưu tiên — experts đã LIVE)
- **Cập nhật:** vì B6-17 tĩnh xác nhận **experts được áp ổn** trên 0.12.0 1-GPU, lý do "né expert vì mong
  manh" yếu đi. Giữ làm phương án dự phòng: nếu sau này thấy expert-LoRA gây bất ổn, dồn rank-32 vào
  attention + in_proj 3-slice (B6-18) + embed (B6-19) + lm_head. **Không ưu tiên** trừ khi có bằng chứng bất ổn.
- **Expected:** +0–0.8pp (giảm variance) 🟡. **Falsify:** điểm ổn định hơn qua các lần grade.

**Thứ tự Đợt 5 (đã bỏ probe runtime — dùng module-map tĩnh B6-17):**
1. **B6-19 embed_tokens** — free coverage **đã xác nhận tĩnh** (module applicable duy nhất 0.86 chưa dùng); rẻ nhất.
2. **B6-18 in_proj 3-slice fan-out** — coverage SSM (in_proj_z/qkv/ba).
3. **B6-21 HP-sweep plain LoRA** — song song, không phụ thuộc coverage.
4. **B6-20 expert-format hygiene** — đảm bảo layout PEFT-3D đúng (low-risk).
5. ~~B6-17 audit runtime~~ **BỎ** (không có GPU; thay bằng module-map tĩnh ở trên). ~~B6-22~~ dự phòng.
Mọi idea Đợt 5 đều **deployable rank-32 LoRA** (coverage/HP, không phá trần), và **không cần GPU để chốt hướng**.
