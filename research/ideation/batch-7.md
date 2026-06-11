# Batch 7 — MERGING / SOUP các LoRA adapter (deep-research-grounded)

> Mục tiêu: 0.86 → 0.88+. Tầng tấn công: **hợp nhất các adapter đã có** — KHÔNG train mới (trừ B7-7/B7-9),
> KHÔNG đụng corpus. Nguồn: deep-research 2026-06-11 (103 agents, 21 nguồn primary, 22 claims
> verify 3-0/2-1, 3 claims bị refute — xem cuối file).
>
> Ràng buộc thi: deliverable = **một** adapter LoRA rank-32 additive, vLLM-loadable trên base nguyên bản.
> ⇒ mọi merge phải cho ra per-module cặp A/B rank ≤ 32 — **không** được nộp full-weight delta hay rank n·32.
>
> Ingredient sẵn có (đều 0.86, cùng base, cùng corpus → thỏa điều kiện "single low-error basin" của
> model-soups): **baseline 0.86, exp21 (LoRA+), exp40 (EMA), exp42 (anchored-L2), exp43 (localized)**.
> Nhóm phụ 0.84 (exp1/exp5/exp10) dùng được cho soup small-α.

## Phát hiện nền tảng (vì sao batch này tồn tại)

1. **`soup_adapters.py` hiện tại merge SAI công thức.** Cả `main()` lẫn `wiseft()` average từng tensor
   — tức average `lora_A` và `lora_B` **tách rời** ("Direct Merging"). Vì `B·A` bất biến với
   `(B·s, s⁻¹·A)`, hai adapter cùng ΔW có thể có factor hoàn toàn khác nhau; average factor tạo
   cross-terms `B_i·A_j` phá ΔW. Verify 3-0 từ 3 nguồn độc lập:
   [KnOTS, arXiv:2410.19735](https://arxiv.org/abs/2410.19735) ("products of misaligned factorizations"),
   [ICML 2025 R2-FM **workshop**](https://icml.cc/virtual/2025/50854) (DM "sacrifices performance"),
   [LoraHub, arXiv:2307.13269](https://arxiv.org/abs/2307.13269) (sum-factor-rồi-nhân có cross-term).
2. **Công thức đúng + giữ rank-32** ([ICML 2025 R2-FM workshop](https://icml.cc/virtual/2025/50854), verify 3-0; bản main-conference cùng kết luận: SeedLoRA ICML 2025, LoRE-Merging EMNLP 2025 — xem [batch-7-search-log.md](batch-7-search-log.md)):
   average các **tích** `ΔW_i = B_i·A_i`, rồi **SVD-truncate về rank 32** — "best of both": chất lượng
   của product-merge + giữ dạng low-rank. PEFT có sẵn (`add_weighted_adapter`, `combination_type="svd"`)
   nhưng ta tự viết trên safetensors (CPU, không cần load 30B) vì adapter có key đặc thù
   (`backbone.lm_head` full-weight, expert slices).
3. **Hệ quả hậu nghiệm: exp17 (seed-soup, Batch-2) regress là kết quả BỊ CONFOUND** — nó dùng chính
   factor-averaging. Verdict "soup vô dụng" rút từ exp17 là **chưa được kiểm chứng đúng cách**.
4. **Ngoại lệ làm dịu:** exp40/42/43 continue-train **từ cùng init 0.86** → factor gần align →
   factor-avg ≈ product-avg với riêng nhóm này. Nhưng trộn thêm exp21/baseline (quỹ đạo train khác)
   thì bắt buộc product-space. Làm product-space cho mọi trường hợp = đúng vô điều kiện, chi phí ≈ 0.
5. **Trung thực về kỳ vọng:** KHÔNG có claim sống sót nào chứng minh same-task soup vượt best-single
   trên **LLM reasoning + greedy decoding**. Bằng chứng vượt-best-single duy nhất là
   [Model Soups](https://arxiv.org/abs/2203.05482) trên vision (CLIP/ViT-G, ImageNet). Mọi paper
   LoRA-merge cross-task (KnOTS, Core Space) đều < 100% normalized vs best single. Batch này rẻ
   (hầu hết zero-train) nên đáng quét, nhưng prior thắng ~20–30%.

**Key tensor map (cho mọi code path dưới):** trong `adapter_model.safetensors`, cặp
`...lora_A.weight` (r×in) / `...lora_B.weight` (out×r) → xử lý product-space; key full-weight
(`backbone.lm_head...`, xem memory `peft-save-embedding-full-weight`) → average trực tiếp (đúng,
không có vấn đề factorization); mọi artifact merge phải qua `offline/deploy_check.py` (gate PHẦN-2C)
trước khi đóng `submission.zip`.

---

## PROBE-M0 (BẮT BUỘC chạy trước) — chẩn đoán OFFLINE (CPU, KHÔNG vLLM)

> **Ràng buộc thật:** Kaggle không chạy được vLLM (torch 2.10 không có wheel) → **không có local accuracy eval.**
> Kênh đo accuracy duy nhất = **submit leaderboard Kaggle** (~5 lượt/ngày). Vì vậy PROBE-M0 **không** đo accuracy
> — nó là đại số tuyến tính offline để **lọc** ứng viên trước khi tiêu một lượt submit. Chi tiết: [plan-batch-7.md](plan-batch-7.md) PHẦN 2.

- **Việc (CPU thuần, không model):** với soup 5×0.86, chạy `merge_lora.py --diag` in 3 số:
  (a) **truncation_energy** = `Σ_{k>32} σ_k² / Σ_k σ_k²` của `ΔW_avg` per-module (năng lượng bị SVD-32 cắt);
  (b) **ingredient_diversity** = cosine Frobenius giữa các `ΔW_i` nguồn; (c) **drift_vs_anchor** so 0.86.
  Cộng `offline/deploy_check.py` trên artifact svd-32 (phải PASS).
- **Vì sao:** nếu các ΔW gần nhau (cùng basin) thì truncation_energy ≈ 0 → SVD-32 gần lossless → idea dưới an toàn.
  Nếu **ingredient_diversity ~1.0** → 5 adapter 0.86 quá giống nhau, soup ≈ identity → **cả nhánh soup khả năng = 0.86**,
  dừng sớm để **không tốn lượt submit nào**. Nếu truncation_energy cao → ưu tiên pairwise (B7-3) thay vì 5-way.
- **Falsify (offline, không tốn submit):** ingredient_diversity ~1.0 ⇒ nhánh soup chết trước khi submit;
  truncation_energy < ~0.05 ⇒ truncation không phải bottleneck, tiến hành.

---

## B7-1 — Product-SVD uniform soup 5×0.86 ⭐⭐ TOP (rẻ nhất/lý thuyết vững nhất)
- **Paper:** [ICML 2025 R2-FM wksp SVD-merging](https://icml.cc/virtual/2025/50854) + [Model Soups](https://arxiv.org/abs/2203.05482)
  (uniform soup các checkpoint cùng init khác hyperparameter "often improves accuracy").
- **Hypothesis:** 5 adapter 0.86 là 5 điểm trong cùng basin với lỗi *khác nhau trên từng problem*;
  trung bình product-space rơi gần tâm basin → flatter minimum → vài problem biên lật pass dưới greedy.
- **Code path:** thêm mode `--svd` vào `soup_adapters.py`: per module, `M=[B_1..B_n]` (out×nr),
  `N=[A_1;..;A_n]` (nr×in, fold weight 1/n); QR hai phía → SVD ma trận nhỏ (nr×nr) →
  `B_new=Q_M·U[:,:32]·√S`, `A_new=√S·Vᵀ[:32]·Q_Nᵀ`. lm_head full-weight: average thẳng. CPU, ~phút.
- **Expected:** +0–1.5pp; xác suất vượt 0.86 ~25%. 🟢 chi phí ≈ 0 (không train).
- **Risk:** (a) truncation loss nếu PROBE-M0 báo phổ rộng; (b) lm_head average có thể kéo lệch nếu
  một ingredient có lm_head drift mạnh — fallback: giữ lm_head của baseline.
- **Falsify:** diag PASS (drift≠0, trunc thấp) → submit (lượt #1 của batch); nếu leaderboard < 0.86 → soup-tâm-basin chết, cân nhắc B7-3 pairwise hoặc dừng nhánh.

## B7-2 — Subset selection bằng offline-diag (greedy-soup, ĐÃ SỬA cho no-vLLM) ⭐⭐
- **Paper:** [Model Soups](https://arxiv.org/abs/2203.05482) — greedy soup (thêm dần ingredient, chỉ giữ
  nếu held-out tăng) ổn định hơn uniform. **Nhưng ta KHÔNG có held-out accuracy** (Kaggle không chạy vLLM).
- **Hypothesis:** uniform 5-way có thể bị một ingredient "kéo lùi"; chọn tập con tốt hơn.
- **Code path (KHÔNG vLLM):** `offline/merge_greedy.py` quanh `merge_lora.py soup` + `--diag`: với từng tập con,
  tính offline-diag (truncation_energy thấp + ingredient_diversity cao = ứng viên tốt) → xếp hạng tập con,
  ra **1** tập con đáng submit. Greedy-by-accuracy gốc không khả thi → thay bằng greedy-by-diag-proxy + 1 submit.
- **Expected:** +0–1.5pp; 🟢. Lưu ý diag là proxy, KHÔNG phải accuracy → không "dominates B7-1" như paper.
- **Risk:** diag-proxy không tương quan accuracy → tập con "đẹp diag" vẫn có thể = 0.86. Chỉ submit nếu B7-1 đã ≥0.86.
- **Falsify:** nếu greedy hội tụ về đúng 1 ingredient (không thêm được ai) → các adapter 0.86 quá gần nhau, soup không có gì để trung bình → ưu tiên B7-8 (diversity).

## B7-3 — Pairwise WiSE-FT α-sweep, product-space ⭐
- **Paper:** WiSE-FT-style interpolation (gốc robust-FT) + [Model Soups](https://arxiv.org/abs/2203.05482).
- **Hypothesis:** nội suy 2 điểm là dạng soup ít rủi ro truncation nhất (rank thực ≤ 64, gần 32);
  tồn tại α giữa baseline↔exp21 (hai quỹ đạo train *khác nhau nhất* trong nhóm 0.86) cho điểm > hai đầu.
- **Code path:** sửa `wiseft()` trong `soup_adapters.py` sang product-space (cùng máy SVD của B7-1,
  weights `[(1-α), α]`); quét α ∈ {0.2, 0.35, 0.5} × các cặp {baseline↔exp21, baseline↔exp43, exp21↔exp40}.
- **Expected:** +0–1pp; 🟢. 9 biến thể build offline, không train, KHÔNG submit hết.
- **Risk:** không có local accuracy để chọn α → dùng diag (drift theo α phải đơn điệu, trunc thấp) chọn **đúng 1** biến thể submit; per-layer α (HMA, EMNLP 2024) là biến thể nâng cao nếu α toàn cục hòa.
- **Falsify:** nếu mọi cặp đều có dạng chữ U (giữa kém hơn hai đầu) → các adapter KHÔNG cùng basin như giả định → kill toàn nhánh interpolation (B7-1/2 cũng khó sống).

## B7-4 — Adapter-strength scaling (task-vector coefficient trên CHÍNH 0.86) ⭐ (rẻ nhất tuyệt đối)
- **Paper:** task-arithmetic scaling coefficient + [PEFT merging docs](https://huggingface.co/docs/peft/developer_guides/model_merging)
  (verify 3-0: weight ≥1.0 "preserve the correct scale", default 1.0).
- **Hypothesis:** 0.86 có thể đang over/under-shoot độ mạnh adapter dưới greedy decoding; một hệ số
  toàn cục s trên ΔW (≡ scale `lora_B` × s) là knob 1-chiều chưa từng quét.
- **Code path:** script 5 dòng: nhân mọi `lora_B.weight` với s ∈ {0.85, 0.9, 0.95, 1.05, 1.1}
  (KHÔNG scale lm_head full-weight — nó là replacement, không phải delta). Không cần SVD, không train.
- **Expected:** +0–0.5pp; xác suất thấp (~10–15%) nhưng chi phí = 1 probe sweep. 🟢.
- **Risk:** gần như không — nghịch lý duy nhất là lm_head full-weight giữ nguyên trong khi delta bị scale → mismatch nhẹ; nếu s tốt ≠ 1, thử thêm biến thể nội suy lm_head về base cùng tỷ lệ.
- **Falsify:** curve theo s phẳng quanh s=1 → 0.86 đã ở đúng scale, kill.

## B7-5 — TIES-SVD / DARE-TIES-SVD trên ΔW products
- **Paper:** TIES + DARE per [PEFT docs](https://huggingface.co/docs/peft/developer_guides/model_merging)
  (trim → sign-elect → average cùng dấu; DARE = drop+rescale; cả hai có biến thể `*_svd` cho LoRA).
  Cảnh báo từ [arXiv:2505.15875](https://arxiv.org/pdf/2505.15875) (verify 3-0): method thiết kế cho
  full-FT hoạt động kém trên LoRA vì magnitude-variance của LoRA lớn hơn nhiều.
- **Hypothesis:** nếu B7-1 hòa 0.86, nguyên nhân có thể là interference triệt tiêu giữa các ΔW;
  sign-election + trimming giữ phần đồng thuận, bỏ phần xung đột.
- **Code path:** thêm `--ties density` vào `soup_adapters.py`: trên mỗi `ΔW_i` (product), trim top-|density|
  theo magnitude → sign-elect per-element → mean cùng dấu → SVD-32. Density ∈ {0.7, 0.9}, weights 1.0.
- **Expected:** +0–0.5pp trên đầu B7-1; same-task nên sign-conflict hiếm → gain kỳ vọng thấp. 🟡.
- **Risk:** trimming trên same-task có thể xóa nhầm tín hiệu chung (mọi ΔW cùng hướng); literature
  (claim refuted "chỉ Task Arithmetic reliable") cho thấy method cầu kỳ thường KÉM hơn summation đơn giản trên LLM.
- **Falsify:** nếu ≤ B7-1 trên probe ở cả hai density → đúng dự đoán "same-task không có sign conflict", kill.

## B7-6 — Module-wise grafting (per-module-group selection)
- **Paper:** không có paper trực tiếp (gap thật — phần MoE/Mamba/per-module trong research KHÔNG có
  claim verify nào); seam nội bộ: exp43 (chỉ in_proj/out_proj) vẫn giữ 0.86 → từng nhóm module có thể
  thay nguồn độc lập mà không vỡ.
- **Hypothesis:** adapter tốt nhất per module-group khác nhau (vd: attention từ exp21, in/out_proj từ
  exp43, experts từ baseline) → ghép "frankenadapter" tốt hơn mọi ingredient nguyên khối.
- **Code path:** mode `--graft spec.json` trong `soup_adapters.py`: map module-group → source adapter
  (hoặc → soup B7-1 của group đó). Không SVD (lấy nguyên A/B từ một nguồn per group → rank-32 tự nhiên).
  Search: greedy per group (4 nhóm: attn / in+out_proj / experts+shared / lm_head) ≤ 8 probe.
- **Expected:** +0–1pp; 🟡. Zero-train nhưng tổ hợp lớn → giữ greedy, không quét cạn.
- **Risk:** module-group KHÔNG độc lập (A của layer này phối hợp với B layer khác qua residual) →
  graft chéo quỹ đạo có thể vỡ tệ hơn soup; bắt đầu bằng graft giữa exp40/42/43 (cùng init, an toàn hơn).
- **Falsify:** nếu mọi graft chéo < min(hai nguồn) trên probe → các quỹ đạo không tương thích per-module, kill.

## B7-7 — Re-seed soup ĐÚNG CÁCH (minh oan exp17) — idea duy nhất cần train mới
- **Paper:** [Model Soups](https://arxiv.org/abs/2203.05482) (điều kiện: cùng init, khác seed/hyperparameter).
- **Hypothesis:** exp17 regress vì factor-averaging (confound đã chứng minh ở Phát-hiện-3), không phải
  vì soup sai. 2–3 run continue-train-từ-0.86 config exp40 (đã biết giữ 0.86) khác seed shuffle/dropout
  → soup product-SVD → diversity thật sự (khác B7-1: các điểm 0.86 hiện có không được thiết kế để đa dạng).
- **Code path:** 2–3 lần Kaggle run config exp40 đổi `SEED` (knob đầu file) → tải adapter →
  `soup_adapters.py --svd` cùng baseline. Lưu ý memory `kaggle-only-no-modal`: chạy trên Kaggle.
- **Expected:** +0.5–1.5pp nếu cơ chế soup-variance-reduction hoạt động trên reasoning; đây là bản
  test SẠCH duy nhất của hypothesis đó. 🟡 chi phí 2–3 Kaggle runs.
- **Risk:** continue-train same-corpus ≈ no-op (bài học batch-6) → các re-seed có thể quá gần nhau,
  soup ≈ identity; đo khoảng cách ‖ΔW_i − ΔW_j‖ trước khi tốn probe.
- **Falsify:** nếu pairwise distance giữa re-seeds < 10% distance(baseline, exp21) → không có diversity để soup, kill không cần probe.

## B7-8 — Small-α diversity soup với nhóm 0.84
- **Paper:** [Model Soups](https://arxiv.org/abs/2203.05482) — ingredient hơi-kém vẫn có thể giúp soup
  (greedy soup quyết định, không phải điểm đơn lẻ).
- **Hypothesis:** exp1/exp5/exp10 (0.84, −0.02) thất bại vì lever của chúng *quá mạnh*, nhưng hướng
  ΔW của chúng chứa thông tin khác-basin (lever khác nhau) → pha α nhỏ vào tâm-soup 0.86 thêm diversity
  mà nhóm 0.86 (toàn optimization-tweak, gần nhau) thiếu.
- **Code path:** product-space weighted soup: weights = [1−α trên soup B7-1, α/3 mỗi adapter 0.84],
  α ∈ {0.1, 0.2}. Cùng máy `--svd`.
- **Expected:** +0–0.8pp; 🟡.
- **Risk:** kéo về phía các config đã regress; α ≤ 0.2 là trần cứng, và chỉ chạy sau khi B7-1/B7-2 cho tín hiệu (cần biết tâm-soup đứng đâu trước).
- **Falsify:** nếu cả α=0.1 lẫn 0.2 đều < B7-1 trên probe → 0.84-group là noise thuần, kill.

## B7-9 — Tail-checkpoint soup (LAWA-style, intra-run)
- **Paper:** dòng latest-weight-averaging/EMA (exp40 đã chứng minh EMA online giữ 0.86 — đây là biến
  thể offline thô hơn nhưng cho phép chọn K hậu nghiệm).
- **Hypothesis:** trong MỘT run continue-train (config exp40), trung bình K checkpoint cuối
  (cách nhau ~100 step) khử nhiễu minibatch cuối quỹ đạo — dạng soup "chắc chắn cùng basin" nhất tồn tại.
- **Code path:** sửa config save: `save_steps=100, save_total_limit=5` trong `run_training()` →
  một Kaggle run → `soup_adapters.py --svd` trên 3–5 checkpoint cuối. **Lưu ý:** cùng-run-cùng-init
  → factor align → đây là trường hợp duy nhất factor-avg cũng chấp nhận được, nhưng dùng `--svd` luôn cho nhất quán.
- **Expected:** +0–0.5pp; chủ yếu là biến thể an toàn để "mua" thêm một lần thử khi đã trả tiền một run. 🟢/🟡.
- **Risk:** trùng cơ chế với EMA exp40 (đã hòa 0.86) → khả năng cao cũng hòa; giá trị chính là gần-free khi đằng nào cũng chạy B7-7.
- **Falsify:** = 0.86 → xác nhận EMA-family đã bão hòa, đóng nhánh intra-run vĩnh viễn.

## B7-10 — Post-merge repair anneal (merge → vá 100 step anchored)
- **Paper:** thực hành "merge-then-finetune-briefly" (không có citation verify riêng — đánh dấu
  practitioner-grade); máy anchored-L2 = exp42 (đã giữ 0.86).
- **Hypothesis:** nếu PROBE-M0 cho thấy `truncation_energy` cao (SVD-32 cắt mất năng lượng thật), thì 100–300
  step continue-train artifact đã-merge trên corpus gốc (full, đúng thứ tự, anchored-L2 về chính
  artifact merge) đủ để "vá" phần năng lượng bị cắt mà không drift.
- **Code path:** Kaggle run: `RESET_WEIGHTS=False` nạp artifact B7-1, `NUM_STEPS=100–300`, `LR=5e-6`,
  anchored-L2 (máy exp42) λ=1e-3 anchor = artifact merge. Corpus gốc nguyên vẹn (luật batch-5).
- **Expected:** +0–0.5pp **chỉ khi** PROBE-M0 báo `truncation_energy` cao; ngược lại skip. 🟡.
- **Risk:** chính là continue-train same-corpus ≈ no-op; ở đây no-op lại là *chấp nhận được* (mục tiêu
  chỉ là phục hồi năng lượng bị cắt, không phải vượt).
- **Falsify:** điều kiện kích hoạt rõ: PROBE-M0 `truncation_energy` > ~0.05. Nếu thấp hơn → SVD-32 đã gần lossless, skip B7-10.

---

## Thứ tự chạy khuyến nghị

1. **PROBE-M0** (offline, không vLLM) — `--diag` soup 5×0.86. Nếu ingredient_diversity ~1.0 → dừng nhánh soup,
   **không tốn submit nào**.
2. Sinh **offline** mọi ứng viên rẻ — **B7-1** (uniform), **B7-4** (scale), **B7-3** (WiSE-FT), **B7-6** (graft) —
   chạy `--diag` cho hết, xếp hạng. Zero-train, zero-submit.
3. **Submit #1 = B7-1**. Đọc điểm leaderboard (kênh accuracy duy nhất).
4. Nếu B7-1 ≥ 0.86: submit thêm **1–2** ứng viên đa-dạng-nhất theo diag. Nếu < 0.86: nhánh soup yếu → cân nhắc dừng.
5. **B7-5 / B7-8** — chỉ khi B7-1 đã lên bảng ≥ 0.86.
6. **B7-7 (+B7-9)** — tốn Kaggle runs; chỉ khi muốn diversity thật và `--diag` xác nhận re-seed đủ khác.
7. **B7-10** — chỉ kích hoạt nếu PROBE-M0 báo truncation_energy cao.

**Quy tắc chung (no-vLLM):** KHÔNG có local accuracy eval — Kaggle không chạy vLLM. Lọc ứng viên bằng
**offline-diag** (CPU: truncation_energy / drift / diversity) + `offline/deploy_check.py` (phải PASS), rồi **submit
ít, best-first** (~3–5 lượt cả batch; Kaggle ~5 submit/ngày). Accuracy chỉ biết sau khi lên leaderboard. Chi tiết
giao thức: [plan-batch-7.md](plan-batch-7.md) PHẦN 2.

## Claims bị REFUTE trong deep-research (KHÔNG được trích dẫn ở batch sau)

- ✗ "Task Arithmetic vượt best-single khi merge ≥4 checkpoint trên LLM, gain tăng theo n" (1-2).
- ✗ "Chỉ Task Arithmetic reliable, method cầu kỳ luôn degrade trên LLM" (1-2 — chiều ngược lại cũng chưa chứng minh).
- ✗ "Linear merge trong Core Space ≡ full-space nên framework vô dụng cho soup same-task" (1-2).

## Nguồn chính (đã verify trong phiên 2026-06-11)

[Model Soups, arXiv:2203.05482](https://arxiv.org/abs/2203.05482) ·
[ICML 2025 R2-FM wksp DM/MM/SVD](https://icml.cc/virtual/2025/50854) ·
[KnOTS, arXiv:2410.19735](https://arxiv.org/abs/2410.19735) ·
[PEFT model-merging docs](https://huggingface.co/docs/peft/developer_guides/model_merging) ·
[Core Space, arXiv:2509.17786](https://arxiv.org/pdf/2509.17786) ·
[DO-Merging, arXiv:2505.15875](https://arxiv.org/pdf/2505.15875) ·
[LoraHub, arXiv:2307.13269](https://arxiv.org/abs/2307.13269)

**KHÔNG dùng:** LoraHub/CMA-ES (bằng chứng âm trên BBH, vẫn dính cross-term), Core Space/KnOTS
(cho cross-task, <100% best-single), DO-Merging (output full-rank, không tự cho rank-32).
