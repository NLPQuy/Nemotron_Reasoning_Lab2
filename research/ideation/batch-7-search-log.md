# Batch-7 search log — paper merging/ensemble TOP-TIER (search 2026-06-11, 4-agent sweep)

> Bổ sung cho [batch-7.md](batch-7.md). Mọi venue đều **verify qua OpenReview / proceedings / ACL Anthology /
> PMLR / CVF** — không tin arXiv suông. Ký hiệu áp dụng cho ràng buộc của ta (merge các adapter rank-32
> SAME-TASK, output = MỘT LoRA rank-32 vLLM-loadable, ưu tiên data-free):
> **(a)** full-delta → cần SVD-recompress-32 · **(b)** native trên factor/low-rank · **(c)** cần data/train · **(d)** data-free.

## ⚠️ ĐÍNH CHÍNH cho batch-7.md

1. **"ICML 2025 DM/MM/SVD" (icml.cc/virtual/2025/50854) KHÔNG phải main-conference** — chỉ là poster
   **ICML 2025 R2-FM Workshop** (Tang, Yadav, Sung, Yoon, Bansal — OpenReview t9FrMviTaP). Nội dung
   (Direct vs Multiplied vs SVD merging) vẫn đúng và là playbook của B7-1, nhưng cite phải ghi "workshop".
2. **Core Space** đã được nhận **NeurIPS 2025** (verify qua arXiv comments field). KnOTS = **ICLR 2025**
   (OpenReview 67X93aZHII). DO-Merging: vẫn không venue, không code.

## NHÓM 1 — Đúng-bài-toán nhất: same-task LoRA/checkpoint soup (đọc trước)

| Paper | Venue (verified) | Method 1 dòng | Vì sao quan trọng với ta | Code |
|---|---|---|---|---|
| **SeedLoRA** | **ICML 2025** (OpenReview 7QH48TtFZX) | merge nhiều LoRA **same-task khác seed**, 2 giai đoạn: giữ thành phần **consistent** chung → fuse thành phần **complementary** theo seed | ⭐ **PAPER ĐÚNG BÀI TOÁN NHẤT TÌM ĐƯỢC**: claim **vượt MỌI single LoRA run** trên **GSM8K/MATH** (MetaMathQA, LLaMA2-7B/Mistral-7B, exact-match). Nói thẳng: naive averaging suboptimal, cần structured fusion. = phiên bản published của B7-7 | github.com/NUS-HPC-AI-Lab/SeedLoRA |
| **Model Stock** | **ECCV 2024 ORAL** (ecva.net + Springer) | chỉ cần **2** fine-tune same-task: nội suy layer-wise về "tâm" phân bố weight theo góc giữa 2 delta, neo vào pretrained; closed-form, **(d)** | đúng regime "ít checkpoint"; thay mean bằng công thức tâm-theo-góc trong `merge_lora.py` gần như free; rank ≤64 → SVD-32 | github.com/naver-ai/model-stock |
| **LoRA-LEGO** | **ICLR 2025** (OpenReview j6fsbpAllN) | coi mỗi rank-row/col là Minimal-Semantic-Unit, gom MSU của k adapter, **k-means về đúng r centroid** → output **đúng rank-r by construction**, **(b)(d)** | một trong 2 method có output rank-32 tự nhiên không cần SVD; drop-in thay soup | github.com/Ablustrund/LoRA-LEGO |
| **IterIS** | **CVPR 2025** (CVF open access) | least-squares layer-wise: giải (A,B) **một adapter rank chọn trước** sao cho output khớp ensemble các LoRA nguồn; cần 1–5% unlabeled prompts, **(b)(c-nhẹ)** | cách published để **distill ensemble → 1 LoRA rank-32**; unlabeled = train.csv là đủ | github.com/HKUST-LongGroup/IterIS-merging |
| **LoRE-Merging** | **EMNLP 2025 Findings** (Anthology 2025.findings-emnlp.1195) | merge = joint **low-rank estimation** của các task vector, không cần base model; output natively low-rank, **(b)(d)** | test trên **GSM8K/MATH/MBPP**, thắng TIES/TA — gần nhất với "merge xong vẫn rank-32, thắng trên math" | (Huawei Noah) |
| **Selective Parameter Merging** | **EMNLP 2024 main** (Anthology 2024.emnlp-main.892) | merge các SFT model train **khác data-order**; element-wise **pick** thắng weighted-avg | đúng nỗi đau corpus-order của ta; thêm bằng chứng "mean trần trụi không phải operator tốt nhất" | — |
| **KnOTS** | **ICLR 2025** (OpenReview 67X93aZHII) | joint-SVD concat các ΔW=BA về basis chung rồi áp TIES/DARE trên hệ số aligned, **(a)(d)** | đã clone `refs/knots`; +4.3% vs naive LoRA-merge | github.com/gstoica27/KnOTS |
| **Core Space** | **NeurIPS 2025** (arXiv comments) | project các adapter vào core-basis chung (lossless proof) rồi merge trong đó, **(b)(d)** | đã clone `refs/core-space` | github.com/apanariello4/core-space-merging |

## NHÓM 2 — Lever TRAINING-TIME làm checkpoint "soupable" (cho B7-7/B7-9, cần train)

| Paper | Venue | Ý chính | Map vào repo |
|---|---|---|---|
| **CoTo** | **ICML 2025** (OpenReview Zha2m39ZoM) | dropout-cả-adapter theo lịch tăng dần trong training → flatten landscape, **tăng linear-mode-connectivity giữa các LoRA train độc lập** → plain-average hoạt động | thêm vào trainer khi chạy B7-7 re-seeds; github.com/zwebzone/coto |
| **LoRI** | **COLM 2025** (OpenReview b8cW86QcOD) | **freeze A** (random projection chung) + train B sparse → merge = cộng B trong cùng basis A, **exact rank-32, không alignment problem** | cộng hưởng trực tiếp exp48 Freeze-A (batch-6): train k seed cùng A đông cứng → soup B trivially đúng | github.com/juzhengz/LoRI |
| **Merge before Forget** | **ICLR 2026** (OpenReview i1Rj7yU6eF) | merge tuần tự các LoRA vào **MỘT LoRA rank cố định** (orthogonal-init A mới + time-aware scaling B/A) | output đúng khuôn deploy; recipe fold chuỗi checkpoint → 1 adapter | — |
| **DisTaC** | **ICLR 2026** (GitHub katoro8989/DisTaC) | pre-condition task vector trước merge: **equalize norm** các delta (+self-distill ngắn) | chẩn đoán norm-disparity: baseline vs exp40/42/43 có ‖ΔW‖ khác nhau → **equalize norm trước soup = 1 dòng code**, thêm vào `--diag` | github.com/katoro8989/DisTaC |

## NHÓM 3 — Combiner data-free đáng biết (đều cần SVD-32 sau merge)

| Paper | Venue | 1 dòng |
|---|---|---|
| Task Arithmetic | **ICLR 2023** (OpenReview 6t0Kwf8-jrj) | τ = θ_ft−θ_pre; cộng/trừ/scale — baseline soup chính là đây |
| TIES | **NeurIPS 2023** (papers.neurips.cc) | trim→sign-elect→mean cùng dấu; đã clone `refs/ties-merging` |
| DARE | **ICML 2024** (PMLR v235) | drop p% + rescale 1/(1−p); premise "delta redundant" yếu với LoRA rank-32 (đã nén sẵn) |
| PCB-Merging | **NeurIPS 2024** (proceedings) | cân bằng intra-salience × inter-competition per-param, data-free; same-task thì degenerate về magnitude-trim |
| TSV-M | **CVPR 2025** (CVF) | SVD per-layer task vector, whiten/decorrelate trước merge — SVD-native, khớp khuôn rank-32 |
| Iso-C/Iso-CTS | **ICML 2025** (PMLR v267) | flatten phổ singular của tổng — RỦI RO cho same-task greedy (đổi "LR hiệu dụng" của hướng trội) |
| TALL-masks/Consensus | **ICML 2024** (OpenReview DWT9uiGjxT) | mask đồng thuận ≥2 task — tái dụng được như **agreement filter giữa các seed** |
| RegMean | **ICLR 2023** (OpenReview FCnohuR6AnM) | least-squares per-layer bằng Gram input — **(c)** cần forward cache, nặng với 30B |
| Fisher merging | **NeurIPS 2022** (proceedings) | average theo diagonal Fisher — **(c)**, đắt ở 30B |
| AdaMerging | **ICLR 2024** (OpenReview nZP6NgD3QY) | học hệ số per-layer bằng entropy unlabeled — ý tưởng "per-layer coefficient" dùng được dạng grid offline |
| EMR-Merging | **NeurIPS 2024 Spotlight** | elect + per-task mask/rescaler **lúc inference** → ❌ vi phạm khuôn 1-adapter |
| Twin-Merging | **NeurIPS 2024** | shared + router động → ❌ router; chỉ giữ insight "shared component (= average) mang phần lớn giá trị" |
| DELLA | **KHÔNG venue** (arXiv-only, verify 2x) | MagPrune — dùng được qua mergekit nhưng cite là preprint |

## NHÓM 4 — Lý thuyết khi nào soup giúp/hại (điều kiện & kỳ vọng)

| Paper | Venue | Finding then chốt |
|---|---|---|
| Model Soups | **ICML 2022** (PMLR v162) | greedy-soup ≥ best-single by construction (trên held-out); gain vision ~0.5–1pp; **NLP gain marginal** |
| DiWA | **NeurIPS 2022** | phân rã bias-variance-covariance: soup thắng khi **diversity chức năng cao + cùng basin**; gain dồn vào distribution-shift |
| Neyshabur 2020 | **NeurIPS 2020** | fine-tune từ cùng pretrained init → cùng basin (điều kiện được thỏa cho pool 0.86) |
| **Juneja** | **ICLR 2023** | ⚠️ **counterweight quan trọng nhất**: same-task BERT fine-tunes có thể rơi vào **các basin RỜI NHAU** (khác heuristic tổng quát hóa) → soup xuyên cluster **HẠI**. Hàm ý: check interpolation-loss/diversity trước khi soup (đúng chức năng `--diag`) |
| Frankle LMC | **ICML 2020** | network "stable to SGD noise" từ sớm → con của cùng init nối tuyến tính được |
| Entezari | **ICLR 2022** | barrier biến mất modulo permutation — shared-init LoRA không có permutation mismatch |
| WARM | **ICML 2024** (PMLR v235) | average N reward model same-task: robust hơn ensemble, thắng single (LLM-scale) |
| HMA | **EMNLP 2024 main** | average RLHF↔SFT-init: **per-layer ratio thắng 1 α toàn cục** (tinh chỉnh cho B7-3) |
| Ajroldi | **ICML 2025** | ⚠️ tempering: EMA/LAWA trong-run chủ yếu mua **tốc độ train**, gain accuracy nhỏ/0 khi run đã tuned — khớp exp40 EMA = hòa 0.86 |
| LAWA-COLM | **COLM 2024** | average checkpoint cách xa trong pre-training thắng final — regime pre-train, yếu cho fine-tune ngắn |
| EMA TMLR | **TMLR 2024** | tham chiếu tốt nhất cho chọn EMA window vs LR |
| Model-merging scaling laws | KHÔNG venue (under review) | **gần như toàn bộ gain nằm ở 2–4 checkpoint đầu**, diminishing mạnh theo k; variance giảm theo k |
| What Matters at Scale (Yadav) | **TMLR 2025** | model càng to merge càng dễ, **method khác nhau càng ít quan trọng ở scale lớn** → mean đơn giản có thể đủ ở 30B |
| WARP | ❌ **REJECTED ICLR 2025** (OpenReview) | SLERP/LERP RLHF policies — chỉ cite làm industrial evidence |
| RAIN-Merging | **ICLR 2026 ORAL** | merge vào reasoning model dễ **phá thinking-format** → sau mọi soup phải kiểm tra format `\boxed{}`/`</think>` yield |
| Long-to-Short merging | KHÔNG venue | merge long-CoT↔short-CoT giữ accuracy, giảm ~55% length — lever nếu trace vượt 7680 token |
| Survey Yang (index) | ACM Comput. Surv. 2026 | github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications — index sống, jump-off point |

## Hệ quả cho batch-7 (cập nhật khuyến nghị)

1. **B7-7 (re-seed soup) được NÂNG HẠNG** — SeedLoRA (ICML 2025) là bằng chứng top-tier ĐẦU TIÊN cho
   "same-task LoRA seed merge > best single trên math exact-match". Nó cũng nói naive mean không đủ →
   nếu chạy B7-7, port **two-stage consistent+complementary fusion** của SeedLoRA (code công khai) thay vì mean.
2. **Thêm ứng viên B7-1b: Model Stock** — thay mean bằng closed-form tâm-theo-góc (2 ingredient + anchor
   base), data-free, vài chục dòng. Cùng họ với B7-1, đáng build cùng lượt offline.
3. **Thêm bước rẻ vào engine: norm-equalization (DisTaC)** — equalize ‖ΔW‖ giữa các ingredient trước soup;
   1 dòng, chữa đúng norm-disparity giữa baseline (train dài) và exp40/42/43 (continue ngắn).
4. **`--diag` được literature chống lưng**: Juneja (ICLR 2023) chứng minh same-task fine-tunes có thể ở
   basin rời nhau → đo diversity/interpolation trước khi soup không phải paranoia mà là điều kiện cần.
5. **Số ingredient: 2–4 là đủ** (scaling-laws + Model Stock) — đừng đốt effort vào soup 10-seed.
6. **Sau mọi soup: kiểm format** (RAIN-Merging ICLR'26 oral) — merge có thể phá `</think>`/`\boxed{}` adherence;
   diag thêm check nhanh: chạy tokenizer-level sanity trên vài completion nếu có điều kiện, hoặc chấp nhận rủi ro khi submit.
7. **Kỳ vọng vẫn khiêm tốn**: Wortsman NLP-gain marginal; Ajroldi nói trong-run averaging ≈ 0 gain khi đã tuned;
   nhưng SeedLoRA là datapoint dương đúng domain → trọng tâm dịch về **cross-seed diversity** (B7-7) hơn là
   soup 5 điểm 0.86 hiện có (vốn có thể quá giống nhau — PROBE-M0 sẽ trả lời).

## Repo — trạng thái clone (cập nhật 2026-06-11, xem [refs/README.md](../../refs/README.md) Batch-7 pass 2)

- ✅ **Đã clone:** `model-stock` (B7-1b), `iteris` (distill→1 LoRA), `lori` (freeze-A merge, có `merge_3_loras.py`),
  `distac` (norm-equalize), `awesome-model-merging` (index), `coto` (⚠️ README-only, code "coming soon").
- ❌ **KHÔNG có code công khai — vét máng đa-chiến-lược 2026-06-11, KẾT LUẬN DỨT KHOÁT (đừng đào lại):**
  - **SeedLoRA** (PMLR v267:38384; tác giả Yong Liu, Di Fu, Shenggan Cheng, …, Yang You): PMLR mục "Software"
    + trang lab trỏ `NUS-HPC-AI-Lab/SeedLoRA` nhưng **GitHub API trả 404 cứng** (repo KHÔNG tồn tại, không phải
    private — listing 60+ repo của org cũng không có). Đã check: 2 OpenReview (7QH48TtFZX + jkCvAAcSDa, `code`
    rỗng), 8 GH đồng tác giả, `gh search code`, papers-with-code, HuggingFace → trắng. **Implement từ paper.**
  - **LoRA-LEGO** (2409.16167; Ziyu Zhao=GH `StyxXuan`, …, Fei Wu): không link ở arXiv HTML body (v1/v3),
    OpenReview `code` rỗng, Awesome-Merging index không có cột code, `StyxXuan` có `smora`/`peft` nhưng KHÔNG có
    LEGO, `Ablustrund/LoRA-LEGO` (URL agent bịa) không tồn tại. **Implement MSU k-means từ paper** (gom A-row/B-col
    qua các adapter → k-means về 32 centroid → ráp lại rank-32) — numpy offline, vào `merge_lora.py` dạng `--lego`.
  - Chiến lược đã quét: org listing đầy đủ (NUS-HPC-AI-Lab, hpcaitech), 13 GH tác giả/đồng-tác-giả, `gh search
    code`×nhiều query, `gh search repos`, papers-with-code API, arXiv HTML body v1/v2/v3, PMLR proceedings,
    2 OpenReview forum/paper, Awesome-Model-Merging README, HuggingFace models. Re-check sau vài tuần.
