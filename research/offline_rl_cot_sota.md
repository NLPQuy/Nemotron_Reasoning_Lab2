# Deep research — leverage CoT bằng offline RL trên adapter đã SFT (2024–2026 SOTA)

> Câu hỏi gốc: "Đã có adapter từ exp21 (SFT/LoRA+ rank-32). Cầm nó continue-train bằng
> offline RL để leverage CoT triệt để được không?"
> Phạm vi: vét SOTA 2025–2026 (HF papers + OpenReview/arXiv) cho các hướng *offline*,
> *verifier-gated*, *learning-from-negatives*, ánh xạ vào ràng buộc của repo này.
> Cập nhật: 2026-06-03.

## 0. Ràng buộc của repo (để chấm điểm độ-fit của từng paper)
- **Off-policy only.** Không có vLLM-in-the-loop khi train (CLAUDE.md dead-end). Mọi method
  cần *regenerate rollout từ policy hiện tại mỗi N step* là loại.
- **LoRA rank 32**, continue-train từ adapter exp21 (`RESET_WEIGHTS=False`).
- **Verifier deterministic có sẵn** cho mọi category (`compare_answer`) → dán nhãn đúng/sai miễn phí.
- **Có sẵn trace SAI dồi dào**: corpus hiện chứa ~6.5% `rule_unknown`/`hypothesis_formed`
  (cryptarithm 92–93% sai) + rollout sai từ `generate_rollouts_vllm.py`. → negatives là tài nguyên, không phải rác.
- **Greedy inference, `max_tokens 7680`** → trajectory train phải < budget mới đóng được `\boxed{}`.

---

## 1. Bốn họ phương pháp, xếp theo độ-fit

### A. Offline RL "thật" trên dữ liệu tĩnh (KHÔNG cần paired, KHÔNG cần on-policy) — fit cao nhất
| Paper | Ý tưởng lõi | Vì sao fit repo |
|---|---|---|
| **OREO** — Offline RL for LLM Multi-Step Reasoning (NeurIPS'24/12-2024, [2412.16145](https://arxiv.org/abs/2412.16145)) | Học **policy + value** qua soft Bellman (max-entropy RL) trên dữ liệu offline. **Không cần paired preference**, credit-assignment token-level (hơn DPO ở chỗ DPO coi mọi token đồng đều). | Offline thuần, dùng đúng trace + nhãn verifier có sẵn. Value head là chi phí thêm nhưng vẫn off-policy. |
| **OXA** — Offline eXploration-Aware FT (03-2026, [2603.16206](https://arxiv.org/abs/2603.16206)) | 2 mục tiêu: (i) **đẩy** dữ liệu teacher-distill verified nhưng low-confidence; (ii) **đè** self-distill **sai** high-confidence → tái phân phối xác suất về phía đúng. +6 Pass@1 / +5 Pass@k vs SFT trên Qwen2.5-1.5B-Math, 6 benchmark. | Khớp y hệt tình huống: trace solver đúng = teacher; rollout model sai high-conf = thứ cần đè. |
| **RSO** — Statistical Rejection Sampling Optimization (ICLR'24, [OpenReview](https://openreview.net/forum?id=xbjSwwrQOe)) | Lấy preference data từ *policy tối ưu ước lượng* bằng rejection sampling rồi mới DPO → ước lượng tốt hơn SLiC/DPO. | Cải thiện chất lượng cặp cho bước DPO; offline. |

### B. Học từ NEGATIVES (trace sai) — fit cao, tận dụng đúng tài sản của repo
| Paper | Ý tưởng lõi | Caveat fit |
|---|---|---|
| **NFT** — Negative-aware Fine-Tuning (NVIDIA Labs, 05-2025, [2505.18116](https://arxiv.org/abs/2505.18116)) | SL thuần dựng **"implicit negative policy"** để tối ưu *trên cả trace sai* thay vì vứt đi. Vượt RFT; **ngang GRPO/DAPO** (7B vượt, 32B ~ DAPO). Chứng minh NFT ≡ GRPO ở strict on-policy. | ⚠️ Bản gốc **iterative, cần sample từ policy hiện tại** → đúng nghĩa on-policy. Muốn dùng phải hạ về **một vòng offline** (sample 1 lần bằng exp21, rồi train) — khi đó nó suy biến gần REDI. |
| **REDI** — Reinforcement Distillation (05-2025, [2505.24850](https://arxiv.org/pdf/2505.24850)) | Tận dụng **trace sai bị loại** (từ teacher hoặc self). 2 stage: (1) SFT trên trace đúng; (2) objective đẩy model **ra xa** trace sai. **Offline, không cần paired, không cần reference model.** | ✅ Fit nhất nhóm này cho ràng buộc off-policy. Stage-1 đã chính là SFT corpus hiện tại. |
| **V-STaR** (02-2024, [2402.06457](https://arxiv.org/abs/2402.06457)) | Train **verifier** bằng cả lời giải đúng *và* sai self-generated (DPO), dùng verifier để chọn ở test. | Repo đã có verifier *cứng* (luật) → ít cần verifier học; nhưng ý "dùng cả sai" trùng REDI/NFT. |
| **Learning from Mistakes: Negative Reasoning Samples** (01-2026, [2601.04992](https://arxiv.org/pdf/2601.04992)) | Negative reasoning samples cải thiện **OOD generalization**. | Tín hiệu: negatives giúp tổng quát hoá, đúng mục tiêu "test split ẩn". |

### C. RFT / STaR (chỉ giữ trace ĐÚNG) — baseline an toàn, nhưng trần thấp
| Paper | Đóng góp | Lưu ý |
|---|---|---|
| **STaR** (2022, [2203.14465](https://arxiv.org/abs/2203.14465)) | Bootstrapping: sinh CoT → verify → FT trên CoT đúng. Nền của RFT. | Kinh điển; trần bị chặn bởi pass-rate hiện tại. |
| **RFT scaling** (2023, [2308.01825](https://arxiv.org/abs/2308.01825)) | RFT > SFT, sinh nhiều reasoning-path đúng cải thiện rõ. | |
| **AdaSTaR** (05-2025, [2505.16322](https://arxiv.org/abs/2505.16322)) | **Adaptive sampling** (diversity + curriculum) sửa imbalance observation; tốt hơn & ít FLOPs hơn STaR. | Trực tiếp dùng được nếu chạy STaR. |
| **B-STaR** (12-2024, [2412.17256](https://arxiv.org/abs/2412.17256)) | Tự cân bằng **exploration/exploitation** giữa các vòng tự cải thiện. | Chống collapse khi lặp. |
| **HS-STaR** (05-2025, [2505.19866](https://arxiv.org/abs/2505.19866)) | Hierarchical sampling, dồn budget vào bài **boundary-level** (khó vừa). | ROI sampling — hợp khi budget rollout hữu hạn. |
| **Exploring Expert Failures** (04-2025, [2504.13145](https://arxiv.org/abs/2504.13145)) | Trích **hành động tốt từ trajectory THẤT BẠI** của expert → vẫn cải thiện. | Cầu nối RFT↔negatives. |

### D. Iterative DPO / preference cho reasoning — đã được kiểm chứng ngang RL, rẻ hơn
| Paper | Kết luận quan trọng |
|---|---|
| **Enhancing LLM Reasoning with Iterative DPO** (03-2025, [2503.12854](https://arxiv.org/abs/2503.12854)) | **Khuyến nghị recipe SFT → iterative DPO** (đừng bỏ SFT). Iterative DPO = "online RL với độ off-policy cao", **ngang/hơn PPO-GRPO mà ít compute hơn nhiều**; không cần critic. **Plateau sau 2–3 vòng**; gain **tương quan mạnh với năng lực base model**; tụt nếu verifier/rejection-sampling kém chất lượng. |
| **Iterative Reasoning Preference Optimization** (Meta, 04-2024, [2404.19733](https://arxiv.org/abs/2404.19733)) | DPO + **NLL term** trên cặp CoT thắng/thua → tăng GSM8K/MATH/ARC. Đây là khuôn iterative-DPO chuẩn cho CoT. |
| **Step-DPO** ([2406.18629](https://arxiv.org/abs/2406.18629)) / **Full-Step-DPO** ([2502.14356](https://arxiv.org/abs/2502.14356)) | Preference **theo từng bước** thay vì cả chuỗi → tốt cho long-chain; cần định vị bước sai. |
| **Self-Training + DPO improves CoT** (07-2024, [2407.18248](https://arxiv.org/abs/2407.18248)) | Self-training + DPO nâng CoT cho model nhỏ, rẻ. |
| **OREO** (mục A) | Tác giả chỉ trích DPO: cần paired + token đồng đều → đây là lý do chọn offline-RL nếu thiếu cặp tốt. |

---

## 2. Trần lý thuyết — điều KHÔNG method nào ở trên vượt được (đọc kỹ)
Đây là phần quyết định kỳ vọng thực tế cho hai category gãy nhất (bit_manip 9.6%, cryptarithm 0%).

- **"The Invisible Leash: Why RLVR May Not Escape Its Origin"** + **Self-Play Variational Problem
  Synthesis** ([2508.14029](https://arxiv.org/pdf/2508.14029)): RLVR **làm sắc (sharpen)** các mode
  *đã có* của base, **không sinh** mode mới; thường tăng Pass@1 nhưng **giảm Pass@k**, entropy sụp.
- **Weak-to-Strong Elicitation** ([2605.17314](https://arxiv.org/html/2605.17314)): on-policy RL FT
  sharpen mode sẵn có, để nguyên hoặc giảm pass@k của base.
- **Can Large Reasoning Models Self-Train?** ([2505.21444](https://arxiv.org/pdf/2505.21444)):
  self-training reward-hack/collapse nếu thiếu tín hiệu ngoài.

**Hệ quả cho repo:** RL/RFT/DPO **chỉ khuếch đại cái model đã đôi khi làm đúng**.
- cipher 36%, equation 31%, gravity 66%, unit 79%, numeral 97% → **có mode đúng để sharpen** ⇒ leverage được.
- **cryptarithm 0% / bit_manip 9.6%** → gần như **không có gì để bootstrap**; đây là vấn đề
  **data + độ dài CoT vượt budget**, *không* phải thiếu tín hiệu RL. Đừng kỳ vọng offline-RL cứu hai cái này.
  (Khớp memory `cryptarithm-unsolved-levers`: chỉ solver-fix mới động được.)

---

## 3. LoRA rank-32 có đủ để chịu giai đoạn RL/preference thứ hai không?
- **Tina** (LoRA-RL trên 1.5B đạt >43% AIME24 với chi phí nhỏ): **LoRA-RL khả thi trên micro-budget**.
- **Plasticity vs. Rigidity: Impact of Low-Rank Adapters on Reasoning on a Micro-Budget**
  ([2601.06677](https://arxiv.org/pdf/2601.06677)): rank thấp đủ cho reasoning nhưng có đánh đổi
  dẻo/cứng → **giai đoạn 2 nên LR thấp** để không ghi đè SFT.
- Rủi ro thực: rank-32 đã "tiêu" gần hết capacity ở SFT ⇒ stage-2 dễ **bão hoà/quên format**.
  Giảm thiểu: LR thấp (≈1e-5–2e-5), ít step, **trộn lại một phần SFT corpus** (anchor), hoặc
  DPO có reference-KL (SimPO reference-free thì phải thêm guard độ dài/format).

---

## 4. Phán quyết & recipe đề xuất cho exp21 → exp28

**Trả lời câu hỏi:** Có — *nhưng* "offline RL" đúng nghĩa ở đây là **offline preference / negative-aware
trên dữ liệu sinh một lần**, KHÔNG phải importance-sampling RL kiểu exp27 (dead-end), và KHÔNG phải
NFT/GRPO on-policy (cần vLLM-in-loop — dead-end).

**Recipe khuyến nghị (rủi ro tăng dần):**
1. **REDI-style 2 stage (fit nhất):** giữ SFT corpus làm stage-1; sinh rollout **một lần** bằng adapter
   exp21 (`generate_rollouts_vllm.py`), verifier tách đúng/sai, stage-2 = objective **đẩy xa trace sai +
   kéo gần trace đúng**, LR≈1e-5, mix một phần SFT. Tận dụng được kho negatives sẵn có. Offline thuần.
2. **Iterative-DPO (bằng chứng mạnh nhất, [2503.12854]):** cặp chosen=CoT đúng / rejected=CoT sai (gồm cả
   `rule_unknown` "bỏ cuộc" sẵn trong corpus). 1–2 vòng (plateau 2–3). Thêm NLL term kiểu IRPO.
   Reference-KL giữ format box.
3. **OXA-style:** nếu muốn 1 objective gọn — đè self-distill sai high-confidence, đẩy teacher-trace đúng low-confidence.
4. **OREO:** nếu thiếu cặp tốt và muốn credit-assignment token-level; chi phí value-head.

**Kỳ vọng trung thực:** gain tập trung ở cipher/equation/gravity (có mode đúng để sharpen). Hai category
0% sẽ **không** nhúc nhích bằng RL — để dành cho solver/format-fix. Đo category-level, đừng nhìn mỗi tổng 0.86.

**Tránh:** exp27 importance-weighted RL; bất kỳ method nào regenerate rollout trong vòng train.

---

---

# PHẦN II — Vét rộng "leverage CoT trajectory" (2025–2026), các họ mới

> Bổ sung sau yêu cầu "tìm thêm, 2025-2026 rất nhiều người làm". Đây là các trục *khác* với
> Phần I (vốn xoay quanh offline-RL/preference/negatives). Mỗi họ chấm độ-fit với ràng buộc repo.

## 6. Selective-loss / token-weighted SFT — ⭐ fit cao nhất, offline thuần, gần-zero infra
Insight chung: SFT phạt **mọi token đồng đều**, trong khi chỉ một thiểu số token "quyết định" đúng/sai.
RL ngầm reweight các token này qua reward. Tái-kỹ-thuật SFT để **dồn loss vào token quan trọng** ⇒
lấy được phần lớn lợi ích "kiểu RL" mà vẫn là SFT ổn định. **Đặc biệt hợp repo** vì trace solver-mirror
(bit_manip 816 dòng) đa số là token "filler" low-entropy → đang làm loãng tín hiệu.

| Paper | Cơ chế | Ghi chú fit |
|---|---|---|
| **Critical Token Fine-Tuning (CFT)** (10-2025, [2510.10974](https://arxiv.org/abs/2510.10974)) | Chỉ update token **functionally-indispensable** (xác định qua counterfactual perturbation); <12% token mà **vượt SFT**, giữ đa dạng token không-critical. | Áp thẳng lên corpus hiện tại; cơ chế mask `weights` của repo đã sẵn để reweight. |
| **PEAR** (02-2026, [2602.01227](https://arxiv.org/html/2602.01227v1)) | Importance-sampling reweight loss SFT ở mức token/block/seq; +14.6% pass@8 (AIME-25) so SFT chuẩn, **chuẩn bị tốt cho RL sau**. | 3 biến thể, chọn token-level. |
| **SFT-GO** (OpenReview [dPJJLv4q3r](https://openreview.net/forum?id=dPJJLv4q3r)) | Nhóm token theo importance, tối ưu **worst-group loss** + CE. | |
| **Beyond 80/20: High-Entropy Minority Tokens** (06-2025, [2506.01939](https://arxiv.org/abs/2506.01939), 190 upvote) | Chỉ **~20% token high-entropy** lái phần lớn gain RLVR. | Gợi ý tiêu chí chọn token (entropy) cho cả SFT-reweight. |
| **DelTA** (05-2026, [2605.21467](https://arxiv.org/abs/2605.21467), 204 upvote) | Discriminative token credit cho RLVR: khuếch đại hướng gradient đặc trưng, giảm nhiễu pattern chung. | Cho nhánh RL nếu đi tiếp. |
| **Not All Tokens Learn Alike** (05-2026, [2605.07660](https://arxiv.org/abs/2605.07660)) | Attention-entropy: "anchor" low-entropy (gradient ổn) vs "explorer" high-entropy (tín hiệu quý). | Lý thuyết cho việc reweight. |

## 7. Nén CoT (CoT compression) — ⭐ trực tiếp gỡ nút thắt `max_tokens 7680`
Repo train trên trace solver dài **vượt budget inference** ⇒ model học quỹ đạo không kịp đóng `\boxed{}`.
Nhóm này dạy CoT ngắn mà giữ độ chính xác — đúng đòn cho bit_manip/cryptarithm.

| Paper | Cơ chế |
|---|---|
| **TokenSkip** (02-2025, [2502.12067](https://arxiv.org/abs/2502.12067)) | Controllable compression: bỏ token ít quan trọng, **−40% token**, gần như không mất accuracy; train-time. |
| **CoT-Valve** (02-2025, [2502.09601](https://arxiv.org/abs/2502.09601)) | Tuning cho CoT **co giãn theo độ khó**; một model, nhiều độ dài. |
| **C3oT** (12-2024, [2412.11664](https://arxiv.org/abs/2412.11664)) | Compressor + conditioned training giữ thông tin cốt lõi, rút gọn. |
| **R1-Compress** (05-2025, [2505.16838](https://arxiv.org/abs/2505.16838)) | Chunk-level compression + search cho Long-CoT. |
| **Extra-CoT** (05-2026, [2602.08324](https://arxiv.org/abs/2602.08324)) | Extreme-ratio compression: SFT + RL + hierarchical reward, ép vào **token budget**. |
| **ASAP** (08-2025, [2508.05988](https://arxiv.org/abs/2508.05988)) | Prune theo first-token surprisal, giữ anchor/logic — cho code reasoning. |
| **DSS-GRPO** (03-2026, [2603.07598](https://arxiv.org/abs/2603.07598)) | RL nén theo segment, tách think/answer, scale theo độ khó. |

## 8. Self-correction / reflection trajectory — dạy model tự sửa, nhiều cái offline
| Paper | Đóng góp | Fit |
|---|---|---|
| **Self-rewarding correction for math** (02-2025, [2502.19613](https://arxiv.org/abs/2502.19613), 82 upvote) | 2-stage: sequential rejection sampling sinh trajectory tự-thưởng + tự-sửa → FT; rồi RL **rule-based**, **không cần reward model / không paired**, **offline-compatible**. | ✅ verifier luật của repo khớp "rule-based signal". |
| **Teaching LRMs Effective Reflection** (01-2026, [2601.12720](https://arxiv.org/abs/2601.12720)) | Self-critique FT + RL với reward "reflection hiệu quả"; lọc critique chất lượng. | |
| **STeP: Synthetic Self-Reflected Trajectories + Partial Masking** (05-2025, [2505.20023](https://arxiv.org/abs/2505.20023)) | Trajectory tự-phản-tỉnh + **partial masking** khi train → ít data hơn. | ✅ repo **đã có** cơ chế mask token — chỉ cần đổi nhãn mask. |
| **Dual-Phase Self-Evolved** (01-2026, [2601.05616](https://arxiv.org/abs/2601.05616)) | SFT + **difficulty-aware rejection sampling** dạy tự-sửa. | |
| **SuperCorrect** (10-2024, [2410.09008](https://arxiv.org/abs/2410.09008)) | Teacher hướng dẫn student sửa qua cross-model DPO. | |

## 9. Reward/Advantage-weighted offline — ⭐ bản "ĐÚNG" của exp27 (không suy biến)
exp27 hỏng vì importance-weight clip eps quá chặt. Dòng này chỉ ra: **offline RL không có tỉ số propensity
⇒ tương đương weighted-SFT**, giải bằng primitive train chuẩn, **không clipping, không degeneration**.

| Paper | Cơ chế |
|---|---|
| **A★-PO / Advantage-Weighted Regression cho LLM** (05-2025, [2505.20686](https://arxiv.org/pdf/2505.20686)) | Regress **optimal advantage** bằng least-squares, offline; bỏ critic, bỏ multiple-generation, **bỏ clipping & reward-normalization heuristic**. |
| **Offline RL by Reward-Weighted Fine-Tuning** (NeurIPS'25 poster, [2506.06964](https://arxiv.org/pdf/2506.06964)) | Chứng minh offline RL ≡ weighted-SFT; thêm variance-reduction bằng chuẩn hoá reward trên nhiều trajectory. |
| **PCL-Reasoner-V1.5** (01-2026, [2601.14716](https://arxiv.org/abs/2601.14716)) | 32B math: SFT + **offline RL** novel → SOTA AIME; bằng chứng offline-RL ăn ở quy mô lớn. |
| **Good/Better SFT prepares for RL** (02-2026, [2602.01058](https://arxiv.org/pdf/2602.01058)) | SFT thế nào để bệ phóng RL tốt hơn — định hướng thiết kế stage-1. |

## 10. Process Reward / step-level (fit thấp hơn — chủ yếu test-time/selection)
Với repo có **verifier luật cứng**, PRM học ít cần; ghi nhận để biết landscape.
- **OmegaPRM** (06-2024, [2406.06592](https://arxiv.org/abs/2406.06592)) — MCTS divide-conquer tự thu process-supervision, không cần người.
- **SPARK** (12-2025, [2512.03244](https://arxiv.org/abs/2512.03244)) — generator+verifier dựng PRM **reference-free**, vượt ground-truth ở vài setting.
- **AdaptiveStep** (02-2025, [2502.13943](https://arxiv.org/abs/2502.13943)) — chia step theo confidence, value-guided decoding.
- **GroundedPRM** (10-2025, [2510.14942](https://arxiv.org/abs/2510.14942)) — tree-guided, fidelity-aware, ít nhãn.

## 11. Data selection trace / "less-is-more" — chống nhiễm, ít mà tinh
- **LIMO / s1** — ~800–1000 trace **được curate** vượt cả RLVR ở vài benchmark; bằng chứng **chất > lượng**.
  → ủng hộ Tier-3 "lọc contaminated CoT" trong [data_status.md](data_status.md): bỏ `rule_unknown` thay vì nhồi thêm.
- **Difficulty-Aware CoT Distillation** ([2509.05226](https://arxiv.org/abs/2509.05226)) — độ dài trace tỉ lệ độ khó.

## 12. Trajectory stitching (fit thấp — vướng greedy single-model)
- **R-Stitch** (07-2025, [2507.17307](https://arxiv.org/abs/2507.17307)) — ghép SLM+LLM theo token lúc decode (entropy-gated). Cần 2 model ở inference → **không hợp** ràng buộc 1-adapter/greedy của cuộc thi. Ghi nhận để loại.

---

## 13. Cập nhật phán quyết sau Phần II
Hai họ MỚI vọt lên trên cả offline-RL vì **rẻ hơn, offline thuần, và đánh đúng bệnh của repo**:

1. **Selective-token / reward-weighted SFT (§6, §9)** — *làm trước*. Đây vừa là bản **đúng** của exp27
   (reward-weighted = offline RL, không clip, không suy biến), vừa sửa "trace verbose làm loãng tín hiệu".
   Tận dụng **đúng cơ chế mask `weights` repo đã có** → chỉ đổi cách tính trọng số token. Gần-zero infra.
2. **CoT compression (§7)** — *song song*. Gỡ trực tiếp nút `max_tokens 7680` cho bit_manip/cryptarithm,
   thứ mà offline-RL **không** chạm tới được (xem §2 trần lý thuyết).
3. **REDI / iterative-DPO / negatives (Phần I)** — *bet chính cho category có mode đúng để sharpen*
   (cipher/equation/gravity).
4. **Self-rewarding correction (§8)** — nếu muốn dạy kỹ năng tự-sửa, offline, khớp verifier luật.

Thứ tự thực thi đề xuất: (1) đo rollout-yield bằng adapter exp21 → (2) thử **CFT/reward-weighted SFT**
trên corpus sạch (rẻ nhất, ít rủi ro) → (3) nếu category có headroom thì REDI/iterative-DPO →
(4) nén CoT cho bit_manip. Mọi bước đo **category-level**.

---

---

# PHẦN III — Bootstrap "đã-giải → chưa-giải": expert iteration / curriculum / synthesis

> Khung mới (yêu cầu của user): dùng trace **đã giải** + sampling verify-gated để **giải thêm bài
> chưa giải trong cùng category** → *mở rộng coverage*, không chỉ tái dùng trace. Đây là họ
> **expert iteration / STaR / ReST-EM**. Enabler đặc thù repo: **`train.csv` có `answer` cho MỌI bài
> kể cả unsolved** → verify-gate được; và các category là **sinh thủ tục (generator)** → có thể tạo
> vô hạn bài có nhãn ở độ khó tuỳ chọn (synthesis rẻ bất thường so với math).

## 14. Bốn câu hỏi con của expert-iteration, ánh xạ vào repo

### 14.A — Nền tảng (canonical)
| Paper | Đóng góp |
|---|---|
| **STaR** ([2203.14465](https://arxiv.org/abs/2203.14465)) / **RFT scaling** ([2308.01825](https://arxiv.org/abs/2308.01825)) | Sinh CoT → verify → FT trên bản đúng; lặp. Nền của tất cả. |
| **ReST-EM** (Beyond Human Data, DeepMind 2023) | Expectation-Maximization self-training **offline**: E-step = sample+verify, M-step = FT trên bản đúng. Bản offline chuẩn cho repo. |
| **Formal Math Statement Curriculum Learning** (OpenAI, [2202.01344](https://arxiv.org/abs/2202.01344)) | Expert iteration giải được **curriculum bài không có ground-truth proof** — bằng chứng kinh điển bootstrap tới bài chưa giải. |
| **Easy-to-Hard Generalization** ([2403.09472](https://arxiv.org/abs/2403.09472)) | Train trên bài **dễ** + reward model → tổng quát sang bài **khó** vượt giám sát người. Khung cho "dùng dễ học khó". |
| **ReGenesis** ([2410.02108](https://arxiv.org/abs/2410.02108)) | Tự tổng hợp reasoning path (general→task-specific), cải thiện **OOD** — quan trọng cho test split ẩn. |

### 14.B — *Chọn bài nào để bootstrap* (curriculum / sampling) — né cold-start & under-determined
| Paper | Cơ chế | Dùng ở repo |
|---|---|---|
| **HS-STaR** ([2505.19866](https://arxiv.org/abs/2505.19866)) ←link bạn chọn | Hierarchical sampling, dồn budget vào bài **boundary-level**, bỏ quá-khó/quá-dễ. | Xếp unsolved theo pass-rate, đánh sát biên năng lực. |
| **AUTO-CEI — Automatic Curriculum Expert Iteration** ([2410.07627](https://arxiv.org/abs/2410.07627)) | Explore trajectory **gần policy**, kéo path sai về đúng; curriculum tự động giảm compounding error. | Khung vòng lặp chính. |
| **Self-Evolving Curriculum** ([2505.14970](https://arxiv.org/abs/2505.14970)) | Chọn bài như **non-stationary bandit** để tối ưu học. | Lịch chọn bài qua các vòng. |
| **Curriculum RL easy→hard** ([2506.06632](https://arxiv.org/abs/2506.06632)) | Lý thuyết: easy→hard cải thiện **sample-efficiency**. | Biện minh thứ tự dễ→khó. |
| **Edge of Learnability — Teaching Models to Teach Themselves** ([2601.18778](https://arxiv.org/abs/2601.18778), 43 upvote, Meta 01-2026) | Tự sinh **curriculum tự động** để giải bài **trước đó bất khả**, qua "stepping stones" + meta-RL ở **rìa học được**. | ⭐ Trực diện câu hỏi: làm sao chạm bài cold-start (cryptarithm 0%) bằng bậc thang khó dần. |

### 14.C — *Cold-start*: lấy thắng lợi đầu trên bài khó (khi pass-rate ≈ 0)
| Paper | Cơ chế |
|---|---|
| **Context-Bootstrapped RL** ([2603.18953](https://arxiv.org/abs/2603.18953)) | Tiêm **demonstration** (few-shot) có annealing curriculum để khởi động exploration trên bài khó. → repo: bơm cấu trúc bài đã-giải vào prompt khi sampling để gỡ cold-start. |
| **h1: Bootstrapping longer-horizon** ([2510.07312](https://arxiv.org/abs/2510.07312)) | **Synthetic problem composition** + curriculum RL, outcome-only reward, không cần dense supervision. |
| **Exploring Expert Failures** ([2504.13145](https://arxiv.org/abs/2504.13145)) | Trích **phần đúng từ trajectory thất bại** → vẫn có tín hiệu khi chưa có bản hoàn toàn đúng. |

### 14.D — *Khi set unsolved cạn / quá khó*: synthesize bài mới cùng category
> Lợi thế repo: category là **generator thủ tục** → tạo bài có nhãn vô hạn, KHÔNG cần self-play LLM để bịa đề.
| Paper | Cơ chế |
|---|---|
| **SwS — Self-aware Weakness-driven Problem Synthesis** ([2506.08989](https://arxiv.org/abs/2506.08989)) | Dò **điểm yếu** của model → sinh bài nhắm đúng điểm yếu, rồi RLVR. ⭐ Khớp: dò loại bit-rule model hay trượt → generator tạo thêm đúng loại đó. |
| **ScaleDiff** ([2509.21070](https://arxiv.org/abs/2509.21070)) | Sinh bài **khó** có chủ đích, train cost-efficient. |
| **Socratic-Zero** ([2509.24726](https://arxiv.org/abs/2509.24726)) / **STP** ([2502.00212](https://arxiv.org/abs/2502.00212)) / **ANCORA** ([2604.27644](https://arxiv.org/abs/2604.27644)) | Self-play proposer–solver–verifier co-evolve sinh curriculum. (Repo có generator nên đây là tuỳ chọn, không bắt buộc.) |
| **Self-Play Variational Problem Synthesis** ([2508.14029](https://arxiv.org/abs/2508.14029)) | Sinh biến thể đề để **duy trì Pass@k / chống collapse** khi lặp RLVR. |

### 14.E — *Chống sụp đổ đa dạng qua các vòng* (bootstrapping error / collapse)
- **B-STaR** ([2412.17256](https://arxiv.org/abs/2412.17256)) — cân bằng explore/exploit động qua vòng.
- **AdaSTaR** ([2505.16322](https://arxiv.org/abs/2505.16322)) — adaptive diversity/curriculum sampling.
- **R-Diverse** ([2602.13103](https://arxiv.org/abs/2602.13103)) / **DARC** ([2601.13761](https://arxiv.org/abs/2601.13761)) / **Self-Guided Self-Play** ([2604.20209](https://arxiv.org/abs/2604.20209)) — chống "diversity illusion", reward-hacking, bootstrapping-error trong vòng self-play.

### 14.F — *Augment bài ĐÃ-GIẢI → bài KHÓ HƠN* (problem evolution / difficulty hiking)
> Đây là câu hỏi của user. Khác §14.D (synthesis from scratch): ở đây **lấy seed đã giải → mutate khó hơn**,
> giữ verify được. ⭐ **Khớp repo một cách hiếm có**: các category đã LÀ "executable spec" (generator +
> verifier trong `reasoners/`), nên chế độ "hardened" dưới đây là **native**, không cần LLM bịa đề.

| Paper | Cơ chế "làm khó hơn" | Map vào repo |
|---|---|---|
| **VeRA — Verified Reasoning Data Augmentation at Scale** ([2602.13217](https://arxiv.org/abs/2602.13217)) | Biến 1 seed → executable spec (template + generator + verifier) → **vô hạn biến thể có nhãn**. **VERA-E** = rewrite tương đương; **VERA-H (hardened)** = **tăng độ phức tạp mà vẫn verify được**. | ⭐ Bản mô tả chính xác hạ tầng repo: generator+verifier đã có ⇒ VERA-H là free. |
| **WizardMath / Evol-Instruct (RLEIF)** ([2308.09583](https://arxiv.org/abs/2308.09583)) | **Upward evolution**: thêm ràng buộc, cụ-thể-hoá, tăng số bước suy luận → khó hơn (8 vòng, 15k→96k). | Thêm toán hạng/độ dài/op cho bit_manip & cryptarithm. |
| **MathForge — "Harder Is Better"** ([2601.20614](https://arxiv.org/abs/2601.20614), 119 upvote, 01-2026) | Difficulty-Aware GRPO + **Multi-Aspect Question Reformulation** (cải biến đề nhiều mặt). | Rất mới, attention cao; khung reformulation. |
| **SAND-Math** ([2507.20527](https://arxiv.org/abs/2507.20527)) | **"Difficulty Hiking"** — nâng độ phức tạp của bài có sẵn. | Trực tiếp "leo độ khó". |
| **EvolProver** ([2510.00732](https://arxiv.org/abs/2510.00732)) | Evolve bài hình thức qua **symmetry & difficulty**. | Tương tự cho bài có cấu trúc. |
| **MetaMath** ([2309.12284](https://arxiv.org/abs/2309.12284)) | Bootstrap đề: rephrase / **backward** (tự đặt ẩn) — đa dạng, không hẳn khó hơn. | Biến thể backward cho equation. |
| **QuestA** ([2507.13266](https://arxiv.org/abs/2507.13266)) | Augment đề bằng **chèn lời giải một phần** → cải thiện pass@k & sample-efficiency lúc RL. | ⭐ Gỡ cold-start: chèn 1 phần CoT đúng vào bài khó. |
| **MathFimer** ([2502.11684](https://arxiv.org/abs/2502.11684)) | Fill-in-the-middle để **giãn bước suy luận** trong lời giải (augment SOLUTION, không phải đề). | Làm CoT dày bước hơn nếu cần. |

**Bằng chứng cổ vũ (đúng loại task của repo):**
- **Self-Improving Transformers Overcome Easy-to-Hard & Length Generalization** ([2502.01612](https://arxiv.org/abs/2502.01612))
  — self-generated curriculum **vượt được easy→hard và length-generalization** trên **arithmetic, string
  manipulation, maze**. Repo chính là arithmetic + bit/string manipulation ⇒ tín hiệu tích cực rằng
  "augment dễ→khó" có thể chạy cho bit_manip/cryptarithm.

**Cảnh báo OOD (đọc trước khi đầu tư):**
- **Query & Response Augmentation Cannot Help OOD Math Generalization** ([2310.05506](https://arxiv.org/abs/2310.05506),
  MuggleMath) — augment giúp **in-domain** nhưng **không** chắc giúp OOD. Test split ẩn có thể là OOD.
- **Superficial Self-Improved Reasoners** ([2503.02103](https://arxiv.org/abs/2503.02103)) — self-improve
  dễ thành **memorize**: in-domain ↑ nhưng OOD ↓; model-merging cứu phần nào.
  → Phải đo trên **held-out / category-level**, không chỉ train accuracy.

## 15. Đúc kết Phần III cho repo
- **Lợi thế hiếm:** nhãn có sẵn cho mọi bài + generator thủ tục ⇒ expert-iteration **và** weakness-driven
  synthesis đều rẻ; bottleneck là *sinh CoT đúng*, không phải nguồn đề.
- **Lộ trình:** (1) ReST-EM/AUTO-CEI vòng verify-gated trên **bit_manip + equation** (có hạt giống);
  (2) chọn bài bằng **HS-STaR/Edge-of-Learnability** (boundary), né under-determined & cold-start-0%;
  (3) cho cryptarithm 0%: thử **Context-Bootstrapped** (bơm exemplar) hoặc **easy→hard** từ bài
  more-determined; nếu vẫn 0 → xác nhận thuộc trần thông tin, dừng;
  (4) **SwS-style**: dò bit-rule hay trượt → generator sinh thêm đúng loại → vòng kế.
- **Cảnh báo:** dùng B-STaR/AdaSTaR để khỏi collapse; đo **Pass@k** (không chỉ Pass@1) để phát hiện
  mất đa dạng (Invisible Leash / Variational-Synthesis).

---

## 5. Nguồn
- OREO — https://arxiv.org/abs/2412.16145
- OXA (Offline eXploration-Aware FT) — https://arxiv.org/abs/2603.16206
- RSO (Statistical Rejection Sampling) — https://openreview.net/forum?id=xbjSwwrQOe
- NFT (NVIDIA Labs) — https://arxiv.org/abs/2505.18116 · code https://github.com/NVlabs/NFT
- REDI (Reinforcement Distillation) — https://arxiv.org/pdf/2505.24850
- V-STaR — https://arxiv.org/abs/2402.06457
- Learning from Mistakes (neg. samples, OOD) — https://arxiv.org/pdf/2601.04992
- STaR — https://arxiv.org/abs/2203.14465 · RFT scaling — https://arxiv.org/abs/2308.01825
- AdaSTaR — https://arxiv.org/abs/2505.16322 · B-STaR — https://arxiv.org/abs/2412.17256 · HS-STaR — https://arxiv.org/abs/2505.19866
- Exploring Expert Failures — https://arxiv.org/abs/2504.13145
- Enhancing LLM Reasoning with Iterative DPO — https://arxiv.org/abs/2503.12854
- Iterative Reasoning Preference Optimization (Meta) — https://arxiv.org/abs/2404.19733
- Step-DPO — https://arxiv.org/abs/2406.18629 · Full-Step-DPO — https://arxiv.org/abs/2502.14356
- Self-Training + DPO — https://arxiv.org/abs/2407.18248
- Invisible Leash / Self-Play Variational Problem Synthesis — https://arxiv.org/pdf/2508.14029
- Weak-to-Strong Elicitation — https://arxiv.org/html/2605.17314
- Can Large Reasoning Models Self-Train? — https://arxiv.org/pdf/2505.21444
- Tina (LoRA-RL) / Plasticity vs Rigidity — https://arxiv.org/pdf/2601.06677

**Phần II — leverage CoT trajectory (rộng):**
- Selective-token SFT: CFT https://arxiv.org/abs/2510.10974 · PEAR https://arxiv.org/html/2602.01227v1 · SFT-GO https://openreview.net/forum?id=dPJJLv4q3r
- Token entropy/credit: 80/20 high-entropy https://arxiv.org/abs/2506.01939 · DelTA https://arxiv.org/abs/2605.21467 · Not All Tokens Learn Alike https://arxiv.org/abs/2605.07660
- CoT compression: TokenSkip https://arxiv.org/abs/2502.12067 · CoT-Valve https://arxiv.org/abs/2502.09601 · C3oT https://arxiv.org/abs/2412.11664 · R1-Compress https://arxiv.org/abs/2505.16838 · Extra-CoT https://arxiv.org/abs/2602.08324 · ASAP https://arxiv.org/abs/2508.05988 · DSS-GRPO https://arxiv.org/abs/2603.07598
- Self-correction: Self-rewarding correction https://arxiv.org/abs/2502.19613 · Effective Reflection https://arxiv.org/abs/2601.12720 · STeP https://arxiv.org/abs/2505.20023 · Dual-Phase https://arxiv.org/abs/2601.05616 · SuperCorrect https://arxiv.org/abs/2410.09008
- Reward/advantage-weighted offline: A★-PO https://arxiv.org/pdf/2505.20686 · Reward-Weighted FT (NeurIPS'25) https://arxiv.org/pdf/2506.06964 · PCL-Reasoner-V1.5 https://arxiv.org/abs/2601.14716 · Good/Better SFT for RL https://arxiv.org/pdf/2602.01058
- PRM (auto): OmegaPRM https://arxiv.org/abs/2406.06592 · SPARK https://arxiv.org/abs/2512.03244 · AdaptiveStep https://arxiv.org/abs/2502.13943 · GroundedPRM https://arxiv.org/abs/2510.14942
- Data selection: Difficulty-Aware Distillation https://arxiv.org/abs/2509.05226 (LIMO/s1 referenced)
- Trajectory stitching (loại): R-Stitch https://arxiv.org/abs/2507.17307

**Phần III — expert iteration / curriculum / synthesis (bootstrap đã-giải→chưa-giải):**
- Canonical: STaR https://arxiv.org/abs/2203.14465 · RFT scaling https://arxiv.org/abs/2308.01825 · ReST-EM "Beyond Human Data" (DeepMind 2023) · Formal Math Curriculum (OpenAI) https://arxiv.org/abs/2202.01344 · Easy-to-Hard https://arxiv.org/abs/2403.09472 · ReGenesis https://arxiv.org/abs/2410.02108
- Chọn bài / curriculum: HS-STaR https://arxiv.org/abs/2505.19866 · AUTO-CEI https://arxiv.org/abs/2410.07627 · Self-Evolving Curriculum https://arxiv.org/abs/2505.14970 · Curriculum-RL easy→hard https://arxiv.org/abs/2506.06632 · Edge-of-Learnability https://arxiv.org/abs/2601.18778
- Cold-start: Context-Bootstrapped RL https://arxiv.org/abs/2603.18953 · h1 https://arxiv.org/abs/2510.07312 · Exploring Expert Failures https://arxiv.org/abs/2504.13145
- Synthesis: SwS https://arxiv.org/abs/2506.08989 · ScaleDiff https://arxiv.org/abs/2509.21070 · Socratic-Zero https://arxiv.org/abs/2509.24726 · STP https://arxiv.org/abs/2502.00212 · ANCORA https://arxiv.org/abs/2604.27644 · Self-Play Variational Synthesis https://arxiv.org/abs/2508.14029
- Chống collapse: B-STaR https://arxiv.org/abs/2412.17256 · AdaSTaR https://arxiv.org/abs/2505.16322 · R-Diverse https://arxiv.org/abs/2602.13103 · DARC https://arxiv.org/abs/2601.13761 · Self-Guided Self-Play https://arxiv.org/abs/2604.20209
- Augment solved→harder (§14.F): VeRA https://arxiv.org/abs/2602.13217 · WizardMath/Evol-Instruct https://arxiv.org/abs/2308.09583 · MathForge https://arxiv.org/abs/2601.20614 · SAND-Math https://arxiv.org/abs/2507.20527 · EvolProver https://arxiv.org/abs/2510.00732 · MetaMath https://arxiv.org/abs/2309.12284 · QuestA https://arxiv.org/abs/2507.13266 · MathFimer https://arxiv.org/abs/2502.11684 · Self-Improving Transformers (easy→hard/length) https://arxiv.org/abs/2502.01612 · MuggleMath OOD caveat https://arxiv.org/abs/2310.05506 · Superficial Self-Improved https://arxiv.org/abs/2503.02103
</content>
</invoke>
