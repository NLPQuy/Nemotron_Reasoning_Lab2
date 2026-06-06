# Idea Batch 4 — grounded trong `refs/` source code (không phải prose)

**Generated**: 2026-06-06
**Khác batch 1–3**: mỗi idea neo vào **code thật** của repo trong [refs/](../../refs/) (clone chính thức của paper), đã đọc tận file/hàm. Cột "code path" là thứ đã `grep`/`Read`, không phải suy từ abstract.
**Target**: 0.86 → 0.88+ | **Exp range**: exp28–exp37
**Ràng buộc** (giữ nguyên): off-policy only (vLLM-in-loop = dead-end), LoRA r32 **continue-train từ adapter SFT 0.86**, verifier luật `compare_answer` miễn phí, budget 7680 tok, greedy 1-pass. RunPod = sinh data offline (1 lần), Kaggle = training.

---

## 0. Feasibility triage — đọc CODE rồi mới chấm (quan trọng)

Đọc entry-point thật cho thấy nhiều method **không** dùng được nguyên trạng vì gắn chặt verl/ray/OpenRLHF/AReaL (on-policy, vLLM-in-loop). Nhưng **stage sinh-data offline** của chúng thì **lift được**.

| Ref | Infra thật trong repo | Dùng được gì cho ta |
|---|---|---|
| [a-po](../../refs/a-po) | `verl/trainer/apo/` (Ray+FSDP, on-policy stage-2) | ✅ **stage-1 offline value-est** (8 gen/prompt → baseline V) — tách khỏi verl được |
| [nft](../../refs/nft) | `ray_nft_trainer.py`, `experience_maker.py` (Ray, on-policy) | ⚠️ chỉ ý tưởng loss; infra bỏ |
| [oreo](../../refs/oreo) | OpenRLHF + value head (`train_pcl.py`) | ⚠️ value-head nặng; bỏ |
| [self-rewarding-correction](../../refs/self-rewarding-correction) | `infer_math/` (vLLM gen offline) **+** `ppo_training/` (verl) | ✅ **infer_math gen-pipeline offline**; bỏ ppo stage |
| [oxa](../../refs/oxa) | `1-prepare_sft_data/` (pure python) + LLaMA-Factory SFT | ✅ **toàn bộ data-prep + SFT** (không RL) |
| [redi](../../refs/redi) | `experiments_trl/open_r1_sft.py` (TRL SFTTrainer) | ✅ **objective port được** vào Continuer loop |
| [dpo-st](../../refs/dpo-st) | `generate.py`/`sft.py`/`dpo.py` (TRL, **không** verl) | ✅ **toàn pipeline** scriptable |
| [tokenskip](../../refs/tokenskip) | `LLMLingua.py` + `data_processing/` (pure python) + LoRA-SFT | ✅ **engine nén + SFT** |
| [self-improving-transformers](../../refs/self-improving-transformers) | `run_self_improve.py` (TRL 0.10, pure) | ✅ **vòng self-gen curriculum** |
| [sws](../../refs/sws) | `src/{problem_generation,data_synthesis}.py` (vLLM gen) | ✅ gen-pipeline; thay vLLM-gen bằng generator của ta |
| [vera](../../refs/vera) | `cli/prepare_vera.py` (LLM-judge oracle) | ✅ khung augment; **thay judge bằng verifier luật** |
| [sand-math](../../refs/sand-math) | `pipeline/*.py` (vLLM gen) + LLaMA-Factory SFT | ✅ pipeline difficulty-hiking |
| [step-dpo](../../refs/step-dpo) | TRL DPO + step-locate | ✅ data-format + objective |

**Nguyên tắc:** ta chỉ lấy **stage offline-data** + **objective** của mỗi ref, ghép vào hạ tầng repo (RunPod gen → `corpus_preprocessed.jsonl` → Kaggle Continuer SFT continue-0.86). Không port verl/ray/OpenRLHF.

---

## 1. Summary table

| # | exp | Idea | Ref + code path thật | Cơ chế (theo code) | GPU gen? | Volume | Code-reuse |
|---|-----|------|----------------------|--------------------|:--:|:--:|:--:|
| 1 | exp28 | **OXA dual data** (LP-promote + UL-suppress) | [oxa/1-prepare_sft_data/make_SFT_LP_data.py](../../refs/oxa/1-prepare_sft_data/make_SFT_LP_data.py), `make_ULloss_data.py` | rank trace verified theo **PPL dưới policy**; upweight high-PPL đúng + unlikelihood low-PPL **sai** | ✓ (PPL+gen) | corpus reweight | 🟢 cao |
| 2 | exp29 | **REDI** 2-stage negative-aware | [redi/experiments_trl/open_r1_sft.py](../../refs/redi/experiments_trl/open_r1_sft.py) | `preprocess_data_filter` tách đúng/sai → SFT đúng + đẩy-xa sai | ✓ (rollout) | +negatives | 🟢 cao |
| 3 | exp30 | **A★-PO** offline optimal-advantage weight | [a-po/preprocess/data_generation/model_generate.py](../../refs/a-po/preprocess/data_generation/), README §Offline | 8 gen/prompt → V (β1=0.5) → advantage-regress (β2=1e-3) thành **per-traj weight** | ✓ (8/prompt) | reweight all | 🟡 (bỏ verl) |
| 4 | exp31 | **dpo-st** self-train RFT + offline DPO | [dpo-st/utils/make_rft_data.py](../../refs/dpo-st/utils/make_rft_data.py), `make_dpo_data.py`, `dpo.py` | gen→Jaccard-dedup→RFT-SFT; rồi cặp đúng/sai→DPO | ✓ (rollout) | +RFT +pairs | 🟢 cao |
| 5 | exp32 | **Step-DPO** step-level preference | [step-dpo](../../refs/step-dpo) (TRL DPO + locate) | định vị **bước sai đầu tiên**, cặp prefix-chung | ✓ | +step pairs | 🟡 |
| 6 | exp33 | **TokenSkip** nén CoT bit_manip | [tokenskip/LLMLingua.py](../../refs/tokenskip/LLMLingua.py), `data_processing/process_utils.py` | LLMLingua-2 rank token importance → prune ratio → SFT | ✗ (CPU) | transform | 🟢 cao |
| 7 | exp34 | **Self-rewarding correction** multi-turn | [self-rewarding-correction/infer_math/](../../refs/self-rewarding-correction/infer_math/) (`gen_hf`,`reward_labeling`,`process_prompt_turn{1,2,3}`,`merge_data`) | turn1→label-reward→turn2-sửa→merge thành trace tự-sửa | ✓ | +correction | 🟢 cao |
| 8 | exp35 | **Self-Improving Transformers** self-gen curriculum | [self-improving-transformers/run_self_improve.py](../../refs/self-improving-transformers/run_self_improve.py) | vòng self-gen + verify-label, easy→hard/length-gen — **đúng domain arithmetic/bit** | ✓ | +curriculum | 🟢 cao |
| 9 | exp36 | **SwS** weakness-driven synthesis | [sws/src/data_synthesis.py](../../refs/sws/src/data_synthesis.py), `problem_generation.py` | concept co-occurrence → dò điểm yếu → sinh bài nhắm yếu | ✗/✓ | +targeted | 🟡 (thay gen) |
| 10 | exp37 | **VeRA-H / SAND** difficulty-hiking solved→harder | [vera/cli/prepare_vera.py](../../refs/vera/cli/prepare_vera.py), [sand-math/pipeline/datageneration.py](../../refs/sand-math/pipeline/datageneration.py) | param-hardened variant của seed đã giải; **thay LLM-judge bằng solver-verify** | ✗ (CPU) | +6k–9k | 🟢 cao |

---

## 2. Chi tiết (mỗi idea: code thật → port plan → distinct → falsify)

### exp28 — OXA dual data (LP-promote + UL-suppress) ⭐
- **Code thật:** [make_SFT_LP_data.py](../../refs/oxa/1-prepare_sft_data/make_SFT_LP_data.py) chọn `GLOBAL_K` response **PPL thấp nhất dưới student** (verified đúng) làm "low-prob teacher" để internalize; [make_ULloss_data.py](../../refs/oxa/1-prepare_sft_data/make_ULloss_data.py) chọn response **SAI** với `MIN_PPL_THRESHOLD=1.2` (high-confidence wrong) cho **unlikelihood loss**. Pure-python heap-select, rồi SFT thường (LLaMA-Factory).
- **Port:** RunPod: `sample_rollouts` (8/bài) + tính PPL mỗi trace dưới adapter 0.86 → 2 tập: (a) đúng-high-PPL → `weight>1`; (b) sai-low-PPL → `sign=−λ` (UL). Map thẳng vào field `weight`/`sign` của corpus (đã thiết kế). Kaggle: continue-0.86, LR 2e-5.
- **Distinct:** không phải exp2 (positional upweight) — đây là **confidence-ranked** trace-level; UL-part là negative learning có chọn lọc theo PPL (khác REDI dùng mọi negative).
- **Falsify:** cipher/equation/gravity +0.5pp; nếu UL làm vỡ format → giảm λ. **Volume:** reweight toàn corpus + ~50k negatives chọn lọc.

### exp29 — REDI 2-stage negative-aware SFT
- **Code thật:** [open_r1_sft.py](../../refs/redi/experiments_trl/open_r1_sft.py) `preprocess_data_filter` (L120) tách correct/incorrect; trainer SFT trên correct + term đẩy-xa incorrect (TRL SFTTrainer custom). Không verl.
- **Port:** nguồn sai = rollout `reward=0` + contaminated 1,167 (`status≠rule_found`). Continuer loop: thêm `sign=−λ` term (`weighted_loss = ce*weight*sign`, mẫu số = |weight|). LR 1e-5, mix SFT anchor.
- **Distinct:** REDI = học từ sai có chủ đích; toàn bộ exp1–27 chưa làm. Khác exp28 (OXA chọn negative theo PPL; REDI dùng all-wrong).
- **Falsify:** format-box giữ + cipher/eq không tụt. **Volume:** ~free (negatives sẵn có).

### exp30 — A★-PO offline optimal-advantage weighting
- **Code thật:** README §Offline + [preprocess/data_generation/model_generate.py](../../refs/a-po/preprocess/data_generation/) — stage-1 sinh **8 response/prompt** từ reference, gather reward, ước V (`beta1=0.5`); stage-2 least-squares advantage-regress (`beta2=1e-3`). Stage-2 gắn verl nhưng **chỉ là weighted-regression** — làm được bằng per-traj weight offline.
- **Port:** baseline `V = soft_max_β1(rewards_group)`; `advantage = reward − V`; `weight = exp(advantage/β2)`-clip hoặc tuyến tính → bake vào corpus `weight`. Đây là **bản nguyên-lý-đúng của exp27** (không importance-clip degenerate).
- **Distinct:** khác exp27 (GSPO importance-clip), khác exp28 (OXA dùng PPL không reward-advantage).
- **Falsify:** so với CE thuần +0.5pp macro; nếu không → reward-signal quá thưa (đo P0).

### exp31 — dpo-st self-train RFT + offline DPO
- **Code thật:** [make_rft_data.py](../../refs/dpo-st/utils/make_rft_data.py) (Jaccard-dedup solution đa dạng → RFT), `make_dpo_data.py` (cặp đúng/sai), `sft.py`/`dpo.py` (TRL, **không verl**). Pipeline gọn, đã chạy GSM8K.
- **Port:** RunPod: gen + Jaccard-dedup (giữ ≤k trace đa dạng đúng) → corpus RFT; + cặp chosen/rejected → `pairs.jsonl`. Kaggle: SFT RFT trước, DPO sau (cache ref-logp offline).
- **Distinct:** khác exp7 (STaR thuần, chưa eval) ở Jaccard-diversity + DPO stage; khác exp18 (SimPO ref-free) ở RFT-warmup + ref-KL.
- **Falsify:** Pass@k không giảm (collapse); cipher/eq +1pp.

### exp32 — Step-DPO step-level preference
- **Code thật:** [step-dpo](../../refs/step-dpo) — định vị **bước sai đầu tiên** trong chain, tạo cặp share-prefix (TRL DPO). Hợp chain dài.
- **Port:** dùng cho bit_manip (chain dài, nhiều bước cột-bit) + equation: verifier per-step (solver biết bước nào lệch) → cặp prefix-chung. `pairs.jsonl` step-level.
- **Distinct:** preference theo **bước** (không phải cả chuỗi như exp18). Cần per-step verify — solver của ta cung cấp được.
- **Falsify:** bit_manip step-accuracy ↑; nếu không định vị được bước sai → fallback full-DPO (exp31).

### exp33 — TokenSkip nén CoT bit_manipulation
- **Code thật:** [LLMLingua.py](../../refs/tokenskip/LLMLingua.py) (engine importance), [process_utils.py](../../refs/tokenskip/data_processing/process_utils.py), `get_llamafactory_input.py` — prune token ít quan trọng theo ratio → SFT (paper dùng **LoRA**, 7.5k mẫu, rẻ).
- **Port:** CPU-only. Chạy LLMLingua trên `reasoning/<id>.txt` bit_manip, ratio 0.5–0.7, giữ token số/`\boxed`. → corpus thay phần bit_manip. Gỡ nút 816-dòng > 7680.
- **Distinct:** khác exp3 (terse rewrite tay, hard-fail −0.28) — đây là prune **controllable** giữ cấu trúc, có engine đo importance.
- **Falsify:** bit_manip pass ↑ **hoặc** truncation-bucket ↓; scope hẹp + ratio nhẹ trước.

### exp34 — Self-rewarding correction multi-turn traces
- **Code thật:** [infer_math/](../../refs/self-rewarding-correction/infer_math/): `gen_hf.py` (gen turn1) → `reward_labeling.py` (chấm đúng/sai, có cả math-normalizer) → `process_prompt_turn2/3.py` (sinh lượt sửa) → `merge_data.py` (ghép thành 1 trace tự-thưởng+tự-sửa). Offline, reward **luật**.
- **Port:** thay `reward_labeling` bằng `compare_answer` của ta. Sinh trace `attempt→self-check→correct` cho cipher/equation/bit. Mask phần sai (cơ chế mask repo có sẵn).
- **Distinct:** khác exp13 (self-verify, append-check) — đây là **multi-turn correction** (sửa thật, không chỉ verify).
- **Falsify:** arithmetic-slip bucket ↓; canh độ dài < 7680.

### exp35 — Self-Improving Transformers self-gen curriculum (arithmetic/bit)
- **Code thật:** [run_self_improve.py](../../refs/self-improving-transformers/run_self_improve.py) + `experiments/run_addition_new.sh`, `run_multiplication_mv.sh` — vòng: gen self-label → verify → FT → tăng độ khó/độ dài. **Đúng domain repo** (addition/mult/copy/maze ≈ bit_manip/arithmetic). Bằng chứng vượt easy→hard & length-gen.
- **Port:** vòng ReST-EM trên bit_manip+equation: 0.86 gen → verify-keep → corpus → (Kaggle FT) → lặp. Length-gen: train độ-dài ngắn → sinh độ-dài dài hơn.
- **Distinct:** khác exp10 (SA curriculum, length-of-trace generic) — đây là **self-generated** label + structural difficulty (số bước/độ dài input), có bằng chứng đúng-domain.
- **Falsify:** Pass@k giữ; bit_manip/equation +3pp. Cảnh báo OOD → đo category-level held-out.

### exp36 — SwS weakness-driven synthesis
- **Code thật:** [data_synthesis.py](../../refs/sws/src/data_synthesis.py) build **concept co-occurrence** + freq; [problem_generation.py](../../refs/sws/src/problem_generation.py) vLLM-gen bài nhắm concept yếu.
- **Port:** "concept" = loại bit-rule / operator-type. Dò loại model hay trượt (từ P0 error-bucket) → **generator thủ tục của ta** sinh thêm đúng loại đó (thay vLLM-gen). Verified-by-construction.
- **Distinct:** khác exp32(VeRA) ở chỗ nhắm **điểm yếu đo được** chứ không hard-hoá đều; khác mọi exp 1–27.
- **Falsify:** category yếu (bit) +pp; nếu generator không phủ được loại yếu → fallback uniform synth.

### exp37 — VeRA-H / SAND difficulty-hiking (solved→harder)
- **Code thật:** [prepare_vera.py](../../refs/vera/cli/prepare_vera.py) (`hardest_variant_messages`, judge oracle chọn biến thể khó nhất); [sand-math/pipeline/datageneration.py](../../refs/sand-math/pipeline/datageneration.py) + `solution_generation.py` + `traindataPrpe.py` (difficulty-hiking → SFT).
- **Port:** repo là "executable spec" → mutate seed `rule_found` (bit/equation) khó hơn (thêm operand/op/độ dài) → **solver-verify thay LLM-judge** → corpus. Free-verified.
- **Distinct:** augment **solved→harder** (khác synth-from-scratch exp24/26 đã thử cho guess). OOD-risk → gate held-out.
- **Falsify:** held-out category-level (không train-acc); nếu không nhích → distribution sai.

---

## 3. Composition & run-order

- **Sinh 1 lần, tách nhiều cách:** chạy `sample_rollouts` (8/bài) + PPL-score **một lần** trên RunPod → cấp data cho exp28(OXA)/29(REDI)/30(A★-PO)/31(dpo-st)/34(correction) cùng lúc.
- **Thuần CPU (rẻ, làm trước):** exp33 (TokenSkip), exp37 (VeRA-H mutate), exp36 (SwS synth).
- **Thứ tự rủi ro tăng dần:**
  1. **P0** measure_yield + PPL + error-bucket (gate).
  2. exp33 → exp37 → exp36 (CPU, coverage/length).
  3. exp28 (OXA) → exp30 (A★-PO) → exp29 (REDI) — reweight/negative trên cùng rollout.
  4. exp35 (self-improve curriculum) cho bit_manip.
  5. exp31 (dpo-st) → exp32 (Step-DPO) — preference, bet 0.87→0.88+, làm sau cùng.

## 4. Guard
- **Hết verl/ray/OpenRLHF/AReaL** — chỉ lift stage offline-data + objective; train bằng Continuer SFT continue-0.86 (LR thấp 1e-5–5e-5).
- **Verifier luật thay mọi LLM-judge** (VeRA) / reward-model (self-rewarding) — ta có `compare_answer` miễn phí.
- **Đã loại (đừng lặp):** beyond-80-20→exp20 (regressed), doremi→exp22, ESFT→exp23, python-constraint→exp24, group_DRO→exp25, GSPO→exp27.
- **Trần cứng:** cryptarithm/guess 0% là info-ceiling — exp35/36/37 sinh được nhưng xác suất move điểm thấp; gain kỳ vọng ở cipher/equation/gravity/bit-length.
- Mọi exp: đo **category-level + Pass@k**.
