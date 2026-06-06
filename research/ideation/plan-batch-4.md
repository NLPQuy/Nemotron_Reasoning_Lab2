# Plan — VÉT MÁNG bước sinh data/trajectory OFFLINE từ toàn bộ `refs/`

> Quét toàn diện **56 ref** (loại `trl`, `stable-baselines3` — chỉ là lib nền). Mỗi ref đã `find` entry-point sinh-data + đọc docstring/đầu file. Mục tiêu: liệt kê **mọi cơ chế sinh trajectory offline** có thể lift, map vào hạ tầng repo (RunPod gen → `corpus_preprocessed.jsonl` → Kaggle Continuer SFT continue-từ-0.86).
> Đi kèm [batch-4.md](batch-4.md). Ràng buộc: off-policy only, LoRA r32, verifier luật `compare_answer` thay mọi LLM-judge/RM, budget 7680 tok, greedy 1-pass.
> **Nguyên tắc lift:** chỉ lấy **stage offline-data + objective**; bỏ verl/ray/OpenRLHF/AReaL/VeRL (on-policy infra). Generator thủ tục của repo (`reasoners/*`) + `train.csv.answer` cho MỌI bài ⇒ verify-gate miễn phí.

---

## 0. Bản đồ 12 pattern sinh trajectory (mọi ref được phân vào đây)

| Pattern | Cơ chế | # ref | GPU? | Sản phẩm |
|---|---|:--:|:--:|---|
| **G1** Rollout → verify-keep-correct (RFT/STaR/ReST-EM) | sample 0.86, giữ trace ĐÚNG | 10 | ✓ | +correct traces |
| **G2** Negative-trace retention (REDI/NFT/OXA-UL) | giữ trace SAI, học đẩy-xa | 4 | ✓/✗ | +negatives (sign/UL) |
| **G3** Advantage/reward-weighted offline (A★-PO/OREO) | rollout → V baseline → per-traj weight | 4 | ✓ | reweight |
| **G4** Preference-pair (DPO/Step-DPO) | (chosen,rejected) ± step-level | 5 | ✓ | pairs.jsonl |
| **G5** CoT compression (TokenSkip/ASAP/…) | prune token trace có sẵn | 5 | ✗ | shorter traces |
| **G6** Self-correction trajectory | attempt→check→sửa→merge | 2 | ✓ | +correction traces |
| **G7** Problem synthesis from-scratch (SwS/ScaleDiff/…) | sinh bài mới nhắm yếu/khó | 7 | ✗/✓ | +new problems |
| **G8** Problem evolution seed→variant (MetaMath/VeRA/SAND/Evol) | mutate bài đã-giải → khó/đa dạng | 5 | ✗ | +variants |
| **G9** Cold-start elicitation (QuestA/CBRL) | chèn partial-sol/few-shot khi gen | 2 | ✓ | unlock hard |
| **G10** Process/step-reward (GroundedPRM/AdaptiveStep) | tree-search → step labels | 2 | ✓ | step-mask |
| **G11** Token-credit selective (Beyond-80/20) | reweight token theo entropy | 1 | ✗ | mask weight |
| **G12** Mixture/selection (DoReMi/GroupDRO/ESFT/decontam) | chọn *bài nào*, không sinh | 5 | ✗ | reweight/filter |

---

## 1. G1 — Rollout → verify-keep-correct (RFT / STaR / ReST-EM)

> Sample 0.86 G lần/bài, verify bằng `compare_answer`, giữ trace ĐÚNG (đặc biệt bài solver chưa giải). Nền của mọi expert-iteration.

| Ref | Script thật | Lift được gì |
|---|---|---|
| [star](../../refs/star) | `arithmetic/gen_sums.py`, `create_finetune_tfrecords.py` | OG STaR: gen→verify→FT loop; mẫu cho vòng lặp |
| [gsm8k-screl](../../refs/gsm8k-screl) | `group_sample_{7b,30b,65b}.sh`, `gen_train.sh` | RFT-scaling: **nhiều** sample/bài → nhiều path đúng |
| [dpo-st](../../refs/dpo-st) | `generate.py`, `utils/make_rft_data.py` | **Jaccard-dedup** giữ trace đa dạng (chống trùng) ⭐ |
| [regenesis](../../refs/regenesis) | `src/reasoning/reasoning_paths_gen.py`, `filter_questions_cannot_answer.py`, `truth_convert_reason.py` | self-discover reasoning-path + lọc bài bất khả |
| [srt](../../refs/srt) | `recipe/r1/data_process.py` | self-train data prep |
| [auto-cei](../../refs/auto-cei) | `src/rl/ei_generation.py`, `src/rl/curriculum.py` | expert-iteration + curriculum chọn bài |
| [b-star](../../refs/b-star) | `train_code/convert4reward_auto_ground_sample.py` | balance explore/exploit (chống collapse) |
| [adastar](../../refs/adastar) | `datasets/data_gsm8k/makesample.py` | adaptive sampling (diversity+curriculum) |
| [self-improving-transformers](../../refs/self-improving-transformers) | `run_self_improve.py` | self-gen easy→hard/length — **đúng domain bit/arith** ⭐ |
| [stp](../../refs/stp) | `RL/generate_and_test.py`, `RL/RL_step1_generate.py`, `RL/prepare_datasets.py` | gen+test split |

**Port → `offline/sample_rollouts.py`** (generalize `generate_rollouts_vllm.py`, bỏ drop pure-group): 1 lần G=8 toàn `train.csv`. Tách: keep-correct (Jaccard-dedup ≤k/bài, từ dpo-st) → corpus; boundary-pick (auto-cei/b-star) cho ROI. **Volume:** ~10k–14k trace đúng mới.

---

## 2. G2 — Negative-trace retention (học từ SAI)

| Ref | Script thật | Lift được gì |
|---|---|---|
| [redi](../../refs/redi) | `experiments_trl/data_preprocess/clean_base_data.py`, `ablation_verifier_signal.py`, `clean_base_shuffled_dpo.py` | tách correct/incorrect; objective đẩy-xa-sai (TRL) ⭐ |
| [oxa](../../refs/oxa) | `1-prepare_sft_data/make_ULloss_data.py` (+`cal_ppl/cal_ppl.py`) | chọn **sai high-confidence** (PPL<1.2) → **unlikelihood** ⭐ |
| [nft](../../refs/nft) | `experience_maker.py` (ý tưởng implicit-negative-policy; infra ray bỏ) | công thức loss; data = rollout sai |
| [learning-from-mistakes-glow](../../refs/learning-from-mistakes-glow) | (sparse) | negatives cải thiện OOD |

**Port → `offline/build_negatives.py`**: nguồn = rollout `reward=0` (từ G1) + contaminated 1,167 (`status≠rule_found`). OXA-style: tính PPL dưới 0.86 (`cal_ppl.py` pattern), giữ **sai-PPL-thấp** → field `sign=−λ`. REDI: dùng all-wrong. **Volume:** ~free (negatives sẵn dồi dào). Kaggle loop: `weighted_loss=ce*weight*sign`, mẫu số=|weight|.

---

## 3. G3 — Advantage/reward-weighted offline (bản ĐÚNG của exp27)

| Ref | Script thật | Lift được gì |
|---|---|---|
| [a-po](../../refs/a-po) | `preprocess/data_generation/model_generate.py`, `preprocess/data_preprocess/{gsm8k,math_dataset,arth,multiply,countdown}.py` | **8 gen/prompt → V(β1=.5) → advantage-regress(β2=1e-3)** ⭐ |
| [oreo](../../refs/oreo) | `examples/gen_balanced.py`, `scratch/construct_sft.py`, `construct_dpo_dataset.py` | balanced gen + construct SFT/DPO từ rollout |
| [oxa](../../refs/oxa) | `make_SFT_LP_data.py`, `post_process_SFT_LP_data.py`, `make_OXA_MLE_data.py` | **PPL-rank** trace đúng → promote low-prob (khó-với-student) ⭐ |
| [pcl-reasoner](../../refs/pcl-reasoner) | `data_preprocess/{decontaminate,dataset_prehandle_and_split}.py` | offline-RL math data prep |

**Port → `offline/build_reward_weighted.py`**: từ rollout G1, `V=soft_max_β1(rewards)`; `advantage=reward−V`; `weight=lin/exp(adv)` clip → bake `weight`. Không importance-clip (khác exp27). **Volume:** reweight toàn ~76k samples. *Đây là cùng rollout với G1/G2/G4 — sinh 1 lần.*

---

## 4. G4 — Preference-pair construction (DPO)

| Ref | Script thật | Lift được gì |
|---|---|---|
| [dpo-st](../../refs/dpo-st) | `utils/make_dpo_data.py`, `dpo.py` | cặp đúng/sai gọn (non-verl) ⭐ |
| [step-dpo](../../refs/step-dpo) | `data_pipeline/generate_dataset.py`, `prepare_for_correction.py` | **định vị bước sai** → cặp share-prefix ⭐ |
| [full-step-dpo](../../refs/full-step-dpo) | (TRL) | step-wise reward toàn chuỗi |
| [iterative-dpo-dpovp](../../refs/iterative-dpo-dpovp) | `openrlhf/datasets/process_reward_dataset.py` | iterative DPO + value-penalty (DPO-VP) |
| [oreo](../../refs/oreo) | `scratch/construct_dpo_dataset.py` | construct pairs từ rollout |

**Port → `offline/build_dpo_pairs.py`** (hoàn thiện `pref_generate.py` stub): mixed-group → (chosen,rejected); step-dpo cho bit_manip chain dài (solver biết bước lệch). Cache `ref_logp` offline. **Volume:** ~2k–3.5k pairs (+ step-level nhiều hơn). Kaggle: DPO+NLL+ref-KL, continue-0.86, LR 1e-5.

---

## 5. G5 — CoT compression (gỡ nút 7680, đặc biệt bit_manip 816 dòng)

| Ref | Script thật | Cơ chế |
|---|---|---|
| [tokenskip](../../refs/tokenskip) | `LLMLingua.py`, `data_processing/process_utils.py`, `get_llamafactory_input.py` | LLMLingua-2 rank importance → prune ratio ⭐ |
| [asap](../../refs/asap) | `prune/prune-stage1.py`, `prune-stage2.py`, `direct_cot.py` | prune theo **first-token surprisal**, giữ anchor |
| [cot-valve](../../refs/cot-valve) | (tuning) | CoT co-giãn theo độ khó |
| [r1-compress](../../refs/r1-compress) | (chunk) | nén chunk-level Long-CoT |
| [extra-cot](../../refs/extra-cot) | (extreme) | ép vào token-budget |

**Port → `offline/compress_traces.py`** (CPU): chạy LLMLingua (tokenskip) HOẶC surprisal-prune (asap) trên `reasoning/<id>.txt` **bit_manip**, ratio 0.5–0.7, giữ token số/`\boxed`. → corpus thay phần bit_manip. **Khác exp3** (terse tay, hard-fail) — controllable + engine đo importance. **Volume:** transform (không bài mới).

---

## 6. G6 — Self-correction trajectory

| Ref | Script thật | Cơ chế |
|---|---|---|
| [self-rewarding-correction](../../refs/self-rewarding-correction) | `infer_math/{gen_hf,reward_labeling,process_prompt_turn1,turn2,turn3,merge_data}.py` | turn1→label→sửa→merge thành trace tự-thưởng+tự-sửa ⭐ |
| [supercorrect](../../refs/supercorrect) | (thought-template) | template-guided self-correct |

**Port → `offline/build_correction_traces.py`**: thay `reward_labeling` bằng `compare_answer`. Pipeline turn1(gen)→label→turn2(sửa nếu sai)→merge. Mask phần sai (cơ chế repo có). Target cipher/equation/bit. **Volume:** ~bài có cả sai+đúng. **Khác exp13** (chỉ verify-append).

---

## 7. G7 — Problem synthesis from-scratch (sinh bài MỚI)

| Ref | Script thật | Cơ chế |
|---|---|---|
| [sws](../../refs/sws) | `src/data_synthesis.py`, `problem_generation.py`, `scripts/synthesis/step1_concepts_extraction.sh`, `step3_concepts_sampling.sh` | **concept co-occurrence → dò yếu → sinh nhắm yếu** ⭐ |
| [scalediff](../../refs/scalediff) | `difficult_identify_generation/{diffgen_vllm,adapt_think_difficulty_vllm,solution_distill_vllm}.py` | xác định bài khó → sinh thêm bài khó |
| [darc](../../refs/darc) | `question_generate/question_generate_difficulty_aware.py` | questioner sinh theo độ khó, chống reward-hack |
| [socratic-zero](../../refs/socratic-zero) | `processors/{question_enhancer,solver_data_processor,reward_calculator}.py` | enhance đề theo **error-analysis** |
| [self-guided-self-play](../../refs/self-guided-self-play) | `sgs/pipeline/step1_data_gen.py` | self-play sinh đề, chống diversity-illusion |
| [mathforge](../../refs/mathforge) | `scripts/generate_reasoning.py`, `src/open_r1/generate.py` | **multi-aspect question reformulation** |
| [h1](../../refs/h1) | `long_reasoning_solution_code_generation.py`, `process_hendrycks_math.py` | synthetic composition longer-horizon |

**Port → `offline/synth_weakness.py`** (SwS-style): "concept" = loại bit-rule/operator-type; dò yếu từ **P0 error-bucket** → **generator thủ tục của repo** sinh thêm đúng loại (thay vLLM-gen) → verified-by-construction. **Volume:** +targeted (nhắm bit_manip yếu). ⚠️ OOD-risk → gate held-out.

---

## 8. G8 — Problem evolution: seed đã-giải → variant (augment solved→harder)

| Ref | Script thật | Cơ chế "biến hoá" |
|---|---|---|
| [metamath](../../refs/metamath) | `code_for_generating_data/code/main_rephrase_question.py`, `main_forward_reasoning.py`, `run_backward.sh`, `main_self_verification.py` | **4 toán tử**: rephrase / forward / **backward (self-ask)** / FOBAR self-verify ⭐ |
| [vera](../../refs/vera) | `cli/prepare_vera.py` (`hardest_variant_messages`) | vera-E (tương đương) + **vera-H (param-hardened)** |
| [sand-math](../../refs/sand-math) | `pipeline/datageneration.py`, `solution_generation.py`, `traindataPrpe.py` | **difficulty hiking** |
| [wizardlm](../../refs/wizardlm) | `training/src/generate.py` (Evol-Instruct) | **upward evolution**: thêm ràng buộc/bước |
| [easy-to-hard](../../refs/easy-to-hard) | `data/prepare_metamath{,_shepherd}.py`, `inference_generate.py` | easy→hard generalization |

**Port → `offline/evolve_solved.py`** (CPU, ⭐ native): repo là executable-spec → mutate seed `rule_found` (bit/equation): thêm operand/op/độ-dài (Evol/SAND/VeRA-H) + **backward self-ask** (MetaMath) → **solver-verify thay LLM-judge**. **Volume:** seed ~1,900 × 3–5 = **+6k–9.5k** verified. OOD-risk → gate.

---

## 9. G9 — Cold-start elicitation (chèn context/partial-sol khi gen)

| Ref | Script thật | Cơ chế |
|---|---|---|
| [questa](../../refs/questa) | `AReaL/datasets/process.py` | **chèn partial-solution** vào prompt → giảm độ khó động |
| [cbrl](../../refs/cbrl) | `data_preprocess/make_qprog_fewshots.py`, `annotate_qprog_fewshots_tags.py`, `make_qprog_sft_parquet.py` | **few-shot demo curriculum** + annealing |

**Port → `offline/coldstart_inject.py`** (GPU): chèn k% CoT đúng/skeleton vào prompt khi sample → verify-keep completion đầy đủ. Chỉ lúc GEN (submission greedy thuần). Target bit_manip-238-gap + equation hard. **Volume:** +vài trăm–1k. ⚠️ cryptarithm có thể vẫn 0 = info-ceiling → nếu yield≈0 thì dừng.

---

## 10. G10 — Process/step-reward supervision (tạo step-mask)

| Ref | Script thật | Cơ chế |
|---|---|---|
| [groundedprm](../../refs/groundedprm) | `pipeline/{root_generation,data_generation}.py`, `data_process/{sampling,construct,extract_path}.py` | **MCTS tree → nhãn từng bước** (fidelity-aware) |
| [adaptivestep-asprm](../../refs/adaptivestep-asprm) | (confidence-split) | chia step theo confidence model |

**Port (tuỳ chọn):** với verifier luật cứng, PRM ít cần. Có thể dùng `extract_path.py` pattern để **gán weight cao cho bước quyết định** trong trace bit_manip (thay vì học PRM riêng). **Volume:** reweight token. Fit thấp hơn G1–G8 → để sau.

---

## 11. G11 — Token-credit selective (ĐÃ THỬ — đừng lặp)

| Ref | Script | Trạng thái |
|---|---|---|
| [beyond-80-20](../../refs/beyond-80-20) | `recipe/rlvr_with_high_entropy_tokens_only/` | = **exp20** (forking-token), **regressed**. Không lặp. |

---

## 12. G12 — Mixture/selection (chọn bài nào — không sinh trajectory)

| Ref | Script | Trạng thái |
|---|---|---|
| [doremi](../../refs/doremi) | `scripts/preprocess_*.py` | = exp22 (mixture reweight), regressed |
| [group_DRO](../../refs/group_DRO) | `dataset_scripts/generate_*.py` | = exp25 (worst-group), regressed |
| [ESFT](../../refs/ESFT) | `scripts/expert/generate_expert_config.py` | = exp23, regressed |
| [pcl-reasoner](../../refs/pcl-reasoner) | `data_preprocess/decontaminate.py` | **decontamination** — lift được (lọc train↔test rò rỉ) |
| [tina](../../refs/tina) | `tina/post_train_hf/preprocess.py` | LoRA-RL micro-budget preprocess |

---

## 13. Build spec — folder `offline/` (vét máng → 9 generator)

```
offline/
  setup_runpod.sh                 # mở rộng (tokenizers, llmlingua, kaggle CLI)
  common/
    verify.py                     # compare_answer/extract_answer/format_ok  (port generate_rollouts_vllm.py)
    tokenize_format.py            # {tokens,mask}  (port corpus.py:56-194)
    vllm_engine.py                # load 0.86 + batched generate
    ppl.py                        # PPL/trace dưới 0.86  (port oxa/cal_ppl)
    corpus_io.py                  # → corpus_preprocessed.jsonl {tokens,mask,weight,sign}
  sample_rollouts.py              # G1/G2/G3/G4 — 1 LẦN, G=8, giữ reward+text+logp   [GPU]
  build_rft.py                    # G1 — Jaccard-dedup keep-correct (dpo-st)          [CPU]
  build_negatives.py              # G2 — PPL-select wrong→sign=−λ (oxa-UL/redi)       [CPU]
  build_reward_weighted.py        # G3 — advantage→weight (a-po/oxa-LP)               [CPU]
  build_dpo_pairs.py              # G4 — pairs ± step-level (dpo-st/step-dpo)          [CPU]
  compress_traces.py              # G5 — LLMLingua/surprisal prune bit_manip          [CPU]
  build_correction_traces.py      # G6 — turn1→label→sửa→merge (self-rewarding)       [GPU]
  synth_weakness.py               # G7 — concept-weakness → generator repo (sws)      [CPU]
  evolve_solved.py                # G8 — mutate solved→harder + backward (metamath/vera/sand) [CPU]
  coldstart_inject.py             # G9 — partial-sol/few-shot inject (questa/cbrl)    [GPU]
  upload_kaggle.py                # đóng gói artifact → kaggle dataset
  measure_yield.py                # P0 — pass-rate + PPL + error-bucket/category
```

**Sinh 1 lần, tách nhiều:** `sample_rollouts.py` (G=8 + PPL) chạy MỘT lần → feed `build_rft`/`build_negatives`/`build_reward_weighted`/`build_dpo_pairs`/`build_correction_traces` cùng lúc. CPU-only generator (`compress_traces`/`synth_weakness`/`evolve_solved`) chạy độc lập, rẻ.

---

## 14. Run-order & volume tổng

| Bậc | Generator | GPU | Volume | Ghi chú |
|---|---|:--:|--:|---|
| 0 | `measure_yield` (P0) | ✓ | — | gate mọi nhánh |
| 1 | `compress_traces` (G5) | ✗ | transform | bit_manip length-fix, rẻ nhất |
| 1 | `evolve_solved` (G8) | ✗ | +6k–9.5k | solved→harder, native |
| 1 | `synth_weakness` (G7) | ✗ | +targeted | nhắm bit yếu |
| 2 | `sample_rollouts` (1 lần) | ✓ | ~76k samples | nguồn G1–G4,G6 |
| 2 | `build_rft`/`build_reward_weighted`/`build_negatives` | ✗ | +10k–14k / reweight / +neg | từ rollout |
| 3 | `build_correction_traces` (G6) | ✓ | +correction | self-sửa |
| 3 | `coldstart_inject` (G9) | ✓ | +<1k | hard cats |
| 4 | `build_dpo_pairs` (G4) | ✗ | +2k–3.5k | bet 0.88+, sau cùng |

**Tổng generatable:** ~17,963 corpus hiện tại → **~100k+** trajectory. **High-value thực:** ~25k–35k (G1+G2+G3+G4+G6+G8+G5) tập trung cipher/equation/gravity (sharpen) + bit_manip (length). cryptarithm/guess 0% = info-ceiling (G7/G8/G9 sinh được nhưng xác suất move điểm thấp).

## 15. Guard
- **Verifier luật thay mọi LLM-judge/RM**: VeRA judge, SwS/Socratic teacher-model, self-rewarding RM → đều thay bằng `compare_answer`.
- **Bỏ hết infra on-policy**: verl (a-po/srt/svs/sws/darc/beyond-80-20), ray (nft), OpenRLHF (oreo/iterative-dpo), AReaL (questa). Chỉ lift stage-data + objective.
- **Token format khớp grader**: mọi generator → `common/tokenize_format.py` (1 nguồn).
- **Continue-0.86**: `RESET_WEIGHTS=False`, `BASE_ADAPTER_ZIP`→0.86, LR 1e-5–5e-5, NUM_STEPS giảm.
- **Đã loại (G11/G12)**: beyond-80-20/doremi/ESFT/group_DRO = exp20/22/23/25 đã regress.
- Mọi exp đo **category-level + Pass@k**; decontaminate (pcl-reasoner) train↔test trước khi tin số.

---

## 16. Deep-dive — kernel thuật toán thật (đọc ruột generator) + lợi thế repo

> Quét sâu vào thân hàm. Phát hiện lớn: **verifier/solver luật của repo thay được CriticLLM / judge / difficulty-classifier** mà các ref phải gọi model phụ — ta làm **deterministic, rẻ hơn, sạch hơn**.

### 16.1 Step-DPO — ta có oracle bước-sai MIỄN PHÍ ⭐
- **Ref làm sao** ([step-dpo/data_pipeline/prepare_for_correction.py](../../refs/step-dpo/data_pipeline/prepare_for_correction.py)): gọi **critic-LLM** sinh "final decision: incorrect at step N", **parse text** `step_num` → cắt solution tại step N → cặp share-prefix. Mong manh (phụ thuộc critic + regex).
- **Ta làm**: solver biết **chuỗi bước đúng chính xác** → định vị **bước lệch đầu tiên** giữa rollout sai và trace solver **bằng so khớp**, không cần critic. Cặp = prefix-chung + (bước đúng / bước sai). Sạch tuyệt đối cho bit_manip (cột-bit) & equation. → exp32 mạnh hơn bản gốc.

### 16.2 dpo-st — luôn neo `chosen` bằng GOLD solver-trace ⭐
- **Ruột** ([make_dpo_data.py](../../refs/dpo-st/utils/make_dpo_data.py):99–102): `positives = [gold_rationale] + model_positives`; dedup **Jaccard** (`compare_similarity`, token-set IoU). Gold solution **luôn** là positive.
- **Áp dụng**: ta có gold = trace `reasoners/*` cho mọi `rule_found` → mọi cặp DPO/RFT neo chosen bằng solver-trace; rejected = rollout sai; dedup Jaccard giữ đa dạng. exp31 free-anchor.

### 16.3 Self-rewarding correction — format trace cụ thể ⭐
- **Ruột** ([process_prompt_turn2.py](../../refs/self-rewarding-correction/infer_math/process_prompt_turn2.py):20–28): trace tự-sửa có **tag bắt buộc** `[VERIFY] correct.` / `[VERIFY] wrong.` sau self-evaluation; nếu `wrong` → revise (turn3).
- **Template ta dựng** (exp34): `attempt → [VERIFY] wrong. → <sửa> → [VERIFY] correct. → \boxed{}`. Reward = `compare_answer` (thay RLHFlow RM). Mask phần `attempt` sai (giữ kỹ-năng-sửa, không học nội-dung-sai).

### 16.4 OXA confidence = PPL chunked-forward ⭐
- **Ruột** ([cal_ppl.py](../../refs/oxa/1-prepare_sft_data/cal_ppl/cal_ppl.py)): `ppl_forward` tính PPL/response, **chunk `step_tokens=128`** để tiết kiệm VRAM; chọn `GLOBAL_K` thấp nhất.
- **Áp dụng** (exp28): `common/ppl.py` tính PPL mỗi trace dưới adapter 0.86 (chunked) → (a) đúng-**PPL-cao** = khó-với-student → `weight>1` (LP-promote); (b) sai-**PPL-thấp** (`<1.2`) = sai-tự-tin → `sign=−λ` (UL-suppress).

### 16.5 Self-Improving = curriculum theo SỐ-CHỮ-SỐ + majority-voting ⭐ (đúng bit_manip)
- **Ruột** ([run_self_improve.py](../../refs/self-improving-transformers/run_self_improve.py):32,52,84): biến `current_n_digits → max_train_digits`; `self_improve_decay_steps`; `majority_voting` để tự-gán nhãn khi không có verifier.
- **Áp dụng** (exp35): bit_manip = bài **n-bit** → train n-bit → 0.86 sinh **(n+k)-bit** → **verify bằng solver** (mạnh hơn majority-voting của ref, vì ta có ground-truth) → thêm corpus → lặp. Length-generalization native cho bit/arith.

### 16.6 SwS = failures → concepts → co-occurrence-sample → synth ⭐
- **Ruột** ([step1_concepts_extraction.sh](../../refs/sws/scripts/synthesis/step1_concepts_extraction.sh) đọc `record/failure_cases.jsonl`; [data_synthesis.py](../../refs/sws/src/data_synthesis.py) `ConceptSampler.build_cooccurrence_stats`): trích concept từ **bài SAI**, dựng ma trận đồng-hiện + freq, sample tổ-hợp concept yếu → sinh bài.
- **Áp dụng** (exp36): "concept" = loại bit-rule / operator-type. P0 → `failure_cases.jsonl`; tag operator; generator repo sinh thêm tổ-hợp operator hay-trượt. Weakness-driven, verified-by-construction.

### 16.7 MetaMath = 4 toán tử, FOBAR backward deterministic cho ta
- **Ruột** ([run_backward.sh](../../refs/metamath/code_for_generating_data/code/run_backward.sh)): `--method_name fobar --num_repeat 2 --temp 0.7` — **mask một số đã biết, hỏi tìm nó cho trước đáp án**. `main_rephrase_question.py` (paraphrase), `main_forward_reasoning.py`, `main_self_verification.py`.
- **Áp dụng** (exp37 nhánh phụ): equation/numeral — **ẩn-hoá một biến đã biết** (FOBAR) → bài backward có nhãn **deterministic** (không cần GPT như ref). Tăng đa dạng, rẻ.

### 16.8 GroundedPRM = terminal-path label (tuỳ chọn, ta đơn-giản-hoá được)
- **Ruột** ([extract_path.py](../../refs/groundedprm/data_process/extract_path.py)): `extract_terminal_paths(tree_root, ground_truth)` + `is_math_correct` — gán nhãn path theo có chạm ground-truth không.
- **Áp dụng**: không cần MCTS — solver-trace đã là path đúng; chỉ cần gán **weight cao cho token bước-quyết-định** (cột-bit mang-nhớ) → reweight có mục tiêu thay vì PRM học riêng. Fit thấp → để sau.

### 16.9 ASAP vs TokenSkip — chọn engine nén
- **ASAP** ([prune-stage1.py](../../refs/asap/prune/prune-stage1.py)): nén bằng **prompt-LLM** ("giữ example/reflection/test minh hoạ core path") → cần model, kém kiểm soát.
- **TokenSkip** ([LLMLingua.py](../../refs/tokenskip/LLMLingua.py)): **LLMLingua-2 importance-score** thuần, ratio chính xác, **CPU**. → exp33 dùng TokenSkip (rẻ, deterministic), ASAP là fallback.

### 16.10 Tổng lợi thế repo (vì sao bản port mạnh hơn ref)
| Ref cần | Repo có sẵn (free) | Lợi |
|---|---|---|
| critic-LLM định-vị-bước (step-dpo) | solver biết chuỗi bước đúng | step-pair sạch, exp32 |
| LLM-judge chọn variant (vera) | solver-verify | augment đúng 100%, exp37 |
| RM chấm self-reward (self-rewarding) | `compare_answer` | reward luật, exp34 |
| difficulty-classifier model (scalediff) | op-count/độ-dài deterministic | curriculum free, exp35/36 |
| majority-voting tự-nhãn (self-improving) | ground-truth `answer` mọi bài | nhãn đúng tuyệt đối, exp35 |
| GPT sinh backward (metamath) | FOBAR ẩn-biến trên generator | backward free, exp37 |

→ **Kết luận deep-dive:** phần lớn chi phí/độ-nhiễu của các ref nằm ở *model phụ để verify/judge/label*. Repo có **verifier luật + generator thủ tục + answer-mọi-bài** ⇒ bỏ được toàn bộ lớp đó. Bottleneck thật vẫn là *sinh CoT đúng cho bit_manip dài* + *trần info cryptarithm* — không phải thiếu cơ chế.

---

## 17. Quét cuối (closure) — 2 kernel bổ sung + 1 xác nhận dead-end

> Mở nốt ruột các ref chưa đọc (wizardlm, Logic-LLM, auto-cei, questa, h1, b-star/adastar). Mọi ref giờ đã quy về G1–G12; không phát sinh pattern thứ 13. Hai thứ đáng thêm + một xác nhận:

### 17.1 ⭐ WizardLM Evol-Instruct = TAXONOMY 5 toán-tử "làm khó hơn" (cho exp37/G8)
- **Ruột** ([Evol_Instruct/depth.py](../../refs/wizardlm/Evol_Instruct/depth.py), [breadth.py](../../refs/wizardlm/Evol_Instruct/breadth.py)): 5 operator chuẩn —
  1. **add-constraints** (`createConstraintsPrompt`): thêm 1 ràng buộc/điều-kiện.
  2. **deepen** (`createDeepenPrompt`): tăng độ sâu/rộng câu hỏi.
  3. **concretize** (`createConcretizingPrompt`): thay khái-niệm-chung bằng cụ-thể.
  4. **add-reasoning-steps** (`createReasoningPrompt`): ép thêm bước suy luận.
  5. **breadth** (`createBreadthPrompt`): sinh bài mới cùng độ-phức-tạp, khác domain.
- **Áp dụng**: `evolve_solved.py` (G8) dùng **menu deterministic** này thay vì mutate tuỳ tiện: bit_manip = add-operand(=constraints) / longer-chain(=add-steps) / nest-op(=deepen); equation = thêm biến(=concretize). Có taxonomy → mutate có hệ thống + verify bằng solver.

### 17.2 ⭐ AUTO-CEI = curriculum có "IDK" + stop-rule hội tụ (cho exp35/G1)
- **Ruột** ([auto-cei/src/rl/curriculum.py](../../refs/auto-cei/src/rl/curriculum.py)): objective `f(precision, idk_rate, Lambda)`; `if_converge(threshold=0.003)`; `idk_threshold` — cho model **"bỏ cuộc" (IDK)** thay vì bịa, **phạt attempt quá dài**, dừng mở-rộng curriculum khi objective hội tụ.
- **Áp dụng**: vòng self-improve bit_manip (exp35) — cho phép trace "không chắc → IDK" (khớp `rule_unknown` sẵn có) để **không bơm trace sai**; stop khi pass-rate hội tụ (khỏi over-train). Chống compounding-error của curriculum dài.

### 17.3 ✗ Logic-LLM = XÁC NHẬN dead-end cryptarithm (đừng đầu tư lại)
- **Ruột** ([Logic-LLM/models/logic_program.py](../../refs/Logic-LLM/models/logic_program.py), [symbolic_solvers/z3_solver/](../../refs/Logic-LLM/models/symbolic_solvers/), `self_refinement.py`): NL→formal→**Z3/Prover9/python-constraint**→giải→refine.
- **Phán quyết**: khớp memory `cryptarithm-unsolved-levers` + **exp24 (python-constraint CSP) đã regress**. Symbolic-solver chỉ *tăng tốc cùng một model* cryptarithm, **không phủ thêm** residual under-determined/guess. → **không có lever mới** ở đây; giữ nguyên kết luận info-ceiling.

### 17.4 Phần còn lại (questa/h1/b-star/adastar/oreo/socratic-zero/supercorrect/cot-valve…) — không kernel mới
- **questa** ([process.py](../../refs/questa/AReaL/datasets/process.py)): inject = **prepend k% token lời-giải** (file `*-prefix.jsonl`); đúng như §G9 đã mô tả, không thêm gì.
- **h1**: composition = ghép bài thành longer-horizon + test-code; với repo, đã phủ bởi G8 (evolve longer-chain).
- **b-star/adastar**: chỉ là *tiêu-chí chọn bài* (balance explore/exploit, adaptive diversity) cho vòng G1 — đã ghi ở §1; không phải generator riêng.
- **oreo/socratic-zero/supercorrect/cot-valve/r1-compress/extra-cot/dss-grpo/full-step-dpo/iterative-dpo/darc/sgs/scalediff/regenesis/srt/stp/mathforge/groundedprm/adaptivestep/tina/pcl/glow**: đều là biến-thể của G1–G10 đã catalog (objective/infra khác, **cơ chế sinh-data trùng**). Lift theo generator tương ứng.

### 17.5 ✅ Closure
**Tất cả 56 ref đã quy về 12 pattern (G1–G12) + 9 generator trong `offline/` (§13).** Quét thêm chỉ tinh-chỉnh *taxonomy mutate* (WizardLM) và *stop-rule curriculum* (Auto-CEI); **không mở thêm pattern**. Sweep kết thúc — sẵn sàng scaffold code.
