# Plan — Mix solver-corpus + rollouts, rồi train (batch-4, P0-grounded)

> **Audience: Codex.** Mỗi task dưới đây là self-contained: mục tiêu / file phải tạo hoặc sửa
> (đường dẫn thật) / input schema / thuật toán từng bước / CLI / cách validate / config train /
> success metric. Làm đúng thứ tự M0→M5. KHÔNG đổi token-format, KHÔNG động on-policy/verl.
>
> Grounded trong **P0 thật** (chạy trên 76,007 trace của `rollouts.jsonl`):
> | dữ kiện P0 | con số | hệ quả |
> |---|---|---|
> | RFT-net (partial, ≥1/8 đúng nhưng chưa all-8) | **477** (bit 275, cipher 146, gravity 25, unit 20, eq_d 4, crypto 6) | trace đúng **đã có sẵn** trong rollouts → RFT free |
> | zero (0/8 đúng, chỉ solver tạo được) | **1149** (crypto_d 603, crypto_g 153, eq_g 114, bit 219, eq_d 56) | cần solver / evolve / compress |
> | ppl✓ vs ppl✗ | ~1.000 vs 1.000–1.015 | **PPL không tách đúng/sai → bỏ exp28 (OXA PPL-rank)** |
> | bit_manip correct p95 length | **7243 / 7680** | 219 bit zero có thể là lỗi token-budget → exp33 cứu được |

Đầu vào đã có ở repo-root: `rollouts.jsonl` (5.0 GB, stage=`probe`, G=8), `yield_probe.json`.
Corpus solver gốc: `nemotron-master/corpus.jsonl` (17,963 ví dụ) + `nemotron-master/problems.jsonl` (status mỗi id).

---

## 0. Sự thật repo phải nắm (đã verify, có line-ref)

| Cần biết | Ở đâu | Nội dung |
|---|---|---|
| Token-format (nguồn DUY NHẤT) | [offline/common/tokenize_format.py](../../offline/common/tokenize_format.py) `make_example*` | completion=`{reasoning}\n</think>\n\boxed{{ans}}<\|im_end\|>`; prompt qua chat-template `enable_thinking=True` mask 0; completion mask 1. **ans lấy từ regex `\boxed{}` của reasoning, fallback `answer`.** Mọi sinh-token PHẢI đi qua hàm này. |
| Verifier (nguồn DUY NHẤT) | [offline/common/verify.py](../../offline/common/verify.py) | `compare_answer/extract_answer/format_ok`. Copy import, KHÔNG sửa logic. |
| Corpus row schema (trainer ăn) | [Continuer:222-236](../../Continuer_Nemotron_Notebook.py#L222-L236) | mỗi dòng `{problem_id, category, tokens, mask, weight, sign}`. |
| weight/sign ĐÃ wired | [Continuer:203](../../Continuer_Nemotron_Notebook.py#L203), [227](../../Continuer_Nemotron_Notebook.py#L227) | `w = weight*sign`; loss token = `per_token_ce * w * mask`. **Milestone A coi như xong** cho phần Modal-worker. |
| Rollout row schema (input) | dòng đầu `rollouts.jsonl` | `{problem_id, category, prompt_token_ids, completion_token_ids, text, pred, answer, reward, old_logp, ppl_approx, stage}`. |
| Continue-0.86 | [Continuer:11-23](../../Continuer_Nemotron_Notebook.py#L11-L23) | `RESET_WEIGHTS=False`, `NUM_STEPS`/`LEARNING_RATE` đầu file. |
| Tổng hợp yield | [offline/measure_yield.py](../../offline/measure_yield.py) | đã chạy → `yield_probe.json`; tái dùng cho từng corpus mới (đo regress). |

**Quy ước Codex:** chạy mọi script bằng `python3 offline/<x>.py` từ repo-root (offline/ tự thêm ROOT vào sys.path). Trong `nemotron-master/` mới dùng `uv run`. Format: `ruff format offline/*.py && ruff check offline/*.py --fix`. Không thêm dep mới ngoài cái đã có trong common/.

---

## PHẦN 1 — Tổ chức lại data (mix solver + rollouts)

### Thiết kế corpus mix
Một corpus train = hợp của 3 nguồn, mỗi nguồn 1 builder, đổ ra `corpus/parts/<name>.jsonl` rồi merge:

| Nguồn | Builder | Nội dung | weight | sign |
|---|---|---|---|---|
| **S** solver-base | (có sẵn) `nemotron-master/corpus.jsonl` → repack | toàn bộ ví dụ `included=true` của solver | 1.0 | +1 |
| **R** RFT-mined | `offline/build_rft.py` (M2) | trace ĐÚNG lọc từ `rollouts.jsonl`, dedup, cap | 1.0 | +1 |
| **N** negatives | `offline/build_negatives.py` (M4) | trace SAI từ `rollouts.jsonl` (REDI) | λ (0.1–0.3) | −1 |

Reweight (A★-PO) là biến thể của R: cùng builder, đặt `weight=advantage` thay vì 1.0.

**Anti-regress (bắt buộc):** nguồn S luôn là anchor — không bao giờ train chỉ R/N. Mọi corpus mix phải chứa S đầy đủ.

### M0 — repack corpus solver thành part-S (CPU, ~2 phút)
- **File mới:** `offline/repack_solver.py`
- **Input:** `nemotron-master/corpus.jsonl` + per-id `nemotron-master/corpus/<id>/synthetic.jsonl`.
- **Schema per-id (ĐÃ VERIFY):** mỗi file `synthetic.jsonl` gồm các **segment** `{type: "masked"|"unmasked", pos: int, tokens: [int]}` (xem [corpus.py:108-128](../../nemotron-master/corpus.py#L108-L128)). KHÔNG có field `mask` sẵn — phải tự dựng.
- **Thuật toán:** với mỗi row `included=true` của `corpus.jsonl`:
  1. mở `corpus/<problem_id>/<segment>` (field `segment`, mặc định `synthetic.jsonl`).
  2. sort segment theo `pos`, `tokens = concat(seg.tokens)`, `mask = [0]*len nếu type=="masked" else [1]*len`, ghép cùng thứ tự.
  3. assert `len(tokens)==len(mask)` và khớp `token_count` của row.
  4. emit `{problem_id, category, tokens, mask, weight:1.0, sign:1.0}` qua `corpus_io.write_rows`.
- **Output:** `corpus/parts/S_solver.jsonl`.
- **Validate:** `wc -l` = số `included=true`; spot-check 3 dòng có `sum(mask)>0` và `len(tokens)==len(mask)`.

### M1 — driver merge + cap + dedup-id
- **File mới:** `offline/build_corpus.py`
- **CLI:** `--parts S_solver.jsonl R_rft.jsonl [N_neg.jsonl] --out corpus/mix_<tag>.jsonl --max_tokens_total <int> --dedup_id`
- **Thuật toán:**
  1. Đọc lần lượt các part (streaming, `corpus_io.read_rollouts`).
  2. `--dedup_id`: nếu cùng `problem_id` xuất hiện ở ≥2 part, giữ thứ tự ưu tiên S > R > N (S không bị R đè); trong cùng part giữ tất cả (R có thể nhiều trace/1 id sau dedup-text của M2).
  3. Cộng dồn `token_count`; nếu vượt `--max_tokens_total` thì dừng (cho phép cap để khớp budget Kaggle).
  4. In summary per-category: #rows, Σunmasked-token, %mỗi nguồn.
- **Output:** `corpus/mix_<tag>.jsonl` (đúng schema trainer).
- **Validate:** chạy `measure_yield`-style đếm category để chắc S vẫn nguyên; in bảng so với corpus stats trong CLAUDE.md.

---

## PHẦN 1.5 — DEPLOY KAGGLE (BẮT BUỘC — KHÔNG có Modal)

> ⚠️ User **chỉ chạy Kaggle**, KHÔNG có Modal. `modal run exp<N>.py` / `modal volume put` = VÔ DỤNG.
> Mọi exp phải chạy như **Kaggle notebook**, corpus là **Kaggle dataset**. exp file PR-1..3 hiện chỉ
> sửa `CORPUS_PATH` nhánh Modal-worker → trên Kaggle chúng train NHẦM corpus gốc. Phải fix 3 việc:

### Fix A — patch BASE notebook đọc single-jsonl trên Kaggle (1 lần, rồi sync vào 10 exp)
Nhánh `IS_KAGGLE` ([Continuer:165-214](../../Continuer_Nemotron_Notebook.py#L165-L214)) đọc **thư mục per-id** `<id>/synthetic.json` + `index.jsonl`, **1 file/id** → KHÔNG chứa được mix single-jsonl (multi-trace/id). Thêm nhánh: nếu `CORPUS_PATH` là `.jsonl` thì đọc dòng-một-row (y hệt reader Modal-worker [Continuer:215-236](../../Continuer_Nemotron_Notebook.py#L215-L236)):
```python
if IS_KAGGLE:
    if CORPUS_PATH.endswith(".jsonl"):          # batch-4 mix
        with open(CORPUS_PATH) as f:
            for line in f:
                rec = json.loads(line.strip())
                tokens, mask = rec["tokens"], rec["mask"]
                if len(tokens) > MAX_SEQ_LEN:
                    tokens, mask = tokens[:MAX_SEQ_LEN], mask[:MAX_SEQ_LEN]
                if not any(mask): continue
                w = float(rec.get("weight",1.0))*float(rec.get("sign",1.0))
                examples.append({"problem_id": rec.get("problem_id",""),
                                 "tokens": tokens[:-1], "targets": tokens[1:],
                                 "weights": [w*float(m) for m in mask[1:]]})
    else:
        <giữ nguyên reader per-id cũ + TRAIN_ORDER_PATH>
```
Bọc trong marker `# >>> KAGGLE_JSONL START/END`. **Không** cần `TRAIN_ORDER_PATH`/`index.jsonl` cho nhánh jsonl (thứ tự = thứ tự build_corpus emit; bật `SHUFFLE_DATASET=True` nếu muốn trộn).

### Fix B — mỗi exp set `CORPUS_PATH` nhánh **Kaggle** ([Continuer:109](../../Continuer_Nemotron_Notebook.py#L109)), KHÔNG phải Modal
Trong marker `# >>> EXP<N>` đổi dòng Kaggle CORPUS_PATH → đường dẫn dataset đã upload, vd:
`CORPUS_PATH = "/kaggle/input/nemotron-mixes/mix_correction.jsonl"`. (Dòng Modal-worker để mặc kệ — không bao giờ chạy.)

### Fix C — upload mix làm Kaggle dataset
Đẩy tất cả `corpus/mix_*.jsonl` thành 1 dataset (vd slug `nemotron-mixes`) qua kaggle CLI:
`kaggle datasets create -p corpus/` (hoặc `version` để update), rồi **attach dataset vào notebook** mỗi exp. Mix lớn (0.5–0.7 GB/file, ~10 file) — cân nhắc tách dataset/exp nếu vượt giới hạn, hoặc gzip.

### Chạy
Mỗi exp = 1 Kaggle notebook (GPU on): paste/upload `exp<N>.py`, attach dataset `nemotron-mixes` + dataset wheels (`mayukh18/nemotron-packages`) + model + adapter-zip (đã hardcode sẵn trong nhánh Kaggle). Run → notebook tự ghi `submission.zip`. Submit.
- `KAGGLE_DATASET` ([Continuer:25](../../Continuer_Nemotron_Notebook.py#L25)) chỉ dùng cho upload-adapter của luồng Modal — trên Kaggle adapter ra thẳng `submission.zip`, **không lo ghi đè** giữa các exp (mỗi notebook output riêng).
- Kiểm `RESET_WEIGHTS=False` + adapter-zip 0.86 ([Continuer:116](../../Continuer_Nemotron_Notebook.py#L116)) có attach để continue-train.

---

## PHẦN 2 — Thí nghiệm (đúng thứ tự ưu tiên P0)

### Ref anchors — ĐỌC TRƯỚC KHI CODE (đừng tự chế thuật toán)
Mỗi builder port từ code thật trong `refs/`. Codex PHẢI mở đúng file/dòng dưới đây, đọc, rồi mới viết — không suy từ abstract.

| Builder | Ref file (đọc) | Hàm/dòng | Port gì (đã verify) |
|---|---|---|---|
| `build_rft.py` (dedup) | [refs/dpo-st/utils/make_rft_data.py](../../refs/dpo-st/utils/make_rft_data.py) | `compare_similarity` (L23-28), `process_rft_train` (L52-130) | Jaccard sim **thr=0.7**, greedy-dedup giữ trace đa dạng, cap = 90th-pct #sol/question. Repo đã có sẵn tương đương ở [offline/common/dedup.py](../../offline/common/dedup.py) `dedup_keep_diverse(thr=0.7)` → **dùng lại, đừng viết lại Jaccard**. |
| `build_rft.py --reward_weight` (A★-PO) | [refs/a-po/README.md](../../refs/a-po/README.md) §Offline + [preprocess/data_generation/model_generate.py](../../refs/a-po/preprocess/data_generation/model_generate.py) | README L95-99 | Stage-1 offline value: **V = β1·logmeanexp(rᵢ/β1), β1=0.5**; advantage = r − V. Stage-2 on-policy regression (β2=1e-3) **KHÔNG port** (cần verl). Ta chỉ lấy advantage làm per-trace weight. |
| `build_negatives.py` (REDI) | [refs/redi/trl/trl/trainer/dpo2_trainer.py](../../refs/redi/trl/trl/trainer/dpo2_trainer.py) | **L1128-1155** (loss_type=`reinforce`) + config [recipes/redi/redi_1_08_1e-6/config.yaml](../../refs/redi/experiments_trl/recipes/redi/redi_1_08_1e-6/config.yaml) | `loss = −1.0·(logp_chosen/len) + 0.8·(logp_rejected/len)`, **reference-free, length-normalized**. coef chuẩn: **chosen=1.0, rejected=0.8**. Optional neg-weighting `w=exp(γ·norm_logp_rej)` (L1147-1151). |
| `compress_traces.py` (TokenSkip) | [refs/tokenskip/LLMLingua.py](../../refs/tokenskip/LLMLingua.py), [data_processing/process_utils.py](../../refs/tokenskip/data_processing/process_utils.py) | — | engine importance + prune-ratio. **Đã wrap trong [offline/compress_traces.py](../../offline/compress_traces.py)** → chỉ đọc CLI hiện có, không re-port. |
| `evolve_solved.py` (exp37/SAND) | [refs/sand-math/pipeline/datageneration.py](../../refs/sand-math/pipeline), [refs/vera/cli/prepare_vera.py](../../refs/vera/cli/prepare_vera.py) | — | difficulty-hiking solved→harder, judge thay bằng solver-verify. **Đã wrap trong [offline/evolve_solved.py](../../offline/evolve_solved.py)** → đọc CLI, không re-port. |
| `build_step_negatives.py` (exp32 Step-DPO) | [refs/step-dpo/data_pipeline/generate_dataset.py](../../refs/step-dpo/data_pipeline/generate_dataset.py), [stepdpo_trainer.py](../../refs/step-dpo/stepdpo_trainer.py) | gen L42-49, trainer L50-72 | cặp share-prefix; prefix=prompt-mask, chỉ continuation tính loss. **Định vị bước sai bằng gold-divergence (không cần GPT-4)** → map vào mask+sign. |
| `build_correction.py` (exp34) | [refs/self-rewarding-correction/infer_math/](../../refs/self-rewarding-correction/infer_math/) | `merge_data.py`, `process_prompt_turn2.py` | attempt→bridge→fix gộp 1 trace; **reward_labeling → `compare_answer`**; fix = gold. |
| `build_length_curriculum.py` (exp35) | [refs/self-improving-transformers/preamble.py](../../refs/self-improving-transformers/preamble.py), [run_self_improve.py](../../refs/self-improving-transformers/run_self_improve.py) | preamble L82-93 | `n_digits` range tăng dần (short→long). Ta phân tầng theo #cột-bit; verified by solver. |
| `synth_weakness.py` (exp36 SwS) | [refs/sws/src/data_synthesis.py](../../refs/sws/src/data_synthesis.py), [problem_generation.py](../../refs/sws/src/problem_generation.py) | `build_cooccurrence_stats` L13-35 | concept freq/co-occur → nhắm family yếu; **gen thủ tục thay vLLM-gen**. |
| `build_corpus.py --difficulty_weight` (exp J / AdaSTaR) | [refs/adastar/utils_adastar.py](../../refs/adastar/utils_adastar.py), [device_inference_adastar_new.py](../../refs/adastar/device_inference_adastar_new.py) | — | adaptive sampling theo accuracy; ta dùng pass-count/8 của P0 làm difficulty → reweight `weight`. |

**Map REDI→repo (quan trọng):** repo train bằng `weighted_loss = per_token_ce·weight·sign`, chia `weight_sum` ([Continuer:203,227](../../Continuer_Nemotron_Notebook.py#L203)). REDI `reinforce` loss ≡ đặt: chosen `sign=+1, weight=1.0`; rejected `sign=−1, weight=0.8`. (minimize CE·(+1) = maximize logp_chosen; minimize CE·(−0.8) = minimize logp_rejected). Chia `weight_sum` của repo ≈ length-normalization của REDI. **Nên λ_negative mặc định = 0.8 (giá trị paper), KHÔNG phải 0.2.**

### Training entry-point — convention `exp<N>.py` (BẮT BUỘC đọc)
Theo CLAUDE.md: mỗi exp = **1 bản copy** `Continuer_Nemotron_Notebook.py` → `exp<N>.py`, mọi sửa đổi bọc trong `# >>> EXP<N> START` / `# >>> EXP<N> END`, header comment ghi: idea + knob đổi + lệnh rollback. **Launch: Kaggle notebook (KHÔNG Modal — user không có Modal).** Xem PHẦN 1.5 cho deploy.

**Điểm mấu chốt:** trainer ĐÃ wired `weight`/`sign` → các exp data-only **KHÔNG sửa code loss**, chỉ sửa **config đầu file + trỏ corpus**:
- [Continuer:11-25](../../Continuer_Nemotron_Notebook.py#L11-L25): `RESET_WEIGHTS=False`, `LEARNING_RATE=1e-5` (KHÔNG để 2e-4 mặc định — continue-0.86 cần LR thấp), `NUM_STEPS` ≈ 1 epoch trên corpus mix.
- `CORPUS_PATH` nhánh **Kaggle** [Continuer:109](../../Continuer_Nemotron_Notebook.py#L109) → `/kaggle/input/<dataset>/mix_<tag>.jsonl` (CẦN Fix A để đọc jsonl). Dòng Modal-worker [Continuer:125](../../Continuer_Nemotron_Notebook.py#L125) bỏ mặc kệ.

**Mapping idea → file** (giữ số hiệu batch-4 cho cái trùng; exp28/exp31 BỎ TRỐNG — OXA bị P0 bác, dpo-st hấp thụ vào F+C):

| Builder/idea | exp file | corpus dùng |
|---|---|---|
| EXP-C REDI | **exp29.py** | mix_redi |
| EXP-D A★-PO | **exp30.py** | mix_apo |
| EXP-F Step-localized REDI | **exp32.py** | mix_stepneg |
| EXP-B TokenSkip | **exp33.py** | mix_rft+compress |
| EXP-G self-correction | **exp34.py** | +correction |
| EXP-H length-curriculum | **exp35.py** | +length |
| EXP-I SwS weakness | **exp36.py** | +weakness |
| EXP-E evolve/SAND | **exp37.py** | +evolve |
| EXP-A RFT-mined *(mới)* | **exp38.py** | mix_rft |
| EXP-J AdaSTaR reweight *(mới)* | **exp39.py** | mix_adastar |

> ⚠️ Nhiều exp chỉ khác base ở `CORPUS_PATH`+LR+steps → diff trong marker rất nhỏ (đó là chủ đích). KHÔNG sửa loss/forward/LoRA trong các file này. Khác biệt thực nằm ở `offline/build_*.py` + file corpus.

> Quy ước mỗi exp: tạo corpus mix → upload Kaggle → train continue-0.86 → submit → đo
> `yield`/leaderboard. Train config mặc định: `RESET_WEIGHTS=False`, `LEARNING_RATE=1e-5`,
> `NUM_STEPS` = sao cho ~1 epoch trên corpus mix (tính từ Σtoken / (BATCH_SIZE*SEQ)).
> **Gate chung:** category-level pass@1 của 5 nhóm mạnh (numeral/unit/gravity/cipher/eq_deduce)
> KHÔNG được tụt > 0.5pp so với baseline 0.86. Nếu tụt → giảm `NUM_STEPS` hoặc λ.

### EXP-A — RFT từ rollouts có sẵn  ⭐ (free win, làm TRƯỚC)
**Mục tiêu:** convert 477 partial-problem (bit 275 / cipher 146 / …) bằng trace đúng đã có.
- **File mới:** `offline/build_rft.py`
- **Input:** `rollouts.jsonl`.
- **Thuật toán:**
  1. Stream, giữ trace `reward>0` AND `format_ok(text)` (double-check bằng `compare_answer(answer, pred)`).
  2. Group theo `problem_id`. **Bỏ** problem all-8-đúng (đã thuộc S, thêm = nhiễu) — chỉ giữ id thuộc tập *partial* (1..7 đúng). Tập này = 477 id; xác định bằng đếm đúng/8.
  3. Dedup-text trong mỗi id: gọi `offline/common/dedup.py:dedup_keep_diverse(items, k=keep_per_id, key=lambda r: r["completion_token_ids"], thr=0.7)` — **port chính xác** từ dpo-st `make_rft_data.compare_similarity` (Jaccard thr 0.7, [refs/dpo-st/utils/make_rft_data.py:23](../../refs/dpo-st/utils/make_rft_data.py#L23)). Mặc định `keep_per_id=2`.
  4. Re-tokenize qua `make_example_from_ids(prompt_token_ids, completion_token_ids, ...)` (token đã có sẵn trong rollout — KHÔNG re-encode text, tránh lệch). category lấy từ rollout.
  5. emit weight=1.0 sign=+1.
- **CLI:** `python3 offline/build_rft.py --rollouts rollouts.jsonl --keep_per_id 2 --partial_only --out corpus/parts/R_rft.jsonl`
- **Validate:** #id ≈ 477; in #rows/category (kỳ vọng bit≫cipher); kiểm tra mọi row `mask[-k:]` cover `\boxed`.
- **Corpus:** `build_corpus.py --parts S_solver R_rft --out corpus/mix_rft.jsonl`
- **Success:** leaderboard > 0.86, hoặc category bit/cipher pass@1 ↑ mà 5-nhóm-mạnh không tụt.

### EXP-B — TokenSkip nén bit_manipulation (exp33)  ⭐ (cứu 219 bit zero)
**Mục tiêu:** rút ngắn CoT bit_manip (p95 7243→<6500) để bài dài đóng được `\boxed` trong budget.
- **File:** `offline/compress_traces.py` **đã tồn tại** — CLI verify rồi (dưới).
- **Thuật toán (đã code, ĐÃ VERIFY):** scope **hardcode `bit_manipulation`** ([compress_traces.py:138-141](../../offline/compress_traces.py#L138-L141), KHÔNG có `--category`). LLMLingua-2 rank importance → prune, giữ `PROTECT_PATTERNS` (`\boxed`, `Applying to`, `Input/Output`, dòng bit `^\d+\s+[01]$`). **Adaptive-ratio có verifier-gate:** thử lần lượt các ratio trong `--ratios`, **chỉ giữ bản nén nào VẪN verify đúng** (`format_ok`+`compare_answer`); bản nào hỏng đáp số bị drop → an toàn, không bịa.
- **CLI (ĐÃ VERIFY):** `python3 offline/compress_traces.py --ratios 0.7,0.6 --output corpus/parts/S_bit_compressed.jsonl` (ratio ≥0.5 bắt buộc; thử 0.7 trước rồi 0.6; CPU-only, cần `uv add llmlingua`).
- **Corpus (compression-only, ablation sạch):** `build_corpus.py --parts S_bit_compressed S_solver --dedup_id --out corpus/mix_tokenskip.jsonl`. Verifier-gate drop ~238/1602 bit → phải **fallback gốc, KHÔNG drop**: S_bit_compressed (1364) liệt-kê-trước thắng, S_solver fill 238 còn lại + non-bit ([dedup bảo vệ id theo part trước, build_corpus.py:220-221](../../offline/build_corpus.py#L220-L221)). **KHÔNG dùng `--drop_category_from_part`** (cần 238 fallback).
  - ⚠️ **KHÔNG thêm R_rft vào mix này**: `--dedup_id` dedup theo `problem_id` → 477 partial-id của RFT đã có trong S_solver nên RFT bị loại sạch (xung đột: dedup cần cho override-bit, nhưng RFT cần additive). exp33 do đó là **compression-only** (đúng cho ablation). Muốn corpus *vừa nén vừa RFT* phải sửa build_corpus cho dedup chỉ áp trong các part S (để R/N luôn additive) — follow-up, chưa làm.
- **Validate:** corpus mix có đủ **1602 bit** (1364 nén + ~238 gốc); median/p95 token bit_manip < bản gốc thuần. Spot-check 3 trace nén vẫn còn đủ bước cột-bit + `\boxed` đúng.
- **Success:** truncation-bucket bit_manip ↓ HOẶC pass@1 bit ↑; 5-nhóm-mạnh giữ nguyên.

### EXP-C — REDI negatives (exp29, đã nuốt exp28)
**Mục tiêu:** đẩy model ra xa trace sai high-confidence (đặc biệt cryptarithm 4845 wrong).
- **File mới:** `offline/build_negatives.py`
- **Ref (đọc trước):** [refs/redi/trl/trl/trainer/dpo2_trainer.py:1128-1155](../../refs/redi/trl/trl/trainer/dpo2_trainer.py#L1128-L1155) + config [redi_1_08_1e-6/config.yaml](../../refs/redi/experiments_trl/recipes/redi/redi_1_08_1e-6/config.yaml). REDI = reference-free `reinforce` loss; coef chuẩn **rejected=0.8**.
- **Lý do bỏ OXA-PPL:** P0 cho ppl✓≈ppl✗ → không lọc/rank theo PPL được; dùng REDI (toàn bộ negative, sign=−λ).
- **Thuật toán:**
  1. Stream, giữ trace `reward==0` (sai) AND `format_ok` (loại trace vỡ format để không dạy format xấu).
  2. Cap số negative/category để không lệch mixture: `--max_per_cat` (mặc định 2000); ưu tiên category nhiều zero (crypto/eq_guess).
  3. Re-tokenize qua `make_example_from_ids`, emit `weight=λ`, `sign=-1`. **λ mặc định = 0.8 (= REDI `reinforce_rejected_coef`, [config.yaml](../../refs/redi/experiments_trl/recipes/redi/redi_1_08_1e-6/config.yaml)), KHÔNG phải 0.2.** Repo chia `weight_sum` ≈ length-norm của REDI.
  4. (Optional, REDI `use_neg_weighting`, [dpo2_trainer.py:1147-1151](../../refs/redi/trl/trl/trainer/dpo2_trainer.py#L1147-L1151)) nhân thêm `exp(γ·norm_logp_rej)` từ `old_logp` của rollout — chỉ bật nếu λ=0.8 phẳng làm vỡ format.
- **CLI:** `python3 offline/build_negatives.py --rollouts rollouts.jsonl --lambda 0.8 --max_per_cat 2000 --out corpus/parts/N_neg.jsonl`
- **Corpus:** `build_corpus.py --parts S_solver R_rft N_neg --out corpus/mix_redi.jsonl`
- **Rủi ro (P0):** model sai *nhất quán* → đè negative chưa chắc tạo mode đúng + có thể vỡ format. **Bắt buộc** ablate λ∈{0.4,0.8} (quanh giá trị paper); theo dõi `format_rate_wrong` (nếu ↑ → giảm λ hoặc bật neg-weighting).
- **Success:** macro pass@1 ↑ mà format không vỡ; nếu chỉ làm tụt → λ=0 (drop EXP-C).

### EXP-D — A★-PO reward-weighted (exp30, scope bit+cipher)
**Mục tiêu:** weight trace đúng theo advantage thay vì 1.0. **Chỉ có tín hiệu ở 421 partial-problem bit+cipher** (zero-cat advantage degenerate → loại).
- **File:** mở rộng `build_rft.py` thêm flag `--reward_weight`.
- **Ref (đọc trước):** [refs/a-po/README.md](../../refs/a-po/README.md) §Offline (L95-99) — β1=0.5 cho value-est offline.
- **Thuật toán:** per problem-group (8 trace): **V = β1·logmeanexp(rewardᵢ/β1)** với β1=0.5 (đúng a-po, [README L98](../../refs/a-po/README.md)); `advantage = reward − V`; chỉ giữ trace đúng, `weight = clip(exp(advantage/β2), 0.5, 3.0)`. **β2 dùng 1.0 (KHÔNG dùng 1e-3 của a-po stage-2** — đó là cho on-policy regression ta không port; β2 nhỏ làm weight nổ). Scope `--categories bit_manipulation,cipher`.
- **CLI:** `python3 offline/build_rft.py --rollouts rollouts.jsonl --reward_weight --categories bit_manipulation,cipher --out corpus/parts/R_apo.jsonl`
- **Distinct:** khác exp27 (GSPO importance-clip degenerate) — đây là weighted-regression thuần, không importance-ratio.
- **Success:** so với EXP-A (weight=1) trên cùng id, +macro pass@1.

### EXP-E — evolve_solved harder (exp37, solver-verified coverage)
**Mục tiêu:** mở rộng coverage bằng biến thể KHÓ HƠN của seed `rule_found`, verify bằng solver (free, CPU).
- **File:** `offline/evolve_solved.py` **đã tồn tại** — CLI verify rồi (dưới).
- **Thuật toán (đã code, ĐÃ VERIFY):** mutate seed `rule_found` qua `--ops` (`add_operand,longer_chain,nest_op`, [evolve_solved.py:63](../../offline/evolve_solved.py#L63)) → solver `GENERATORS[cat]` giải → chỉ giữ `compare_answer` đúng → `make_example`. Hỗ trợ sẵn `bit_manipulation,equation_numeric_deduce,equation_numeric_guess`.
- **CLI (ĐÃ VERIFY):** `python3 offline/evolve_solved.py --categories bit_manipulation,equation_numeric_deduce --variants_per_seed 3 --max_seeds 0 --output corpus/parts/E_evolve.jsonl` (`--max_seeds 0` = dùng hết seed; flag là `--categories`/`--variants_per_seed`, KHÔNG phải `--category`/`--n`).
- **Corpus:** thêm part E vào mix tốt nhất từ A/B.
- **Gate OOD (bắt buộc):** đo trên **held-out category-level**, KHÔNG train-acc — synth-from-harder dễ lệch distribution. Nếu held-out không nhích → drop.

### EXP-F — Step-localized REDI: negative prefix-masked (exp32 Step-DPO, reframe về weight/sign)
**Mục tiêu:** thay vì phạt CẢ trace sai (EXP-C), chỉ phạt **đúng bước sai đầu tiên** — surgical, nhắm thẳng `arithmetic_slip`.
- **File mới:** `offline/build_step_negatives.py`
- **Ref (đọc trước):** [refs/step-dpo/data_pipeline/generate_dataset.py:42-49](../../refs/step-dpo/data_pipeline/generate_dataset.py#L42-L49) (cặp share-prefix, tách theo `\nStep `), [refs/step-dpo/stepdpo_trainer.py:50-72](../../refs/step-dpo/stepdpo_trainer.py#L50-L72) (prefix = prompt-mask, chỉ continuation tính loss).
- **Lợi thế data hiện tại:** Step-DPO gốc dùng **GPT-4** để định vị bước sai vì không có gold. Ta **CÓ gold solver trace** → định vị divergence **xác định, miễn phí**: so token-by-token gold-trace vs wrong-rollout, điểm khác đầu tiên = bước sai.
- **Thuật toán:** với mỗi problem có gold (part-S/solver) + ≥1 wrong rollout cùng id:
  1. align gold completion_ids vs wrong completion_ids, tìm index `d` đầu tiên khác nhau.
  2. emit 1 row từ wrong trace: `tokens = prompt + wrong_completion`, `mask`: prefix `[0..d)` = 0 (đã đúng, không học), suffix `[d..]` = 1; `weight=λ_step` (0.5), `sign=-1` → chỉ đè đúng phần lệch.
  3. (giữ part-S gold như positive bình thường — không cần đổi).
- **CLI:** `python3 offline/build_step_negatives.py --rollouts rollouts.jsonl --gold_corpus corpus/parts/S_solver.jsonl --lambda_step 0.5 --categories bit_manipulation,cipher,equation_numeric_deduce --out corpus/parts/F_stepneg.jsonl`
- **Distinct:** khác EXP-C (đè toàn trace) — đây là token-level credit-assignment, đúng tinh thần Step-DPO mà KHÔNG cần DPO-trainer (map vào mask+sign sẵn có).
- **Data fit:** chỉ category có gold (bit/cipher/equation). Nhắm 421 partial + 219 bit-zero (gold từ solver).
- **Success:** `arithmetic_slip_rate_wrong` bit/cipher ↓; format giữ. Rủi ro thấp hơn EXP-C vì negative cực hẹp.

### EXP-G — Synthetic self-correction traces (exp34)
**Mục tiêu:** dạy model **tự sửa** — bucket lỗi lớn nhất là slip; chèn pha "phát hiện sai → sửa đúng".
- **File mới:** `offline/build_correction.py`
- **Ref (đọc trước):** [refs/self-rewarding-correction/infer_math/](../../refs/self-rewarding-correction/infer_math/) (`gen_hf`→`reward_labeling`→`process_prompt_turn2`→`merge_data`). **Thay `reward_labeling` bằng `compare_answer`** của ta.
- **Lợi thế data:** thay vì để model tự-thưởng (P0 cho thấy nó tự tin sai, ppl≈1.0 → không tự phát hiện), ta **dựng trace correction bằng gold**: lấy wrong rollout làm "lần thử 1", chèn câu chuyển ("Let me recheck — column/bit X mismatches…"), rồi nối **gold continuation** làm "lần sửa".
- **Thuật toán:** với problem có gold + wrong rollout:
  1. `attempt = wrong_completion` (cắt trước `\boxed`); `bridge` = template ngắn cố định báo sai; `fix = gold_reasoning + \boxed{gold}`.
  2. `make_example(reasoning = attempt + bridge + fix, answer = gold)`. **Mask:** phần `attempt` = 0 (không học cách sai), `bridge+fix` = 1.
  3. cap độ dài < 7680 (attempt+fix có thể dài — ưu tiên category trace ngắn: cipher/equation; bit_manip skip nếu attempt đã ~6700).
- **CLI:** `python3 offline/build_correction.py --rollouts rollouts.jsonl --gold_corpus corpus/parts/S_solver.jsonl --categories cipher,equation_numeric_deduce --max_len 7600 --out corpus/parts/G_correct.jsonl`
- **Distinct:** khác exp13 (self-verify chỉ append-check) — đây là **sửa thật** (wrong→gold) trong 1 trace.
- **Data fit:** category trace NGẮN (cipher 2046, equation tránh). KHÔNG dùng bit_manip (đã sát budget). Success: slip-bucket ↓.

### EXP-H — Length-generalization curriculum cho bit_manip (exp35)
**Mục tiêu:** P0 cho 219 bit-zero ≈ trace dài sát trần (p95 7243). Dạy short→long để model giải bài dài bằng ít token hơn.
- **File mới:** `offline/build_length_curriculum.py` (gen bằng solver, CPU) — hoặc tái dùng `evolve_solved.py` với op `longer_chain`.
- **Ref (đọc trước):** [refs/self-improving-transformers/preamble.py:82-93](../../refs/self-improving-transformers/preamble.py#L82-L93) (`n_digits` range tăng theo `self_improve_round`), [run_self_improve.py:52,84](../../refs/self-improving-transformers/run_self_improve.py#L52). Ý lõi: train trên độ-dài nhỏ → generalize lên dài.
- **Thuật toán:** phân tầng bit_manip theo **số bước/độ dài input** (proxy: `N_BITS`/số cột). Tạo nhiều biến thể ở tầng NGẮN (verified by solver) → upweight; giữ tầng dài làm held-out đo length-gen. (ReST-EM: nếu có vòng 2, train rồi gen lại — nhưng 1-shot offline trước.)
- **CLI:** `python3 offline/build_length_curriculum.py --category bit_manipulation --short_max_cols 6 --variants 5 --out corpus/parts/H_lengthcur.jsonl`
- **Distinct:** khác exp10 (SA-curriculum theo độ-dài-trace generic) — đây là **structural difficulty** (số cột bit), có bằng chứng đúng-domain.
- **Data fit:** kết hợp EXP-B (compress) — vừa ngắn hoá vừa dạy short→long. Success: bit-zero ↓ trên **held-out tầng dài**.

### EXP-I — SwS weakness-targeted synthesis (exp36)
**Mục tiêu:** sinh thêm đúng **loại bit-rule/operator model hay trượt**, verified-by-construction.
- **File mới:** `offline/synth_weakness.py`
- **Ref (đọc trước):** [refs/sws/src/data_synthesis.py:13-35](../../refs/sws/src/data_synthesis.py#L13-L35) (`build_cooccurrence_stats`: concept freq + co-occurrence). [problem_generation.py](../../refs/sws/src/problem_generation.py) (gen nhắm concept — **thay vLLM-gen bằng generator thủ tục của ta**).
- **Thuật toán:**
  1. P0-extension: với mỗi wrong bit_manip rollout, gán "concept" = rule-family (`PAIR_FAMILIES`/op-type từ `reasoners/bit_manipulation.py`). Đếm freq lỗi per-family.
  2. Lấy top-k family yếu nhất → dùng generator solver sinh thêm bài đúng family đó → verify → corpus.
- **CLI:** `python3 offline/synth_weakness.py --rollouts rollouts.jsonl --top_k_families 5 --n_per_family 400 --out corpus/parts/I_weakness.jsonl`
- **Distinct:** khác EXP-E (evolve đều tay) — nhắm **điểm yếu đo được** từ error-bucket.
- **Data fit:** cần map family được; nếu generator không phủ family yếu → fallback EXP-E uniform. Success: family yếu pass@1 ↑.

### EXP-J — Difficulty-adaptive resampling (AdaSTaR, free reweight)
**Mục tiêu:** thao tác THUẦN trên mixture — upsample 477 boundary-problem, downweight 7874 mastered. Zero gen, làm cùng `build_corpus.py`.
- **File:** mở rộng `build_corpus.py` thêm `--difficulty_weight`.
- **Ref (đọc trước):** [refs/adastar/utils_adastar.py](../../refs/adastar/utils_adastar.py), [device_inference_adastar_new.py](../../refs/adastar/device_inference_adastar_new.py) — adaptive sampling theo accuracy (diversity+curriculum, dồn budget vào boundary).
- **Thuật toán:** dùng pass-count/8 từ P0 làm difficulty `c∈{0..8}`: `weight(problem) = w_boundary nếu 1≤c≤7 (boundary, ×1.5–2.0); ×1.0 nếu c=8 (mastered); c=0 để solver lo (Tier-0/EXP-E)`. Bake vào field `weight` của row part-S/R tương ứng id.
- **CLI:** `build_corpus.py --parts S_solver R_rft --difficulty_weight --passcount_json yield_probe_passcount.json --out corpus/mix_adastar.jsonl` (cần dump pass-count/id — mở rộng `measure_yield.py` hoặc script P0 đã có).
- **Distinct:** khác mọi exp khác — không thêm trace, chỉ **đổi trọng số theo difficulty đo được**. Rẻ nhất, rủi ro thấp nhất.
- **Data fit:** trực tiếp dùng 477 partial của P0. Success: boundary pass@1 ↑ mà mastered không tụt; layer được lên BẤT KỲ corpus A–I.

### Tier-0 (song song, ngoài batch-4) — solver cryptarithm/guess
870 zero-problem (crypto_d 603 + crypto_g 153 + eq_g 114) **chỉ** solver tạo trace được; sampling đã chứng minh bất lực (pass@k≈pass@1). Đây là lever lớn nhất nhưng nằm trong `nemotron-master/reasoners/` (xem [[cryptarithm-solver-wired]], [research/cryptarithm_gap_plan.md](../cryptarithm_gap_plan.md)) — track riêng, không thuộc 10 exp (A–J) trên. Output của nó cũng đổ vào part-S, nên hưởng lợi từ cùng `build_corpus.py`.

---

## PHẦN 3 — Run matrix & gating

| Round | Corpus | Builders | Bet | Cost |
|---|---|---|---|---|
| 1 | mix_rft | S + R(A) | partial-convert bit/cipher | free (no GPU gen) |
| 2 | mix_tokenskip | S(bit nén, B) — compression-only, KHÔNG R (dedup_id giết RFT) | + cứu 219 bit zero | CPU |
| 3 | mix_adastar | S + R + difficulty-reweight (J) | boundary upsample, zero gen | free |
| 4 | mix_redi | S + R + N(λ, C) | learning-from-negatives full-trace | free |
| 5 | mix_stepneg | S + R + F(step-neg) | negative token-level (thay/bổ C) | free |
| 6 | mix_apo | S + R(adv, D) | advantage-weight bit/cipher | free |
| 7 | +correction | best(1-6) + G | self-correction cipher/eq | free |
| 8 | +evolve/length | best + E + H | coverage harder + length-gen | CPU |
| 9 | +weakness | best + I | family-targeted synth | CPU |

**Thứ tự chạy = thứ tự bảng** (rủi ro/chi phí/độ-mới tăng dần; round 1-3 free + an toàn nhất làm trước). Mỗi round layer lên best-so-far, KHÔNG cộng dồn mù — drop part nào làm tụt. Sau mỗi round:
1. `python3 offline/measure_yield.py --rollouts <new_rollouts nếu có> --output yield_r<N>.json` (hoặc đo qua submission).
2. Copy `tracker/rounds/round_template.md`→`round_<N>.md`, fill; append `tracker/leaderboard.md`.
3. So 5-nhóm-mạnh: tụt >0.5pp ⇒ rollback round đó.

**CẤM:** đổi PROMPT_SUFFIX / token-format; train R hoặc N mà không có S; on-policy/vLLM-in-loop; thêm dep ngoài common/; re-encode text khi đã có `*_token_ids` trong rollout (dùng `make_example_from_ids`).

**Definition of done batch-4-mix:** ≥1 round > 0.86 trên leaderboard, ghi đủ tracker, builders có validate pass + `ruff` sạch.
