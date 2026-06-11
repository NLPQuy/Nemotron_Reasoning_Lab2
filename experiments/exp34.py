# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # EXP34 — Paraphrase traces (Shape-of-Thought)  (Batch-4 Idea 6, P12, T1)
# ============================================================
# Paper: arXiv:2512.22255 — NO official repo. Dùng Claude API.
# Upstream: nemotron-master/paraphrase_instances.py (đọc trước — có thể cần extend)
#   + reasoning/ → reasoning_paraphrased/ (paraphrased traces)
#   + corpus.py rebuild với traces mới
# Change: reasoning/ traces được paraphrase → diverse vocabulary, giữ nguyên structure
# Rollback: dùng reasoning/ gốc (đừng overwrite — viết vào reasoning_paraphrased/ riêng)
# Expected: +0.5–1.5 pp (gradient saturation relief cho template-rigid traces)
# Cost estimate: ~2–3M tokens via Claude Haiku API (~$6–10)
# ============================================================

# %% [code] {"jupyter":{"outputs_hidden":false}}
import os

NEMOTRON_MASTER = os.path.join(os.path.dirname(__file__), "..", "nemotron-master")

# Paths
REASONING_DIR = os.path.join(NEMOTRON_MASTER, "reasoning")
PARAPHRASE_OUT = os.path.join(NEMOTRON_MASTER, "reasoning_paraphrased")

# Verification gates
MAX_TOKEN_COUNT = 7600        # hard limit — traces dài hơn bị discard
MIN_LOGPROB_IMPROVEMENT = 0.1 # avg per-token logprob của adapter 0.86 phải tăng > 0.1 trên paraphrased

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Bước 0 — Đọc paraphrase_instances.py

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(f"Đọc trước: {os.path.join(NEMOTRON_MASTER, 'paraphrase_instances.py')}")
print()
print("Key questions:")
print("1. Hàm generate() đang paraphrase problems hay traces?")
print("2. Verification gate đã có chưa (compare_answer + token count)?")
print("3. Claude API được call như thế nào — sync hay async?")
print()
print("Nếu paraphrase_instances.py chỉ xử lý problem instances (không phải reasoning/*.txt):")
print("→ Thêm mode='traces' với input=reasoning/, output=reasoning_paraphrased/")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Bước 1 — Pre-check Shape-of-Thought transfer

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Trước khi paraphrase toàn bộ (~8,600 traces):
# 1. Paraphrase 100 traces ngẫu nhiên
# 2. Tính avg per-token logprob của adapter 0.86 trên paraphrased vs original
# 3. Nếu logprob improvement < MIN_LOGPROB_IMPROVEMENT → Shape-of-Thought không transfer → ABORT

PRE_CHECK_SAMPLE = 100
print(f"Pre-check: paraphrase {PRE_CHECK_SAMPLE} traces → compare logprob với adapter 0.86")
print(f"Threshold: avg logprob improvement >= {MIN_LOGPROB_IMPROVEMENT}")
print()
print("Script: uv run python3 paraphrase_instances.py --mode traces --sample 100 --verify")
print("Sau đó: uv run python3 compute_corpus_logprobs.py (từ exp30) để compare logprobs")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Bước 2 — Full paraphrase run

# %% [code] {"jupyter":{"outputs_hidden":false}}
PARAPHRASE_CMD = f"""
cd {NEMOTRON_MASTER} && \\
uv run python3 paraphrase_instances.py \\
  --mode   traces \\
  --input  reasoning/ \\
  --out    reasoning_paraphrased/ \\
  --verify \\
  --max_tokens {MAX_TOKEN_COUNT}
# KHÔNG overwrite reasoning/ — giữ original để rollback
"""
print("Full paraphrase command (sau pre-check passed):")
print(PARAPHRASE_CMD)
print()
print("Verification rate expected: >95% (fail chủ yếu do boxed answer thay đổi)")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Bước 3 — Rebuild corpus với paraphrased traces

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Sửa corpus.py để đọc từ reasoning_paraphrased/ thay vì reasoning/
# Hoặc thêm env var: REASONING_DIR=reasoning_paraphrased/ uv run python3 corpus.py

CORPUS_CMD = f"""
cd {NEMOTRON_MASTER} && \\
REASONING_DIR=reasoning_paraphrased uv run python3 corpus.py
# Nếu corpus.py dùng hardcoded path "reasoning/" → sửa thành os.environ.get("REASONING_DIR", "reasoning/")
"""
print("Rebuild corpus:")
print(CORPUS_CMD)
print()
print("Sau rebuild: pack_kaggle_snapshot.py --tag exp34 → upload → train")
