# Search Log — Batch 3 — 2026-06-02

(Live searches only; every primary paper below traces to one of these queries. Backend returned 529/Overloaded for ~10 min at session start; queries retried until healthy, then run.)

## Tier 1: In-Field (LLM reasoning SFT / PEFT / RL)
- Query: "LoRA+ efficient low rank adaptation different learning rates Hayou 2024" → picked [arXiv:2402.12354, ICML'24]
- Query: "high-entropy minority tokens reasoning reinforcement learning 2025" → picked [arXiv:2506.01939, NeurIPS'25]
- Query: "DoReMi domain reweighting data mixture language model training NeurIPS 2023" → picked [arXiv:2305.10429, NeurIPS'23 spotlight]
- Query: "GSPO group sequence policy optimization mixture of experts RL Qwen 2025" → picked [arXiv:2507.18071, 2025]

## Tier 2: Adjacent (robustness / MoE-PEFT)
- Query: "GroupDRO distributionally robust optimization worst-group Sagawa ICLR 2020" → picked [arXiv:1911.08731, ICLR'20]
- Query: "ESFT expert-specialized fine-tuning mixture of experts DeepSeek EMNLP 2024" → picked [arXiv:2407.01906, EMNLP'24 main]

## Tier 3: Cross-Domain (RL goal-relabeling / symbolic constraint solving)
- Query: "hindsight experience replay goal relabeling reinforcement learning Andrychowicz 2017" → picked [arXiv:1707.01495, NeurIPS'17]
- Query: "distilling symbolic solver reasoning chain-of-thought synthetic data constraint satisfaction LLM" → picked [arXiv:2512.03272; supporting arXiv:2505.13252; Logic-LM EMNLP'23 (github-confirmed)]

## Devil's-advocate (top-1 = LoRA+)
- Query: "LoRA+ learning rate ratio limitations negative results does not improve reproduction" → contrast [arXiv:2602.04998 "Learning Rate Matters: Vanilla LoRA May Suffice"; arXiv:2410.09692 ALLoRA]
- Query: "LoRA+ sensitivity learning rate ratio hyperparameter instability critique" → reinforced contrast [arXiv:2602.04998]

## Totals
- Queries used (successful): 10 / 19  (plus ~9 failed 529 retries, not counted)
- Summaries read: ~40 / 45
- Full reads: 0 / 10 (search-snippet summaries sufficed)
- Wall-clock: ~12 min active (excl. backend-outage wait)
- Saturation events: none
