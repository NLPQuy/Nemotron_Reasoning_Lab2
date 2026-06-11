"""Generate greedy predictions on the eval slice with vLLM (Batch-3, RUN ON KAGGLE/GPU).

This is the only GPU step of the slice workflow. It mirrors the competition inference
contract exactly: greedy (temperature 0), max_tokens 7680, max_model_len 8192, and the
SAME prompt construction as corpus.py / the grader (chat template + PROMPT_SUFFIX +
enable_thinking=True). Output is written as preds.jsonl ({id, output}) which eval_slice.py
then scores with the grader's own compare_answer.

NOT runnable on a typical local box (Nemotron-3-Nano-30B-A3B is a 30B MoE). Run inside a
GPU notebook (Kaggle/Modal) with vLLM installed and the trained LoRA adapter available.

Usage (inside a GPU notebook):
    uv run python3 infer_slice.py \
        --base   <base-model-path-or-hf-id> \
        --adapter <path-to-extracted-adapter-dir>   # the dir from submission.zip (has adapter_config.json) \
        --slice  eval_slice.jsonl \
        --out    preds.jsonl
Then locally: uv run python3 eval_slice.py --preds preds.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Must match corpus.py / query.py exactly.
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)
MAX_TOKENS = 7680
MAX_MODEL_LEN = 8192
MAX_LORA_RANK = 32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="base model path or HF id")
    parser.add_argument(
        "--adapter",
        default=None,
        help="path to the extracted LoRA adapter dir (with adapter_config.json). "
        "Omit to evaluate the base model with no adapter.",
    )
    parser.add_argument("--slice", default="eval_slice.jsonl")
    parser.add_argument("--out", default="preds.jsonl")
    # >>> EXP29 START — sampling support (defaults reproduce the old greedy behaviour).
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="sampling temperature; 0.0 = greedy (default, unchanged behaviour)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="completions per problem; 1 keeps the legacy {id, output} schema, "
        ">1 writes {id, outputs:[...]} for DPO rollout collection",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="sampling seed (only meaningful when temperature > 0)",
    )
    parser.add_argument(
        "--gen_chunk",
        type=int,
        default=2,
        help="hf backend: generate this many samples per model.generate() call and "
        "loop, to bound peak memory (num_return_sequences=n_samples at once OOMs the "
        "30B on the naive Mamba path). 0 = all n_samples in one call (legacy).",
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="vllm (fast, matches grader) or hf (transformers.generate fallback when "
        "vLLM is unavailable, e.g. no compatible wheel)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_TOKENS,
        help="cap on generated tokens. Default 7680 = competition contract (use for "
        "eval). Lower it for hf rollouts to fit a time budget.",
    )
    # <<< EXP29 END
    args = parser.parse_args()
    cap = args.max_new_tokens

    from transformers import AutoTokenizer

    slice_recs = [
        json.loads(line)
        for line in Path(args.slice).read_text().splitlines()
        if line.strip()
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    # prompt_token_ids = [
    #     tokenizer.apply_chat_template(
    #         [{"role": "user", "content": rec["prompt"] + PROMPT_SUFFIX}],
    #         tokenize=True,
    #         add_generation_prompt=True,
    #         enable_thinking=True,
    #     )
    #     for rec in slice_recs
    # ]
    prompt_token_ids = []

    for rec in slice_recs:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": rec["prompt"] + PROMPT_SUFFIX}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        # apply_chat_template may return list[int], a tokenizers.Encoding, or a
        # BatchEncoding depending on tokenizer/transformers version. Coerce to list[int]
        # so torch.tensor([ids]) below never sees an Encoding object.
        if hasattr(ids, "ids"):  # tokenizers.Encoding
            ids = ids.ids
        elif hasattr(ids, "input_ids"):  # BatchEncoding
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):  # nested batch-of-1
            ids = ids[0]
        ids = [int(t) for t in ids]

        prompt_token_ids.append(ids)
        
    # >>> EXP29 START — two backends. per_rec_samples[i] = list of (text, n_output_tokens)
    # for slice_recs[i]. vllm matches the grader; hf is a no-vLLM fallback.
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(
            model=args.base,
            trust_remote_code=True,
            enable_lora=args.adapter is not None,
            max_lora_rank=MAX_LORA_RANK,
            max_model_len=MAX_MODEL_LEN,
        )
        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=cap,
            n=args.n_samples,
            seed=args.seed if args.temperature > 0 else None,
        )
        lora_req = (
            LoRARequest("slice_adapter", 1, args.adapter) if args.adapter else None
        )
        outputs = llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling,
            lora_request=lora_req,
        )
        per_rec_samples = [
            [(s.text, len(s.token_ids)) for s in o.outputs] for o in outputs
        ]
    else:  # hf — transformers.generate (slower, no vLLM dependency)
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        if args.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
            # Merge LoRA + drop the PEFT wrapper: its generate() bypasses NemotronH's
            # prepare_inputs_for_generation, so the hybrid Mamba cache is never built ->
            # no-cache O(n^2) generation that effectively hangs. Merging restores the
            # model's native cached generation path.
            model = model.merge_and_unload()
        model.eval()
        if model.generation_config is not None:
            model.generation_config.use_cache = True

        # NemotronH only caches when an actual NemotronHHybridDynamicCache is passed;
        # transformers' generate() auto-builds a plain DynamicCache (wrong type), so the
        # model logs "no cache will be returned" and runs cacheless O(n^2) (effective
        # hang). Resolve the hybrid-cache class from the model's remote-code module and
        # pass a FRESH instance per generate() call.
        import importlib

        _HybridCache = None
        try:
            _mod = importlib.import_module(model.__class__.__module__)
            _HybridCache = getattr(_mod, "NemotronHHybridDynamicCache", None)
        except Exception:
            _HybridCache = None

        def _make_cache(batch_size: int):
            if _HybridCache is None:
                return None
            for kw in (
                {"config": model.config, "batch_size": batch_size},
                {"config": model.config, "max_batch_size": batch_size},
            ):
                try:
                    return _HybridCache(
                        **kw, dtype=model.dtype, device=model.device
                    )
                except TypeError:
                    continue
            for args_pos in ((model.config, batch_size), (model.config,)):
                try:
                    return _HybridCache(*args_pos)
                except Exception:
                    continue
            # Signature didn't match any attempt: surface it so it can be fixed exactly.
            import inspect

            try:
                print(
                    "WARNING: could not build NemotronHHybridDynamicCache; its __init__ "
                    f"signature is {inspect.signature(_HybridCache.__init__)} "
                    "-> falling back to cacheless (slow). Report this signature to patch."
                )
            except Exception:
                pass
            return None

        if _HybridCache is None:
            print(
                "WARNING: NemotronHHybridDynamicCache not found -> generation will be "
                "cacheless (very slow). Check the base model's remote code."
            )

        do_sample = args.temperature > 0
        if do_sample:
            torch.manual_seed(args.seed)
        gen_kwargs = {
            "max_new_tokens": cap,
            "do_sample": do_sample,
            "use_cache": True,  # required: NemotronH recomputes the whole seq each step otherwise
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature

        # Generate n_samples in chunks (default 2) instead of one
        # num_return_sequences=n_samples call: the 30B on the naive Mamba path
        # materialises all parallel sequences' activations at once -> OOM. Chunking
        # bounds peak memory; greedy (n_samples=1) is a single chunk, unchanged.
        chunk = args.gen_chunk if args.gen_chunk and args.gen_chunk > 0 else args.n_samples
        per_rec_samples = []
        for ids in prompt_token_ids:
            input_ids = torch.tensor([ids], device=model.device)
            plen = len(ids)
            samples = []
            remaining = args.n_samples
            while remaining > 0:
                k = min(chunk, remaining)
                cache = _make_cache(k)  # fresh hybrid cache per call (None => cacheless)
                call_kwargs = dict(gen_kwargs)
                if cache is not None:
                    call_kwargs["past_key_values"] = cache
                with torch.no_grad():
                    gen = model.generate(
                        input_ids, num_return_sequences=k, **call_kwargs
                    )
                for g in gen:
                    new = g[plen:]
                    text = tokenizer.decode(new, skip_special_tokens=True)
                    samples.append((text, int(new.shape[0])))
                remaining -= k
                del gen
                torch.cuda.empty_cache()
            per_rec_samples.append(samples)

    # n_samples==1 keeps the legacy single-output schema (eval_slice.py reads "output");
    # n_samples>1 emits an "outputs" list for build_dpo_pairs.py.
    with open(args.out, "w") as f:
        for rec, samples in zip(slice_recs, per_rec_samples):
            if args.n_samples == 1:
                text, n_tokens = samples[0]
                record: dict = {
                    "id": rec["id"],
                    "output": text,
                    "n_output_tokens": n_tokens,
                    "hit_cap": n_tokens >= cap,
                }
            else:
                record = {
                    "id": rec["id"],
                    "outputs": [t for t, _ in samples],
                    "n_output_tokens": [n for _, n in samples],
                    "hit_cap": [n >= cap for _, n in samples],
                }
            f.write(json.dumps(record) + "\n")

    n_cap = sum(1 for s in per_rec_samples for (_, n) in s if n >= cap)
    total_gen = len(slice_recs) * args.n_samples
    print(f"Wrote {len(slice_recs)} records ({total_gen} completions) -> {args.out}")
    print(f"hit max_tokens cap ({cap}): {n_cap} (truncation-risk count)")
    print("Next (local): uv run python3 eval_slice.py --preds", args.out)
    # <<< EXP29 END


if __name__ == "__main__":
    main()
