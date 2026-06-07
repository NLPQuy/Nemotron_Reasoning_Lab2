import gc
import os
from typing import Any


def load_engine(
    model_path: str,
    adapter_path: str | None,
    max_model_len: int = 8192,
) -> tuple[Any, Any, Any | None]:
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    lora_request = None
    if adapter_path and os.path.isfile(
        os.path.join(adapter_path, "adapter_config.json")
    ):
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("offline_adapter", 1, adapter_path)
    return llm, tokenizer, lora_request


def sample(
    llm: Any,
    prompt_token_ids: list[list[int]],
    lora_request: Any | None,
    *,
    n: int,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_tokens: int = 7680,
    logprobs: int = 1,
) -> Any:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        logprobs=logprobs,
    )
    prompts = [
        TokensPrompt(prompt_token_ids=[int(token_id) for token_id in ids])
        for ids in prompt_token_ids
    ]
    return llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )


def free(llm: Any) -> None:
    import torch

    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment

        destroy_distributed_environment()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
