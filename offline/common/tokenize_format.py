import re
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from transformers import AutoTokenizer

from .verify import PROMPT_SUFFIX

TOKEN_LIMIT = 8192


def load_tokenizers(
    model_path: str, tokenizer_json_path: str | Path
) -> tuple[Any, Tokenizer]:
    chat_tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    comp_tok = Tokenizer.from_file(str(tokenizer_json_path))
    return chat_tok, comp_tok


def make_example(
    prompt_text: str,
    reasoning_text: str,
    answer: str,
    *,
    chat_tok: Any,
    comp_tok: Tokenizer,
    problem_id: str = "",
    category: str = "rollout",
    weight: float = 1.0,
    sign: float = 1.0,
) -> dict:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", reasoning_text)
    ans = boxed[-1] if boxed else answer
    completion = (
        f"{reasoning_text.rstrip(chr(10))}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    )
    comp_ids = comp_tok.encode(completion, add_special_tokens=False).ids
    prompt_ids = chat_tok.apply_chat_template(
        [{"role": "user", "content": prompt_text + PROMPT_SUFFIX}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return make_example_from_ids(
        prompt_ids,
        comp_ids,
        problem_id=problem_id,
        category=category,
        weight=weight,
        sign=sign,
    )


def make_example_from_ids(
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    problem_id: str,
    category: str,
    weight: float = 1.0,
    sign: float = 1.0,
) -> dict:
    tokens = list(prompt_ids) + list(completion_ids)
    mask = [0] * len(prompt_ids) + [1] * len(completion_ids)
    if len(tokens) > TOKEN_LIMIT:
        tokens = tokens[:TOKEN_LIMIT]
        mask = mask[:TOKEN_LIMIT]
    return {
        "problem_id": problem_id,
        "category": category,
        "tokens": tokens,
        "mask": mask,
        "weight": float(weight),
        "sign": float(sign),
    }
