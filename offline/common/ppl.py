import math
from typing import Any, Sequence


def ppl_from_logp(old_logp: Sequence[float], mask: Sequence[int | float]) -> float:
    vals = [float(lp) for lp, m in zip(old_logp, mask) if int(m) == 1]
    if not vals:
        return math.inf
    mean_logp = sum(vals) / len(vals)
    try:
        return math.exp(-mean_logp)
    except OverflowError:
        return math.inf


def trace_ppl_hf(
    model: Any,
    tokenizer: Any,
    text: str,
    comp_mask: Sequence[int | float],
    step_tokens: int = 128,
) -> float:
    import torch

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(model.device)
    mask = torch.tensor(list(comp_mask), device=model.device, dtype=torch.bool)
    if mask.numel() > input_ids.shape[1]:
        mask = mask[: input_ids.shape[1]]
    elif mask.numel() < input_ids.shape[1]:
        pad = torch.zeros(
            input_ids.shape[1] - mask.numel(), device=model.device, dtype=torch.bool
        )
        mask = torch.cat([mask, pad], dim=0)

    nlls: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[1] - 1, step_tokens):
            end = min(input_ids.shape[1], start + step_tokens + 1)
            chunk = input_ids[:, start:end]
            out = model(chunk)
            log_probs = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
            labels = chunk[:, 1:]
            token_nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            label_mask = mask[start + 1 : end].unsqueeze(0)
            if label_mask.any():
                nlls.append(token_nll[label_mask])

    if not nlls:
        return math.inf
    mean_nll = torch.cat(nlls).mean().item()
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return math.inf
