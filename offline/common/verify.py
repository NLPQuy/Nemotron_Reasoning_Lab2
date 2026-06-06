import re


PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


def extract_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    return ""


def compare_answer(stored: str, pred: str) -> bool:
    stored = stored.strip()
    pred = pred.strip()
    if re.fullmatch(r"[01]+", stored):
        return pred.lower() == stored.lower()
    try:
        stored_num = float(stored)
        pred_num = float(pred)
        if stored_num == 0:
            return abs(pred_num) < 1e-2
        return abs(stored_num - pred_num) / abs(stored_num) <= 1e-2
    except ValueError:
        return pred.lower() == stored.lower()


def format_ok(text: str) -> bool:
    return len(re.findall(r"\\boxed\{", text)) >= 1
