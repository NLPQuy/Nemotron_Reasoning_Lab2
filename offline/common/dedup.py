from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def jaccard(a_ids: Sequence[int], b_ids: Sequence[int]) -> float:
    a = set(a_ids)
    b = set(b_ids)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def dedup_keep_diverse(
    items: Sequence[T],
    k: int,
    key: Callable[[T], Sequence[int]],
    thr: float = 0.7,
    anchor: T | None = None,
) -> list[T]:
    kept: list[T] = []
    kept_ids: list[Sequence[int]] = []

    if anchor is not None and k > 0:
        kept.append(anchor)
        kept_ids.append(key(anchor))

    for item in items:
        if len(kept) >= k:
            break
        item_ids = key(item)
        if all(jaccard(item_ids, prev_ids) < thr for prev_ids in kept_ids):
            kept.append(item)
            kept_ids.append(item_ids)

    return kept[:k]
