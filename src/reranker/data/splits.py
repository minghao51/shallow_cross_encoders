from __future__ import annotations

import random
from collections.abc import Callable
from typing import TypeVar

from reranker.config import get_settings

RowT = TypeVar("RowT")


def _normalize_ratios(ratios: tuple[float, float, float]) -> tuple[float, float, float]:
    total = sum(ratios)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return tuple(ratio / total for ratio in ratios)  # type: ignore[return-value]


def partition_rows(
    rows: list[RowT],
    key_fn: Callable[[RowT], str],
    split: str,
    seed: int | None = None,
    ratios: tuple[float, float, float] | None = None,
) -> list[RowT]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    if not rows:
        return []

    settings = get_settings()
    resolved_seed = settings.synthetic_data.seed if seed is None else seed
    resolved_ratios = _normalize_ratios((0.7, 0.15, 0.15) if ratios is None else ratios)

    grouped: dict[str, list[RowT]] = {}
    for row in rows:
        grouped.setdefault(key_fn(row), []).append(row)

    group_keys = list(grouped)
    random.Random(resolved_seed).shuffle(group_keys)

    total_groups = len(group_keys)
    train_cut = max(1, int(round(total_groups * resolved_ratios[0])))
    validation_cut = int(round(total_groups * resolved_ratios[1]))
    if train_cut >= total_groups and total_groups > 1:
        train_cut = total_groups - 1
    if train_cut + validation_cut >= total_groups and total_groups > 2:
        validation_cut = max(1, total_groups - train_cut - 1)

    train_keys = set(group_keys[:train_cut])
    validation_keys = set(group_keys[train_cut : train_cut + validation_cut])
    test_keys = set(group_keys) - train_keys - validation_keys
    if not test_keys:
        spill_key = group_keys[-1]
        train_keys.discard(spill_key)
        validation_keys.discard(spill_key)
        test_keys = {spill_key}

    selected_keys = {
        "train": train_keys,
        "validation": validation_keys,
        "test": test_keys,
    }[split]
    return [row for row in rows if key_fn(row) in selected_keys]
