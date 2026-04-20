from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


def dcg_at_k(relevances: list[float], k: int) -> float:
    gains = [((2**rel) - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k])]
    return float(sum(gains))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0:
        return 0.0
    return dcg_at_k(relevances, k) / denom


def reciprocal_rank(binary_relevances: list[int]) -> float:
    for idx, rel in enumerate(binary_relevances, start=1):
        if rel:
            return 1.0 / idx
    return 0.0


def precision_at_k(binary_relevances: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = binary_relevances[:k]
    if not top:
        return 0.0
    return float(sum(top)) / len(top)


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    compared = list(zip(y_true, y_pred, strict=False))
    if not compared:
        return 0.0
    return float(sum(int(a == b) for a, b in compared)) / len(compared)


def mrr(relevance_scores: list[list[float]], k: int | None = None) -> float:
    """Mean Reciprocal Rank across queries.

    Args:
        relevance_scores: List of ranked relevance lists per query.
                         Each inner list is document relevance scores (higher = more relevant).
        k: Maximum rank to consider. If None, uses full list length.

    Returns:
        MRR score (0.0 to 1.0)
    """
    if not relevance_scores:
        return 0.0

    reciprocal_ranks: list[float] = []
    for scores in relevance_scores:
        if k is not None:
            scores = scores[:k]
        for idx, score in enumerate(scores, start=1):
            if score > 0:
                reciprocal_ranks.append(1.0 / idx)
                break
        else:
            reciprocal_ranks.append(0.0)

    return float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0


def map(relevance_scores: list[list[float]], k: int | None = None) -> float:
    """Mean Average Precision across queries.

    Args:
        relevance_scores: List of ranked relevance lists per query.
                         Each inner list is document relevance scores (higher = more relevant).
        k: Maximum rank to consider. If None, uses full list length.

    Returns:
        MAP score (0.0 to 1.0)
    """
    if not relevance_scores:
        return 0.0

    average_precisions: list[float] = []
    for scores in relevance_scores:
        if k is not None:
            scores = scores[:k]
        precisions: list[float] = []
        relevant_count = 0
        for idx, score in enumerate(scores, start=1):
            if score > 0:
                relevant_count += 1
                precisions.append(relevant_count / idx)
        if precisions:
            average_precisions.append(sum(precisions) / len(precisions))
        else:
            average_precisions.append(0.0)

    return float(sum(average_precisions) / len(average_precisions)) if average_precisions else 0.0


@dataclass(slots=True)
class LatencyTracker:
    samples_ms: list[float] = field(default_factory=list)

    @contextmanager
    def measure(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.samples_ms.append((time.perf_counter() - start) * 1000)

    def summary(self) -> dict[str, float]:
        if not self.samples_ms:
            return {"p50": 0.0, "p99": 0.0, "mean": 0.0}
        ordered = sorted(self.samples_ms)
        p50 = float(np.percentile(np.asarray(ordered, dtype=np.float64), 50, method="nearest"))
        p99 = float(np.percentile(np.asarray(ordered, dtype=np.float64), 99, method="nearest"))
        return {
            "p50": p50,
            "p99": p99,
            "mean": float(statistics.fmean(ordered)),
        }
