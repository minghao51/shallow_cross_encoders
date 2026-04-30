"""Evaluation metrics and latency tracking utilities."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k.

    Args:
        relevances: Relevance scores in ranked order.
        k: Rank position to compute up to.

    Returns:
        DCG value.
    """
    gains = [((2**rel) - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k])]
    return float(sum(gains))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at k.

    Args:
        relevances: Relevance scores in ranked order.
        k: Rank position.

    Returns:
        NDCG value (0.0 to 1.0).
    """
    ideal = sorted(relevances, reverse=True)
    denom = dcg_at_k(ideal, k)
    if denom == 0:
        return 0.0
    return dcg_at_k(relevances, k) / denom


def reciprocal_rank(binary_relevances: list[int]) -> float:
    """Compute the reciprocal rank of the first relevant document.

    Args:
        binary_relevances: Binary relevance indicators in ranked order.

    Returns:
        Reciprocal rank (0.0 to 1.0), or 0.0 if no relevant doc found.
    """
    for idx, rel in enumerate(binary_relevances, start=1):
        if rel:
            return 1.0 / idx
    return 0.0


def precision_at_k(binary_relevances: list[int], k: int) -> float:
    """Compute Precision at k.

    Args:
        binary_relevances: Binary relevance indicators in ranked order.
        k: Rank position.

    Returns:
        Precision at k (0.0 to 1.0).
    """
    if k <= 0:
        return 0.0
    top = binary_relevances[:k]
    if not top:
        return 0.0
    return float(sum(top)) / len(top)


def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy score (0.0 to 1.0).
    """
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


def mean_average_precision(relevance_scores: list[list[float]], k: int | None = None) -> float:
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
    """Track latency samples and compute summary statistics."""

    samples_ms: list[float] = field(default_factory=list)

    @contextmanager
    def measure(self) -> Iterator[None]:
        """Context manager to measure and record elapsed time in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.samples_ms.append((time.perf_counter() - start) * 1000)

    def summary(self) -> dict[str, float]:
        """Compute p50, p99, and mean latency from recorded samples.

        Returns:
            Dict with keys p50, p99, mean (in milliseconds).
        """
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
