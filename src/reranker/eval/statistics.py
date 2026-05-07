"""Statistical significance testing for benchmark results.

Provides bootstrap confidence intervals, Wilcoxon signed-rank tests,
and strategy comparison with CI-overlap detection.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def bootstrap_ci(
    scores: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a list of per-query metric scores.

    Args:
        scores: Per-query metric values (e.g., per-query NDCG@10).
        n_resamples: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        random_seed: RNG seed for reproducibility.

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if len(scores) < 2:
        return (float(scores[0]) if scores else 0.0, float(scores[0]) if scores else 0.0)

    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower = float(np.percentile(bootstrap_means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0)))
    return (lower, upper)


def wilcoxon_signed_rank(
    a: list[float],
    b: list[float],
) -> dict[str, float]:
    """Paired Wilcoxon signed-rank test between two lists of per-query scores.

    Args:
        a: Per-query scores for strategy A.
        b: Per-query scores for strategy B (paired with a).

    Returns:
        Dict with 'W' (test statistic) and 'p_value' (two-sided).
    """
    if len(a) != len(b):
        raise ValueError(f"Lists must have same length: {len(a)} vs {len(b)}")

    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diff = arr_a - arr_b
    non_zero = diff[diff != 0]

    if len(non_zero) < 2:
        return {"W": 0.0, "p_value": 1.0}

    abs_diff = np.abs(non_zero)
    sorted_idx = np.argsort(abs_diff)
    sorted_abs = abs_diff[sorted_idx]
    ranks = np.empty_like(sorted_abs)

    i = 0
    while i < len(sorted_abs):
        j = i
        while j < len(sorted_abs) and math.isclose(sorted_abs[j], sorted_abs[i]):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    signs = np.sign(non_zero[sorted_idx])
    W_pos = np.sum(ranks[signs > 0])
    W_neg = np.sum(ranks[signs < 0])
    W = min(W_pos, W_neg)

    n = len(non_zero)
    if n < 10:
        return {"W": float(W), "p_value": 0.5}

    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sigma
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))

    return {"W": float(W), "p_value": float(p_value)}


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz and Stegun)."""
    if x < 0:
        return 1.0 - _normal_cdf(-x)
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    t = 1.0 / (1.0 + b0 * x)
    phi = 0.3989422804014327 * math.exp(-x * x / 2.0)
    return 1.0 - phi * (b1 * t + b2 * t**2 + b3 * t**3 + b4 * t**4 + b5 * t**5)


def compare_strategies(
    name_a: str,
    per_query_a: list[float],
    name_b: str,
    per_query_b: list[float],
    metric_name: str = "NDCG@10",
) -> dict[str, Any]:
    """Compare two strategies with bootstrap CI and Wilcoxon test.

    Args:
        name_a: Name of strategy A.
        per_query_a: Per-query metric scores for A.
        name_b: Name of strategy B.
        per_query_b: Per-query metric scores for B (paired with A).
        metric_name: Human-readable metric name for output.

    Returns:
        Dict with comparison results including means, CIs, delta, and significance.
    """
    mean_a = float(np.mean(per_query_a)) if per_query_a else 0.0
    mean_b = float(np.mean(per_query_b)) if per_query_b else 0.0
    ci_a = bootstrap_ci(per_query_a)
    ci_b = bootstrap_ci(per_query_b)
    wilcoxon = wilcoxon_signed_rank(per_query_a, per_query_b)

    delta = mean_a - mean_b
    ci_overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])

    return {
        "metric": metric_name,
        "strategy_a": name_a,
        "strategy_b": name_b,
        f"{name_a}_mean": mean_a,
        f"{name_b}_mean": mean_b,
        f"{name_a}_ci_95": ci_a,
        f"{name_b}_ci_95": ci_b,
        "delta": delta,
        "ci_overlap": ci_overlap,
        "wilcoxon_W": wilcoxon["W"],
        "wilcoxon_p_value": wilcoxon["p_value"],
        "significant_at_005": wilcoxon["p_value"] < 0.05 and not ci_overlap,
    }
