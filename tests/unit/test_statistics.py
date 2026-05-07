"""Tests for statistical significance utilities."""

from __future__ import annotations

import numpy as np
import pytest

from reranker.eval.statistics import (
    _normal_cdf,
    bootstrap_ci,
    compare_strategies,
    wilcoxon_signed_rank,
)


class TestBootstrapCI:
    def test_returns_lower_upper_tuple(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        lo, hi = bootstrap_ci(scores)
        assert lo <= hi
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0

    def test_single_value_returns_identical(self):
        lo, hi = bootstrap_ci([0.75])
        assert lo == 0.75
        assert hi == 0.75

    def test_empty_returns_zero(self):
        lo, hi = bootstrap_ci([])
        assert lo == 0.0
        assert hi == 0.0

    def test_deterministic_seed(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        lo1, hi1 = bootstrap_ci(scores, random_seed=42)
        lo2, hi2 = bootstrap_ci(scores, random_seed=42)
        assert lo1 == lo2
        assert hi1 == hi2

    def test_different_seed_produces_different_result(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        lo1, hi1 = bootstrap_ci(scores, random_seed=42)
        lo2, hi2 = bootstrap_ci(scores, random_seed=99)
        assert (lo1, hi1) != (lo2, hi2)

    def test_ci_contains_mean(self):
        scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75, 0.85, 0.95]
        lo, hi = bootstrap_ci(scores)
        mean = float(np.mean(scores))
        assert lo <= mean <= hi


class TestWilcoxonSignedRank:
    def test_identical_scores(self):
        a = [0.5, 0.6, 0.7]
        result = wilcoxon_signed_rank(a, a)
        assert result["p_value"] == 1.0

    def test_different_scores(self):
        a = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        b = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = wilcoxon_signed_rank(a, b)
        assert "W" in result
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            wilcoxon_signed_rank([0.5, 0.6], [0.7])

    def test_small_sample_returns_p50(self):
        result = wilcoxon_signed_rank([0.9, 0.8], [0.1, 0.2])
        assert result["p_value"] == 0.5

    def test_all_equal_returns_high_p(self):
        a = [0.5, 0.5, 0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = wilcoxon_signed_rank(a, b)
        assert result["p_value"] == 1.0

    def test_clearly_different_returns_low_p(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0.5, 0.1, 30).tolist()
        b = rng.normal(0.3, 0.1, 30).tolist()
        result = wilcoxon_signed_rank(a, b)
        assert result["p_value"] < 0.05


class TestNormalCDF:
    def test_zero(self):
        assert _normal_cdf(0.0) == pytest.approx(0.5, abs=1e-4)

    def test_one(self):
        assert _normal_cdf(1.0) == pytest.approx(0.8413, abs=1e-2)

    def test_negative(self):
        assert _normal_cdf(-1.0) == pytest.approx(1.0 - _normal_cdf(1.0), abs=1e-4)

    def test_two(self):
        assert _normal_cdf(2.0) == pytest.approx(0.9772, abs=1e-2)


class TestCompareStrategies:
    def test_returns_all_keys(self):
        a = [0.5, 0.6, 0.7, 0.8, 0.9]
        b = [0.4, 0.5, 0.6, 0.7, 0.8]
        result = compare_strategies("A", a, "B", b)
        assert result["metric"] == "NDCG@10"
        assert "A_mean" in result
        assert "B_mean" in result
        assert "A_ci_95" in result
        assert "B_ci_95" in result
        assert "delta" in result
        assert "ci_overlap" in result
        assert "wilcoxon_p_value" in result
        assert "significant_at_005" in result

    def test_a_better_than_b(self):
        a = [0.9, 0.8, 0.7, 0.8, 0.9]
        b = [0.4, 0.5, 0.3, 0.4, 0.5]
        result = compare_strategies("A", a, "B", b)
        assert result["delta"] > 0
        assert result["A_mean"] > result["B_mean"]

    def test_better_metric_name(self):
        a = [0.5, 0.6, 0.7]
        b = [0.4, 0.5, 0.6]
        result = compare_strategies("X", a, "Y", b, metric_name="MAP@10")
        assert result["metric"] == "MAP@10"
