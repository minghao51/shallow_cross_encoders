"""Unit tests for eval/metrics.py module."""

from __future__ import annotations

import time

import numpy as np

from reranker.eval.metrics import (
    LatencyTracker,
    accuracy,
    dcg_at_k,
    map,
    mrr,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


class TestDCG:
    """Tests for dcg_at_k function."""

    def test_dcg_with_perfect_ranking(self) -> None:
        """DCG with perfect ranking (highest scores first)."""
        relevances = [3.0, 2.0, 1.0, 0.0]
        result = dcg_at_k(relevances, k=4)
        # DCG should be positive for good ranking
        assert result > 0

    def test_dcg_with_zero_relevances(self) -> None:
        """DCG with all zero relevances should be zero."""
        relevances = [0.0, 0.0, 0.0]
        result = dcg_at_k(relevances, k=3)
        assert result == 0.0

    def test_dcg_truncates_at_k(self) -> None:
        """DCG should only consider first k items."""
        relevances = [3.0, 2.0, 1.0, 0.0]
        result_k2 = dcg_at_k(relevances, k=2)
        result_k4 = dcg_at_k(relevances, k=4)
        # DCG with k=2 should be different from k=4
        assert result_k2 != result_k4
        # DCG with k=2 should be higher since we only consider top 2
        assert result_k2 > 0

    def test_dcg_with_empty_list(self) -> None:
        """DCG with empty list should be zero."""
        result = dcg_at_k([], k=5)
        assert result == 0.0

    def test_dcg_with_k_larger_than_list(self) -> None:
        """DCG should handle k larger than list length."""
        relevances = [1.0, 2.0]
        result = dcg_at_k(relevances, k=10)
        assert result > 0

    def test_dcg_formula_components(self) -> None:
        """Verify DCG uses the correct formula: (2^rel - 1) / log2(i + 2)."""
        relevances = [3.0, 2.0]
        result = dcg_at_k(relevances, k=2)
        # First item: (2^3 - 1) / log2(2) = 7 / 1 = 7
        # Second item: (2^2 - 1) / log2(3) = 3 / 1.585 = 1.893
        expected = 7.0 + (3.0 / np.log2(3))
        assert abs(result - expected) < 0.01


class TestNDCG:
    """Tests for ndcg_at_k function."""

    def test_ndcg_perfect_ranking(self) -> None:
        """NDCG should be 1.0 for perfect ranking."""
        relevances = [3.0, 2.0, 1.0, 0.0]
        result = ndcg_at_k(relevances, k=4)
        assert result == 1.0

    def test_ndcg_worst_ranking(self) -> None:
        """NDCG should be less than 1.0 for imperfect ranking."""
        relevances = [0.0, 1.0, 2.0, 3.0]
        result = ndcg_at_k(relevances, k=4)
        assert result < 1.0
        assert result > 0

    def test_ndcg_with_zero_relevances(self) -> None:
        """NDCG with all zero relevances should be 0 (or 1 by convention)."""
        relevances = [0.0, 0.0, 0.0]
        result = ndcg_at_k(relevances, k=3)
        # When all relevances are zero, DCG is 0, so NDCG is 0
        assert result == 0.0

    def test_ndcg_with_empty_list(self) -> None:
        """NDCG with empty list should be 0."""
        result = ndcg_at_k([], k=5)
        assert result == 0.0

    def test_ndcg_normalizes_by_ideal(self) -> None:
        """NDCG should normalize by ideal DCG."""
        relevances = [2.0, 1.0, 3.0]
        result = ndcg_at_k(relevances, k=3)
        # Should be between 0 and 1
        assert 0 <= result <= 1.0

    def test_ndcg_different_k_values(self) -> None:
        """NDCG should vary with different k values."""
        relevances = [1.0, 3.0, 2.0, 0.0]
        result_k2 = ndcg_at_k(relevances, k=2)
        result_k4 = ndcg_at_k(relevances, k=4)
        # Results should differ
        assert result_k2 != result_k4


class TestReciprocalRank:
    """Tests for reciprocal_rank function."""

    def test_reciprocal_rank_first_position(self) -> None:
        """RR should be 1.0 when relevant item is first."""
        relevances = [1, 0, 0, 0]
        result = reciprocal_rank(relevances)
        assert result == 1.0

    def test_reciprocal_rank_second_position(self) -> None:
        """RR should be 0.5 when relevant item is second."""
        relevances = [0, 1, 0, 0]
        result = reciprocal_rank(relevances)
        assert result == 0.5

    def test_reciprocal_rank_third_position(self) -> None:
        """RR should be 1/3 when relevant item is third."""
        relevances = [0, 0, 1, 0]
        result = reciprocal_rank(relevances)
        assert abs(result - 1.0 / 3.0) < 0.001

    def test_reciprocal_rank_no_relevant(self) -> None:
        """RR should be 0.0 when no relevant items."""
        relevances = [0, 0, 0, 0]
        result = reciprocal_rank(relevances)
        assert result == 0.0

    def test_reciprocal_rank_multiple_relevant(self) -> None:
        """RR should return position of first relevant item."""
        relevances = [0, 1, 1, 1]
        result = reciprocal_rank(relevances)
        # First relevant is at position 2
        assert result == 0.5

    def test_reciprocal_rank_empty_list(self) -> None:
        """RR should be 0.0 for empty list."""
        result = reciprocal_rank([])
        assert result == 0.0


class TestPrecisionAtK:
    """Tests for precision_at_k function."""

    def test_precision_at_k_all_relevant(self) -> None:
        """P@k should be 1.0 when all top-k are relevant."""
        relevances = [1, 1, 1, 0]
        result = precision_at_k(relevances, k=3)
        assert result == 1.0

    def test_precision_at_k_half_relevant(self) -> None:
        """P@k should be 0.5 when half of top-k are relevant."""
        relevances = [1, 0, 1, 0]
        result = precision_at_k(relevances, k=4)
        assert result == 0.5

    def test_precision_at_k_no_relevant(self) -> None:
        """P@k should be 0.0 when none are relevant."""
        relevances = [0, 0, 0, 0]
        result = precision_at_k(relevances, k=3)
        assert result == 0.0

    def test_precision_at_k_with_k_zero(self) -> None:
        """P@k should be 0.0 when k is 0."""
        relevances = [1, 1, 1]
        result = precision_at_k(relevances, k=0)
        assert result == 0.0

    def test_precision_at_k_truncates_at_k(self) -> None:
        """P@k should only consider first k items."""
        relevances = [1, 0, 1, 1]
        result_k2 = precision_at_k(relevances, k=2)
        result_k4 = precision_at_k(relevances, k=4)
        assert result_k2 == 0.5  # 1 out of 2
        assert result_k4 == 0.75  # 3 out of 4

    def test_precision_at_k_empty_list(self) -> None:
        """P@k should be 0.0 for empty list."""
        result = precision_at_k([], k=3)
        assert result == 0.0

    def test_precision_at_k_larger_than_list(self) -> None:
        """P@k should handle k larger than list length."""
        relevances = [1, 0]
        result = precision_at_k(relevances, k=5)
        assert result == 0.5  # 1 out of 2


class TestAccuracy:
    """Tests for accuracy function."""

    def test_accuracy_perfect(self) -> None:
        """Accuracy should be 1.0 for perfect predictions."""
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 0]
        result = accuracy(y_true, y_pred)
        assert result == 1.0

    def test_accuracy_half_correct(self) -> None:
        """Accuracy should be 0.5 for half correct predictions."""
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 0, 0]
        result = accuracy(y_true, y_pred)
        assert result == 0.5

    def test_accuracy_all_wrong(self) -> None:
        """Accuracy should be 0.0 for all wrong predictions."""
        y_true = [1, 0, 1, 0]
        y_pred = [0, 1, 0, 1]
        result = accuracy(y_true, y_pred)
        assert result == 0.0


class TestMRR:
    def test_mrr_averages_first_relevant_rank_per_query(self) -> None:
        result = mrr([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        expected = (0.5 + (1.0 / 3.0) + 0.0) / 3.0
        assert abs(result - expected) < 1e-6

    def test_mrr_respects_k(self) -> None:
        assert mrr([[0, 0, 1]], k=2) == 0.0


class TestMAP:
    def test_map_averages_query_average_precision(self) -> None:
        result = map([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
        expected = ([(1.0 + (2.0 / 3.0)) / 2.0, ((1.0 / 2.0) + (2.0 / 3.0)) / 2.0, 0.0])
        assert abs(result - (sum(expected) / 3.0)) < 1e-6

    def test_map_respects_k(self) -> None:
        assert map([[1, 0, 1]], k=2) == 1.0

    def test_accuracy_empty_lists(self) -> None:
        """Accuracy should be 0.0 for empty lists."""
        result = accuracy([], [])
        assert result == 0.0

    def test_accuracy_mismatched_lengths(self) -> None:
        """Accuracy should handle mismatched lengths (uses strict=False)."""
        y_true = [1, 0, 1]
        y_pred = [1, 0]
        result = accuracy(y_true, y_pred)
        # Should only compare up to shorter length
        assert result == 1.0  # First two match

    def test_accuracy_single_element(self) -> None:
        """Accuracy should work with single element."""
        assert accuracy([1], [1]) == 1.0
        assert accuracy([1], [0]) == 0.0


class TestLatencyTracker:
    """Tests for LatencyTracker class."""

    def test_latency_tracker_initial_state(self) -> None:
        """LatencyTracker should start with empty samples."""
        tracker = LatencyTracker()
        assert len(tracker.samples_ms) == 0
        summary = tracker.summary()
        assert summary["p50"] == 0.0
        assert summary["p99"] == 0.0
        assert summary["mean"] == 0.0

    def test_latency_tracker_measure_single(self) -> None:
        """LatencyTracker should record a single measurement."""
        tracker = LatencyTracker()
        with tracker.measure():
            time.sleep(0.001)  # 1ms

        assert len(tracker.samples_ms) == 1
        assert tracker.samples_ms[0] >= 1.0  # At least 1ms

    def test_latency_tracker_measure_multiple(self) -> None:
        """LatencyTracker should record multiple measurements."""
        tracker = LatencyTracker()
        for _ in range(3):
            with tracker.measure():
                time.sleep(0.001)

        assert len(tracker.samples_ms) == 3

    def test_latency_tracker_summary_percentiles(self) -> None:
        """LatencyTracker summary should compute correct percentiles."""
        tracker = LatencyTracker()
        # Add known samples
        tracker.samples_ms = [10.0, 20.0, 30.0, 40.0, 50.0]

        summary = tracker.summary()
        assert summary["p50"] == 30.0  # Median
        assert summary["p99"] == 50.0  # 99th percentile (max in this case)
        assert summary["mean"] == 30.0  # Average

    def test_latency_tracker_summary_mean(self) -> None:
        """LatencyTracker summary should compute correct mean."""
        tracker = LatencyTracker()
        tracker.samples_ms = [10.0, 20.0, 30.0]

        summary = tracker.summary()
        assert summary["mean"] == 20.0  # (10 + 20 + 30) / 3

    def test_latency_tracker_context_manager_exception(self) -> None:
        """LatencyTracker should record time even if exception occurs."""
        tracker = LatencyTracker()

        try:
            with tracker.measure():
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert len(tracker.samples_ms) == 1
        assert tracker.samples_ms[0] >= 0

    def test_latency_tracker_measure_with_variable_time(self) -> None:
        """LatencyTracker should record different durations."""
        tracker = LatencyTracker()

        with tracker.measure():
            time.sleep(0.001)

        with tracker.measure():
            time.sleep(0.002)

        assert len(tracker.samples_ms) == 2
        assert tracker.samples_ms[0] >= 1.0
        assert tracker.samples_ms[1] >= 2.0

    def test_latency_tracker_p99_calculation(self) -> None:
        """Test p99 percentile calculation with more samples."""
        tracker = LatencyTracker()
        # Create 100 samples from 1 to 100
        tracker.samples_ms = list(range(1, 101))

        summary = tracker.summary()
        # p99 should be close to 99
        assert 98 <= summary["p99"] <= 100

    def test_latency_tracker_p50_calculation(self) -> None:
        """Test p50 (median) calculation with odd number of samples."""
        tracker = LatencyTracker()
        tracker.samples_ms = [10, 20, 30, 40, 50]

        summary = tracker.summary()
        assert summary["p50"] == 30.0  # True median
