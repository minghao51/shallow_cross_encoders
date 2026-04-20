"""Tests for CascadeReranker."""

from reranker.protocols import RankedDoc
from reranker.strategies import CascadeConfig, CascadeReranker, ConfidenceMetric


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self, scores: list[float], strategy_name: str = "mock"):
        self.scores = scores
        self.strategy_name = strategy_name

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        return [
            RankedDoc(doc=doc, score=score, rank=idx + 1, metadata={"strategy": self.strategy_name})
            for idx, (doc, score) in enumerate(zip(docs, self.scores, strict=False))
        ]


def test_cascade_fast_path():
    """Test that high confidence queries use primary only."""
    primary = MockReranker([0.8, 0.6, 0.4], "primary")
    fallback = MockReranker([0.9, 0.7, 0.5], "fallback")

    cascade = CascadeReranker(primary, fallback, config=CascadeConfig(confidence_threshold=0.6))

    results = cascade.rerank("test query", ["doc1", "doc2", "doc3"])

    # Should use primary (max score 0.8 >= threshold 0.6)
    assert results[0].score == 0.8
    assert results[0].metadata["fallback_used"] is False
    assert results[0].metadata["confidence"] == 0.8


def test_cascade_fallback():
    """Test that low confidence queries trigger fallback."""
    primary = MockReranker([0.4, 0.3, 0.2], "primary")
    fallback = MockReranker([0.9, 0.7, 0.5], "fallback")

    cascade = CascadeReranker(primary, fallback, config=CascadeConfig(confidence_threshold=0.6))

    results = cascade.rerank("test query", ["doc1", "doc2", "doc3"])

    # Should use fallback (max score 0.4 < threshold 0.6)
    assert results[0].score == 0.9
    assert results[0].metadata["fallback_used"] is True
    assert results[0].metadata["confidence"] == 0.4


def test_cascade_threshold_configurable():
    """Test confidence threshold is configurable."""
    primary = MockReranker([0.5, 0.4, 0.3], "primary")
    fallback = MockReranker([0.9, 0.7, 0.5], "fallback")

    # Low threshold - should use primary
    cascade_low = CascadeReranker(primary, fallback, config=CascadeConfig(confidence_threshold=0.4))
    results_low = cascade_low.rerank("test query", ["doc1", "doc2", "doc3"])
    assert results_low[0].metadata["fallback_used"] is False

    # High threshold - should use fallback
    cascade_high = CascadeReranker(
        primary, fallback, config=CascadeConfig(confidence_threshold=0.6)
    )
    results_high = cascade_high.rerank("test query", ["doc1", "doc2", "doc3"])
    assert results_high[0].metadata["fallback_used"] is True


def test_cascade_max_score_metric():
    """Test MAX_SCORE confidence metric."""
    primary = MockReranker([0.7, 0.5, 0.3], "primary")
    fallback = MockReranker([0.9, 0.7, 0.5], "fallback")

    cascade = CascadeReranker(
        primary,
        fallback,
        config=CascadeConfig(
            confidence_threshold=0.6, confidence_metric=ConfidenceMetric.MAX_SCORE
        ),
    )

    results = cascade.rerank("test query", ["doc1", "doc2", "doc3"])

    # Max score is 0.7, should use primary
    assert results[0].metadata["confidence"] == 0.7
    assert results[0].metadata["metric"] == "max_score"
    assert results[0].metadata["fallback_used"] is False


def test_cascade_top_margin_metric():
    """Test TOP_MARGIN confidence metric."""
    primary = MockReranker([0.7, 0.6, 0.3], "primary")
    fallback = MockReranker([0.9, 0.7, 0.5], "fallback")

    cascade = CascadeReranker(
        primary,
        fallback,
        config=CascadeConfig(
            confidence_threshold=0.1, confidence_metric=ConfidenceMetric.TOP_MARGIN
        ),
    )

    results = cascade.rerank("test query", ["doc1", "doc2", "doc3"])

    # Top margin is 0.7 - 0.6 = 0.1, at threshold boundary
    assert abs(results[0].metadata["confidence"] - 0.1) < 0.01
    assert results[0].metadata["metric"] == "top_margin"


def test_cascade_score_variance_metric():
    """Test SCORE_VARIANCE confidence metric."""
    primary = MockReranker([0.8, 0.5, 0.2], "primary")
    fallback = MockReranker([0.9, 0.7, 0.5], "fallback")

    cascade = CascadeReranker(
        primary,
        fallback,
        config=CascadeConfig(
            confidence_threshold=0.06, confidence_metric=ConfidenceMetric.SCORE_VARIANCE
        ),
    )

    results = cascade.rerank("test query", ["doc1", "doc2", "doc3"])

    # Variance of [0.8, 0.5, 0.2] is ~0.09
    assert abs(results[0].metadata["confidence"] - 0.09) < 0.01
    assert results[0].metadata["metric"] == "score_variance"
    assert results[0].metadata["fallback_used"] is False


def test_cascade_stats():
    """Test get_stats() returns correct counters."""
    primary = MockReranker([0.8, 0.6, 0.4], "primary")
    fallback = MockReranker([0.9, 0.7, 0.5], "fallback")

    cascade = CascadeReranker(primary, fallback, config=CascadeConfig(confidence_threshold=0.6))

    # Run 3 queries: 2 fast path, 1 fallback
    cascade.rerank("query1", ["doc1", "doc2", "doc3"])  # Fast (0.8 >= 0.6)
    cascade.rerank("query2", ["doc1", "doc2", "doc3"])  # Fast (0.8 >= 0.6)
    primary.scores = [0.4, 0.3, 0.2]  # Lower scores for next query
    cascade.rerank("query3", ["doc1", "doc2", "doc3"])  # Fallback (0.4 < 0.6)

    stats = cascade.get_stats()

    assert stats["total_queries"] == 3
    assert stats["fallback_count"] == 1
    assert stats["fallback_rate"] == 1 / 3
    assert abs(stats["avg_confidence"] - (0.8 + 0.8 + 0.4) / 3) < 0.01


def test_cascade_reset_stats():
    """Test reset_stats() clears counters."""
    primary = MockReranker([0.8, 0.6, 0.4], "primary")
    fallback = MockReranker([0.9, 0.7, 0.5], "fallback")

    cascade = CascadeReranker(primary, fallback, config=CascadeConfig(confidence_threshold=0.6))

    cascade.rerank("query1", ["doc1", "doc2", "doc3"])
    cascade.rerank("query2", ["doc1", "doc2", "doc3"])

    assert cascade.get_stats()["total_queries"] == 2

    cascade.reset_stats()

    assert cascade.get_stats()["total_queries"] == 0
    assert cascade.get_stats()["fallback_count"] == 0


def test_cascade_metadata():
    """Test metadata correctly records which path was taken."""
    primary = MockReranker([0.8, 0.6, 0.4], "primary")
    fallback = MockReranker([0.9, 0.7, 0.5], "fallback")

    cascade = CascadeReranker(primary, fallback, config=CascadeConfig(confidence_threshold=0.6))

    results = cascade.rerank("test query", ["doc1", "doc2", "doc3"])

    # Check all metadata fields
    assert "strategy" in results[0].metadata
    assert "fallback_used" in results[0].metadata
    assert "confidence" in results[0].metadata
    assert "metric" in results[0].metadata
    assert "threshold" in results[0].metadata

    assert results[0].metadata["strategy"] == "cascade"
    assert results[0].metadata["threshold"] == 0.6


def test_cascade_empty_docs():
    """Test behavior with empty document list."""
    primary = MockReranker([], "primary")
    fallback = MockReranker([], "fallback")

    cascade = CascadeReranker(primary, fallback, config=CascadeConfig(confidence_threshold=0.6))

    results = cascade.rerank("test query", [])

    assert results == []


def test_cascade_fallback_always():
    """Test fallback_strategy='always' always uses fallback."""
    primary = MockReranker([0.9, 0.8, 0.7], "primary")
    fallback = MockReranker([0.5, 0.4, 0.3], "fallback")

    cascade = CascadeReranker(
        primary,
        fallback,
        config=CascadeConfig(confidence_threshold=0.6, fallback_strategy="always"),
    )

    results = cascade.rerank("test query", ["doc1", "doc2", "doc3"])

    # Should use fallback despite high confidence
    assert results[0].score == 0.5  # From fallback
    assert results[0].metadata["fallback_used"] is True


def test_cascade_fallback_never():
    """Test fallback_strategy='never' never uses fallback."""
    primary = MockReranker([0.3, 0.2, 0.1], "primary")
    fallback = MockReranker([0.9, 0.8, 0.7], "fallback")

    cascade = CascadeReranker(
        primary,
        fallback,
        config=CascadeConfig(confidence_threshold=0.6, fallback_strategy="never"),
    )

    results = cascade.rerank("test query", ["doc1", "doc2", "doc3"])

    # Should use primary despite low confidence
    assert results[0].score == 0.3  # From primary
    assert results[0].metadata["fallback_used"] is False
