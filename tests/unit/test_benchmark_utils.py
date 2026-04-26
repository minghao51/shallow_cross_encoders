"""Tests for benchmark utilities."""

from unittest.mock import MagicMock, patch

import pytest


def test_evaluate_reranker_on_rows_empty():
    """Test evaluate_reranker_on_rows raises ValueError for empty rows."""
    from reranker.eval.benchmark_utils import evaluate_reranker_on_rows

    with pytest.raises(ValueError, match="rows cannot be empty"):
        evaluate_reranker_on_rows([], MagicMock())


def test_evaluate_reranker_on_rows_none():
    """Test evaluate_reranker_on_rows raises ValueError for None reranker."""
    from reranker.eval.benchmark_utils import evaluate_reranker_on_rows

    rows = [{"query": "test", "doc": "doc", "score": 1}]
    with pytest.raises(ValueError, match="reranker cannot be None"):
        evaluate_reranker_on_rows(rows, None)


def test_evaluate_reranker_on_rows_basic():
    """Test evaluate_reranker_on_rows with basic data."""
    from reranker.eval.benchmark_utils import evaluate_reranker_on_rows

    # Mock reranker
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"doc": "relevant doc", "score": 0.9, "rank": 1},
        {"doc": "irrelevant doc", "score": 0.1, "rank": 2},
    ]

    rows = [
        {
            "query": "test query",
            "doc": "relevant doc",
            "score": 2,
        },
        {
            "query": "test query",
            "doc": "irrelevant doc",
            "score": 0,
        },
    ]

    results = evaluate_reranker_on_rows(rows, mock_reranker)

    assert "ndcg@10" in results
    assert "mrr" in results
    assert "p@1" in results
    assert "latency_p50_ms" in results
    assert "latency_p99_ms" in results
    assert "queries_evaluated" in results
    assert results["queries_evaluated"] == 1


def test_evaluate_reranker_on_rows_tracks_queries_without_relevant_docs():
    """Test query counting/latency are still reported when no docs are relevant."""
    from reranker.eval.benchmark_utils import evaluate_reranker_on_rows

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"doc": "doc one", "score": 0.2, "rank": 1},
        {"doc": "doc two", "score": 0.1, "rank": 2},
    ]

    rows = [
        {"query": "test query", "doc": "doc one", "score": 0},
        {"query": "test query", "doc": "doc two", "score": 0},
    ]

    results = evaluate_reranker_on_rows(rows, mock_reranker)

    assert results["queries_evaluated"] == 1
    assert results["ndcg@10"] == 0.0
    assert results["mrr"] == 0.0
    assert results["p@1"] == 0.0
    assert results["latency_p50_ms"] >= 0.0


def test_train_strategies_empty():
    """Test train_strategies raises ValueError for empty config."""
    from reranker.eval.benchmark_utils import train_strategies

    train_rows = [{"query": "test", "doc": "doc", "score": 1}]
    with pytest.raises(ValueError, match="strategies_config cannot be empty"):
        train_strategies(train_rows, {})


def test_train_strategies_empty_rows():
    """Test train_strategies raises ValueError for empty rows."""
    from reranker.eval.benchmark_utils import train_strategies

    with pytest.raises(ValueError, match="train_rows cannot be empty"):
        train_strategies([], {"hybrid": {}})


def test_train_strategies_basic():
    """Test train_strategies with basic config."""
    from unittest.mock import MagicMock

    train_rows = [
        {"query": "test query", "doc": "relevant doc", "score": 1},
        {"query": "test query", "doc": "irrelevant doc", "score": 0},
    ]

    strategies_config = {
        "hybrid": {"adapters": []},
    }

    # Mock the HybridFusionReranker to avoid actual training
    with patch("reranker.strategies.hybrid.HybridFusionReranker") as MockHybrid, patch(
        "reranker.strategies.hybrid.KeywordMatchAdapter"
    ):
        mock_instance = MagicMock()
        mock_instance.is_fitted = True
        MockHybrid.return_value = mock_instance

        # Import after patching
        from reranker.eval.benchmark_utils import train_strategies

        strategies = train_strategies(train_rows, strategies_config)

        assert "hybrid" in strategies
