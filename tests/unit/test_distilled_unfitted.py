"""Regression test: DistilledPairwiseRanker raises on unfitted rerank/compare."""

import pytest

from reranker.strategies.distilled import DistilledPairwiseRanker


def test_unfitted_rerank_raises_runtime_error():
    ranker = DistilledPairwiseRanker()
    assert ranker.is_fitted is False
    with pytest.raises(RuntimeError, match="fitted"):
        ranker.rerank("query", ["doc1", "doc2"])


def test_unfitted_compare_raises_runtime_error():
    ranker = DistilledPairwiseRanker()
    assert ranker.is_fitted is False
    with pytest.raises(RuntimeError, match="fitted"):
        ranker.compare("query", "doc1", "doc2")
