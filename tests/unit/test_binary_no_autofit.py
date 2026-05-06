"""Regression test for C-2: BinaryQuantizedReranker must not auto-fit."""

import pytest

from reranker.strategies.binary_reranker import BinaryQuantizedReranker


def test_unfitted_rerank_raises_runtime_error():
    reranker = BinaryQuantizedReranker()
    with pytest.raises(RuntimeError, match="not fitted"):
        reranker.rerank("query", ["doc1", "doc2"])


def test_unfitted_score_raises_runtime_error():
    reranker = BinaryQuantizedReranker()
    with pytest.raises(RuntimeError, match="fitted"):
        reranker.score("query", ["doc1"])
