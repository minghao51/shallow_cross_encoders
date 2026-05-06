"""Test lifecycle consistency: is_fitted and RuntimeError on unfitted rerank."""

import pytest

from reranker.protocols import RankedDoc
from reranker.strategies.binary_reranker import BinaryQuantizedReranker
from reranker.strategies.cascade import CascadeConfig, CascadeReranker
from reranker.strategies.late_interaction import StaticColBERTReranker
from reranker.strategies.splade import SPLADEReranker


class _MockReranker:
    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        return [RankedDoc(doc=d, score=1.0, rank=i + 1) for i, d in enumerate(docs)]


def test_cascade_is_fitted_delegates():
    cascade = CascadeReranker(_MockReranker(), _MockReranker(), CascadeConfig())
    assert cascade.is_fitted is True


def test_binary_reranker_starts_unfitted():
    reranker = BinaryQuantizedReranker()
    assert reranker.is_fitted is False


def test_static_colbert_starts_unfitted():
    reranker = StaticColBERTReranker()
    assert reranker.is_fitted is False


def test_splade_starts_unfitted():
    reranker = SPLADEReranker()
    assert reranker.is_fitted is False


def test_unfitted_static_colbert_raises():
    reranker = StaticColBERTReranker()
    with pytest.raises(RuntimeError, match="not fitted"):
        reranker.rerank("q", ["doc"])


def test_unfitted_splade_raises():
    reranker = SPLADEReranker()
    with pytest.raises(RuntimeError, match="not fitted"):
        reranker.rerank("q", ["doc"])
