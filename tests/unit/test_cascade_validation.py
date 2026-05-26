"""Test CascadeReranker fallback_strategy validation and WeightingMode enum."""

import pytest

from reranker.strategies.cascade import CascadeConfig, CascadeReranker, FallbackStrategy
from reranker.strategies.hybrid import WeightingMode
from reranker.types import RankedDoc


class _MockReranker:
    def __init__(self, fitted: bool = True) -> None:
        self.is_fitted = fitted

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        return [RankedDoc(doc=d, score=1.0, rank=i + 1) for i, d in enumerate(docs)]


class _MockNoFitted:
    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        return [RankedDoc(doc=d, score=1.0, rank=i + 1) for i, d in enumerate(docs)]


def test_fallback_strategy_valid_values():
    for val in ("flashrank", "always", "never"):
        config = CascadeConfig(fallback_strategy=val)
        assert config.fallback_strategy == val


def test_fallback_strategy_enum_values():
    assert FallbackStrategy.FLASHRANK == "flashrank"
    assert FallbackStrategy.ALWAYS == "always"
    assert FallbackStrategy.NEVER == "never"


def test_fallback_strategy_invalid_raises():
    with pytest.raises(ValueError, match="Invalid fallback_strategy"):
        CascadeConfig(fallback_strategy="invalid")


def test_weighting_mode_enum_values():
    assert WeightingMode.STATIC == "static"
    assert WeightingMode.LEARNED == "learned"
    assert WeightingMode.META_ROUTER == "meta_router"


def test_weighting_mode_invalid_raises():
    with pytest.raises(ValueError):
        WeightingMode("invalid")


def test_cascade_is_fitted_property():
    cascade = CascadeReranker(_MockReranker(True), _MockReranker(True))
    assert cascade.is_fitted is True


def test_cascade_is_fitted_false_when_primary_unfitted():
    cascade = CascadeReranker(_MockReranker(False), _MockReranker(True))
    assert cascade.is_fitted is False


def test_cascade_is_fitted_true_when_no_is_fitted_attr():
    cascade = CascadeReranker(_MockNoFitted(), _MockNoFitted())
    assert cascade.is_fitted is True
