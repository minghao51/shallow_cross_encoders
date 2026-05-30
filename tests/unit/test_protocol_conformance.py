"""Protocol conformance tests — verify all strategies implement BaseReranker."""

from __future__ import annotations

import pytest

from reranker.persistence_mixin import SaveableReranker
from reranker.protocols import BaseReranker, NotFittedError
from reranker.strategies import (
    BinaryQuantizedReranker,
    DistilledPairwiseRanker,
    HybridFusionReranker,
    SPLADEReranker,
    StaticColBERTReranker,
)

STRATEGY_CLASSES = [
    HybridFusionReranker,
    DistilledPairwiseRanker,
    BinaryQuantizedReranker,
    StaticColBERTReranker,
    SPLADEReranker,
]


SCORELESS_STRATEGIES = {DistilledPairwiseRanker}


class TestProtocolConformance:
    @pytest.mark.parametrize("strategy_cls", STRATEGY_CLASSES)
    def test_strategy_is_saveable_reranker(self, strategy_cls) -> None:
        assert issubclass(strategy_cls, SaveableReranker)

    @pytest.mark.parametrize("strategy_cls", STRATEGY_CLASSES)
    def test_strategy_implements_base_reranker(self, strategy_cls) -> None:
        instance = strategy_cls()
        assert isinstance(instance, BaseReranker)

    @pytest.mark.parametrize("strategy_cls", STRATEGY_CLASSES)
    def test_strategy_has_is_fitted(self, strategy_cls) -> None:
        instance = strategy_cls()
        assert hasattr(instance, "is_fitted")
        assert isinstance(instance.is_fitted, bool)

    @pytest.mark.parametrize("strategy_cls", STRATEGY_CLASSES)
    def test_strategy_rerank_is_callable(self, strategy_cls) -> None:
        instance = strategy_cls()
        assert callable(instance.rerank)
        assert not instance.is_fitted
        with pytest.raises(NotFittedError):
            instance.rerank("test query", ["doc one", "doc two"])

    @pytest.mark.parametrize(
        "strategy_cls",
        [s for s in STRATEGY_CLASSES if s not in SCORELESS_STRATEGIES],
    )
    def test_strategy_score_is_callable(self, strategy_cls) -> None:
        instance = strategy_cls()
        assert callable(instance.score)

    @pytest.mark.parametrize("strategy_cls", STRATEGY_CLASSES)
    def test_strategy_has_artifact_type(self, strategy_cls) -> None:
        instance = strategy_cls()
        assert hasattr(instance, "_artifact_type")
        assert isinstance(instance._artifact_type, str)
        assert len(instance._artifact_type) > 0

    @pytest.mark.parametrize("strategy_cls", STRATEGY_CLASSES)
    def test_strategy_save_load_is_callable(self, strategy_cls) -> None:
        instance = strategy_cls()
        assert callable(instance.save)
        assert callable(instance.load)
