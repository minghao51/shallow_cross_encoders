from __future__ import annotations

import pytest

from reranker.protocols import RankedDoc
from reranker.strategies.multi import MultiReranker, MultiRerankerConfig


class _StubReranker:
    def __init__(self, ranked: list[tuple[str, float]]) -> None:
        self._ranked = ranked

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        del query
        allowed = set(docs)
        return [
            RankedDoc(doc=doc, score=score, rank=rank)
            for rank, (doc, score) in enumerate(self._ranked, start=1)
            if doc in allowed
        ]


def test_multi_single_reranker_preserves_component_order() -> None:
    docs = ["a", "b", "c"]
    ranker = _StubReranker([("c", 0.9), ("a", 0.7), ("b", 0.1)])
    multi = MultiReranker([("stub", ranker)])

    ranked = multi.rerank("query", docs)

    assert [r.doc for r in ranked] == ["c", "a", "b"]
    assert ranked[0].metadata["strategy"] == "multi_stub"


def test_multi_rrf_aligns_scores_to_original_doc_ids() -> None:
    docs = ["a", "b", "c"]
    ranker_a = _StubReranker([("c", 3.0), ("b", 2.0), ("a", 1.0)])
    ranker_b = _StubReranker([("c", 30.0), ("b", 20.0), ("a", 10.0)])
    multi = MultiReranker([("ra", ranker_a), ("rb", ranker_b)], MultiRerankerConfig(rrf_k=60))

    ranked = multi.rerank("query", docs)

    assert [r.doc for r in ranked] == ["c", "b", "a"]
    assert ranked[0].metadata["strategy"] == "multi_rrf"


def test_multi_raises_for_weights_length_mismatch() -> None:
    with pytest.raises(ValueError, match="weights length must match number of rerankers"):
        MultiReranker(
            [("a", _StubReranker([("a", 1.0)])), ("b", _StubReranker([("a", 1.0)]))],
            MultiRerankerConfig(weights=[1.0]),
        )
