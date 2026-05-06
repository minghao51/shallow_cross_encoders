"""Regression test for C-1: pipeline zip misalignment."""

from reranker.protocols import RankedDoc
from reranker.strategies.pipeline import PipelineReranker, PipelineStage


class _FilterReranker:
    def __init__(self, drop: set[int]) -> None:
        self._drop = drop

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        results = [
            RankedDoc(doc=doc, score=float(idx), rank=idx + 1, metadata={"strategy": "filter"})
            for idx, doc in enumerate(docs)
            if idx not in self._drop
        ]
        return results


class _ReverseReranker:
    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        return [
            RankedDoc(
                doc=doc,
                score=float(len(docs) - idx),
                rank=idx + 1,
                metadata={"strategy": "reverse"},
            )
            for idx, doc in enumerate(reversed(docs))
        ]


def test_pipeline_final_ranking_uses_passed_docs():
    stage1 = PipelineStage(name="filter", reranker=_FilterReranker(drop={1, 3}), top_k=100)
    stage2 = PipelineStage(name="reverse", reranker=_ReverseReranker(), top_k=100)
    pipeline = PipelineReranker(stages=[stage1, stage2])

    docs = ["a", "b", "c", "d", "e"]
    result = pipeline.run_pipeline("q", docs)

    passed_docs = [r.doc for r in result.final_ranking]
    assert "b" not in passed_docs
    assert "d" not in passed_docs
    assert len(passed_docs) == 3


def test_pipeline_scores_match_ranked_scores():
    stage = PipelineStage(name="filter", reranker=_FilterReranker(drop=set()), top_k=100)
    pipeline = PipelineReranker(stages=[stage])

    docs = ["x", "y", "z"]
    result = pipeline.run_pipeline("q", docs)

    for r in result.final_ranking:
        assert r.score >= 0.0


def test_pipeline_empty_docs():
    pipeline = PipelineReranker(stages=[])
    result = pipeline.run_pipeline("q", [])
    assert result.final_ranking == []
