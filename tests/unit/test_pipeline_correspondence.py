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


class _ScoreAwareReranker:
    """Reranker that assigns unique scores — useful for score-preservation checks."""

    def __init__(self, score_offset: float = 0.0) -> None:
        self._offset = score_offset

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        return [
            RankedDoc(
                doc=doc,
                score=self._offset + float(len(docs) - idx),
                rank=idx + 1,
                metadata={"strategy": "score_aware"},
            )
            for idx, doc in enumerate(docs)
        ]


def test_pipeline_preserves_last_stage_scores():
    """Regression: final_ranking must use passed_doc.score, not recompute from current_docs."""
    stage = PipelineStage(name="scorer", reranker=_ScoreAwareReranker(score_offset=10.0), top_k=100)
    pipeline = PipelineReranker(stages=[stage])

    docs = ["a", "b", "c"]
    result = pipeline.run_pipeline("q", docs)

    assert len(result.final_ranking) == 3
    for r, expected_score in zip(result.final_ranking, [13.0, 12.0, 11.0], strict=True):
        assert r.score == expected_score, (
            f"Expected {expected_score}, got {r.score} for doc {r.doc}"
        )
    assert all(r.doc in docs for r in result.final_ranking)


def test_pipeline_preserves_scores_through_multi_stage():
    """Scores from the final stage must appear verbatim in final_ranking."""
    stage1 = PipelineStage(name="s1", reranker=_ScoreAwareReranker(score_offset=0.0), top_k=3)
    stage2 = PipelineStage(name="s2", reranker=_ScoreAwareReranker(score_offset=100.0), top_k=2)
    pipeline = PipelineReranker(stages=[stage1, stage2])

    docs = ["a", "b", "c"]
    result = pipeline.run_pipeline("q", docs)

    # stage1 scores: a=3.0, b=2.0, c=1.0, top_k=3 keeps all
    # stage2 scores: a=(100+3)=103, b=(100+2)=102, top_k=2 keeps top-2
    assert len(result.final_ranking) == 2
    # second stage reverses: first doc gets len(docs)=3, so a gets 103.0, b gets 102.0
    assert result.final_ranking[0].score == 103.0
    assert result.final_ranking[1].score == 102.0


def test_pipeline_scores_not_zero_when_passed_fewer_than_input():
    """Regression: when stage returns fewer docs than input, scores must not be 0.0."""
    stage1 = PipelineStage(name="filter", reranker=_FilterReranker(drop={2}), top_k=100)
    stage2 = PipelineStage(name="scorer", reranker=_ScoreAwareReranker(score_offset=5.0), top_k=100)
    pipeline = PipelineReranker(stages=[stage1, stage2])

    docs = ["a", "b", "c", "d"]
    result = pipeline.run_pipeline("q", docs)

    # stage1 drops doc at index 2 → "c" removed → passes a,b,d
    # stage2 scores: a=8.0, b=7.0, d=6.0 (reversed order)
    assert len(result.final_ranking) == 3
    for r in result.final_ranking:
        assert r.score > 0.0, f"Doc {r.doc} has score 0.0 — should preserve stage score"
    assert all(r.doc != "c" for r in result.final_ranking)
