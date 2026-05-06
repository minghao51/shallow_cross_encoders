import pytest

from reranker.strategies.distilled import DistilledPairwiseRanker


@pytest.fixture
def fitted_ranker():
    ranker = DistilledPairwiseRanker()
    queries = ["python", "python", "java"]
    doc_as = ["python programming", "python snake", "java coffee"]
    doc_bs = ["java programming", "python snake", "python code"]
    labels = [1, 1, 0]
    ranker.fit(queries, doc_as, doc_bs, labels)
    return ranker


class TestMergeRankDeque:
    def test_merge_rank_basic(self, fitted_ranker) -> None:
        docs = ["python programming", "java programming", "python snake"]
        result = fitted_ranker.rerank("python", docs)
        assert len(result) == 3
        assert all(r.doc in docs for r in result)
        ranks = [r.rank for r in result]
        assert ranks == sorted(ranks)

    def test_merge_rank_two_docs(self, fitted_ranker) -> None:
        docs = ["python programming", "java programming"]
        result = fitted_ranker.rerank("python", docs)
        assert len(result) == 2

    def test_merge_rank_single_doc(self, fitted_ranker) -> None:
        docs = ["python programming"]
        result = fitted_ranker.rerank("python", docs)
        assert len(result) == 1
        assert result[0].rank == 1

    def test_merge_rank_uses_deque(self) -> None:
        import inspect

        source = inspect.getsource(DistilledPairwiseRanker._merge_rank)
        assert "deque" in source
        assert ".pop(0)" not in source
        assert ".popleft()" in source

    def test_merge_rank_many_docs(self, fitted_ranker) -> None:
        original_max = fitted_ranker.full_tournament_max_docs
        fitted_ranker.full_tournament_max_docs = 5
        docs = [f"document number {i}" for i in range(20)]
        result = fitted_ranker.rerank("python", docs)
        assert len(result) == 20
        ranks = [r.rank for r in result]
        assert ranks == sorted(ranks)
        fitted_ranker.full_tournament_max_docs = original_max
