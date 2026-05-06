import pytest

from reranker.strategies.hybrid import HybridFusionReranker


@pytest.fixture
def fitted_ranker():
    ranker = HybridFusionReranker()
    queries = ["python", "python", "java"]
    doc_as = ["python programming language", "python snake", "java coffee"]
    doc_bs = ["java programming language", "python code", "python code"]
    labels = [1, 1, 0]
    ranker.fit(queries, doc_as, doc_bs, labels)
    return ranker


class TestHybridBatch:
    def test_batch_matches_individual(self, fitted_ranker) -> None:
        queries = ["python", "java"]
        docs_list = [
            ["python programming language", "java programming language"],
            ["java coffee", "python code"],
        ]
        batch_results = fitted_ranker.rerank_batch(queries, docs_list)
        assert len(batch_results) == 2
        for q_idx, query in enumerate(queries):
            individual = fitted_ranker.rerank(query, docs_list[q_idx])
            batch = batch_results[q_idx]
            assert len(batch) == len(individual)
            for b, i in zip(batch, individual, strict=True):
                assert b.doc == i.doc

    def test_batch_empty_queries(self, fitted_ranker) -> None:
        result = fitted_ranker.rerank_batch([], [])
        assert result == []

    def test_batch_empty_docs_for_one_query(self, fitted_ranker) -> None:
        queries = ["python", "java"]
        docs_list = [
            ["python programming language"],
            [],
        ]
        results = fitted_ranker.rerank_batch(queries, docs_list)
        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[1] == []

    def test_batch_returns_ranked_docs(self, fitted_ranker) -> None:
        queries = ["python"]
        docs_list = [["python programming language", "java programming language"]]
        results = fitted_ranker.rerank_batch(queries, docs_list)
        assert len(results) == 1
        for ranked_doc in results[0]:
            assert ranked_doc.rank >= 1
            assert ranked_doc.score is not None

    def test_batch_mismatched_lengths_raises(self, fitted_ranker) -> None:
        with pytest.raises(ValueError, match="same length"):
            fitted_ranker.rerank_batch(["python", "java"], [["python programming language"]])
