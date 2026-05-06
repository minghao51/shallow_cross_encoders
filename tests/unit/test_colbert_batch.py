import numpy as np
import pytest

from reranker.strategies.late_interaction import StaticColBERTReranker


@pytest.fixture
def fitted_ranker():
    ranker = StaticColBERTReranker(use_salience=False, quantization_mode="none")
    docs = [
        "python programming language",
        "java programming language",
        "machine learning algorithms",
    ]
    ranker.fit(docs)
    return ranker


class TestColBERTBatch:
    def test_batch_matches_individual(self, fitted_ranker) -> None:
        queries = ["python", "java"]
        docs = [
            "python programming language",
            "java programming language",
            "machine learning algorithms",
        ]
        batch_results = fitted_ranker.rerank_batch(queries, docs)
        assert len(batch_results) == 2
        for q_idx, query in enumerate(queries):
            individual = fitted_ranker.rerank(query, docs)
            batch = batch_results[q_idx]
            assert len(batch) == len(individual)
            for b, i in zip(batch, individual, strict=True):
                assert b.doc == i.doc
                assert abs(b.score - i.score) < 1e-5

    def test_batch_empty_queries(self, fitted_ranker) -> None:
        result = fitted_ranker.rerank_batch([], ["doc1"])
        assert result == []

    def test_batch_returns_correct_count(self, fitted_ranker) -> None:
        queries = ["python", "java", "machine learning"]
        docs = ["python programming language", "java programming language"]
        results = fitted_ranker.rerank_batch(queries, docs)
        assert len(results) == 3
        for q_result in results:
            assert len(q_result) == 2

    def test_batch_unfitted_raises(self) -> None:
        ranker = StaticColBERTReranker(use_salience=False, quantization_mode="none")
        with pytest.raises(RuntimeError, match="not fitted"):
            ranker.rerank_batch(["python"], ["python programming language"])


class TestColBERTTokenIndexReuse:
    def test_prebuilt_indices_match(self, fitted_ranker) -> None:
        query = "python"
        docs = ["python programming language", "java programming language"]
        prebuilt = list(fitted_ranker._index)
        normal_scores = fitted_ranker.score(query, docs)
        prebuilt_scores = fitted_ranker.score(query, docs, prebuilt_indices=prebuilt)
        np.testing.assert_allclose(normal_scores, prebuilt_scores, rtol=1e-6)

    def test_prebuilt_indices_rerank(self, fitted_ranker) -> None:
        query = "python"
        docs = ["python programming language", "java programming language"]
        prebuilt = list(fitted_ranker._index)
        normal = fitted_ranker.rerank(query, docs)
        prebuilt_result = fitted_ranker.rerank(query, docs, prebuilt_indices=prebuilt)
        assert len(normal) == len(prebuilt_result)
        for n, p in zip(normal, prebuilt_result, strict=True):
            assert n.doc == p.doc
            assert abs(n.score - p.score) < 1e-5
