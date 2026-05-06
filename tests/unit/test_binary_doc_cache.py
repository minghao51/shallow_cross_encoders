import numpy as np

from reranker.strategies.binary_reranker import BinaryQuantizedReranker


class TestBinaryDocCache:
    def test_cache_populated_after_score(self) -> None:
        ranker = BinaryQuantizedReranker()
        queries = ["python"]
        docs = ["python programming", "java programming"]
        labels = [1, 0]
        ranker.fit(queries, docs, labels)
        ranker.score("python", docs)
        assert len(ranker._doc_encoding_cache) == 2
        for doc in docs:
            assert doc in ranker._doc_encoding_cache
            vec, bits = ranker._doc_encoding_cache[doc]
            assert vec.shape[0] == ranker.embedder.dimension
            assert bits.shape[0] == ranker.embedder.dimension

    def test_cache_hit_avoids_reencode(self) -> None:
        ranker = BinaryQuantizedReranker()
        queries = ["python"]
        docs = ["python programming", "java programming"]
        labels = [1, 0]
        ranker.fit(queries, docs, labels)
        score1 = ranker.score("python", docs)
        assert len(ranker._doc_encoding_cache) == 2
        score2 = ranker.score("python", docs)
        np.testing.assert_allclose(score1, score2, rtol=1e-6)

    def test_cache_invalidated_on_refit(self) -> None:
        ranker = BinaryQuantizedReranker()
        ranker.fit(["python"], ["python programming"], [1])
        ranker.score("python", ["python programming"])
        assert len(ranker._doc_encoding_cache) > 0
        ranker.fit(["java"], ["java programming"], [1])
        assert (
            len(ranker._doc_encoding_cache) == 0
            or "python programming" not in ranker._doc_encoding_cache
        )

    def test_cache_with_new_docs(self) -> None:
        ranker = BinaryQuantizedReranker()
        ranker.fit(["python"], ["python programming", "java programming"], [1, 0])
        ranker.score("python", ["python programming"])
        score = ranker.score("python", ["python programming", "new document"])
        assert score.shape[0] == 2
        assert "new document" in ranker._doc_encoding_cache

    def test_cache_lru_eviction(self) -> None:
        ranker = BinaryQuantizedReranker()
        ranker._doc_encoding_cache_max_size = 3
        ranker.fit(["python"], ["doc1", "doc2"], [1, 0])
        ranker.score("python", ["doc1", "doc2"])
        # Access doc1 to make it recently used
        ranker.score("python", ["doc1"])
        # Add doc3 and doc4, should evict doc2 (least recently used)
        ranker.score("python", ["doc3", "doc4"])
        assert len(ranker._doc_encoding_cache) == 3
        assert "doc1" in ranker._doc_encoding_cache
        assert "doc2" not in ranker._doc_encoding_cache
        assert "doc3" in ranker._doc_encoding_cache
        assert "doc4" in ranker._doc_encoding_cache
