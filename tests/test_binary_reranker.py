import numpy as np
import pytest

from reranker.strategies.binary_reranker import BinaryQuantizedReranker


@pytest.mark.unit
class TestBinaryQuantizedReranker:
    def test_quantize_basic(self):
        vectors = np.array([[1.0, -0.5, 0.3, -0.1]], dtype=np.float32)
        bits = BinaryQuantizedReranker._quantize(vectors)
        expected = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(bits, expected)

    def test_hamming_distance(self):
        query_bits = np.array([1, 0, 1, 0], dtype=np.uint8)
        doc_bits = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0]],
            dtype=np.uint8,
        )
        dists = BinaryQuantizedReranker._hamming_distances(query_bits, doc_bits)
        assert dists[0] == 0
        assert dists[1] == 4
        assert dists[2] == 1

    def test_rerank_basic(self):
        docs = [
            "python dataclass field default factory",
            "javascript array spread operator clone",
        ]
        reranker = BinaryQuantizedReranker()
        reranker.fit(["python dataclass"], docs, [1, 0])
        result = reranker.rerank("python dataclass", docs)
        assert len(result) == 2
        assert result[0].metadata["strategy"] == "binary_reranker"

    def test_rerank_empty(self):
        reranker = BinaryQuantizedReranker()
        assert reranker.rerank("query", []) == []

    def test_score_without_fit_raises(self):
        reranker = BinaryQuantizedReranker()
        with pytest.raises(RuntimeError, match="must be fitted"):
            reranker.score("query", ["doc"])

    def test_fit_empty(self):
        reranker = BinaryQuantizedReranker()
        reranker.fit([], [], [])
        assert reranker.is_fitted
        assert reranker._doc_vectors.shape[0] == 0

    def test_save_load_roundtrip(self, tmp_path):
        docs = ["python dataclass mutable default", "javascript array spread"]
        reranker = BinaryQuantizedReranker()
        reranker.fit(["python dataclass"], docs, [1, 0])
        path = tmp_path / "binary_reranker.pkl"
        reranker.save(path)
        loaded = BinaryQuantizedReranker.load(path)
        assert loaded.is_fitted
        assert loaded.hamming_top_k == reranker.hamming_top_k
        assert loaded.bilinear_top_k == reranker.bilinear_top_k

    def test_bilinear_weights_shape(self):
        docs = [
            "python dataclass field default factory mutable list",
            "javascript array spread operator clone object",
            "python typing annotations type hints",
        ]
        queries = ["python dataclass", "javascript array", "python typing"]
        labels = [1, 0, 1]
        reranker = BinaryQuantizedReranker()
        reranker.fit(queries, docs, labels)
        assert reranker._bilinear_weights is not None
        assert reranker._bilinear_weights.shape == (reranker.embedder.dimension,)

    def test_two_stage_ranking(self):
        docs = [
            "python dataclass field default factory mutable list",
            "javascript array spread operator clone object",
            "python typing annotations type hints",
            "rust ownership borrow checker memory safety",
            "python dataclass vs namedtuple comparison",
        ]
        queries = ["python dataclass"] * len(docs)
        labels = [1, 0, 0, 0, 1]
        reranker = BinaryQuantizedReranker(hamming_top_k=3, bilinear_top_k=2)
        reranker.fit(queries, docs, labels)
        result = reranker.rerank("python dataclass", docs)
        assert len(result) == 5
        assert result[0].score >= result[-1].score

    def test_single_label_fallback(self):
        docs = ["python dataclass field", "javascript array spread"]
        reranker = BinaryQuantizedReranker()
        reranker.fit(["python"], docs, [1, 1])
        assert reranker._bilinear_model is not None
        result = reranker.rerank("python", docs)
        assert len(result) == 2
