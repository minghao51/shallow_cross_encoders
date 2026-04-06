import numpy as np
import pytest

from reranker.strategies.late_interaction import StaticColBERTReranker


@pytest.mark.unit
class TestStaticColBERTReranker:
    def test_rerank_basic(self):
        docs = [
            "python dataclass field default factory",
            "javascript array spread operator clone",
            "python dataclass mutable default list",
        ]
        reranker = StaticColBERTReranker()
        reranker.fit(docs)
        result = reranker.rerank("python dataclass default", docs)
        assert len(result) == 3
        assert result[0].rank == 1
        assert result[0].metadata["strategy"] == "late_interaction"

    def test_rerank_empty(self):
        reranker = StaticColBERTReranker()
        assert reranker.rerank("query", []) == []

    def test_maxsim_identical_tokens(self):
        query_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        doc_vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        score = StaticColBERTReranker._maxsim(query_vecs, doc_vecs)
        assert score == pytest.approx(2.0, abs=1e-5)

    def test_maxsim_empty(self):
        empty = np.zeros((0, 2), dtype=np.float32)
        vecs = np.array([[1.0, 0.0]], dtype=np.float32)
        assert StaticColBERTReranker._maxsim(empty, vecs) == 0.0
        assert StaticColBERTReranker._maxsim(vecs, empty) == 0.0

    def test_token_pruning(self):
        reranker = StaticColBERTReranker(top_k_tokens=3)
        tokens = ["a", "b", "c", "d", "e", "f"]
        vectors = np.random.randn(6, 8).astype(np.float32)
        index = reranker._prune_tokens(tokens, vectors)
        assert len(index.tokens) == 3
        assert index.vectors.shape == (3, 8)

    def test_salience_computation(self):
        reranker = StaticColBERTReranker()
        tokens = ["the", "cat", "the", "dog", "the"]
        vectors = np.random.randn(5, 8).astype(np.float32)
        salience = reranker._compute_salience(tokens, vectors)
        assert salience.shape == (5,)
        assert salience[0] == salience[2] == salience[4]

    def test_save_load_roundtrip(self, tmp_path):
        docs = ["python dataclass mutable default", "javascript array spread"]
        reranker = StaticColBERTReranker()
        reranker.fit(docs)
        path = tmp_path / "late_interaction.pkl"
        reranker.save(path)
        loaded = StaticColBERTReranker.load(path)
        assert loaded.is_fitted
        assert len(loaded._index) == 2

    def test_score_without_fit_raises(self):
        reranker = StaticColBERTReranker()
        with pytest.raises(RuntimeError, match="must be fitted"):
            reranker.score("query", ["doc"])

    def test_rerank_auto_fit(self):
        docs = ["python dataclass field", "javascript array"]
        reranker = StaticColBERTReranker()
        result = reranker.rerank("python", docs)
        assert len(result) == 2
        assert reranker.is_fitted

    def test_query_coverage_effect(self):
        relevant = "python dataclass field default factory mutable list"
        irrelevant = "weather patterns monsoon climate change"
        docs = [relevant, irrelevant]
        reranker = StaticColBERTReranker()
        reranker.fit(docs)
        result = reranker.rerank("python dataclass", docs)
        assert result[0].doc == relevant
        assert result[0].score > result[1].score
