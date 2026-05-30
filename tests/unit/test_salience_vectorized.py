import numpy as np
import pytest

from reranker.strategies.late_interaction import StaticColBERTReranker


class TestSalienceVectorized:
    def test_salience_matches_expected(self) -> None:
        ranker = StaticColBERTReranker(use_salience=True)
        tokens = ["alpha", "beta", "alpha", "gamma", "beta", "alpha"]
        vectors = np.random.randn(6, 32).astype(np.float32)
        salience = ranker._compute_salience(tokens, vectors)
        assert salience.shape == (6,)
        assert salience.dtype == np.float32
        assert np.all(salience >= 0)

    def test_salience_higher_for_frequent_tokens(self) -> None:
        ranker = StaticColBERTReranker(use_salience=True)
        tokens = ["alpha", "beta", "alpha", "alpha", "beta"]
        vectors = np.random.randn(5, 16).astype(np.float32)
        salience = ranker._compute_salience(tokens, vectors)
        alpha_salience = salience[0]
        beta_salience = salience[1]
        assert alpha_salience > 0
        assert beta_salience > 0

    def test_salience_empty_vectors(self) -> None:
        ranker = StaticColBERTReranker(use_salience=True)
        vectors = np.zeros((0, 16), dtype=np.float32)
        salience = ranker._compute_salience([], vectors)
        assert salience.shape == (0,)

    def test_salience_single_token(self) -> None:
        ranker = StaticColBERTReranker(use_salience=True)
        tokens = ["word"]
        vectors = np.random.randn(1, 16).astype(np.float32)
        salience = ranker._compute_salience(tokens, vectors)
        assert salience.shape == (1,)
        assert salience[0] > 0

    def test_salience_handles_repeated_tokens(self) -> None:
        ranker = StaticColBERTReranker()
        tokens = ["hello", "world", "hello"]
        vectors = ranker.embedder.encode(tokens)
        salience = ranker._compute_salience(tokens, vectors)
        assert len(salience) == 3
        assert salience[0] == pytest.approx(salience[2], rel=0.001)
