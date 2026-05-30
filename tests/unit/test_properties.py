"""Property-based tests for reranker invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

from reranker.eval.metrics import ndcg_at_k
from reranker.lexical import BM25Engine
from reranker.quantization import dequantize, dequantize_4bit, quantize, quantize_4bit

pytestmark = pytest.mark.slow


class TestCosineSimilarityBounds:
    @given(
        st_np.arrays(
            np.float32,
            st.integers(min_value=1, max_value=16),
            elements=st.floats(-100.0, 100.0),
        ),
        st_np.arrays(
            np.float32,
            st.integers(min_value=1, max_value=16),
            elements=st.floats(-100.0, 100.0),
        ),
    )
    def test_cosine_in_bounds_nonzero(self, a: np.ndarray, b: np.ndarray) -> None:
        a = a.ravel()
        b = b.ravel()
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        assert -1.0 - 1e-5 <= sim <= 1.0 + 1e-5


class TestNDCGMonotonicity:
    @given(
        st.lists(st.floats(min_value=0.0, max_value=5.0), min_size=2, max_size=10),
        st.integers(min_value=1, max_value=10),
    )
    def test_ndcg_improves_with_better_first_relevance(
        self, relevances: list[float], k: int
    ) -> None:
        if k > len(relevances):
            return
        base = ndcg_at_k(relevances, k=k)
        improved = list(relevances)
        if improved:
            improved[0] = max(improved) + 1.0
            better = ndcg_at_k(improved, k=k)
            assert better >= base


class TestBM25Monotonicity:
    @given(
        st.text(min_size=1, max_size=50),
        st.lists(st.text(min_size=1, max_size=200), min_size=1, max_size=10),
    )
    @settings(max_examples=50)
    def test_bm25_partial_match(self, query: str, docs: list[str]) -> None:
        bm25 = BM25Engine()
        bm25.fit(docs)
        scores = bm25.score(query)
        assert scores.shape[0] == len(docs)
        assert np.all(scores >= 0.0)

    @given(
        st.text(min_size=3, max_size=30, alphabet="abcdefghijklmnopqrstuvwxyz "),
    )
    @settings(max_examples=50)
    def test_bm25_full_match_higher_than_partial(self, query: str) -> None:
        bm25 = BM25Engine()
        docs = [query, query[: len(query) // 2]]
        bm25.fit(docs)
        scores = bm25.score(query)
        assert scores[0] >= scores[1]


class TestQuantizationRoundtrip:
    @given(
        st_np.arrays(
            np.float32,
            st.integers(min_value=1, max_value=16),
            elements=st.floats(-10.0, 10.0),
        ),
    )
    @settings(max_examples=100)
    def test_quantize_dequantize_int8_roundtrip(self, vectors: np.ndarray) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        result = quantize(vectors, mode="int8")
        recovered = dequantize(result)
        max_error = np.max(np.abs(vectors - recovered))
        tolerance = 0.25 * (np.max(vectors) - np.min(vectors)) / 255.0 + 1e-4
        assert max_error <= tolerance, f"max_error={max_error}, tol={tolerance}"

    @given(
        st_np.arrays(
            np.float32,
            st.integers(min_value=1, max_value=16),
            elements=st.floats(-10.0, 10.0),
        ),
    )
    @settings(max_examples=100)
    def test_quantize_4bit_roundtrip(self, vectors: np.ndarray) -> None:
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] % 2 != 0:
            vectors = vectors[:, :-1]
        if vectors.shape[1] == 0:
            return
        result = quantize_4bit(vectors)
        recovered = dequantize_4bit(result)
        max_error = np.max(np.abs(vectors - recovered))
        tolerance = 0.25 * (np.max(vectors) - np.min(vectors)) / 15.0 + 1e-4
        assert max_error <= tolerance, f"max_error={max_error}, tol={tolerance}"
