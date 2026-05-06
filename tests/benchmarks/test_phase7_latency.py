"""Phase 7 latency regression benchmarks.

Verifies that batch operations, shared caching, and algorithmic
optimizations deliver measurable speedups on standard workloads.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from reranker.embedder import Embedder
from reranker.embedding_cache import get_shared_cache, reset_shared_cache
from reranker.strategies.hybrid import HybridFusionReranker
from reranker.strategies.late_interaction import StaticColBERTReranker


class TestBatchSpeedup:
    """Verify rerank_batch is faster than individual rerank calls."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        reset_shared_cache()
        yield
        reset_shared_cache()

    def test_hybrid_batch_faster_than_individual(self) -> None:
        ranker = HybridFusionReranker()
        queries = ["python"] * 3
        doc_as = ["python programming", "python snake", "java coffee"]
        doc_bs = ["java programming", "python code", "python code"]
        labels = [1, 1, 0]
        ranker.fit(queries, doc_as, doc_bs, labels)

        queries = [f"query number {i}" for i in range(5)]
        docs_list = [
            [f"doc {j} about programming languages and software engineering" for j in range(5)]
            for _ in range(5)
        ]

        # Disable caches to isolate pure batching benefit
        original_cache = ranker.embedder._cache
        original_encode_cache = ranker.embedder._encode_cache
        ranker.embedder._cache = None
        ranker.embedder._encode_cache = None

        try:
            # Time batch
            start = time.perf_counter()
            for _ in range(20):
                ranker.rerank_batch(queries, docs_list)
            batch_time = time.perf_counter() - start

            # Time individual
            start = time.perf_counter()
            for _ in range(20):
                for q, d in zip(queries, docs_list, strict=False):
                    ranker.rerank(q, d)
            individual_time = time.perf_counter() - start

            assert batch_time < individual_time, (
                f"rerank_batch ({batch_time:.4f}s) should be faster than "
                f"individual calls ({individual_time:.4f}s)"
            )
        finally:
            ranker.embedder._cache = original_cache
            ranker.embedder._encode_cache = original_encode_cache

    def test_colbert_batch_faster_than_individual(self) -> None:
        ranker = StaticColBERTReranker(use_salience=False, quantization_mode="none")
        docs = [
            "python programming language",
            "java programming language",
            "machine learning algorithms",
            "deep neural networks",
            "natural language processing",
        ]
        ranker.fit(docs)

        queries = ["python", "java", "machine learning"]

        # Warm up
        ranker.rerank_batch(queries, docs)
        for q in queries:
            ranker.rerank(q, docs)

        # Time batch
        start = time.perf_counter()
        for _ in range(10):
            ranker.rerank_batch(queries, docs)
        batch_time = time.perf_counter() - start

        # Time individual
        start = time.perf_counter()
        for _ in range(10):
            for q in queries:
                ranker.rerank(q, docs)
        individual_time = time.perf_counter() - start

        # Allow minor runtime jitter; batch should not be materially slower.
        assert batch_time <= individual_time * 1.15, (
            f"rerank_batch ({batch_time:.4f}s) should not be materially slower than "
            f"individual calls ({individual_time:.4f}s)"
        )


class TestSharedCacheSpeedup:
    """Verify shared embedding cache avoids redundant encoding."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        reset_shared_cache()
        yield
        reset_shared_cache()

    def test_shared_cache_reduces_encoding_time(self) -> None:
        embedder = Embedder()
        texts = ["hello world", "foo bar", "baz qux"]

        # Cold run
        start = time.perf_counter()
        embedder.encode(texts)
        cold_time = time.perf_counter() - start

        # Warm run (should hit shared cache)
        start = time.perf_counter()
        embedder.encode(texts)
        warm_time = time.perf_counter() - start

        cache = get_shared_cache()
        assert cache.stats()["size"] == len(texts)
        assert warm_time < cold_time, (
            f"Warm encode ({warm_time:.6f}s) should be faster than cold encode ({cold_time:.6f}s)"
        )

    def test_multi_strategy_shared_cache(self) -> None:
        """Two strategies sharing the same embedder should share cache hits."""
        embedder = Embedder()
        docs = ["python programming", "java programming"]
        queries = ["python", "java"]

        # First strategy encodes docs and queries
        ranker1 = HybridFusionReranker(embedder=embedder)
        ranker1.fit(queries, docs, docs, [1, 0])
        ranker1.rerank("python", docs)

        cache = get_shared_cache()
        hits_after_first = cache.stats()["size"]

        # Second strategy with same embedder should get cache hits for the
        # same query and doc strings.
        ranker2 = HybridFusionReranker(embedder=embedder)
        ranker2.fit(queries, docs, docs, [1, 0])
        ranker2.rerank("python", docs)

        hits_after_second = cache.stats()["size"]
        assert hits_after_second == hits_after_first, (
            "Shared cache should not re-encode same texts across strategies"
        )


class TestHundredDocLatency:
    """Verify end-to-end latency on 100-document workloads."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        reset_shared_cache()
        yield
        reset_shared_cache()

    def test_colbert_100_doc_latency(self) -> None:
        ranker = StaticColBERTReranker(use_salience=False, quantization_mode="none")
        docs = [f"document number {i} about programming languages" for i in range(100)]
        ranker.fit(docs)

        # Warm up
        ranker.rerank("python", docs)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            ranker.rerank("python", docs)
            times.append(time.perf_counter() - start)

        median_latency = float(np.median(times))
        # Threshold is generous to account for CI variance; the real
        # target is avoiding regressions, not absolute milliseconds.
        assert median_latency < 5.0, (
            f"ColBERT 100-doc median latency ({median_latency:.3f}s) exceeds 5s threshold"
        )

    def test_hybrid_100_doc_latency(self) -> None:
        ranker = HybridFusionReranker()
        queries = ["python"] * 10
        doc_as = [f"doc a {i}" for i in range(10)]
        doc_bs = [f"doc b {i}" for i in range(10)]
        labels = [1] * 10
        ranker.fit(queries, doc_as, doc_bs, labels)

        docs = [f"document number {i} about programming languages" for i in range(100)]

        # Warm up
        ranker.rerank("python", docs)

        times = []
        for _ in range(5):
            start = time.perf_counter()
            ranker.rerank("python", docs)
            times.append(time.perf_counter() - start)

        median_latency = float(np.median(times))
        assert median_latency < 10.0, (
            f"Hybrid 100-doc median latency ({median_latency:.3f}s) exceeds 10s threshold"
        )
