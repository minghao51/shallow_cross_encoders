"""Thread-safety tests for EmbeddingCache."""

from __future__ import annotations

import concurrent.futures

import numpy as np

from reranker.embedder import Embedder
from reranker.embedding_cache import EmbeddingCache, get_shared_cache, reset_shared_cache


class TestEmbeddingCacheThreadSafety:
    def test_concurrent_cache_access(self) -> None:
        cache = EmbeddingCache(max_size=1000, ttl_seconds=60)
        embedder = Embedder()
        texts = [f"doc {i}" for i in range(50)]

        def worker(_: int) -> np.ndarray:
            return cache.get_or_encode(texts, embedder)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker, range(8)))

        first = results[0]
        for r in results[1:]:
            np.testing.assert_array_equal(first, r)

    def test_concurrent_cache_no_corruption(self) -> None:
        cache = EmbeddingCache(max_size=1000, ttl_seconds=60)
        embedder = Embedder()

        def worker(idx: int) -> np.ndarray:
            texts = [f"doc_{idx}_{i}" for i in range(10)]
            return cache.get_or_encode(texts, embedder)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(worker, range(8)))

        stats = cache.stats()
        assert stats["size"] == 80
        assert stats["enabled"] == 1

    def test_concurrent_shared_cache(self) -> None:
        reset_shared_cache()

        def worker(_: int) -> None:
            cache = get_shared_cache()
            embedder = Embedder()
            texts = [f"shared_{_}_{i}" for i in range(5)]
            cache.get_or_encode(texts, embedder)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(worker, range(4)))
