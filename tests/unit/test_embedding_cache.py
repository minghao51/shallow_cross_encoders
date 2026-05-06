import numpy as np
import pytest

from reranker.embedder import Embedder
from reranker.embedding_cache import EmbeddingCache, get_shared_cache, reset_shared_cache


@pytest.fixture(autouse=True)
def _clean_global_cache():
    reset_shared_cache()
    yield
    reset_shared_cache()


class TestEmbeddingCache:
    def test_cache_miss_encodes(self) -> None:
        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        embedder = Embedder()
        texts = ["hello world", "foo bar"]
        result = cache.get_or_encode(texts, embedder)
        assert result.shape[0] == 2
        assert result.shape[1] == embedder.dimension

    def test_cache_hit_avoids_reencode(self) -> None:
        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        embedder = Embedder()
        texts = ["hello world"]
        first = cache.get_or_encode(texts, embedder)
        second = cache.get_or_encode(texts, embedder)
        np.testing.assert_array_equal(first, second)
        stats = cache.stats()
        assert stats["size"] == 1

    def test_cache_mixed_hit_miss(self) -> None:
        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        embedder = Embedder()
        cache.get_or_encode(["alpha"], embedder)
        result = cache.get_or_encode(["alpha", "beta"], embedder)
        assert result.shape[0] == 2

    def test_cache_clear(self) -> None:
        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        embedder = Embedder()
        cache.get_or_encode(["hello"], embedder)
        cache.clear()
        stats = cache.stats()
        assert stats["size"] == 0

    def test_cache_invalidate_single(self) -> None:
        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        embedder = Embedder()
        cache.get_or_encode(["hello"], embedder)
        cache.invalidate("hello", embedder.model_name)
        stats = cache.stats()
        assert stats["size"] == 0

    def test_shared_cache_singleton(self) -> None:
        a = get_shared_cache()
        b = get_shared_cache()
        assert a is b

    def test_shared_cache_reset(self) -> None:
        a = get_shared_cache()
        reset_shared_cache()
        b = get_shared_cache()
        assert a is not b

    def test_empty_texts(self) -> None:
        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        embedder = Embedder()
        result = cache.get_or_encode([], embedder)
        assert result.shape[0] == 0

    def test_cache_key_isolates_normalize_mode(self) -> None:
        class _FakeEmbedder:
            def __init__(self, normalize: bool) -> None:
                self.model_name = "fake-model"
                self.dimension = 2
                self.normalize = normalize
                self.backend_name = "fake"
                self.calls = 0

            def _encode_raw(self, texts: list[str]) -> np.ndarray:
                self.calls += 1
                val = 1.0 if self.normalize else 2.0
                return np.full((len(texts), self.dimension), val, dtype=np.float32)

        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        a = _FakeEmbedder(normalize=True)
        b = _FakeEmbedder(normalize=False)
        av = cache.get_or_encode(["same"], a)
        bv = cache.get_or_encode(["same"], b)
        assert a.calls == 1
        assert b.calls == 1
        assert float(av[0, 0]) != float(bv[0, 0])

    def test_cache_key_isolates_backend_and_dimension(self) -> None:
        class _FakeEmbedder:
            def __init__(self, backend_name: str, dimension: int) -> None:
                self.model_name = "fake-model"
                self.dimension = dimension
                self.normalize = True
                self.backend_name = backend_name
                self.calls = 0

            def _encode_raw(self, texts: list[str]) -> np.ndarray:
                self.calls += 1
                return np.full((len(texts), self.dimension), 3.0, dtype=np.float32)

        cache = EmbeddingCache(max_size=100, ttl_seconds=60)
        a = _FakeEmbedder(backend_name="hashed", dimension=2)
        b = _FakeEmbedder(backend_name="model2vec", dimension=2)
        c = _FakeEmbedder(backend_name="hashed", dimension=3)
        cache.get_or_encode(["same"], a)
        cache.get_or_encode(["same"], b)
        cache.get_or_encode(["same"], c)
        assert a.calls == 1
        assert b.calls == 1
        assert c.calls == 1
