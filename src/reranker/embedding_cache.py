from __future__ import annotations

import threading
from typing import Any

import numpy as np

from reranker.config import get_settings

try:
    from cachetools import TTLCache
except Exception:
    TTLCache = None  # type: ignore[assignment,misc]


class EmbeddingCache:
    def __init__(self, max_size: int | None = None, ttl_seconds: int | None = None) -> None:
        settings = get_settings().embedding_cache
        self._max_size = max_size or settings.max_size
        self._ttl_seconds = ttl_seconds or settings.ttl_seconds
        self._cache: Any = None
        self._lock = threading.Lock()
        if TTLCache is not None:
            self._cache = TTLCache(maxsize=self._max_size, ttl=self._ttl_seconds)

    @property
    def enabled(self) -> bool:
        return self._cache is not None

    @staticmethod
    def _embedder_fingerprint(embedder: Any) -> tuple[str, str, int, str]:
        model_name = str(getattr(embedder, "model_name", "unknown"))
        normalize = str(bool(getattr(embedder, "normalize", False)))
        dimension = int(getattr(embedder, "dimension", 0))
        backend_name = str(getattr(embedder, "backend_name", "unknown"))
        return (model_name, normalize, dimension, backend_name)

    def _key(self, text: str, embedder: Any) -> tuple[str, str, str, int, str]:
        model_name, normalize, dimension, backend_name = self._embedder_fingerprint(embedder)
        return (text, model_name, normalize, dimension, backend_name)

    def get_or_encode(self, texts: list[str], embedder: Any) -> np.ndarray:
        if not texts:
            return np.zeros((0, embedder.dimension), dtype=np.float32)
        if self._cache is None:
            return embedder.encode(texts)

        result: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        with self._lock:
            for i, text in enumerate(texts):
                cached = self._cache.get(self._key(text, embedder))
                if cached is not None:
                    result[i] = cached
                else:
                    uncached_indices.append(i)

        if not uncached_indices:
            return np.stack([v for v in result if v is not None])

        uncached_texts = [texts[i] for i in uncached_indices]
        if hasattr(embedder, "_encode_raw"):
            vectors = embedder._encode_raw(uncached_texts)
        else:
            vectors = embedder.encode(uncached_texts)

        with self._lock:
            for idx, vec in zip(uncached_indices, vectors, strict=True):
                self._cache[self._key(texts[idx], embedder)] = vec
                result[idx] = vec

        return np.stack([v for v in result if v is not None])

    def invalidate(self, text: str, model_name: str) -> None:
        if self._cache is None:
            return
        with self._lock:
            dead_keys = [k for k in self._cache.keys() if k[0] == text and k[1] == model_name]
            for key in dead_keys:
                self._cache.pop(key, None)

    def clear(self) -> None:
        if self._cache is None:
            return
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, int]:
        if self._cache is None:
            return {"enabled": 0, "size": 0, "max_size": 0}
        with self._lock:
            return {
                "enabled": 1,
                "size": len(self._cache),
                "max_size": self._max_size,
            }


_global_cache: EmbeddingCache | None = None
_global_cache_lock = threading.Lock()


def get_shared_cache() -> EmbeddingCache:
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = EmbeddingCache()
    return _global_cache


def reset_shared_cache() -> None:
    global _global_cache
    with _global_cache_lock:
        if _global_cache is not None:
            _global_cache.clear()
        _global_cache = None
