from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reranker.config import get_settings
from reranker.deps import check_model2vec

try:
    from cachetools import TTLCache
except Exception:
    TTLCache = None


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


@dataclass(slots=True)
class Embedder:
    """Static embedding wrapper with a deterministic offline fallback."""

    model_name: str = field(default_factory=lambda: get_settings().embedder.model_name)
    dimension: int = field(default_factory=lambda: get_settings().embedder.dimension)
    normalize: bool = field(default_factory=lambda: get_settings().embedder.normalize)
    _backend: Any = field(init=False, default=None, repr=False)
    backend_name: str = field(init=False, default="hashed")
    _encode_cache: Any = field(init=False, default=None, repr=False)

    def _sync_backend_dimension(self) -> None:
        if self._backend is None:
            return
        try:
            probe_vectors = np.asarray(
                self._backend.encode([""], normalize=self.normalize),
                dtype=np.float32,
            )
        except Exception:
            return
        if probe_vectors.ndim == 2 and probe_vectors.shape[1] > 0:
            self.dimension = int(probe_vectors.shape[1])

    def _init_cache(self) -> None:
        if TTLCache is None:
            return
        settings = get_settings()
        max_size = getattr(settings.embedder, "cache_max_size", 10000)
        ttl = getattr(settings.embedder, "cache_ttl_seconds", 3600)
        self._encode_cache = TTLCache(maxsize=max_size, ttl=ttl)

    def __post_init__(self) -> None:
        self._backend = None
        self.backend_name = "hashed"
        self._init_cache()
        backend_cls, status = check_model2vec()
        if backend_cls is not None:
            try:
                self._backend = backend_cls.from_pretrained(self.model_name)
                self.backend_name = status.backend
                self._sync_backend_dimension()
            except Exception:
                warnings.warn(
                    (
                        f"Unable to load model2vec model '{self.model_name}'; "
                        "falling back to deterministic hashed embeddings."
                    ),
                    stacklevel=2,
                )
                self._backend = None
                self.backend_name = "hashed"

    def _encode_hashed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._simple_tokenize(text):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, index] += sign
        if self.normalize:
            matrix = _normalize_rows(matrix)
        return matrix

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        if self._backend is None:
            return self._encode_hashed(texts)
        if self._encode_cache is not None:
            result = []
            uncached = []
            for i, text in enumerate(texts):
                cached = self._encode_cache.get(text)
                if cached is not None:
                    result.append(cached)
                else:
                    result.append(None)
                    uncached.append(i)
            if not uncached:
                return np.stack(result)
            vectors = self._backend.encode([texts[i] for i in uncached], normalize=self.normalize)
            vectors = np.asarray(vectors, dtype=np.float32)
            for idx, vec in zip(uncached, vectors, strict=True):
                self._encode_cache[texts[idx]] = vec
                result[idx] = vec
            return np.stack(result)
        vectors = self._backend.encode(texts, normalize=self.normalize)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 2 and vectors.shape[1] > 0:
            self.dimension = int(vectors.shape[1])
        return vectors

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words, handling CJK and other scripts.

        Tries to use the model's tokenizer if available (via model2vec internals),
        otherwise falls back to a simple Unicode-aware regex tokenizer.
        """
        if self._backend is not None:
            try:
                if hasattr(self._backend, "tokenizer") and self._backend.tokenizer is not None:
                    tokens = self._backend.tokenizer.encode(text, add_special_tokens=False)
                    return self._backend.tokenizer.convert_ids_to_tokens(tokens)
            except Exception:
                pass

        return self._simple_tokenize(text)

    @staticmethod
    def _simple_tokenize(text: str) -> list[str]:
        """Simple Unicode-aware word tokenization.

        Splits on whitespace and groups consecutive alphabetic characters
        into words. For CJK characters, creates character bigrams to enable
        partial matching without a language-specific tokenizer.
        """
        text = text.lower()
        words: list[str] = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            if text[i].isascii() and text[i].isalpha():
                start = i
                while i < len(text) and text[i].isascii() and text[i].isalnum():
                    i += 1
                words.append(text[start:i])
            elif not text[i].isascii():
                start = i
                while i < len(text) and not text[i].isascii():
                    i += 1
                char_seq = text[start:i]
                if len(char_seq) >= 2:
                    for j in range(len(char_seq) - 1):
                        words.append(char_seq[j : j + 2])
                else:
                    words.append(char_seq)
            else:
                words.append(text[i])
                i += 1
        return words

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.clip(np.dot(a, b), 0.0, 1.0))

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "model_name": self.model_name,
            "dimension": self.dimension,
        }
