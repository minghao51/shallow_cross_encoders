"""Binary quantised reranker using Hamming distance and bilinear scoring.

Stage 1: Quick Hamming-distance filter over binary-quantised embeddings.
Stage 2: Learned bilinear (query^T W doc) re-scoring for top candidates.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import structlog
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from reranker.config import get_settings
from reranker.embedder import Embedder
from reranker.persistence_mixin import SaveableReranker
from reranker.protocols import EmbedderProtocol
from reranker.types import RankedDoc
from reranker.utils import rank_docs

logger = structlog.get_logger(__name__)


class BinaryQuantizedReranker(SaveableReranker):
    """Fast reranker using binary quantization and Hamming distance."""

    _artifact_type = "binary_reranker"

    def __init__(
        self,
        embedder: EmbedderProtocol | None = None,
        hamming_top_k: int | None = None,
        bilinear_top_k: int | None = None,
        random_state: int | None = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder or Embedder()
        self.hamming_top_k = (
            hamming_top_k if hamming_top_k is not None else settings.binary_reranker.hamming_top_k
        )
        self.bilinear_top_k = (
            bilinear_top_k
            if bilinear_top_k is not None
            else settings.binary_reranker.bilinear_top_k
        )
        self.random_state = (
            random_state if random_state is not None else settings.binary_reranker.random_state
        )
        self._doc_vectors: np.ndarray | None = None
        self._doc_bits: np.ndarray | None = None
        self._bilinear_weights: np.ndarray | None = None
        self._bilinear_model: LogisticRegression | DummyClassifier | None = None
        self._doc_encoding_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._doc_encoding_cache_max_size = 10_000
        self.is_fitted = False

    @staticmethod
    def _quantize(vectors: np.ndarray) -> np.ndarray:
        return (vectors > 0).astype(np.uint8)

    @staticmethod
    def _hamming_distances(query_bits: np.ndarray, doc_bits: np.ndarray) -> np.ndarray:
        if query_bits.ndim == 1:
            query_bits = query_bits[np.newaxis, :]
        return np.count_nonzero(query_bits != doc_bits, axis=1).astype(np.float32)

    def _fit_bilinear(
        self,
        query_vectors: np.ndarray,
        doc_vectors: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if query_vectors.shape[0] == 0:
            self._bilinear_weights = np.ones(self.embedder.dimension, dtype=np.float32)
            self._bilinear_model = DummyClassifier(strategy="constant", constant=0)
            self._bilinear_model.fit(np.zeros((1, 1)), np.array([0]))
            return

        elementwise_products = query_vectors * doc_vectors
        self._bilinear_model = LogisticRegression(
            C=1.0,
            max_iter=500,
            random_state=self.random_state,
        )
        try:
            self._bilinear_model.fit(elementwise_products, labels)
            if hasattr(self._bilinear_model, "coef_"):
                self._bilinear_weights = np.abs(self._bilinear_model.coef_[0]).astype(np.float32)
            else:
                self._bilinear_weights = np.ones(self.embedder.dimension, dtype=np.float32)
        except Exception as exc:
            logger.warning(
                "Bilinear model fit failed (%s). Falling back to uniform weights.",
                exc,
                exc_info=True,
            )
            self._bilinear_weights = np.ones(self.embedder.dimension, dtype=np.float32)
            self._bilinear_model = DummyClassifier(strategy="constant", constant=0)
            self._bilinear_model.fit(np.zeros((1, 1)), np.array([0]))

    def _bilinear_score(self, query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
        if self._bilinear_weights is None:
            return float(np.dot(query_vec, doc_vec))
        return float(np.dot(query_vec * self._bilinear_weights, doc_vec))

    def fit(
        self,
        queries: list[str],
        docs: list[str],
        labels: list[int],
    ) -> BinaryQuantizedReranker:
        if not queries or not docs:
            self._doc_vectors = np.zeros((0, self.embedder.dimension), dtype=np.float32)
            self._doc_bits = np.zeros((0, self.embedder.dimension), dtype=np.uint8)
            self._bilinear_weights = np.ones(self.embedder.dimension, dtype=np.float32)
            self._doc_encoding_cache.clear()
            self.is_fitted = True
            return self

        all_docs = list(set(docs))
        doc_vectors = self.embedder.encode(all_docs)
        self._doc_vectors = doc_vectors
        self._doc_bits = self._quantize(doc_vectors)
        self._doc_encoding_cache.clear()
        for doc_text, vec, bits in zip(all_docs, doc_vectors, self._doc_bits, strict=True):
            self._doc_encoding_cache[doc_text] = (vec, bits)

        y = np.asarray(labels, dtype=np.int32)
        if len(set(y.tolist())) < 2:
            self._bilinear_weights = np.ones(self.embedder.dimension, dtype=np.float32)
            self._bilinear_model = DummyClassifier(strategy="most_frequent")
            self._bilinear_model.fit(
                np.zeros((max(len(y), 1), 1)), y if len(y) > 0 else np.array([0])
            )
        else:
            query_vectors = self.embedder.encode(queries)
            doc_texts_for_training = [str(doc) for doc in docs]
            doc_vectors_for_training = self.embedder.encode(doc_texts_for_training)
            if query_vectors.shape[0] == 1 and doc_vectors_for_training.shape[0] > 1:
                query_vectors = np.tile(query_vectors, (doc_vectors_for_training.shape[0], 1))
            self._fit_bilinear(query_vectors, doc_vectors_for_training, y)

        self.is_fitted = True
        return self

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        self._require_fitted("BinaryQuantizedReranker")
        if not docs:
            return np.zeros(0, dtype=np.float32)

        if self._doc_vectors is None or self._doc_vectors.shape[0] == 0:
            return np.zeros(len(docs), dtype=np.float32)

        query_vec = self.embedder.encode([query])[0]
        query_bits = self._quantize(query_vec[np.newaxis, :])[0]

        uncached_docs = [
            (i, doc) for i, doc in enumerate(docs) if doc not in self._doc_encoding_cache
        ]
        if uncached_docs:
            uncached_texts = [doc for _, doc in uncached_docs]
            new_vectors = self.embedder.encode(uncached_texts)
            new_bits = self._quantize(new_vectors)
            for (_, doc_text), vec, bits in zip(uncached_docs, new_vectors, new_bits, strict=True):
                self._doc_encoding_cache[doc_text] = (vec, bits)
                if len(self._doc_encoding_cache) > self._doc_encoding_cache_max_size:
                    self._doc_encoding_cache.popitem(last=False)

        doc_vectors = np.zeros((len(docs), self.embedder.dimension), dtype=np.float32)
        doc_bits = np.zeros((len(docs), self.embedder.dimension), dtype=np.uint8)
        for i, doc in enumerate(docs):
            entry = self._doc_encoding_cache[doc]
            self._doc_encoding_cache.move_to_end(doc)
            vec, bits = entry
            doc_vectors[i] = vec
            doc_bits[i] = bits

        hamming_dists = self._hamming_distances(query_bits, doc_bits)
        max_dist = max(float(hamming_dists.max()), 1.0)
        hamming_scores = 1.0 - (hamming_dists / max_dist)

        top_k_indices = np.argsort(hamming_scores)[-self.hamming_top_k :]

        final_scores = hamming_scores.copy()
        bilinear_indices = top_k_indices[-self.bilinear_top_k :]
        for idx in bilinear_indices:
            final_scores[idx] = self._bilinear_score(query_vec, doc_vectors[idx])

        return final_scores

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        if not docs:
            return []
        self._require_fitted("BinaryQuantizedReranker")
        scores = self.score(query, docs)
        return rank_docs(docs, scores, "binary_reranker")

    def _save_metadata(self) -> dict:
        return {
            "embedder_model_name": self.embedder.model_name,
            "hamming_top_k": self.hamming_top_k,
            "bilinear_top_k": self.bilinear_top_k,
        }

    def _save_weights(self) -> dict:
        return {
            "doc_vectors": self._doc_vectors,
            "doc_bits": self._doc_bits,
            "bilinear_weights": self._bilinear_weights,
            "bilinear_model": self._bilinear_model,
        }

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: Embedder | None = None,
    ) -> BinaryQuantizedReranker:
        payload = cls._load_payload(path, expected_type=cls._artifact_type)
        instance = cls(
            embedder=embedder
            or Embedder(str(payload.get("embedder_model_name", "minishlab/potion-base-32M"))),
            hamming_top_k=payload.get("hamming_top_k"),
            bilinear_top_k=payload.get("bilinear_top_k"),
        )
        instance._doc_vectors = payload.get("doc_vectors")
        instance._doc_bits = payload.get("doc_bits")
        instance._bilinear_weights = payload.get("bilinear_weights")
        instance._bilinear_model = payload.get("bilinear_model")
        instance.is_fitted = True
        return instance
