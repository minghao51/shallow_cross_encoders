"""Binary quantised reranker using Hamming distance and bilinear scoring.

Stage 1: Quick Hamming-distance filter over binary-quantised embeddings.
Stage 2: Learned bilinear (query^T W doc) re-scoring for top candidates.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from reranker.config import get_settings
from reranker.embedder import Embedder
from reranker.persistence import save_safe, try_load_safe_or_warn
from reranker.protocols import RankedDoc
from reranker.utils import (
    load_pickle,
)

logger = logging.getLogger("reranker.strategies.binary_reranker")


class BinaryQuantizedReranker:
    """Fast reranker using binary quantization and Hamming distance.

    Stage 1: Quantize embeddings to bits, rank by Hamming distance (ultra-fast).
    Stage 2: For top-k candidates, apply bilinear interaction score = q^T W d
             where W is a learned diagonal weight matrix.

    This provides parameterized similarity — instead of assuming all dimensions
    are equally important, W learns which semantic dimensions are high-signal
    for specific query types.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
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
                "Bilinear model fit failed (%s). Falling back to uniform weights. "
                "This usually means the training data has too few samples or no variance.",
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
        """Fit the reranker by encoding docs and learning bilinear weights.

        Args:
            queries: List of query strings.
            docs: List of document strings.
            labels: Relevance labels for each (query, doc) pair.

        Returns:
            Self, fitted.
        """
        if not queries or not docs:
            self._doc_vectors = np.zeros((0, self.embedder.dimension), dtype=np.float32)
            self._doc_bits = np.zeros((0, self.embedder.dimension), dtype=np.uint8)
            self._bilinear_weights = np.ones(self.embedder.dimension, dtype=np.float32)
            self.is_fitted = True
            return self

        all_docs = list(set(docs))
        doc_vectors = self.embedder.encode(all_docs)
        self._doc_vectors = doc_vectors
        self._doc_bits = self._quantize(doc_vectors)

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
        """Score documents using Hamming distance + bilinear re-scoring.

        Args:
            query: Search query.
            docs: Documents to score.

        Returns:
            Array of relevance scores.

        Raises:
            RuntimeError: If the reranker has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("BinaryQuantizedReranker must be fitted before scoring.")
        if not docs:
            return np.zeros(0, dtype=np.float32)

        if self._doc_vectors is None or self._doc_vectors.shape[0] == 0:
            return np.zeros(len(docs), dtype=np.float32)

        query_vec = self.embedder.encode([query])[0]
        query_bits = self._quantize(query_vec[np.newaxis, :])[0]

        doc_vectors = self.embedder.encode(docs)
        doc_bits = self._quantize(doc_vectors)

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
        """Rerank documents by binary quantised scoring.

        Auto-fits on the provided documents if not already fitted.

        Args:
            query: Search query.
            docs: Documents to rerank.

        Returns:
            Ranked list of RankedDoc.
        """
        if not docs:
            return []

        if not self.is_fitted:
            self.fit([query], docs, [1] * len(docs))

        scores = self.score(query, docs)
        ranked = sorted(
            zip(docs, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            RankedDoc(
                doc=doc, score=float(score), rank=rank, metadata={"strategy": "binary_reranker"}
            )
            for rank, (doc, score) in enumerate(ranked, start=1)
        ]

    def save(self, path: str | Path) -> None:
        """Persist the binary reranker to disk.

        Args:
            path: Destination file path.
        """
        save_safe(
            path,
            artifact_type="binary_reranker",
            metadata={
                "embedder_model_name": self.embedder.model_name,
                "hamming_top_k": self.hamming_top_k,
                "bilinear_top_k": self.bilinear_top_k,
            },
            weights={
                "doc_vectors": self._doc_vectors,
                "doc_bits": self._doc_bits,
                "bilinear_weights": self._bilinear_weights,
                "bilinear_model": self._bilinear_model,
            },
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: Embedder | None = None,
    ) -> BinaryQuantizedReranker:
        """Load a saved BinaryQuantizedReranker from disk.

        Args:
            path: Path to the saved artifact.
            embedder: Optional embedder override.

        Returns:
            Loaded BinaryQuantizedReranker instance.
        """
        payload = try_load_safe_or_warn(
            path,
            expected_type="binary_reranker",
            legacy_loader=load_pickle,
        )
        instance = cls(
            embedder=embedder or Embedder(payload.get("embedder_model_name")),
            hamming_top_k=payload.get("hamming_top_k"),
            bilinear_top_k=payload.get("bilinear_top_k"),
        )
        instance._doc_vectors = payload.get("doc_vectors")
        instance._doc_bits = payload.get("doc_bits")
        instance._bilinear_weights = payload.get("bilinear_weights")
        instance._bilinear_model = payload.get("bilinear_model")
        instance.is_fitted = True
        return instance
