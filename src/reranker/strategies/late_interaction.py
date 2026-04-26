from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reranker.config import get_settings
from reranker.embedder import Embedder
from reranker.persistence import save_safe, try_load_safe_or_warn
from reranker.protocols import RankedDoc
from reranker.quantization import QuantizationResult, dequantize, quantize
from reranker.utils import (
    load_pickle,
)


@dataclass(slots=True)
class TokenIndex:
    """Stores token-level embeddings for a single document."""

    text: str
    tokens: list[str]
    vectors: np.ndarray  # shape: (num_tokens, dim)
    salience: np.ndarray | None = None
    quantized: QuantizationResult | None = None


class StaticColBERTReranker:
    """Late interaction reranker using token-level MaxSim scoring.

    Instead of collapsing a document into a single vector, this stores
    individual token embeddings and computes MaxSim at query time:
        score = sum_{q_t in query} max_{d_t in doc} cosine(q_t, d_t)

    This captures term-level alignment that mean-pooling loses, while
    remaining CPU-efficient because the vectors are static and pre-computed.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        top_k_tokens: int | None = None,
        use_salience: bool = False,
        quantization_mode: str | None = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder or Embedder()
        self.top_k_tokens = (
            top_k_tokens if top_k_tokens is not None else settings.late_interaction.top_k_tokens
        )
        self.use_salience = use_salience
        self.quantization_mode = (
            quantization_mode
            if quantization_mode is not None
            else settings.late_interaction.quantization
        )
        self._index: list[TokenIndex] = []
        self.is_fitted = False

    def _tokenize(self, text: str) -> list[str]:
        return self.embedder.tokenize(text.lower())

    def _encode_tokens(self, tokens: list[str]) -> np.ndarray:
        if not tokens:
            return np.zeros((0, self.embedder.dimension), dtype=np.float32)
        return self.embedder.encode(tokens)

    def _compute_salience(self, tokens: list[str], vectors: np.ndarray) -> np.ndarray:
        if vectors.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        tf = np.zeros(vectors.shape[0], dtype=np.float32)
        unique, counts = np.unique(tokens, return_counts=True)
        for tok, cnt in zip(unique, counts, strict=False):
            indices = [i for i, t in enumerate(tokens) if t == tok]
            for idx in indices:
                tf[idx] = cnt
        idf = np.log(1 + len(tokens) / (tf + 1))
        return tf * idf

    def _prune_tokens(
        self,
        doc_text: str | list[str],
        tokens: list[str] | np.ndarray,
        vectors: np.ndarray | None = None,
    ) -> TokenIndex:
        if vectors is None:
            actual_doc_text = " ".join(doc_text) if isinstance(doc_text, list) else str(doc_text)
            actual_tokens = list(doc_text) if isinstance(doc_text, list) else [str(doc_text)]
            actual_vectors = np.asarray(tokens, dtype=np.float32)
        else:
            actual_doc_text = str(doc_text)
            actual_tokens = list(tokens)
            actual_vectors = vectors
        if actual_vectors.shape[0] <= self.top_k_tokens:
            salience = (
                self._compute_salience(actual_tokens, actual_vectors) if self.use_salience else None
            )
            return TokenIndex(
                text=actual_doc_text,
                tokens=actual_tokens,
                vectors=actual_vectors,
                salience=salience,
            )

        if self.use_salience:
            salience = self._compute_salience(actual_tokens, actual_vectors)
            top_indices = np.argsort(salience)[-self.top_k_tokens :]
            return TokenIndex(
                text=actual_doc_text,
                tokens=[actual_tokens[i] for i in top_indices],
                vectors=actual_vectors[top_indices],
                salience=salience[top_indices],
            )

        top_indices = np.arange(min(self.top_k_tokens, len(actual_tokens)))
        return TokenIndex(
            text=actual_doc_text,
            tokens=[actual_tokens[i] for i in top_indices],
            vectors=actual_vectors[top_indices],
            salience=None,
        )

    def fit(self, docs: list[str]) -> StaticColBERTReranker:
        self._index = []
        for doc in docs:
            tokens = self._tokenize(doc)
            vectors = self._encode_tokens(tokens)
            entry = self._prune_tokens(doc, tokens, vectors)
            if self.quantization_mode != "none" and entry.vectors.shape[0] > 0:
                entry.quantized = quantize(
                    entry.vectors,
                    mode=self.quantization_mode,
                )
            self._index.append(entry)
        self.is_fitted = True
        return self

    @staticmethod
    def _maxsim(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
        if query_vectors.shape[0] == 0 or doc_vectors.shape[0] == 0:
            return 0.0

        q_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        d_norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0, 1.0, q_norms)
        d_norms = np.where(d_norms == 0, 1.0, d_norms)

        q_normalized = query_vectors / q_norms
        d_normalized = doc_vectors / d_norms

        sim_matrix = q_normalized @ d_normalized.T
        max_sims = np.max(sim_matrix, axis=1)
        return float(np.sum(max_sims))

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StaticColBERTReranker must be fitted before scoring.")
        if not docs:
            return np.zeros(0, dtype=np.float32)

        query_tokens = self._tokenize(query)
        query_vectors = self._encode_tokens(query_tokens)

        doc_to_index = {entry.text: entry for entry in self._index}
        scores = np.zeros(len(docs), dtype=np.float32)
        for idx, doc_text in enumerate(docs):
            doc_index = doc_to_index.get(doc_text)
            if doc_index is None:
                scores[idx] = 0.0
                continue

            if doc_index.quantized is not None:
                doc_vectors = dequantize(doc_index.quantized)
            else:
                doc_vectors = doc_index.vectors

            if doc_vectors.shape[0] == 0:
                scores[idx] = 0.0
                continue

            if doc_index.salience is not None:
                doc_vectors = doc_vectors * doc_index.salience[:, np.newaxis]

            scores[idx] = self._maxsim(query_vectors, doc_vectors)

        return scores

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        if not docs:
            return []

        if not self.is_fitted:
            self.fit(docs)

        scores = self.score(query, docs)
        ranked = sorted(
            zip(docs, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            RankedDoc(
                doc=doc,
                score=float(score),
                rank=rank,
                metadata={"strategy": "late_interaction"},
            )
            for rank, (doc, score) in enumerate(ranked, start=1)
        ]

    def save(self, path: str | Path) -> None:
        index_data = []
        for entry in self._index:
            item: dict[str, Any] = {
                "text": entry.text,
                "tokens": entry.tokens,
                "salience": entry.salience,
                "quantization_mode": self.quantization_mode,
            }
            if entry.quantized is not None and entry.quantized.mode != "none":
                item["quantized_codes"] = entry.quantized.codes
                item["quantized_mode"] = entry.quantized.mode
                item["quantized_original_shape"] = entry.quantized.original_shape
                if entry.quantized.scale is not None:
                    item["quantized_scale"] = entry.quantized.scale
                if entry.quantized.min_val is not None:
                    item["quantized_min_val"] = entry.quantized.min_val
                item["vectors"] = np.zeros((0,), dtype=np.float32)
            else:
                item["vectors"] = entry.vectors
            index_data.append(item)
        save_safe(
            path,
            artifact_type="late_interaction_reranker",
            metadata={
                "embedder_model_name": self.embedder.model_name,
                "top_k_tokens": self.top_k_tokens,
                "use_salience": self.use_salience,
                "quantization_mode": self.quantization_mode,
            },
            weights={"index_data": index_data},
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: Embedder | None = None,
    ) -> StaticColBERTReranker:
        payload = try_load_safe_or_warn(
            path,
            expected_type="late_interaction_reranker",
            legacy_loader=load_pickle,
        )
        q_mode = payload.get("quantization_mode", "none")
        instance = cls(
            embedder=embedder or Embedder(payload.get("embedder_model_name")),
            top_k_tokens=payload.get("top_k_tokens"),
            use_salience=payload.get("use_salience", False),
            quantization_mode=q_mode,
        )
        index_data = payload.get("index_data", [])
        instance._index = []
        for item in index_data:
            quantized = None
            if "quantized_codes" in item and q_mode != "none":
                quantized = QuantizationResult(
                    codes=item["quantized_codes"],
                    codebook=None,
                    scale=item.get("quantized_scale"),
                    min_val=item.get("quantized_min_val"),
                    mode=item.get("quantized_mode", q_mode),
                    original_shape=tuple(item.get("quantized_original_shape", (0,))),
                )
                vectors = dequantize(quantized)
            else:
                vectors = item["vectors"]
            instance._index.append(
                TokenIndex(
                    text=item.get("text", ""),
                    tokens=item["tokens"],
                    vectors=vectors,
                    salience=item.get("salience"),
                    quantized=quantized,
                )
            )
        instance.is_fitted = True
        return instance
