"""Late-interaction reranker using token-level MaxSim scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reranker.config import get_settings
from reranker.embedder import Embedder, _normalize_rows
from reranker.persistence_mixin import SaveableReranker
from reranker.protocols import EmbedderProtocol
from reranker.quantization import QuantizationResult, dequantize, quantize
from reranker.types import RankedDoc
from reranker.utils import rank_docs


@dataclass(slots=True)
class TokenIndex:
    """Stores token-level embeddings for a single document."""

    text: str
    tokens: list[str]
    vectors: np.ndarray  # shape: (num_tokens, dim)
    salience: np.ndarray | None = None
    quantized: QuantizationResult | None = None


class StaticColBERTReranker(SaveableReranker):
    """Late interaction reranker using token-level MaxSim scoring."""

    _artifact_type = "late_interaction_reranker"

    def __init__(
        self,
        embedder: EmbedderProtocol | None = None,
        top_k_tokens: int | None = None,
        use_salience: bool | None = None,
        quantization_mode: str | None = None,
    ) -> None:
        settings = get_settings()
        self.embedder = embedder or Embedder()
        self.top_k_tokens = (
            top_k_tokens if top_k_tokens is not None else settings.late_interaction.top_k_tokens
        )
        self.use_salience = (
            use_salience if use_salience is not None else settings.late_interaction.use_salience
        )
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
        token_arr = np.array(tokens, dtype=object)
        _, inverse, counts = np.unique(token_arr, return_inverse=True, return_counts=True)
        tf = counts[inverse].astype(np.float32)
        idf = np.log(1 + len(tokens) / (tf + 1))
        return tf * idf

    def _prune_tokens(
        self, tokens: list[str], vectors: np.ndarray, doc_text: str = ""
    ) -> TokenIndex:
        if vectors.shape[0] <= self.top_k_tokens:
            salience = self._compute_salience(tokens, vectors) if self.use_salience else None
            return TokenIndex(
                text=doc_text or " ".join(tokens),
                tokens=tokens,
                vectors=_normalize_rows(vectors),
                salience=salience,
            )

        if self.use_salience:
            salience = self._compute_salience(tokens, vectors)
            top_indices = np.argsort(salience)[-self.top_k_tokens :]
            return TokenIndex(
                text=doc_text or " ".join([tokens[i] for i in top_indices]),
                tokens=[tokens[i] for i in top_indices],
                vectors=_normalize_rows(vectors[top_indices]),
                salience=salience[top_indices],
            )

        top_indices = np.arange(min(self.top_k_tokens, len(tokens)))
        return TokenIndex(
            text=doc_text or " ".join([tokens[i] for i in top_indices]),
            tokens=[tokens[i] for i in top_indices],
            vectors=_normalize_rows(vectors[top_indices]),
            salience=None,
        )

    def fit(self, docs: list[str]) -> StaticColBERTReranker:
        self._index = []
        all_tokens: list[str] = []
        doc_token_lengths: list[int] = []
        per_doc_tokens: list[list[str]] = []
        for doc in docs:
            tokens = self._tokenize(doc)
            per_doc_tokens.append(tokens)
            all_tokens.extend(tokens)
            doc_token_lengths.append(len(tokens))

        if all_tokens:
            all_vectors = self.embedder.encode(all_tokens)
        else:
            all_vectors = np.zeros((0, self.embedder.dimension), dtype=np.float32)

        offset = 0
        for i, doc in enumerate(docs):
            n_tokens = doc_token_lengths[i]
            token_vecs = (
                all_vectors[offset : offset + n_tokens]
                if n_tokens > 0
                else np.zeros((0, self.embedder.dimension), dtype=np.float32)
            )
            offset += n_tokens
            entry = self._prune_tokens(per_doc_tokens[i], token_vecs, doc_text=doc)
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
        q_norms = np.where(q_norms == 0, 1.0, q_norms)
        q_normalized = query_vectors / q_norms

        sim_matrix = q_normalized @ doc_vectors.T
        max_sims = np.max(sim_matrix, axis=1)
        return float(np.sum(max_sims))

    def score(
        self,
        query: str,
        docs: list[str],
        *,
        prebuilt_indices: list[TokenIndex] | None = None,
    ) -> np.ndarray:
        self._require_fitted("StaticColBERTReranker")
        if not docs:
            return np.zeros(0, dtype=np.float32)

        query_tokens = self._tokenize(query)
        query_vectors = self._encode_tokens(query_tokens)

        if prebuilt_indices is not None:
            doc_to_index = {entry.text: entry for entry in prebuilt_indices}
        else:
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

    def rerank(
        self,
        query: str,
        docs: list[str],
        *,
        prebuilt_indices: list[TokenIndex] | None = None,
    ) -> list[RankedDoc]:
        if not docs:
            return []
        self._require_fitted("StaticColBERTReranker")
        scores = self.score(query, docs, prebuilt_indices=prebuilt_indices)
        return rank_docs(docs, scores, "late_interaction")

    def rerank_batch(
        self,
        queries: list[str],
        docs: list[str],
    ) -> list[list[RankedDoc]]:
        if not queries:
            return []
        self._require_fitted("StaticColBERTReranker")
        all_query_tokens = [self._tokenize(q) for q in queries]
        flat_tokens = [t for tokens in all_query_tokens for t in tokens]
        if flat_tokens:
            all_vectors = self._encode_tokens(flat_tokens)
        else:
            all_vectors = np.zeros((0, self.embedder.dimension), dtype=np.float32)
        query_vectors_list: list[np.ndarray] = []
        offset = 0
        for tokens in all_query_tokens:
            n = len(tokens)
            if n == 0:
                query_vectors_list.append(np.zeros((0, self.embedder.dimension), dtype=np.float32))
            else:
                query_vectors_list.append(all_vectors[offset : offset + n])
                offset += n

        doc_to_index = {entry.text: entry for entry in self._index}
        results: list[list[RankedDoc]] = []
        for q_idx, _query in enumerate(queries):
            q_vecs = query_vectors_list[q_idx]
            scores = np.zeros(len(docs), dtype=np.float32)
            for d_idx, doc_text in enumerate(docs):
                entry = doc_to_index.get(doc_text)
                if entry is None:
                    continue
                doc_vectors = (
                    dequantize(entry.quantized) if entry.quantized is not None else entry.vectors
                )
                if doc_vectors.shape[0] == 0:
                    continue
                if entry.salience is not None:
                    doc_vectors = doc_vectors * entry.salience[:, np.newaxis]
                scores[d_idx] = self._maxsim(q_vecs, doc_vectors)
            results.append(rank_docs(docs, scores, "late_interaction"))
        return results

    def _save_metadata(self) -> dict:
        return {
            "embedder_model_name": self.embedder.model_name,
            "top_k_tokens": self.top_k_tokens,
            "use_salience": self.use_salience,
            "quantization_mode": self.quantization_mode,
        }

    def _save_weights(self) -> dict:
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
        return {"index_data": index_data}

    @classmethod
    def load(
        cls,
        path: str | Path,
        embedder: Embedder | None = None,
    ) -> StaticColBERTReranker:
        payload = cls._load_payload(path, expected_type=cls._artifact_type)
        q_mode = payload.get("quantization_mode", "none")
        instance = cls(
            embedder=embedder
            or Embedder(str(payload.get("embedder_model_name", "minishlab/potion-base-32M"))),
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
