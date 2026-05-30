from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reranker.persistence_mixin import SaveableReranker
from reranker.types import RankedDoc
from reranker.utils import rank_docs


class SPLADEReranker(SaveableReranker):
    """Sparse encoder reranker using SPLADE-style sparse embeddings."""

    _artifact_type = "splade_reranker"

    DEFAULT_MODELS = {
        "en": "naver/splade-cocondenser-ensembledistil",
        "multilingual": "naver/splade-base-es-en",
    }

    def __init__(
        self,
        model_name: str | None = None,
        top_k_terms: int = 128,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODELS["en"]
        self.top_k_terms = top_k_terms
        self._encoder: Any = None
        self._index: list[dict[str, float]] = []
        self._query_cache: dict[str, dict[str, float]] = {}
        self.is_fitted = False

    def _load_encoder(self) -> None:
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import SparseEncoder

            self._encoder = SparseEncoder(self.model_name)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SPLADE. "
                "Install with: pip install sentence-transformers",
            ) from e

    def fit(self, docs: list[str]) -> SPLADEReranker:
        self._load_encoder()
        sparse_embeddings = self._encoder.encode(
            docs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_dict=True,
        )
        self._index = []
        for sparse_vec in sparse_embeddings:
            if isinstance(sparse_vec, dict):
                top_items = sorted(sparse_vec.items(), key=lambda x: x[1], reverse=True)
                self._index.append({str(k): float(v) for k, v in top_items[: self.top_k_terms]})
            else:
                self._index.append({})
        self.is_fitted = True
        return self

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        self._require_fitted("SPLADEReranker")
        if not docs:
            return np.zeros(0, dtype=np.float32)

        if query in self._query_cache:
            query_terms = self._query_cache[query]
        else:
            query_sparse = self._encoder.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_dict=True,
            )
            if isinstance(query_sparse, list):
                query_dict = query_sparse[0] if query_sparse else {}
            else:
                query_dict = query_sparse or {}

            query_terms = {str(k): float(v) for k, v in query_dict.items()}
            self._query_cache[query] = query_terms
        scores = np.zeros(len(docs), dtype=np.float32)

        for idx, doc_dict in enumerate(self._index):
            if not doc_dict or not query_terms:
                scores[idx] = 0.0
                continue
            scores[idx] = self._maxsim_score(query_terms, doc_dict)

        return scores

    def _maxsim_score(self, query_terms: dict[str, float], doc_terms: dict[str, float]) -> float:
        score = 0.0
        for term, query_weight in query_terms.items():
            if term in doc_terms:
                score += query_weight * doc_terms[term]
        return score

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        if not docs:
            return []
        self._require_fitted("SPLADEReranker")
        scores = self.score(query, docs)
        return rank_docs(docs, scores, "splade")

    def _save_metadata(self) -> dict:
        return {"embedder_model_name": self.model_name, "top_k_terms": self.top_k_terms}

    def _save_weights(self) -> dict:
        return {"index": self._index}

    @classmethod
    def load(cls, path: str | Path) -> SPLADEReranker:
        payload = cls._load_payload(path, expected_type=cls._artifact_type)
        instance = cls(
            model_name=payload.get("embedder_model_name"),
            top_k_terms=payload.get("top_k_terms", 128),
        )
        instance._index = payload.get("index", [])
        instance.is_fitted = True
        return instance
