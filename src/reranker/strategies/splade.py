from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reranker.protocols import RankedDoc
from reranker.utils import (
    build_artifact_metadata,
    dump_pickle,
    load_pickle,
    validate_artifact_metadata,
)


class SPLADEReranker:
    """Sparse encoder reranker using SPLADE-style sparse embeddings.

    SPLADE produces sparse vectors where non-zero dimensions correspond to
    informative terms. This provides:
    - Interpretable term-level importance scores
    - Natural combination of lexical (BM25-like) and semantic matching
    - Efficient sparse-dot-product scoring

    Uses pretrained SPLADE model for encoding. For CPU-efficient inference,
    we use the sparse encoder's term-based MaxSim scoring.

    Args:
        model_name: Name of the pretrained SPLADE/SparseEncoder model.
                    Default: "naver/splade-base-es-en" (multilingual)
                    Other options: "naver/splade-v2-max", "naver/splade-v2-base"
        top_k_terms: Number of top terms to keep per document for scoring.
                    Higher = more precise but slower. Default: 128.
    """

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
        if not self.is_fitted:
            raise RuntimeError("SPLADEReranker must be fitted before scoring.")
        if not docs:
            return np.zeros(0, dtype=np.float32)

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
        scores = np.zeros(len(docs), dtype=np.float32)

        for idx, doc_dict in enumerate(self._index):
            if not doc_dict or not query_terms:
                scores[idx] = 0.0
                continue
            score = self._maxsim_score(query_terms, doc_dict)
            scores[idx] = score

        return scores

    def _maxsim_score(self, query_terms: dict[str, float], doc_terms: dict[str, float]) -> float:
        """Compute MaxSim score between query and document sparse vectors.

        For each query term, find its weight in the document and sum.
        """
        score = 0.0
        for term, query_weight in query_terms.items():
            if term in doc_terms:
                score += min(query_weight, doc_terms[term])
        return score

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
                metadata={"strategy": "splade"},
            )
            for rank, (doc, score) in enumerate(ranked, start=1)
        ]

    def save(self, path: str | Path) -> None:
        dump_pickle(
            path,
            build_artifact_metadata(
                "splade_reranker",
                format_name="pickle",
                embedder_model_name=self.model_name,
                extra={
                    "index": self._index,
                    "top_k_terms": self.top_k_terms,
                },
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> SPLADEReranker:
        payload = load_pickle(path)
        validate_artifact_metadata(
            payload,
            expected_type="splade_reranker",
            expected_formats={"pickle"},
        )
        instance = cls(
            model_name=payload.get("embedder_model_name"),
            top_k_terms=payload.get("top_k_terms", 128),
        )
        instance._index = payload.get("index", [])
        instance.is_fitted = True
        return instance
