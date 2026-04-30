"""SentenceTransformer cross-encoder adapter for benchmarking."""

from __future__ import annotations

from typing import Any

from reranker.protocols import RankedDoc


class SentenceTransformerWrapper:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: uv sync --extra sentence-transformers"
            ) from e

        self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        if not docs:
            return []

        model = self._load_model()

        pairs = [[query, doc] for doc in docs]
        scores = model.predict(pairs)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            RankedDoc(
                doc=docs[idx],
                score=float(score),
                rank=rank,
                metadata={"strategy": "cross_encoder"},
            )
            for rank, (idx, score) in enumerate(indexed_scores, start=1)
        ]
