"""FlashRank adapter for external reranking.

This module provides a wrapper around FlashRank models to enable
consistent usage across different benchmarking scripts.
"""

from __future__ import annotations

from typing import Any

from reranker.protocols import RankedDoc


class FlashRankWrapper:
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._ranker = None

    def _load_ranker(self) -> Any:
        if self._ranker is not None:
            return self._ranker

        try:
            from flashrank import Ranker
        except ImportError as e:
            raise ImportError(
                "flashrank is not installed. Install with: uv sync --extra flashrank"
            ) from e

        self._ranker = Ranker(model_name=self.model_name)
        return self._ranker

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        if not docs:
            return []

        ranker = self._load_ranker()

        from flashrank import RerankRequest

        passages = [{"id": str(i), "text": doc} for i, doc in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        ranked = []
        for rank, result in enumerate(results, start=1):
            idx = int(result["id"])
            ranked.append(
                RankedDoc(
                    doc=docs[idx],
                    score=float(result.get("score", 0.0)),
                    rank=rank,
                    metadata={"strategy": "flashrank"},
                )
            )

        return ranked
