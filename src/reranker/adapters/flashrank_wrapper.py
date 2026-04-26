"""FlashRank adapter for external reranking.

This module provides a wrapper around FlashRank models to enable
consistent usage across different benchmarking scripts.
"""

from __future__ import annotations

from typing import Any


class FlashRankWrapper:
    """Wrapper for FlashRank reranking models.

    This class provides a consistent interface for FlashRank models,
    handling conversion between different data formats used in benchmarks.

    Example:
        >>> wrapper = FlashRankWrapper("ms-marco-TinyBERT-L-2-v2")
        >>> ranked = wrapper.rerank("query", ["doc1", "doc2", "doc3"])
        >>> for doc in ranked:
        ...     print(f"{doc['rank']}: {doc['doc']} (score: {doc['score']})")
    """

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        """Initialize FlashRank wrapper.

        Args:
            model_name: FlashRank model name.
                       Common choices: "ms-marco-TinyBERT-L-2-v2",
                                      "ms-marco-MiniLM-L-12-v2"

        Raises:
            ImportError: If flashrank is not installed.
        """
        self.model_name = model_name
        self._ranker = None

    def _load_ranker(self) -> Any:
        """Lazy load FlashRank ranker.

        Returns:
            FlashRank Ranker instance.

        Raises:
            ImportError: If flashrank is not installed.
        """
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

    def rerank(self, query: str, docs: list[str]) -> list[dict[str, Any]]:
        """Rerank documents using FlashRank.

        Args:
            query: Query text.
            docs: List of document texts to rerank.

        Returns:
            List of dictionaries with keys:
            - doc: Original document text
            - score: Relevance score from FlashRank
            - rank: Ranking position (1-indexed)

        Raises:
            ImportError: If flashrank is not installed.
        """
        if not docs:
            return []

        ranker = self._load_ranker()

        from flashrank import RerankRequest

        # Prepare passages for flashrank
        passages = [{"id": str(i), "text": doc} for i, doc in enumerate(docs)]

        # Create rerank request
        request = RerankRequest(query=query, passages=passages)

        # Get reranking results
        results = ranker.rerank(request)

        # Map results back to original docs
        ranked = []
        for result in results:
            idx = int(result["id"])
            ranked.append(
                {
                    "doc": docs[idx],
                    "score": float(result.get("score", 0.0)),
                    "rank": 0,
                }
            )

        # Assign ranks
        for rank, doc in enumerate(ranked, start=1):
            doc["rank"] = rank

        return ranked
