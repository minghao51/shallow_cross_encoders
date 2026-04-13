"""FlashRank ensemble wrapper for multi-teacher distillation.

This module provides a wrapper around multiple FlashRank models to serve
as teachers for ensemble distillation. It averages predictions from multiple
teacher models (e.g., TinyBERT and MiniLM) to generate soft labels for
training a student model.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FlashRankEnsemble:
    """Ensemble of FlashRank models for multi-teacher distillation.

    This class wraps multiple FlashRank rerankers and averages their
    predictions to generate ensemble scores. These ensemble scores serve
    as teacher labels for distilling knowledge into a student model.

    Example:
        >>> ensemble = FlashRankEnsemble(
        ...     models=["ms-marco-TinyBERT-L-2-v2", "ms-marco-MiniLM-L-12-v2"]
        ... )
        >>> scores = ensemble.score_batch("python tutorial", ["doc1", "doc2"])
        >>> # Returns averaged scores from both teacher models
    """

    def __init__(self, models: list[str]) -> None:
        """Initialize the ensemble with a list of FlashRank model names.

        Args:
            models: List of FlashRank model names to ensemble.
                   Common choices: "ms-marco-TinyBERT-L-2-v2",
                                  "ms-marco-MiniLM-L-12-v2"

        Raises:
            ValueError: If models list is empty.
        """
        if not models:
            raise ValueError("models list cannot be empty")

        self.models = models
        self._rankers: list[Any] | None = None  # Lazy loaded

    def _load_rankers(self) -> None:
        """Lazy load FlashRank rankers.

        Raises:
            ImportError: If flashrank is not installed.
        """
        if self._rankers is not None:
            return

        try:
            from flashrank import Ranker
        except ImportError as e:
            raise ImportError(
                "flashrank is not installed. Install with: uv sync --extra flashrank"
            ) from e

        self._rankers = [Ranker(model_name=model) for model in self.models]

    def score_batch(self, query: str, docs: list[str]) -> np.ndarray:
        """Score documents using ensemble of FlashRank models.

        Args:
            query: The query text.
            docs: List of document texts to score.

        Returns:
            np.ndarray: Averaged scores across all teacher models.
                       Shape: (len(docs),), dtype: np.float32

        Raises:
            ImportError: If flashrank is not installed.
        """
        if not docs:
            return np.zeros(0, dtype=np.float32)

        # Lazy load rankers on first use
        self._load_rankers()

        from flashrank import RerankRequest

        # Collect scores from all teacher models
        all_scores: list[np.ndarray] = []

        for ranker in self._rankers:
            # Prepare passages for flashrank
            passages = [{"id": str(i), "text": doc} for i, doc in enumerate(docs)]

            # Create rerank request
            request = RerankRequest(query=query, passages=passages)

            # Get reranking results
            results = ranker.rerank(request)

            # Map results back to original doc order
            scores = np.zeros(len(docs), dtype=np.float32)
            for result in results:
                doc_idx = int(result["id"])
                scores[doc_idx] = float(result.get("score", 0.0))

            all_scores.append(scores)

        # Average scores across all teacher models
        ensemble_scores = np.mean(all_scores, axis=0).astype(np.float32)

        return ensemble_scores
