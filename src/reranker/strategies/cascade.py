"""Cascading reranker with confidence-based fallback.

Fast path: Use distilled model (Hybrid/Distilled Pairwise)
Fallback: Use FlashRank when confidence is low
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from reranker.protocols import BaseReranker, RankedDoc


class ConfidenceMetric(StrEnum):
    """Available confidence metrics for cascade triggering."""

    MAX_SCORE = "max_score"
    TOP_MARGIN = "top_margin"
    SCORE_VARIANCE = "score_variance"
    NORMALIZED_MAX = "normalized_max"


@dataclass(slots=True)
class CascadeConfig:
    """Configuration for cascading reranker.

    Attributes:
        confidence_threshold: Threshold below which fallback is triggered (0-1)
        confidence_metric: Metric to compute confidence score
        fallback_strategy: When to use fallback ("flashrank", "always", "never")
    """

    confidence_threshold: float = 0.6
    confidence_metric: ConfidenceMetric = ConfidenceMetric.TOP_MARGIN
    fallback_strategy: str = "flashrank"


class CascadeReranker:
    """Cascading reranker with confidence-based fallback.

    Uses a fast distilled model (Hybrid Fusion, Distilled Pairwise) for common cases,
    falling back to a slower but more accurate model (FlashRank) when confidence is low.

    Example:
        ```python
        from reranker.strategies import CascadeReranker, HybridFusionReranker
        from reranker.strategies.flashrank_ensemble import FlashRankEnsemble

        primary = HybridFusionReranker()
        fallback = FlashRankEnsemble(models=["ms-marco-TinyBERT-L-2-v2"])
        cascade = CascadeReranker(primary, fallback)

        results = cascade.rerank("python tutorial", docs)
        stats = cascade.get_stats()
        print(f"Fallback rate: {stats['fallback_rate']:.1%}")
        ```
    """

    def __init__(
        self,
        primary: BaseReranker,
        fallback: BaseReranker,
        config: CascadeConfig | None = None,
    ) -> None:
        """Initialize cascade reranker.

        Args:
            primary: Fast distilled model (Hybrid Fusion, Distilled Pairwise)
            fallback: Slow but accurate model (FlashRank)
            config: Cascade configuration
        """
        self.primary = primary
        self.fallback = fallback
        self.config = config or CascadeConfig()
        self.is_fitted = False

        # Metrics tracking
        self._total_queries: int = 0
        self._fallback_count: int = 0
        self._confidence_sum: float = 0.0

    def _compute_confidence(self, results: list[RankedDoc]) -> float:
        """Compute confidence score based on configured metric.

        Args:
            results: Ranked results from primary reranker

        Returns:
            Confidence score (higher = more certain)
        """
        if not results:
            return 0.0

        scores = [r.score for r in results]

        match self.config.confidence_metric:
            case ConfidenceMetric.MAX_SCORE:
                return max(scores)
            case ConfidenceMetric.TOP_MARGIN:
                if len(scores) >= 2:
                    sorted_scores = sorted(scores, reverse=True)
                    return sorted_scores[0] - sorted_scores[1]
                return max(scores)
            case ConfidenceMetric.SCORE_VARIANCE:
                import statistics

                if len(scores) > 1:
                    return statistics.variance(scores)
                return 0.0
            case ConfidenceMetric.NORMALIZED_MAX:
                score_max = max(scores)
                score_min = min(scores)
                score_range = score_max - score_min
                if score_range == 0:
                    return 1.0
                return score_max / score_range if score_range > 0 else 0.0
            case _:
                return max(scores)

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        """Rerank documents with confidence-based fallback.

        Args:
            query: Search query
            docs: Documents to rerank

        Returns:
            Ranked documents with cascade metadata
        """
        if not docs:
            return []

        # Run primary reranker
        results = self.primary.rerank(query, docs)

        # Compute confidence
        confidence = self._compute_confidence(results)

        # Track metrics
        self._total_queries += 1
        self._confidence_sum += confidence

        # Determine if fallback should be used
        use_fallback = self.config.fallback_strategy == "always" or (
            self.config.fallback_strategy == "flashrank"
            and confidence < self.config.confidence_threshold
        )

        # Fallback if needed
        if use_fallback:
            results = self.fallback.rerank(query, docs)
            self._fallback_count += 1
            fallback_used = True
        else:
            fallback_used = False

        # Add metadata
        for r in results:
            r.metadata.update(
                {
                    "strategy": "cascade",
                    "fallback_used": fallback_used,
                    "confidence": confidence,
                    "metric": self.config.confidence_metric.value,
                    "threshold": self.config.confidence_threshold,
                }
            )

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get cascade statistics.

        Returns:
            Dictionary with total_queries, fallback_count, fallback_rate, avg_confidence
        """
        fallback_rate = (
            self._fallback_count / self._total_queries if self._total_queries > 0 else 0.0
        )
        avg_confidence = (
            self._confidence_sum / self._total_queries if self._total_queries > 0 else 0.0
        )
        return {
            "total_queries": self._total_queries,
            "fallback_count": self._fallback_count,
            "fallback_rate": fallback_rate,
            "avg_confidence": avg_confidence,
        }

    def reset_stats(self) -> None:
        """Reset cascade statistics counters."""
        self._total_queries = 0
        self._fallback_count = 0
        self._confidence_sum = 0.0
