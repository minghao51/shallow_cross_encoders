"""Cascading reranker with confidence-based fallback."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import structlog

from reranker.protocols import BaseReranker, RankedDoc

logger = structlog.get_logger(__name__)


class ConfidenceMetric(StrEnum):
    MAX_SCORE = "max_score"
    TOP_MARGIN = "top_margin"
    SCORE_VARIANCE = "score_variance"
    NORMALIZED_MAX = "normalized_max"


class FallbackStrategy(StrEnum):
    FLASHRANK = "flashrank"
    ALWAYS = "always"
    NEVER = "never"


@dataclass(slots=True)
class CascadeConfig:
    confidence_threshold: float = 0.6
    confidence_metric: ConfidenceMetric = ConfidenceMetric.TOP_MARGIN
    fallback_strategy: FallbackStrategy = FallbackStrategy.FLASHRANK

    def __post_init__(self) -> None:
        if isinstance(self.fallback_strategy, str) and not isinstance(
            self.fallback_strategy, FallbackStrategy
        ):
            try:
                self.fallback_strategy = FallbackStrategy(self.fallback_strategy)
            except ValueError:
                raise ValueError(
                    f"Invalid fallback_strategy '{self.fallback_strategy}'. "
                    f"Must be one of: {', '.join(s.value for s in FallbackStrategy)}"
                ) from None


class CascadeReranker:
    """Cascading reranker with confidence-based fallback."""

    def __init__(
        self,
        primary: BaseReranker,
        fallback: BaseReranker,
        config: CascadeConfig | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.config = config or CascadeConfig()
        self._total_queries: int = 0
        self._fallback_count: int = 0
        self._confidence_sum: float = 0.0

    @property
    def is_fitted(self) -> bool:
        return bool(
            getattr(self.primary, "is_fitted", True) and getattr(self.fallback, "is_fitted", True)
        )

    def _compute_confidence(self, results: list[RankedDoc]) -> float:
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
        if not docs:
            return []

        results = self.primary.rerank(query, docs)
        confidence = self._compute_confidence(results)
        self._total_queries += 1
        self._confidence_sum += confidence

        use_fallback = self.config.fallback_strategy == FallbackStrategy.ALWAYS or (
            self.config.fallback_strategy == FallbackStrategy.FLASHRANK
            and confidence < self.config.confidence_threshold
        )

        if use_fallback:
            results = self.fallback.rerank(query, docs)
            self._fallback_count += 1
            fallback_used = True
        else:
            fallback_used = False

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
        self._total_queries = 0
        self._fallback_count = 0
        self._confidence_sum = 0.0
