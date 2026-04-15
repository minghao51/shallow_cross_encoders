from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from reranker.protocols import RankedDoc
from reranker.utils import rrf_from_scores


@runtime_checkable
class Reranker(Protocol):
    """Protocol for rerankers that can be used with MultiReranker."""

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        """Rerank documents for a query."""
        ...


@dataclass
class MultiRerankerConfig:
    """Configuration for MultiReranker."""

    rrf_k: int = 60
    weights: list[float] | None = None


class MultiReranker:
    """Combine multiple rerankers using Reciprocal Rank Fusion.

    This strategy takes multiple rerankers, runs each one independently,
    and fuses their results using RRF. This provides a robust ensemble
    that often outperforms any single reranker.

    Key benefits:
    - Simple and robust fusion (no training required)
    - Each ranker contributes based on its ranking quality
    - Handles different score distributions gracefully
    - Low latency overhead (parallel scoring)

    Args:
        rerankers: List of (name, reranker) tuples to combine.
        config: Optional configuration with rrf_k and weights.
    """

    def __init__(
        self,
        rerankers: list[tuple[str, Reranker]],
        config: MultiRerankerConfig | None = None,
    ) -> None:
        self.rerankers = rerankers
        self.config = config or MultiRerankerConfig()
        self.weights = self.config.weights or [1.0] * len(rerankers)

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        if not docs:
            return []

        if len(self.rerankers) == 1:
            name, ranker = self.rerankers[0]
            return [
                RankedDoc(
                    doc=doc,
                    score=score,
                    rank=rank,
                    metadata={"strategy": f"multi_{name}"},
                )
                for rank, (doc, score) in enumerate(
                    zip(docs, [r.score for r in ranker.rerank(query, docs)], strict=False),
                    start=1,
                )
            ]

        score_arrays: list[np.ndarray] = []
        metadata_list: list[dict] = []

        for (name, ranker), weight in zip(self.rerankers, self.weights, strict=False):
            results = ranker.rerank(query, docs)
            scores = np.array([r.score for r in results], dtype=np.float32)
            if weight != 1.0:
                scores = np.clip(scores, 0, None) * weight
            score_arrays.append(scores)
            metadata_list.append({"name": name, "weight": weight})

        if len(score_arrays) == 1:
            fused_scores = score_arrays[0]
        else:
            fused_scores = rrf_from_scores(score_arrays, k=self.config.rrf_k)

        ranked_indices = np.argsort(-fused_scores)
        ranked_docs = [docs[i] for i in ranked_indices]
        ranked_scores = [fused_scores[i] for i in ranked_indices]

        return [
            RankedDoc(
                doc=doc,
                score=float(score),
                rank=rank,
                metadata={
                    "strategy": "multi_rrf",
                    "component_strategies": [m["name"] for m in metadata_list],
                    "rrf_k": self.config.rrf_k,
                },
            )
            for rank, (doc, score) in enumerate(
                zip(ranked_docs, ranked_scores, strict=False), start=1
            )
        ]

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(
            "MultiReranker does not support save/load because it wraps "
            "multiple reranker instances. Save individual rerankers instead."
        )

    @classmethod
    def load(cls, path: str | Path) -> MultiReranker:
        raise NotImplementedError(
            "MultiReranker does not support save/load because it wraps "
            "multiple reranker instances. Load individual rerankers instead."
        )
