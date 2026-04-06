"""Public package surface for the reranking toolkit."""

from reranker.protocols import BaseReranker, HeuristicAdapter, RankedDoc
from reranker.strategies.consistency import (
    Claim,
    ClaimSet,
    ConsistencyEngine,
    Contradiction,
)
from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter

__all__ = [
    "BaseReranker",
    "Claim",
    "ClaimSet",
    "ConsistencyEngine",
    "Contradiction",
    "DistilledPairwiseRanker",
    "HeuristicAdapter",
    "HybridFusionReranker",
    "KeywordMatchAdapter",
    "RankedDoc",
]
