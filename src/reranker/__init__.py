"""Public package surface for the reranking toolkit."""

from reranker.config import load_yaml_config, settings_from_dict, settings_from_yaml
from reranker.protocols import BaseReranker, HeuristicAdapter, RankedDoc
from reranker.strategies.consistency import (
    Claim,
    ClaimSet,
    ConsistencyEngine,
    Contradiction,
)
from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter
from reranker.strategies.late_interaction import StaticColBERTReranker
from reranker.strategies.meta_router import MetaRouter

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
    "MetaRouter",
    "RankedDoc",
    "StaticColBERTReranker",
    "load_yaml_config",
    "settings_from_dict",
    "settings_from_yaml",
]
