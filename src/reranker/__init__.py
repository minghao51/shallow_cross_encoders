"""Public package surface for the reranking toolkit."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from reranker.config import load_yaml_config, settings_from_dict, settings_from_yaml
from reranker.embedding_cache import EmbeddingCache, get_shared_cache
from reranker.heuristics.keyword import KeywordMatchAdapter
from reranker.protocols import BaseReranker, HeuristicAdapter, RankedDoc

__all__ = [
    "BaseReranker",
    "BinaryQuantizedReranker",
    "Claim",
    "ClaimSet",
    "ConsistencyEngine",
    "Contradiction",
    "DistilledPairwiseRanker",
    "EmbeddingCache",
    "FlashRankEnsemble",
    "FlashRankWrapper",
    "get_shared_cache",
    "HeuristicAdapter",
    "HybridFusionReranker",
    "KeywordMatchAdapter",
    "MetaRouter",
    "RankedDoc",
    "SentenceTransformerWrapper",
    "StaticColBERTReranker",
    "load_yaml_config",
    "settings_from_dict",
    "settings_from_yaml",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "BinaryQuantizedReranker": ("reranker.strategies.binary_reranker", "BinaryQuantizedReranker"),
    "Claim": ("reranker.strategies.consistency", "Claim"),
    "ClaimSet": ("reranker.strategies.consistency", "ClaimSet"),
    "ConsistencyEngine": ("reranker.strategies.consistency", "ConsistencyEngine"),
    "Contradiction": ("reranker.strategies.consistency", "Contradiction"),
    "DistilledPairwiseRanker": ("reranker.strategies.distilled", "DistilledPairwiseRanker"),
    "FlashRankEnsemble": ("reranker.strategies.flashrank_ensemble", "FlashRankEnsemble"),
    "FlashRankWrapper": ("reranker.strategies.flashrank_ensemble", "FlashRankWrapper"),
    "HybridFusionReranker": ("reranker.strategies.hybrid", "HybridFusionReranker"),
    "SentenceTransformerWrapper": (
        "reranker.strategies.flashrank_ensemble",
        "SentenceTransformerWrapper",
    ),
    "StaticColBERTReranker": ("reranker.strategies.late_interaction", "StaticColBERTReranker"),
    "MetaRouter": ("reranker.strategies.meta_router", "MetaRouter"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'reranker' has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
