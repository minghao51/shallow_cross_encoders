"""Ranking and consistency strategies."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from reranker.heuristics.keyword import KeywordMatchAdapter

__all__ = [
    "BinaryQuantizedReranker",
    "CascadeConfig",
    "CascadeReranker",
    "ConfidenceMetric",
    "ConsistencyEngine",
    "DistilledPairwiseRanker",
    "FallbackStrategy",
    "FlashRankEnsemble",
    "FlashRankWrapper",
    "HybridFusionReranker",
    "KeywordMatchAdapter",
    "MultiReranker",
    "MultiRerankerConfig",
    "PipelineResult",
    "PipelineReranker",
    "PipelineStage",
    "SPLADEReranker",
    "SentenceTransformerWrapper",
    "StaticColBERTReranker",
    "TokenIndex",
    "WeightingMode",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "BinaryQuantizedReranker": ("reranker.strategies.binary_reranker", "BinaryQuantizedReranker"),
    "CascadeConfig": ("reranker.strategies.cascade", "CascadeConfig"),
    "CascadeReranker": ("reranker.strategies.cascade", "CascadeReranker"),
    "ConfidenceMetric": ("reranker.strategies.cascade", "ConfidenceMetric"),
    "FallbackStrategy": ("reranker.strategies.cascade", "FallbackStrategy"),
    "ConsistencyEngine": ("reranker.strategies.consistency", "ConsistencyEngine"),
    "DistilledPairwiseRanker": ("reranker.strategies.distilled", "DistilledPairwiseRanker"),
    "FlashRankEnsemble": ("reranker.strategies.flashrank_ensemble", "FlashRankEnsemble"),
    "FlashRankWrapper": ("reranker.strategies.flashrank_ensemble", "FlashRankWrapper"),
    "HybridFusionReranker": ("reranker.strategies.hybrid", "HybridFusionReranker"),
    "WeightingMode": ("reranker.strategies.hybrid", "WeightingMode"),
    "StaticColBERTReranker": ("reranker.strategies.late_interaction", "StaticColBERTReranker"),
    "TokenIndex": ("reranker.strategies.late_interaction", "TokenIndex"),
    "MultiReranker": ("reranker.strategies.multi", "MultiReranker"),
    "MultiRerankerConfig": ("reranker.strategies.multi", "MultiRerankerConfig"),
    "PipelineReranker": ("reranker.strategies.pipeline", "PipelineReranker"),
    "PipelineResult": ("reranker.strategies.pipeline", "PipelineResult"),
    "PipelineStage": ("reranker.strategies.pipeline", "PipelineStage"),
    "SPLADEReranker": ("reranker.strategies.splade", "SPLADEReranker"),
    "SentenceTransformerWrapper": (
        "reranker.strategies.flashrank_ensemble",
        "SentenceTransformerWrapper",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module 'reranker.strategies' has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
