"""Synthetic data generation - seeds, prompts, models, and generator."""

from __future__ import annotations

from reranker.data.client import OpenRouterClient
from reranker.data.synth._generator import SyntheticDataGenerator
from reranker.data.synth._models import (
    ContradictionRecord,
    DatasetManifest,
    HardNegativeRecord,
    ListwisePreferenceRecord,
    PairRecord,
    PreferenceRecord,
    QueryExpansionRecord,
)
from reranker.data.synth._prompts import (
    CONTRADICTION_BATCH_PROMPT,
    CONTRADICTION_PROMPT,
    HARD_NEGATIVE_BATCH_PROMPT,
    HARD_NEGATIVE_PROMPT,
    LISTWISE_BATCH_PROMPT,
    LISTWISE_PROMPT,
    PAIR_BATCH_PROMPT,
    PAIR_PROMPT,
    PREFERENCE_BATCH_PROMPT,
    PREFERENCE_PROMPT,
    QUERY_EXPANSION_BATCH_PROMPT,
    QUERY_EXPANSION_PROMPT,
)
from reranker.data.synth._seeds import DEFAULT_PAIR_SEEDS
from reranker.data.synth.generator import (
    get_expanded_seeds,
    iter_contradictions,
    iter_hard_negatives,
    iter_listwise_preferences,
    iter_pairs,
    iter_preferences,
    iter_query_expansions,
)

__all__ = [
    "CONTRADICTION_BATCH_PROMPT",
    "CONTRADICTION_PROMPT",
    "DEFAULT_PAIR_SEEDS",
    "HARD_NEGATIVE_BATCH_PROMPT",
    "HARD_NEGATIVE_PROMPT",
    "LISTWISE_BATCH_PROMPT",
    "LISTWISE_PROMPT",
    "OpenRouterClient",
    "PAIR_BATCH_PROMPT",
    "PAIR_PROMPT",
    "PREFERENCE_BATCH_PROMPT",
    "PREFERENCE_PROMPT",
    "QUERY_EXPANSION_BATCH_PROMPT",
    "QUERY_EXPANSION_PROMPT",
    "ContradictionRecord",
    "DatasetManifest",
    "HardNegativeRecord",
    "ListwisePreferenceRecord",
    "PairRecord",
    "PreferenceRecord",
    "QueryExpansionRecord",
    "SyntheticDataGenerator",
    "get_expanded_seeds",
    "iter_contradictions",
    "iter_hard_negatives",
    "iter_listwise_preferences",
    "iter_pairs",
    "iter_preferences",
    "iter_query_expansions",
]
