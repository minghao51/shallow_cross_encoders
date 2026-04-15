"""Typed helpers for expanded offline dataset generation."""

from reranker.data._expanded.contradictions import (
    generate_expanded_contradictions,
    iter_expanded_contradictions,
)
from reranker.data._expanded.pairs import generate_expanded_pairs, iter_expanded_pairs
from reranker.data._expanded.preferences import (
    generate_expanded_preferences,
    iter_expanded_preferences,
)
from reranker.data._expanded.seeds import DOMAIN_SEEDS

__all__ = [
    "DOMAIN_SEEDS",
    "generate_expanded_contradictions",
    "generate_expanded_pairs",
    "generate_expanded_preferences",
    "iter_expanded_contradictions",
    "iter_expanded_pairs",
    "iter_expanded_preferences",
]
