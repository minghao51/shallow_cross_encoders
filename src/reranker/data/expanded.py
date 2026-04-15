"""Compatibility exports for expanded offline dataset generation."""

from reranker.data._expanded import (
    DOMAIN_SEEDS,
    generate_expanded_contradictions,
    generate_expanded_pairs,
    generate_expanded_preferences,
    iter_expanded_contradictions,
    iter_expanded_pairs,
    iter_expanded_preferences,
)

__all__ = [
    "DOMAIN_SEEDS",
    "generate_expanded_contradictions",
    "generate_expanded_pairs",
    "generate_expanded_preferences",
    "iter_expanded_contradictions",
    "iter_expanded_pairs",
    "iter_expanded_preferences",
]
