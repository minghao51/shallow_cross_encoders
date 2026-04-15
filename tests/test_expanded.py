from __future__ import annotations

from reranker.data.expanded import (
    DOMAIN_SEEDS,
    generate_expanded_contradictions,
    generate_expanded_pairs,
    generate_expanded_preferences,
    iter_expanded_contradictions,
    iter_expanded_pairs,
    iter_expanded_preferences,
)


def test_expanded_compatibility_exports_exist() -> None:
    assert DOMAIN_SEEDS
    assert callable(generate_expanded_pairs)
    assert callable(generate_expanded_preferences)
    assert callable(generate_expanded_contradictions)


def test_expanded_pair_iterator_matches_list_wrapper() -> None:
    streamed = list(iter_expanded_pairs(target_count=50, seed=42))
    materialized = generate_expanded_pairs(target_count=50, seed=42)
    assert streamed == materialized


def test_expanded_preference_iterator_matches_list_wrapper() -> None:
    streamed = list(iter_expanded_preferences(target_count=40, seed=42))
    materialized = generate_expanded_preferences(target_count=40, seed=42)
    assert streamed == materialized


def test_expanded_contradiction_iterator_matches_list_wrapper() -> None:
    streamed = list(iter_expanded_contradictions(contradiction_count=12, control_count=4, seed=42))
    materialized = generate_expanded_contradictions(
        contradiction_count=12,
        control_count=4,
        seed=42,
    )
    assert streamed == materialized
