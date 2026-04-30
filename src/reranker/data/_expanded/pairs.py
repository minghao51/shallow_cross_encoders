"""Expanded offline pair generation with streaming support."""

from __future__ import annotations

import random
from collections.abc import Iterator

from reranker.data._expanded.helpers import iter_topics, limited_shuffle, sample_cross_domain_topics
from reranker.data._expanded.seeds import DOMAIN_SEEDS
from reranker.data._expanded.types import ExpandedPairRecord, ExpandedSeedMap


def iter_expanded_pairs(
    target_count: int = 10000,
    seed: int = 42,
    *,
    seed_map: ExpandedSeedMap | None = None,
) -> Iterator[ExpandedPairRecord]:
    """Yield expanded pair records without building the final list eagerly.

    Args:
        target_count: Number of pairs to generate.
        seed: Random seed.
        seed_map: Optional custom seed map. Defaults to DOMAIN_SEEDS.

    Yields:
        ExpandedPairRecord dicts with query, doc, score, domain.
    """
    """Yield expanded pair records without building the final list eagerly."""
    active_seed_map = DOMAIN_SEEDS if seed_map is None else seed_map
    rng = random.Random(seed)
    records: list[ExpandedPairRecord] = []

    for domain, topic in iter_topics(active_seed_map):
        query = topic["query"]
        for score, doc in topic["docs"].items():
            records.append({"query": query, "doc": doc, "score": score, "domain": domain})

        other_topics = [
            candidate for candidate in active_seed_map[domain] if candidate["query"] != query
        ]
        for other_topic in other_topics:
            records.append(
                {"query": query, "doc": other_topic["docs"][1], "score": 1, "domain": domain}
            )
            records.append(
                {"query": query, "doc": other_topic["docs"][0], "score": 0, "domain": domain}
            )

        for _, cross_topic in sample_cross_domain_topics(
            active_seed_map,
            domain=domain,
            rng=rng,
            sample_size=4,
        ):
            records.append(
                {"query": query, "doc": cross_topic["docs"][0], "score": 0, "domain": domain}
            )

    for domain, topic in iter_topics(active_seed_map):
        query = topic["query"]
        other_topics = [
            candidate for candidate in active_seed_map[domain] if candidate["query"] != query
        ]
        for other_topic in other_topics:
            records.append(
                {"query": query, "doc": other_topic["docs"][2], "score": 2, "domain": domain}
            )
            records.append({"query": query, "doc": topic["docs"][3], "score": 3, "domain": domain})

    yield from limited_shuffle(records, limit=target_count, rng=rng)


def generate_expanded_pairs(
    target_count: int = 10000,
    seed: int = 42,
) -> list[ExpandedPairRecord]:
    """Generate an expanded pair dataset with balanced relevance labels.

    Args:
        target_count: Number of pairs to generate.
        seed: Random seed.

    Returns:
        List of ExpandedPairRecord dicts.
    """
    """Generate an expanded pair dataset with balanced relevance labels."""
    return list(iter_expanded_pairs(target_count=target_count, seed=seed))
