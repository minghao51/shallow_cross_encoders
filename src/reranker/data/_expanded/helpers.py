"""Helper utilities for expanded dataset generation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from random import Random
from typing import TypeVar

from reranker.data._expanded.types import ExpandedSeedMap, TopicDocs

T = TypeVar("T")


def iter_topics(seed_map: ExpandedSeedMap) -> Iterator[tuple[str, TopicDocs]]:
    """Iterate over (domain, topic) pairs from a seed map.

    Args:
        seed_map: Expanded seed map keyed by domain.

    Yields:
        (domain, topic) tuples.
    """
    for domain, topics in seed_map.items():
        for topic in topics:
            yield domain, topic


def sample_cross_domain_topics(
    seed_map: ExpandedSeedMap,
    *,
    domain: str,
    rng: Random,
    sample_size: int,
) -> list[tuple[str, TopicDocs]]:
    """Sample topics from domains other than the given one.

    Args:
        seed_map: Expanded seed map.
        domain: Domain to exclude from sampling.
        rng: Random number generator.
        sample_size: Maximum number of topics to sample.

    Returns:
        List of (domain, topic) tuples from other domains.
    """
    other_domain_topics = [
        (candidate_domain, topic)
        for candidate_domain, topic in iter_topics(seed_map)
        if candidate_domain != domain
    ]
    if not other_domain_topics:
        return []
    return rng.sample(other_domain_topics, min(sample_size, len(other_domain_topics)))


def limited_shuffle(records: Iterable[T], *, limit: int, rng: Random) -> list[T]:
    """Shuffle an iterable and return up to `limit` items.

    Args:
        records: Iterable of items to shuffle.
        limit: Maximum number of items to return.
        rng: Random number generator.

    Returns:
        Shuffled list limited to `limit` items.
    """
    materialized = list(records)
    rng.shuffle(materialized)
    return materialized[:limit]
