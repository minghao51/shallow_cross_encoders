from __future__ import annotations

from collections.abc import Iterable, Iterator
from random import Random
from typing import TypeVar

from reranker.data._expanded.types import ExpandedSeedMap, TopicDocs

T = TypeVar("T")


def iter_topics(seed_map: ExpandedSeedMap) -> Iterator[tuple[str, TopicDocs]]:
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
    other_domain_topics = [
        (candidate_domain, topic)
        for candidate_domain, topic in iter_topics(seed_map)
        if candidate_domain != domain
    ]
    if not other_domain_topics:
        return []
    return rng.sample(other_domain_topics, min(sample_size, len(other_domain_topics)))


def limited_shuffle(records: Iterable[T], *, limit: int, rng: Random) -> list[T]:
    materialized = list(records)
    rng.shuffle(materialized)
    return materialized[:limit]
