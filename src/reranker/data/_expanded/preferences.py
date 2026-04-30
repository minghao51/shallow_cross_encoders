"""Expanded offline preference generation with streaming support."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Literal

from reranker.data._expanded.helpers import iter_topics, limited_shuffle
from reranker.data._expanded.seeds import DOMAIN_SEEDS
from reranker.data._expanded.types import ExpandedPreferenceRecord, ExpandedSeedMap


def _build_preference_record(
    *,
    query: str,
    doc_a: str,
    doc_b: str,
    preferred: Literal["A", "B"],
    confidence: float,
    domain: str,
) -> ExpandedPreferenceRecord:
    """Construct an ExpandedPreferenceRecord dict.

    Args:
        query: Query string.
        doc_a: First document.
        doc_b: Second document.
        preferred: Which document is preferred.
        confidence: Confidence score 0.0-1.0.
        domain: Domain label.

    Returns:
        ExpandedPreferenceRecord dict.
    """
    return {
        "query": query,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "preferred": preferred,
        "confidence": round(confidence, 2),
        "domain": domain,
    }


def iter_expanded_preferences(
    target_count: int = 5000,
    seed: int = 42,
    *,
    seed_map: ExpandedSeedMap | None = None,
) -> Iterator[ExpandedPreferenceRecord]:
    """Yield expanded pairwise preferences with deterministic shuffling.

    Args:
        target_count: Number of preference records.
        seed: Random seed.
        seed_map: Optional custom seed map. Defaults to DOMAIN_SEEDS.

    Yields:
        ExpandedPreferenceRecord dicts.
    """
    """Yield expanded pairwise preferences with deterministic shuffling."""
    active_seed_map = DOMAIN_SEEDS if seed_map is None else seed_map
    rng = random.Random(seed)
    records: list[ExpandedPreferenceRecord] = []

    for domain, topic in iter_topics(active_seed_map):
        query = topic["query"]
        scores = [0, 1, 2, 3]

        for low_idx, score_low in enumerate(scores):
            for score_high in scores[low_idx + 1 :]:
                doc_high = topic["docs"][score_high]
                doc_low = topic["docs"][score_low]
                confidence = (score_high - score_low) / 3.0
                if rng.random() < 0.5:
                    records.append(
                        _build_preference_record(
                            query=query,
                            doc_a=doc_high,
                            doc_b=doc_low,
                            preferred="A",
                            confidence=confidence,
                            domain=domain,
                        )
                    )
                else:
                    records.append(
                        _build_preference_record(
                            query=query,
                            doc_a=doc_low,
                            doc_b=doc_high,
                            preferred="B",
                            confidence=confidence,
                            domain=domain,
                        )
                    )

        other_topics = [
            candidate for candidate in active_seed_map[domain] if candidate["query"] != query
        ]
        for other_topic in other_topics:
            if rng.random() < 0.5:
                records.append(
                    _build_preference_record(
                        query=query,
                        doc_a=topic["docs"][3],
                        doc_b=other_topic["docs"][3],
                        preferred="A",
                        confidence=0.5,
                        domain=domain,
                    )
                )
            else:
                records.append(
                    _build_preference_record(
                        query=query,
                        doc_a=other_topic["docs"][3],
                        doc_b=topic["docs"][3],
                        preferred="B",
                        confidence=0.5,
                        domain=domain,
                    )
                )

            for higher_score, lower_score in ((3, 2), (2, 1)):
                if rng.random() < 0.5:
                    records.append(
                        _build_preference_record(
                            query=query,
                            doc_a=topic["docs"][higher_score],
                            doc_b=topic["docs"][lower_score],
                            preferred="A",
                            confidence=0.33,
                            domain=domain,
                        )
                    )
                else:
                    records.append(
                        _build_preference_record(
                            query=query,
                            doc_a=topic["docs"][lower_score],
                            doc_b=topic["docs"][higher_score],
                            preferred="B",
                            confidence=0.33,
                            domain=domain,
                        )
                    )

    yield from limited_shuffle(records, limit=target_count, rng=rng)


def generate_expanded_preferences(
    target_count: int = 5000,
    seed: int = 42,
) -> list[ExpandedPreferenceRecord]:
    """Generate an expanded preference dataset with balanced A/B labels.

    Args:
        target_count: Number of preference records.
        seed: Random seed.

    Returns:
        List of ExpandedPreferenceRecord dicts.
    """
    """Generate an expanded preference dataset with balanced A/B labels."""
    return list(iter_expanded_preferences(target_count=target_count, seed=seed))
