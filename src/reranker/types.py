"""Core data types shared across the reranking toolkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RankedDoc:
    """A single ranked document result.

    Attributes:
        doc: The document text.
        score: Relevance score (higher = more relevant).
        rank: 1-based rank position.
        metadata: Arbitrary metadata (strategy name, stage info, etc.).
    """

    doc: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
