"""Type definitions for expanded offline dataset generation."""

from __future__ import annotations

from typing import Literal, TypedDict


class TopicDocs(TypedDict):
    """A topic with a query and docs mapped by relevance score."""

    query: str
    docs: dict[int, str]


ExpandedSeedMap = dict[str, list[TopicDocs]]


class ExpandedPairRecord(TypedDict):
    """A generated pair record in the expanded dataset."""

    query: str
    doc: str
    score: int
    domain: str


class ExpandedPreferenceRecord(TypedDict):
    """A generated preference record in the expanded dataset."""

    query: str
    doc_a: str
    doc_b: str
    preferred: Literal["A", "B"]
    confidence: float
    domain: str


class ExpandedContradictionRecord(TypedDict):
    """A generated contradiction or control record in the expanded dataset."""

    subject: str
    doc_a: str
    doc_b: str
    contradicted_field: str
    value_a: str
    value_b: str
    is_contradiction: bool
