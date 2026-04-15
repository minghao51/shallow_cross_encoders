from __future__ import annotations

from typing import Literal, TypedDict


class TopicDocs(TypedDict):
    query: str
    docs: dict[int, str]


ExpandedSeedMap = dict[str, list[TopicDocs]]


class ExpandedPairRecord(TypedDict):
    query: str
    doc: str
    score: int
    domain: str


class ExpandedPreferenceRecord(TypedDict):
    query: str
    doc_a: str
    doc_b: str
    preferred: Literal["A", "B"]
    confidence: float
    domain: str


class ExpandedContradictionRecord(TypedDict):
    subject: str
    doc_a: str
    doc_b: str
    contradicted_field: str
    value_a: str
    value_b: str
    is_contradiction: bool
