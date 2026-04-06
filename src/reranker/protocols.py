from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(slots=True)
class RankedDoc:
    doc: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class HeuristicAdapter(Protocol):
    """Inject domain-specific scalar features into feature construction."""

    def compute(self, query: str, doc: str) -> dict[str, float]: ...


@runtime_checkable
class BaseReranker(Protocol):
    """Common contract every ranking strategy resolves to."""

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]: ...
