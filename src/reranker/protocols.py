from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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


@runtime_checkable
class TrainableReranker(Protocol):
    """Training contract for rerankers that support pairwise or pointwise learning."""

    def fit(
        self,
        queries: list[str],
        doc_as: list[str],
        doc_bs: list[str],
        labels: list[int],
    ) -> Any: ...

    def fit_pointwise(
        self,
        queries: list[str],
        docs: list[str],
        scores: list[float],
        use_regression: bool = True,
    ) -> Any: ...

    def save(self, path: str | Path) -> None: ...


@runtime_checkable
class SaveableReranker(Protocol):
    """Persistence contract for rerankers."""

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> Any: ...
