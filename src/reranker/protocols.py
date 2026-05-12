"""Protocols and data types implemented by all ranking strategies.

Defines the core interfaces that every reranker, adapter, and
persistence-aware component must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from reranker.persistence import save_safe, try_load_safe_or_warn
from reranker.utils import load_pickle


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


@runtime_checkable
class HeuristicAdapter(Protocol):
    """Protocol for domain-specific scalar feature adapters."""

    def compute(self, query: str, doc: str) -> dict[str, float]: ...


@runtime_checkable
class BaseReranker(Protocol):
    """Common contract every ranking strategy must satisfy."""

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]: ...


class SaveableReranker:
    """Base class providing DRY save/load via the persistence layer.

    Subclasses define ``_artifact_type`` and override
    ``_save_metadata()`` / ``_save_weights()`` to control what gets
    serialized. The concrete ``save()`` method handles the rest.

    Each subclass still provides its own ``load()`` classmethod since
    reconstruction logic varies per strategy.
    """

    _artifact_type: str = ""

    def _save_metadata(self) -> dict[str, Any]:
        return {
            "embedder_model_name": getattr(getattr(self, "embedder", None), "model_name", "unknown")
        }

    def _save_weights(self) -> dict[str, Any]:
        return {}

    def save(self, path: str | Path) -> None:
        save_safe(
            path,
            artifact_type=self._artifact_type,
            metadata=self._save_metadata(),
            weights=self._save_weights(),
        )

    @staticmethod
    def _load_payload(path: str | Path, expected_type: str) -> dict[str, Any]:
        return try_load_safe_or_warn(
            path,
            expected_type=expected_type,
            legacy_loader=load_pickle,
        )
