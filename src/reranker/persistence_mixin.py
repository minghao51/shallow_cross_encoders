"""SaveableReranker mixin for DRY save/load via the persistence layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reranker.persistence import save_safe, try_load_safe_or_warn
from reranker.utils import load_pickle

MAX_QUERY_LENGTH = 10000
MAX_DOCS = 10000


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

    def _require_fitted(self, strategy_name: str | None = None) -> None:
        if not getattr(self, "is_fitted", False):
            name = strategy_name or type(self).__name__
            from reranker.protocols import NotFittedError

            raise NotFittedError(f"{name} is not fitted. Call fit() or load() first.")

    @staticmethod
    def _validate_inputs(query: str, docs: list[str]) -> None:
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds max length {MAX_QUERY_LENGTH}")
        if len(docs) > MAX_DOCS:
            raise ValueError(f"Too many documents ({len(docs)} > {MAX_DOCS})")
