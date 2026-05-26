"""SaveableReranker mixin for DRY save/load via the persistence layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reranker.persistence import save_safe, try_load_safe_or_warn
from reranker.utils import load_pickle


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
