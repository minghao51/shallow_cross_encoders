"""Persistence (save/load) logic for the hybrid fusion reranker."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import joblib

from reranker.config import get_settings
from reranker.embedder import Embedder
from reranker.utils import (
    build_artifact_metadata,
    read_json,
    validate_artifact_metadata,
    write_json,
)

if TYPE_CHECKING:
    from reranker.protocols import HeuristicAdapter
    from reranker.strategies.hybrid import HybridFusionReranker


class HybridPersistence:
    """Handles saving and loading of the hybrid fusion reranker."""

    def __init__(self, reranker: HybridFusionReranker) -> None:
        self._reranker = reranker

    def _save_metadata(self) -> dict:
        reranker = self._reranker
        adapter_types = [type(adapter).__name__ for adapter in reranker.adapters]
        return {
            "embedder_model_name": reranker.embedder.model_name,
            "feature_names": reranker.feature_names_,
            "feature_registry": reranker._feature_builder.feature_registry,
            "adapter_types": adapter_types,
            "has_router": reranker._router is not None and reranker._router.is_fitted,
        }

    def _save_weights(self) -> dict:
        reranker = self._reranker
        router_payload = None
        if reranker._router is not None and reranker._router.is_fitted:
            router_payload = reranker._router
        return {"model": reranker.model, "router": router_payload}

    def save(self, path: str | Path) -> None:
        reranker = self._reranker
        target = Path(path)
        if reranker.model_backend == "xgboost" and target.suffix == ".json":
            reranker.model.save_model(str(target))
            meta = build_artifact_metadata(
                "hybrid_reranker",
                format_name="xgboost-json",
                embedder_model_name=reranker.embedder.model_name,
                extra=self._save_metadata(),
            )
            write_json(target.with_suffix(".meta.json"), meta)

            if reranker._router is not None and reranker._router.is_fitted:
                joblib.dump(reranker._router, target.with_suffix(".router.joblib"))
            return
        from reranker.persistence import save_safe

        save_safe(
            target,
            artifact_type=reranker._artifact_type,
            metadata=self._save_metadata(),
            weights=self._save_weights(),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        adapters: list[HeuristicAdapter] | None = None,
        embedder: Embedder | None = None,
    ) -> HybridFusionReranker:
        from reranker.protocols import SaveableReranker
        from reranker.strategies.hybrid import HybridFusionReranker
        from reranker.strategies.meta_router import MetaRouter

        target = Path(path)
        if target.suffix == ".json":
            from xgboost import XGBClassifier  # type: ignore

            meta_path = target.with_suffix(".meta.json")
            payload = read_json(meta_path)
            validate_artifact_metadata(
                payload,
                expected_type=HybridFusionReranker._artifact_type,
                expected_formats={"xgboost-json"},
            )
            instance = HybridFusionReranker(
                adapters=adapters,
                embedder=embedder or Embedder(payload["embedder_model_name"]),
            )
            instance.model = XGBClassifier()
            instance.model.load_model(str(target))
            instance.model_backend = "xgboost"
            instance._feature_builder.feature_registry = dict(payload.get("feature_registry", {}))
            instance.is_fitted = True
            router_path = target.with_suffix(".router.joblib")
            if payload.get("has_router") and router_path.exists():
                loaded_router = joblib.load(router_path)
                if isinstance(loaded_router, MetaRouter):
                    instance._router = loaded_router
                else:
                    raise TypeError(
                        f"Expected MetaRouter in {router_path}, got {type(loaded_router).__name__}."
                    )
            return instance

        payload = SaveableReranker._load_payload(
            target, expected_type=HybridFusionReranker._artifact_type
        )
        embedder_model_name = payload.get("embedder_model_name")
        instance = HybridFusionReranker(
            adapters=adapters,
            embedder=embedder
            or Embedder(
                str(embedder_model_name)
                if embedder_model_name is not None
                else get_settings().embedder.model_name
            ),
        )
        instance.model = payload["model"]
        instance.model_backend = (
            "xgboost" if instance.model.__class__.__module__.startswith("xgboost") else "sklearn"
        )
        instance._feature_builder.feature_registry = dict(payload.get("feature_registry", {}))
        instance.is_fitted = True
        router_data = payload.get("router")
        if isinstance(router_data, MetaRouter):
            instance._router = router_data
        elif isinstance(router_data, bytes):
            import pickle

            warnings.warn(
                "Loading legacy byte-encoded MetaRouter payload. Re-save model to migrate.",
                UserWarning,
                stacklevel=2,
            )
            loaded_router = pickle.loads(router_data)
            if isinstance(loaded_router, MetaRouter):
                instance._router = loaded_router
            else:
                raise TypeError(
                    f"Expected MetaRouter in legacy payload, got {type(loaded_router).__name__}."
                )
        return instance
