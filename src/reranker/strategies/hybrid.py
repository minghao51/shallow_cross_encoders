"""Hybrid fusion reranker combining semantic and lexical features."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from cachetools import LRUCache
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from reranker.config import get_settings
from reranker.deps import check_xgboost
from reranker.embedder import Embedder
from reranker.lexical import BM25Engine
from reranker.persistence import is_legacy_pickle_allowed
from reranker.protocols import (
    EmbedderProtocol,
    HeuristicAdapter,
    RankedDoc,
    SaveableReranker,
)
from reranker.strategies.hybrid_features import HybridFeatureBuilder
from reranker.strategies.hybrid_training import (
    HybridTrainer,
    WeightingMode,
    auto_label_queries,
)
from reranker.strategies.meta_router import DEFAULT_PROFILE, WEIGHT_PROFILES, MetaRouter
from reranker.utils import (
    build_artifact_metadata,
    rank_docs,
    read_json,
    validate_artifact_metadata,
    write_json,
)

__all__ = [
    "HybridFusionReranker",
    "WeightingMode",
]

_BM25_CACHE_MAX_SIZE = 128
_BM25_CACHE_THRESHOLD = 500


def _is_xgboost_model(model: Any) -> bool:
    return model.__class__.__module__.startswith("xgboost")


def _make_model(regress: bool = False, random_state: int | None = None) -> Any:
    settings = get_settings()
    resolved_random_state = settings.hybrid.random_state if random_state is None else random_state
    xgb_module, _ = check_xgboost()
    if xgb_module is not None:
        cls = xgb_module.XGBRegressor if regress else xgb_module.XGBClassifier
        kwargs: dict[str, Any] = {
            "n_estimators": settings.hybrid.xgb_n_estimators,
            "max_depth": settings.hybrid.xgb_max_depth,
            "learning_rate": settings.hybrid.xgb_learning_rate,
            "subsample": settings.hybrid.xgb_subsample,
            "colsample_bytree": settings.hybrid.xgb_colsample_bytree,
            "random_state": resolved_random_state,
        }
        if regress:
            kwargs["objective"] = "reg:squarederror"
        else:
            kwargs["eval_metric"] = "logloss"
        return cls(**kwargs)
    return (
        GradientBoostingRegressor(random_state=resolved_random_state)
        if regress
        else GradientBoostingClassifier(random_state=resolved_random_state)
    )


def _make_classifier(random_state: int | None = None) -> Any:
    return _make_model(regress=False, random_state=random_state)


def _make_regressor(random_state: int | None = None) -> Any:
    return _make_model(regress=True, random_state=random_state)


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
        target = Path(path)
        if target.suffix == ".json":
            from xgboost import XGBClassifier

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

            allow_legacy_pickle = is_legacy_pickle_allowed()
            if not allow_legacy_pickle:
                raise RuntimeError(
                    "Legacy byte-encoded MetaRouter loading is disabled by default. "
                    "Set RERANKER_ALLOW_LEGACY_PICKLE=1 to load legacy artifacts."
                )
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


class HybridFusionReranker(SaveableReranker):
    """Reranker that fuses semantic similarity and lexical (BM25) features."""

    _artifact_type = "hybrid_reranker"

    def __init__(
        self,
        adapters: list[HeuristicAdapter] | None = None,
        embedder: EmbedderProtocol | None = None,
        random_state: int | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.adapters = adapters or []
        self.model = _make_classifier(random_state=random_state)
        self.model_backend = "xgboost" if _is_xgboost_model(self.model) else "sklearn"
        self.is_fitted = False
        self._router: MetaRouter | None = None
        self._bm25_cache: LRUCache = LRUCache(maxsize=_BM25_CACHE_MAX_SIZE)
        self._cached_feature_map: dict[str, int | None] | None = None

        self._feature_builder = HybridFeatureBuilder(self.embedder, self.adapters)
        self._trainer = HybridTrainer(self.embedder, self._feature_builder)
        self._persistence = HybridPersistence(self)

    @property
    def _feature_registry(self) -> dict[str, int]:
        return self._feature_builder.feature_registry

    @_feature_registry.setter
    def _feature_registry(self, value: dict[str, int]) -> None:
        self._feature_builder.feature_registry = value

    def _init_feature_registry(self, adapter_names: list[str] | None = None) -> None:
        self._feature_builder.init_feature_registry(adapter_names)

    @property
    def feature_names_(self) -> list[str]:
        return self._feature_builder.feature_names

    def _build_features(
        self,
        query: str,
        docs: list[str],
        *,
        bm25: BM25Engine | None = None,
        query_vec: np.ndarray | None = None,
        d_vecs: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._feature_builder.build_features(
            query, docs, bm25=bm25, query_vec=query_vec, d_vecs=d_vecs, is_fitted=self.is_fitted
        )

    def _register_adapter_feature_names(self, query: str, docs: list[str]) -> None:
        self._feature_builder.register_adapter_feature_names(query, docs)

    def fit(
        self, queries: list[str], doc_as: list[str], doc_bs: list[str], labels: list[int]
    ) -> HybridFusionReranker:
        result = self._trainer.fit(
            queries,
            doc_as,
            doc_bs,
            labels,
            is_fitted=self.is_fitted,
            make_classifier=lambda: _make_classifier(
                random_state=get_settings().hybrid.random_state
            ),
            is_xgboost=_is_xgboost_model,
        )
        self.model = result.model
        self.model_backend = result.model_backend
        self.is_fitted = True
        return self

    def fit_pointwise(
        self,
        queries: list[str],
        docs: list[str],
        scores: list[float],
        use_regression: bool = True,
    ) -> HybridFusionReranker:
        result = self._trainer.fit_pointwise(
            queries,
            docs,
            scores,
            is_fitted=self.is_fitted,
            make_classifier=lambda: _make_classifier(
                random_state=get_settings().hybrid.random_state
            ),
            make_regressor=lambda: _make_regressor(random_state=get_settings().hybrid.random_state),
            is_xgboost=_is_xgboost_model,
            use_regression=use_regression,
        )
        if result is None:
            return self
        self.model = result.model
        self.model_backend = result.model_backend
        self.is_fitted = True
        if result.router is not None:
            self._router = result.router
        return self

    def _auto_label_queries(
        self, queries: list[str], docs: list[str], scores: list[float]
    ) -> list[int]:
        return auto_label_queries(self.embedder, queries, docs, scores)

    def _resolve_weights(self, query: str) -> dict[str, float]:
        settings = get_settings().hybrid
        weighting_mode = WeightingMode(settings.weighting_mode)

        if (
            weighting_mode == WeightingMode.META_ROUTER
            and self._router is not None
            and self._router.is_fitted
        ):
            weights = self._router.get_weights(query)
            fallback = WEIGHT_PROFILES[DEFAULT_PROFILE]
            return {key: weights.get(key, fallback[key]) for key in fallback}

        if weighting_mode == WeightingMode.LEARNED:
            return {}

        return {
            "sem_score": settings.weights.sem_score,
            "bm25_score": settings.weights.bm25_score,
            "token_overlap_ratio": settings.weights.token_overlap_ratio,
            "query_coverage_ratio": settings.weights.query_coverage_ratio,
            "shared_token_char_sum": settings.weights.shared_token_char_sum,
            "exact_phrase_match": settings.weights.exact_phrase_match,
            "keyword_hit_rate": settings.weights.keyword_hit_rate,
        }

    _FEATURE_WEIGHT_KEYS = (
        "sem_score",
        "bm25_score",
        "token_overlap_ratio",
        "query_coverage_ratio",
        "shared_token_char_sum",
        "exact_phrase_match",
        "keyword_hit_rate",
    )

    def _apply_weights(
        self,
        X: np.ndarray,
        weight_map: dict[str, float],
        query: str,
    ) -> np.ndarray:
        blended = np.zeros(X.shape[0], dtype=np.float32)
        if self._cached_feature_map is None:
            self._cached_feature_map = {
                name: self._feature_registry.get(name) for name in self._FEATURE_WEIGHT_KEYS
            }
        for name, weight in weight_map.items():
            idx = self._cached_feature_map.get(name)
            if idx is None or weight == 0.0:
                continue
            if name == "shared_token_char_sum":
                norm = max(float(len(query.replace("_", " ").split())), 1.0)
                blended += weight * (X[:, idx] / norm)
            else:
                blended += weight * X[:, idx]
        return blended

    @staticmethod
    def _model_predict(model: Any, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] > 1:
                return np.asarray(probs[:, 1], dtype=np.float32)
            return np.asarray(probs[:, 0], dtype=np.float32)
        if hasattr(model, "predict"):
            return np.asarray(model.predict(X), dtype=np.float32)
        return np.zeros(X.shape[0], dtype=np.float32)

    def score(
        self,
        query: str,
        docs: list[str],
        *,
        bm25: BM25Engine | None = None,
        query_vec: np.ndarray | None = None,
        d_vecs: np.ndarray | None = None,
    ) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)
        self._require_fitted("HybridFusionReranker")
        X = self._build_features(query, docs, bm25=bm25, query_vec=query_vec, d_vecs=d_vecs)
        weighting_mode = WeightingMode(get_settings().hybrid.weighting_mode)

        weight_map = self._resolve_weights(query)
        if weight_map:
            blended = self._apply_weights(X, weight_map, query)
        else:
            blended = np.zeros(X.shape[0], dtype=np.float32)

        if weighting_mode == WeightingMode.LEARNED:
            return self._model_predict(self.model, X)

        model_scores = self._model_predict(self.model, X)
        return np.asarray((model_scores + blended) / 2.0, dtype=np.float32)

    def rerank(
        self,
        query: str,
        docs: list[str],
        *,
        bm25: BM25Engine | None = None,
        query_vec: np.ndarray | None = None,
        d_vecs: np.ndarray | None = None,
    ) -> list[RankedDoc]:
        self._validate_inputs(query, docs)
        cache_key = tuple(docs)
        lexical = bm25 or self._bm25_cache.get(cache_key)
        if lexical is None and docs:
            lexical = BM25Engine(tokenize_fn=self.embedder.tokenize)
            lexical.fit(docs)
            if len(docs) <= _BM25_CACHE_THRESHOLD:
                self._bm25_cache[cache_key] = lexical
        scores = self.score(query, docs, bm25=lexical, query_vec=query_vec, d_vecs=d_vecs)
        return rank_docs(docs, scores, "hybrid")

    def rerank_batch(
        self,
        queries: list[str],
        docs_list: list[list[str]],
    ) -> list[list[RankedDoc]]:
        if not queries:
            return []
        if len(queries) != len(docs_list):
            raise ValueError("queries and docs_list must have the same length for rerank_batch.")

        query_vectors = self.embedder.encode(queries)

        all_docs = list({doc for docs in docs_list for doc in docs})
        if all_docs:
            all_doc_vectors = self.embedder.encode(all_docs)
            doc_vec_map = {doc: vec for doc, vec in zip(all_docs, all_doc_vectors, strict=True)}
        else:
            doc_vec_map = {}

        results: list[list[RankedDoc]] = []
        for q_idx, (query, docs) in enumerate(zip(queries, docs_list, strict=True)):
            if not docs:
                results.append([])
                continue
            cache_key = tuple(docs)
            lexical = self._bm25_cache.get(cache_key)
            if lexical is None:
                lexical = BM25Engine(tokenize_fn=self.embedder.tokenize)
                lexical.fit(docs)
                if len(docs) <= _BM25_CACHE_THRESHOLD:
                    self._bm25_cache[cache_key] = lexical
            q_vec = query_vectors[q_idx]
            d_vecs = np.stack([doc_vec_map[doc] for doc in docs])
            results.append(self.rerank(query, docs, bm25=lexical, query_vec=q_vec, d_vecs=d_vecs))
        return results

    def _save_metadata(self) -> dict:
        return self._persistence._save_metadata()

    def _save_weights(self) -> dict:
        return self._persistence._save_weights()

    def save(self, path: str | Path) -> None:
        self._persistence.save(path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        adapters: list[HeuristicAdapter] | None = None,
        embedder: Embedder | None = None,
    ) -> HybridFusionReranker:
        return HybridPersistence.load(path, adapters=adapters, embedder=embedder)
