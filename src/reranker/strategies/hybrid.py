"""Hybrid fusion reranker combining semantic and lexical features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reranker.config import get_settings
from reranker.embedder import Embedder
from reranker.lexical import BM25Engine
from reranker.protocols import (
    EmbedderProtocol,
    HeuristicAdapter,
    NotFittedError,
    RankedDoc,
    SaveableReranker,
)
from reranker.strategies.hybrid_features import (
    WeightingMode,
    _is_xgboost_model,
    _make_classifier,
    _make_model,  # noqa: F401
    _make_regressor,  # noqa: F401
)
from reranker.strategies.hybrid_persistence import HybridPersistence
from reranker.strategies.hybrid_training import HybridTrainer
from reranker.strategies.meta_router import MetaRouter
from reranker.utils import rank_docs

__all__ = [
    "HybridFusionReranker",
    "WeightingMode",
    "_is_xgboost_model",
    "_make_classifier",
    "_make_model",
    "_make_regressor",
]


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

        from reranker.strategies.hybrid_features import HybridFeatureBuilder

        self._feature_builder = HybridFeatureBuilder(self.embedder, self.adapters)
        self._trainer = HybridTrainer(self)
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
        return self._trainer.fit(queries, doc_as, doc_bs, labels)

    def fit_pointwise(
        self,
        queries: list[str],
        docs: list[str],
        scores: list[float],
        use_regression: bool = True,
    ) -> HybridFusionReranker:
        return self._trainer.fit_pointwise(queries, docs, scores, use_regression)

    def _auto_label_queries(
        self, queries: list[str], docs: list[str], scores: list[float]
    ) -> list[int]:
        return self._trainer._auto_label_queries(queries, docs, scores)

    def _resolve_weights(self, query: str) -> dict[str, float]:
        settings = get_settings().hybrid
        weighting_mode = WeightingMode(settings.weighting_mode)

        if (
            weighting_mode == WeightingMode.META_ROUTER
            and self._router is not None
            and self._router.is_fitted
        ):
            weights = self._router.get_weights(query)
            return {
                "sem_score": weights.get("sem_score", 0.25),
                "bm25_score": weights.get("bm25_score", 0.20),
                "token_overlap_ratio": weights.get("token_overlap_ratio", 0.15),
                "query_coverage_ratio": weights.get("query_coverage_ratio", 0.20),
                "shared_token_char_sum": weights.get("shared_token_char_sum", 0.10),
                "exact_phrase_match": weights.get("exact_phrase_match", 0.10),
                "keyword_hit_rate": weights.get("keyword_hit_rate", 0.05),
            }

        if weighting_mode == WeightingMode.LEARNED:
            return {}

        return {
            "sem_score": settings.weight_sem_score,
            "bm25_score": settings.weight_bm25_score,
            "token_overlap_ratio": settings.weight_token_overlap,
            "query_coverage_ratio": settings.weight_query_coverage,
            "shared_token_char_sum": settings.weight_shared_char,
            "exact_phrase_match": settings.weight_exact_phrase,
            "keyword_hit_rate": settings.weight_keyword_hit,
        }

    def _apply_weights(
        self,
        X: np.ndarray,
        weight_map: dict[str, float],
        query: str,
    ) -> np.ndarray:
        blended = np.zeros(X.shape[0], dtype=np.float32)
        feature_map = {
            "sem_score": self._feature_registry.get("sem_score"),
            "bm25_score": self._feature_registry.get("bm25_score"),
            "token_overlap_ratio": self._feature_registry.get("token_overlap_ratio"),
            "query_coverage_ratio": self._feature_registry.get("query_coverage_ratio"),
            "shared_token_char_sum": self._feature_registry.get("shared_token_char_sum"),
            "exact_phrase_match": self._feature_registry.get("exact_phrase_match"),
            "keyword_hit_rate": self._feature_registry.get("keyword_hit_rate"),
        }
        for name, weight in weight_map.items():
            idx = feature_map.get(name)
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
        X = self._build_features(query, docs, bm25=bm25, query_vec=query_vec, d_vecs=d_vecs)
        weighting_mode = WeightingMode(get_settings().hybrid.weighting_mode)

        weight_map = self._resolve_weights(query)
        if weight_map:
            blended = self._apply_weights(X, weight_map, query)
        else:
            blended = np.zeros(X.shape[0], dtype=np.float32)

        if not self.is_fitted:
            raise NotFittedError("HybridFusionReranker is not fitted. Call fit() or load() first.")

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
        lexical = bm25
        if lexical is None and docs:
            lexical = BM25Engine(tokenize_fn=self.embedder.tokenize)
            lexical.fit(docs)
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
            lexical = BM25Engine(tokenize_fn=self.embedder.tokenize)
            lexical.fit(docs)
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
