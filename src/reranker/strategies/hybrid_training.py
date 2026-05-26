"""Training logic for the hybrid fusion reranker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.dummy import DummyClassifier

from reranker.config import get_settings
from reranker.strategies.hybrid_features import (
    WeightingMode,
    _is_xgboost_model,
    _make_classifier,
    _make_regressor,
)

if TYPE_CHECKING:
    from reranker.strategies.hybrid import HybridFusionReranker


class HybridTrainer:
    """Encapsulates pairwise and pointwise training for the hybrid reranker."""

    def __init__(self, reranker: HybridFusionReranker) -> None:
        self._reranker = reranker

    def fit(
        self, queries: list[str], doc_as: list[str], doc_bs: list[str], labels: list[int]
    ) -> HybridFusionReranker:
        reranker = self._reranker
        fb = reranker._feature_builder
        fb.init_feature_registry()
        for query, doc_a, doc_b in zip(queries, doc_as, doc_bs, strict=False):
            fb.register_adapter_feature_names(query, [doc_a, doc_b])

        samples = []
        for query, doc_a, doc_b in zip(queries, doc_as, doc_bs, strict=False):
            features_a = fb.build_features(query, [doc_a], is_fitted=reranker.is_fitted)[0]
            features_b = fb.build_features(query, [doc_b], is_fitted=reranker.is_fitted)[0]
            samples.append(features_a - features_b)

        if not samples:
            fb.init_feature_registry()
            feature_count = len(fb.feature_registry)
            samples = [np.zeros(feature_count, dtype=np.float32)]
            labels = [0]
        X = np.vstack(samples)
        y = np.asarray(labels[: len(samples)], dtype=np.int32)
        if len(set(y.tolist())) < 2:
            reranker.model = DummyClassifier(strategy="constant", constant=int(y[0]))
        reranker.model.fit(X, y)
        reranker.model_backend = "xgboost" if _is_xgboost_model(reranker.model) else "sklearn"
        reranker.is_fitted = True
        return reranker

    def fit_pointwise(
        self,
        queries: list[str],
        docs: list[str],
        scores: list[float],
        use_regression: bool = True,
    ) -> HybridFusionReranker:
        reranker = self._reranker
        fb = reranker._feature_builder
        fb.init_feature_registry()
        for query, doc in zip(queries, docs, strict=False):
            fb.register_adapter_feature_names(query, [doc])

        samples = [
            fb.build_features(query, [doc], is_fitted=reranker.is_fitted)[0]
            for query, doc in zip(queries, docs, strict=False)
        ]
        if not samples:
            return reranker

        X = np.vstack(samples)
        y = np.asarray(scores[: len(samples)], dtype=np.float32)

        if use_regression:
            reranker.model = _make_regressor(random_state=get_settings().hybrid.random_state)
            reranker.model.fit(X, y)
            reranker.model_backend = "xgboost" if _is_xgboost_model(reranker.model) else "sklearn"
        else:
            reranker.model = _make_classifier(random_state=get_settings().hybrid.random_state)
            threshold = np.median(y)
            y_binary = (y >= threshold).astype(int)
            reranker.model.fit(X, y_binary)

        reranker.is_fitted = True

        settings = get_settings()
        if (
            settings.meta_router.enabled
            and WeightingMode(settings.hybrid.weighting_mode) == WeightingMode.META_ROUTER
        ):
            from reranker.strategies.meta_router import MetaRouter

            reranker._router = MetaRouter(embedder=reranker.embedder)
            router_categories = self._reranker._auto_label_queries(queries, docs, scores)
            reranker._router.fit(queries, router_categories)

        return reranker

    def _auto_label_queries(
        self, queries: list[str], docs: list[str], scores: list[float]
    ) -> list[int]:
        from reranker.lexical import BM25Engine

        reranker = self._reranker
        router_categories = max(1, min(get_settings().meta_router.n_categories, 3))
        query_groups: dict[str, list[tuple[str, float]]] = {}
        for q, d, s in zip(queries, docs, scores, strict=False):
            query_groups.setdefault(q, []).append((d, s))
        category_by_query: dict[str, int] = {}
        query_embedding_cache: dict[str, np.ndarray] = {}

        for query, group in query_groups.items():
            if len(group) < 2:
                category_by_query[query] = 0
                continue
            group_docs = [d for d, _ in group]
            group_scores = np.array([s for _, s in group], dtype=np.float32)
            bm25 = BM25Engine(tokenize_fn=reranker.embedder.tokenize)
            bm25.fit(group_docs)
            bm25_scores = bm25.score(query)
            query_vec = query_embedding_cache.setdefault(
                query, reranker.embedder.encode([query])[0]
            )
            doc_vectors = reranker.embedder.encode(group_docs)
            sem_score = float(
                reranker.embedder.similarity(
                    query_vec,
                    doc_vectors[int(np.argmax(group_scores))],
                )
            )
            bm25_best = float(bm25_scores.max()) if bm25_scores.size > 0 else 0.0
            if router_categories >= 3:
                score_gap = abs(bm25_best - sem_score)
                score_scale = max(abs(bm25_best), abs(sem_score), 1.0)
                if score_gap <= 0.1 * score_scale:
                    category_by_query[query] = 2
                    continue
            category_by_query[query] = 0 if bm25_best > sem_score else 1
        return [category_by_query.get(query, 0) for query in queries]
