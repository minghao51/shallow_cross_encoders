"""Training logic for hybrid fusion reranker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier

from reranker.config import get_settings
from reranker.protocols import EmbedderProtocol
from reranker.strategies.hybrid_features import HybridFeatureBuilder

_AUTO_LABEL_SCORE_GAP_RATIO = 0.1


class WeightingMode(StrEnum):
    STATIC = "static"
    LEARNED = "learned"
    META_ROUTER = "meta_router"


@dataclass
class FitResult:
    """Result of a training operation."""

    model: Any
    model_backend: str
    router: Any = None


def auto_label_queries(
    embedder: EmbedderProtocol,
    queries: list[str],
    docs: list[str],
    scores: list[float],
) -> list[int]:
    n_categories = max(1, min(get_settings().meta_router.n_categories, 3))
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
        from reranker import lexical as lexical_module

        bm25 = lexical_module.BM25Engine(tokenize_fn=embedder.tokenize)
        bm25.fit(group_docs)
        bm25_scores = bm25.score(query)
        query_vec = query_embedding_cache.setdefault(query, embedder.encode([query])[0])
        doc_vectors = embedder.encode(group_docs)
        sem_score = float(
            embedder.similarity(
                query_vec,
                doc_vectors[int(np.argmax(group_scores))],
            )
        )
        bm25_best = float(bm25_scores.max()) if bm25_scores.size > 0 else 0.0
        if n_categories >= 3:
            score_gap = abs(bm25_best - sem_score)
            score_scale = max(abs(bm25_best), abs(sem_score), 1.0)
            if score_gap <= _AUTO_LABEL_SCORE_GAP_RATIO * score_scale:
                category_by_query[query] = 2
                continue
        category_by_query[query] = 0 if bm25_best > sem_score else 1
    return [category_by_query.get(query, 0) for query in queries]


class HybridTrainer:
    """Encapsulates pairwise and pointwise training for the hybrid reranker.

    Receives explicit dependencies (embedder, feature_builder) instead of
    the full reranker, breaking the circular delegation pattern.
    """

    def __init__(self, embedder: EmbedderProtocol, feature_builder: HybridFeatureBuilder) -> None:
        self.embedder = embedder
        self.feature_builder = feature_builder

    def fit(
        self,
        queries: list[str],
        doc_as: list[str],
        doc_bs: list[str],
        labels: list[int],
        *,
        is_fitted: bool,
        make_classifier: Any,
        is_xgboost: Any,
    ) -> FitResult:
        fb = self.feature_builder
        fb.init_feature_registry()
        for query, doc_a, doc_b in zip(queries, doc_as, doc_bs, strict=False):
            fb.register_adapter_feature_names(query, [doc_a, doc_b])

        samples = []
        for query, doc_a, doc_b in zip(queries, doc_as, doc_bs, strict=False):
            features_a = fb.build_features(query, [doc_a], is_fitted=is_fitted)[0]
            features_b = fb.build_features(query, [doc_b], is_fitted=is_fitted)[0]
            samples.append(features_a - features_b)

        if not samples:
            fb.init_feature_registry()
            feature_count = len(fb.feature_registry)
            samples = [np.zeros(feature_count, dtype=np.float32)]
            labels = [0]
        X = np.vstack(samples)
        y = np.asarray(labels[: len(samples)], dtype=np.int32)
        model: Any
        if len(set(y.tolist())) < 2:
            model = DummyClassifier(strategy="constant", constant=int(y[0]))
        else:
            model = make_classifier()
        model.fit(X, y)
        backend = "xgboost" if is_xgboost(model) else "sklearn"
        return FitResult(model=model, model_backend=backend)

    def fit_pointwise(
        self,
        queries: list[str],
        docs: list[str],
        scores: list[float],
        *,
        is_fitted: bool,
        make_classifier: Any,
        make_regressor: Any,
        is_xgboost: Any,
        use_regression: bool = True,
    ) -> FitResult | None:
        fb = self.feature_builder
        fb.init_feature_registry()
        for query, doc in zip(queries, docs, strict=False):
            fb.register_adapter_feature_names(query, [doc])

        samples = [
            fb.build_features(query, [doc], is_fitted=is_fitted)[0]
            for query, doc in zip(queries, docs, strict=False)
        ]
        if not samples:
            return None

        X = np.vstack(samples)
        y = np.asarray(scores[: len(samples)], dtype=np.float32)
        model: Any
        backend: str

        if use_regression:
            model = make_regressor()
            model.fit(X, y)
            backend = "xgboost" if is_xgboost(model) else "sklearn"
        else:
            model = make_classifier()
            threshold = np.median(y)
            y_binary = (y >= threshold).astype(int)
            model.fit(X, y_binary)
            backend = "xgboost" if is_xgboost(model) else "sklearn"

        router = None
        settings = get_settings()

        if (
            settings.meta_router.enabled
            and WeightingMode(settings.hybrid.weighting_mode) == WeightingMode.META_ROUTER
        ):
            from reranker.strategies.meta_router import MetaRouter

            router = MetaRouter(embedder=self.embedder)
            router_categories = auto_label_queries(self.embedder, queries, docs, scores)
            router.fit(queries, router_categories)

        return FitResult(model=model, model_backend=backend, router=router)
