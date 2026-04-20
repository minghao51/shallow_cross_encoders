from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from reranker.config import get_settings
from reranker.deps import check_xgboost
from reranker.embedder import Embedder
from reranker.lexical import BM25Engine
from reranker.protocols import HeuristicAdapter, RankedDoc
from reranker.utils import (
    build_artifact_metadata,
    dump_pickle,
    load_pickle,
    read_json,
    validate_artifact_metadata,
    write_json,
)


def _make_classifier(random_state: int | None = None) -> Any:
    settings = get_settings()
    resolved_random_state = settings.hybrid.random_state if random_state is None else random_state
    xgb_module, _ = check_xgboost()
    if xgb_module is not None:
        return xgb_module.XGBClassifier(
            n_estimators=settings.hybrid.xgb_n_estimators,
            max_depth=settings.hybrid.xgb_max_depth,
            learning_rate=settings.hybrid.xgb_learning_rate,
            subsample=settings.hybrid.xgb_subsample,
            colsample_bytree=settings.hybrid.xgb_colsample_bytree,
            eval_metric="logloss",
            random_state=resolved_random_state,
        )
    return GradientBoostingClassifier(random_state=resolved_random_state)


def _make_regressor(random_state: int | None = None) -> Any:
    settings = get_settings()
    resolved_random_state = settings.hybrid.random_state if random_state is None else random_state
    xgb_module, _ = check_xgboost()
    if xgb_module is not None:
        return xgb_module.XGBRegressor(
            n_estimators=settings.hybrid.xgb_n_estimators,
            max_depth=settings.hybrid.xgb_max_depth,
            learning_rate=settings.hybrid.xgb_learning_rate,
            subsample=settings.hybrid.xgb_subsample,
            colsample_bytree=settings.hybrid.xgb_colsample_bytree,
            objective="reg:squarederror",
            random_state=resolved_random_state,
        )
    return GradientBoostingRegressor(random_state=resolved_random_state)


BASE_FEATURES = [
    "sem_score",
    "bm25_score",
    "vec_norm_diff",
    "token_overlap_ratio",
    "query_coverage_ratio",
    "shared_token_char_sum",
    "exact_phrase_match",
    "query_len",
    "doc_len",
]


@dataclass(slots=True)
class KeywordMatchAdapter:
    """Example adapter that emits a simple term hit-rate signal."""

    tokenize_fn: Callable[[str], list[str]] | None = None

    def compute(self, query: str, doc: str) -> dict[str, float]:
        tokenize = self.tokenize_fn or (lambda t: t.lower().split())
        terms = tokenize(query.lower())
        doc_lower = doc.lower()
        hit_rate = sum(1 for term in terms if term in doc_lower) / max(len(terms), 1)
        return {"keyword_hit_rate": hit_rate}


class HybridFusionReranker:
    def __init__(
        self,
        adapters: list[HeuristicAdapter] | None = None,
        embedder: Embedder | None = None,
        random_state: int | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.adapters = adapters or []
        self.model = _make_classifier(random_state=random_state)
        self.model_backend = (
            "xgboost" if self.model.__class__.__module__.startswith("xgboost") else "sklearn"
        )
        self._feature_registry: dict[str, int] = {}
        self.is_fitted = False

    def _init_feature_registry(self, adapter_names: list[str] | None = None) -> None:
        self._feature_registry = {name: idx for idx, name in enumerate(BASE_FEATURES)}
        if adapter_names:
            for name in adapter_names:
                if name not in self._feature_registry:
                    self._feature_registry[name] = len(self._feature_registry)

    def _get_feature_index(self, name: str) -> int:
        if name not in self._feature_registry:
            self._feature_registry[name] = len(self._feature_registry)
        return self._feature_registry[name]

    def _adapter_feature_names(self, query: str, doc: str) -> list[str]:
        names: list[str] = []
        for adapter in self.adapters:
            names.extend(adapter.compute(query, doc).keys())
        return names

    def _register_adapter_feature_names(self, query: str, docs: list[str]) -> None:
        if not self._feature_registry:
            self._init_feature_registry()
        for doc in docs:
            for name in self._adapter_feature_names(query, doc):
                self._get_feature_index(name)

    def _build_features(
        self, query: str, docs: list[str], *, bm25: BM25Engine | None = None
    ) -> np.ndarray:
        if not docs:
            if not self._feature_registry:
                self._init_feature_registry()
            return np.zeros((0, len(self._feature_registry)), dtype=np.float32)

        q_vec = self.embedder.encode([query])[0]
        d_vecs = self.embedder.encode(docs)
        lexical = bm25
        if lexical is None:
            lexical = BM25Engine(tokenize_fn=self.embedder.tokenize)
            lexical.fit(docs)
        bm25_scores = lexical.score(query)

        rows: list[dict[str, float]] = []

        for idx, doc in enumerate(docs):
            query_tokens = self.embedder.tokenize(query.lower())
            doc_tokens = self.embedder.tokenize(doc.lower())
            query_terms = set(query_tokens)
            doc_terms = set(doc_tokens)
            shared_terms = query_terms & doc_terms
            overlap = len(shared_terms)
            row_dict: dict[str, float] = {
                "sem_score": float(np.dot(q_vec, d_vecs[idx])),
                "bm25_score": float(bm25_scores[idx]) if bm25_scores.size else 0.0,
                "vec_norm_diff": float(np.linalg.norm(q_vec - d_vecs[idx])),
                "token_overlap_ratio": float(overlap / max(len(query_terms | doc_terms), 1)),
                "query_coverage_ratio": float(overlap / max(len(query_terms), 1)),
                "shared_token_char_sum": float(sum(len(term) for term in shared_terms)),
                "exact_phrase_match": float(1.0 if query.lower() in doc.lower() else 0.0),
                "query_len": float(len(self.embedder.tokenize(query))),
                "doc_len": float(len(self.embedder.tokenize(doc))),
            }
            for adapter in self.adapters:
                row_dict.update(adapter.compute(query, doc))
            rows.append(row_dict)

        if not self._feature_registry:
            self._init_feature_registry()
        if not self.is_fitted:
            for row_dict in rows:
                for name in row_dict:
                    self._get_feature_index(name)

        feature_names = list(self._feature_registry.keys())
        n_features = len(feature_names)
        result = np.zeros((len(rows), n_features), dtype=np.float32)
        for i, row_dict in enumerate(rows):
            for name, value in row_dict.items():
                idx_feat = self._feature_registry.get(name)
                if idx_feat is not None and idx_feat < n_features:
                    result[i, idx_feat] = value
        return result

    @property
    def feature_names_(self) -> list[str]:
        return list(self._feature_registry.keys())

    def fit(
        self, queries: list[str], doc_as: list[str], doc_bs: list[str], labels: list[int]
    ) -> HybridFusionReranker:
        self._init_feature_registry()
        for query, doc_a, doc_b in zip(queries, doc_as, doc_bs, strict=False):
            self._register_adapter_feature_names(query, [doc_a, doc_b])

        samples = []
        for query, doc_a, doc_b in zip(queries, doc_as, doc_bs, strict=False):
            features_a = self._build_features(query, [doc_a])[0]
            features_b = self._build_features(query, [doc_b])[0]
            samples.append(features_a - features_b)

        if not samples:
            self._init_feature_registry()
            feature_count = len(self._feature_registry)
            samples = [np.zeros(feature_count, dtype=np.float32)]
            labels = [0]
        X = np.vstack(samples)
        y = np.asarray(labels[: len(samples)], dtype=np.int32)
        if len(set(y.tolist())) < 2:
            self.model = DummyClassifier(strategy="constant", constant=int(y[0]))
        self.model.fit(X, y)
        self.model_backend = (
            "xgboost" if self.model.__class__.__module__.startswith("xgboost") else "sklearn"
        )
        self.is_fitted = True
        return self

    def fit_pointwise(
        self,
        queries: list[str],
        docs: list[str],
        scores: list[float],
        use_regression: bool = True,
    ) -> HybridFusionReranker:
        self._init_feature_registry()
        for query, doc in zip(queries, docs, strict=False):
            self._register_adapter_feature_names(query, [doc])

        samples = [
            self._build_features(query, [doc])[0] for query, doc in zip(queries, docs, strict=False)
        ]
        if not samples:
            return self

        X = np.vstack(samples)
        y = np.asarray(scores[: len(samples)], dtype=np.float32)

        if use_regression:
            self.model = _make_regressor(random_state=get_settings().hybrid.random_state)
            self.model.fit(X, y)
            self.model_backend = (
                "xgboost" if self.model.__class__.__module__.startswith("xgboost") else "sklearn"
            )
        else:
            self.model = _make_classifier(random_state=get_settings().hybrid.random_state)
            threshold = np.median(y)
            y_binary = (y >= threshold).astype(int)
            self.model.fit(X, y_binary)

        self.is_fitted = True
        return self

    def score(self, query: str, docs: list[str], *, bm25: BM25Engine | None = None) -> np.ndarray:
        if not docs:
            return np.zeros(0, dtype=np.float32)
        X = self._build_features(query, docs, bm25=bm25)
        settings = get_settings().hybrid

        sem_idx = self._feature_registry.get("sem_score")
        bm25_idx = self._feature_registry.get("bm25_score")
        overlap_idx = self._feature_registry.get("token_overlap_ratio")
        coverage_idx = self._feature_registry.get("query_coverage_ratio")
        shared_idx = self._feature_registry.get("shared_token_char_sum")
        exact_idx = self._feature_registry.get("exact_phrase_match")
        keyword_idx = self._feature_registry.get("keyword_hit_rate")

        blended = np.zeros(X.shape[0], dtype=np.float32)
        if sem_idx is not None:
            blended += settings.weight_sem_score * X[:, sem_idx]
        if bm25_idx is not None:
            blended += settings.weight_bm25_score * X[:, bm25_idx]
        if overlap_idx is not None:
            blended += settings.weight_token_overlap * X[:, overlap_idx]
        if coverage_idx is not None:
            blended += settings.weight_query_coverage * X[:, coverage_idx]
        if shared_idx is not None:
            blended += settings.weight_shared_char * (
                X[:, shared_idx] / max(float(len(query.replace("_", " ").split())), 1.0)
            )
        if exact_idx is not None:
            blended += settings.weight_exact_phrase * X[:, exact_idx]
        if keyword_idx is not None:
            blended += settings.weight_keyword_hit * X[:, keyword_idx]

        if not self.is_fitted:
            return blended

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] > 1:
                model_scores = np.asarray(probs[:, 1], dtype=np.float32)
            else:
                model_scores = np.asarray(probs[:, 0], dtype=np.float32)
        elif hasattr(self.model, "predict"):
            model_scores = np.asarray(self.model.predict(X), dtype=np.float32)
        else:
            model_scores = np.zeros(X.shape[0], dtype=np.float32)
        return np.asarray((model_scores + blended) / 2.0, dtype=np.float32)

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        if docs:
            lexical = BM25Engine()
            lexical.fit(docs)
        else:
            lexical = None
        scores = self.score(query, docs, bm25=lexical)
        ranked = sorted(
            zip(docs, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            RankedDoc(doc=doc, score=float(score), rank=rank, metadata={"strategy": "hybrid"})
            for rank, (doc, score) in enumerate(ranked, start=1)
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        adapter_types = [type(adapter).__name__ for adapter in self.adapters]
        if self.model_backend == "xgboost" and target.suffix == ".json":
            self.model.save_model(str(target))
            write_json(
                target.with_suffix(".meta.json"),
                build_artifact_metadata(
                    "hybrid_reranker",
                    format_name="xgboost-json",
                    embedder_model_name=self.embedder.model_name,
                    extra={
                        "feature_names": self.feature_names_,
                        "feature_registry": self._feature_registry,
                        "adapter_types": adapter_types,
                    },
                ),
            )
            return
        dump_pickle(
            target,
            build_artifact_metadata(
                "hybrid_reranker",
                format_name="pickle",
                embedder_model_name=self.embedder.model_name,
                extra={
                    "model": self.model,
                    "feature_names": self.feature_names_,
                    "feature_registry": self._feature_registry,
                    "adapter_types": adapter_types,
                },
            ),
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
            from xgboost import XGBClassifier  # type: ignore

            meta_path = target.with_suffix(".meta.json")
            payload = read_json(meta_path)
            validate_artifact_metadata(
                payload,
                expected_type="hybrid_reranker",
                expected_formats={"xgboost-json"},
            )
            instance = cls(
                adapters=adapters,
                embedder=embedder or Embedder(payload["embedder_model_name"]),
            )
            instance.model = XGBClassifier()
            instance.model.load_model(str(target))
            instance.model_backend = "xgboost"
            instance._feature_registry = dict(payload.get("feature_registry", {}))
            instance.is_fitted = True
            return instance

        payload = load_pickle(target)
        validate_artifact_metadata(
            payload,
            expected_type="hybrid_reranker",
            expected_formats={"pickle"},
        )
        instance = cls(
            adapters=adapters,
            embedder=embedder or Embedder(payload["embedder_model_name"]),
        )
        instance.model = payload["model"]
        instance.model_backend = (
            "xgboost" if instance.model.__class__.__module__.startswith("xgboost") else "sklearn"
        )
        instance._feature_registry = dict(payload.get("feature_registry", {}))
        instance.is_fitted = True
        return instance
