from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from reranker.config import get_settings
from reranker.embedder import Embedder

WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "navigational": {
        "sem_score": 0.10,
        "bm25_score": 0.40,
        "token_overlap_ratio": 0.15,
        "query_coverage_ratio": 0.15,
        "shared_token_char_sum": 0.05,
        "exact_phrase_match": 0.10,
        "keyword_hit_rate": 0.05,
    },
    "informational": {
        "sem_score": 0.40,
        "bm25_score": 0.10,
        "token_overlap_ratio": 0.10,
        "query_coverage_ratio": 0.10,
        "shared_token_char_sum": 0.05,
        "exact_phrase_match": 0.05,
        "keyword_hit_rate": 0.20,
    },
    "balanced": {
        "sem_score": 0.25,
        "bm25_score": 0.20,
        "token_overlap_ratio": 0.15,
        "query_coverage_ratio": 0.20,
        "shared_token_char_sum": 0.10,
        "exact_phrase_match": 0.05,
        "keyword_hit_rate": 0.05,
    },
}

DEFAULT_PROFILE = "balanced"


@dataclass(slots=True)
class MetaRouter:
    embedder: Embedder = field(default_factory=Embedder)
    model: Any = None
    is_fitted: bool = False
    n_categories: int = 2
    min_samples_leaf: int = 5

    def __post_init__(self) -> None:
        settings = get_settings().meta_router
        self.n_categories = max(1, min(settings.n_categories, len(WEIGHT_PROFILES)))
        self.min_samples_leaf = settings.min_samples_leaf
        if settings.model_type == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=200,
                random_state=42,
            )
        else:
            self.model = DecisionTreeClassifier(
                max_leaf_nodes=self.n_categories,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )

    def _query_features(self, query: str) -> np.ndarray:
        tokens = self.embedder.tokenize(query.lower())
        q_vec = self.embedder.encode([query])[0]
        avg_token_len = np.mean([len(t) for t in tokens]) if tokens else 0.0
        has_numbers = float(any(c.isdigit() for c in query))
        has_special = float(any(not c.isalnum() and not c.isspace() for c in query))
        capital_ratio = sum(1 for c in query if c.isupper()) / max(len(query), 1)
        return np.array(
            [
                float(len(tokens)),
                avg_token_len,
                float(q_vec.mean()),
                float(q_vec.std()),
                has_numbers,
                has_special,
                capital_ratio,
            ],
            dtype=np.float32,
        )

    def fit(self, queries: list[str], categories: list[int]) -> MetaRouter:
        if len(set(categories)) < 2:
            self.is_fitted = False
            return self
        X = np.vstack([self._query_features(q) for q in queries])
        y = np.asarray(categories, dtype=np.int32)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, query: str) -> int:
        if not self.is_fitted:
            return 0
        features = self._query_features(query).reshape(1, -1)
        return int(self.model.predict(features)[0])

    def get_weights(self, query: str) -> dict[str, float]:
        profile_names = list(WEIGHT_PROFILES.keys())
        if not self.is_fitted:
            return WEIGHT_PROFILES[DEFAULT_PROFILE]
        cat_idx = self.predict(query)
        if cat_idx < len(profile_names):
            return WEIGHT_PROFILES[profile_names[cat_idx]]
        return WEIGHT_PROFILES[DEFAULT_PROFILE]

    def predict_proba(self, query: str) -> np.ndarray:
        if not self.is_fitted:
            n = len(WEIGHT_PROFILES)
            return np.ones(n, dtype=np.float32) / n
        features = self._query_features(query).reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(features)[0]
        pred = int(self.model.predict(features)[0])
        n = max(len(WEIGHT_PROFILES), pred + 1)
        probs = np.zeros(n, dtype=np.float32)
        probs[pred] = 1.0
        return probs
