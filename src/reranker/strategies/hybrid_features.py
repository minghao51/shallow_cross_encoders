"""Feature engineering for hybrid fusion reranker."""

from __future__ import annotations

import numpy as np

from reranker.lexical import BM25Engine
from reranker.protocols import EmbedderProtocol, HeuristicAdapter

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


class HybridFeatureBuilder:
    """Builds feature matrices from query-document pairs."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        adapters: list[HeuristicAdapter] | None = None,
    ) -> None:
        self.embedder = embedder
        self.adapters = adapters or []
        self._feature_registry: dict[str, int] = {}

    def init_feature_registry(self, adapter_names: list[str] | None = None) -> None:
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

    def register_adapter_feature_names(self, query: str, docs: list[str]) -> None:
        if not self._feature_registry:
            self.init_feature_registry()
        for doc in docs:
            for name in self._adapter_feature_names(query, doc):
                self._get_feature_index(name)

    def build_features(
        self,
        query: str,
        docs: list[str],
        *,
        bm25: BM25Engine | None = None,
        query_vec: np.ndarray | None = None,
        d_vecs: np.ndarray | None = None,
        is_fitted: bool = False,
    ) -> np.ndarray:
        if not docs:
            if not self._feature_registry:
                self.init_feature_registry()
            return np.zeros((0, len(self._feature_registry)), dtype=np.float32)

        q_vec = query_vec if query_vec is not None else self.embedder.encode([query])[0]
        d_vecs = d_vecs if d_vecs is not None else self.embedder.encode(docs)
        lexical = bm25
        if lexical is None:
            lexical = BM25Engine(tokenize_fn=self.embedder.tokenize)
            lexical.fit(docs)
        bm25_scores = lexical.score(query)

        query_lower = query.lower()
        query_tokens = self.embedder.tokenize(query_lower)
        query_terms = set(query_tokens)
        query_len = float(len(query_tokens))

        rows: list[dict[str, float]] = []
        for idx, doc in enumerate(docs):
            doc_lower = doc.lower()
            doc_tokens = self.embedder.tokenize(doc_lower)
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
                "exact_phrase_match": float(1.0 if query_lower in doc_lower else 0.0),
                "query_len": query_len,
                "doc_len": float(len(doc_tokens)),
            }
            for adapter in self.adapters:
                row_dict.update(adapter.compute(query, doc))
            rows.append(row_dict)

        if not self._feature_registry:
            self.init_feature_registry()
        if not is_fitted:
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
    def feature_names(self) -> list[str]:
        return list(self._feature_registry.keys())

    @property
    def feature_registry(self) -> dict[str, int]:
        return self._feature_registry

    @feature_registry.setter
    def feature_registry(self, value: dict[str, int]) -> None:
        self._feature_registry = value
