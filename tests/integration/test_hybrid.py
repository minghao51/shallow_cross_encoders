from pathlib import Path

import numpy as np

from reranker.config import (
    apply_settings_override,
    clear_settings_override,
    reset_settings_cache,
    settings_from_dict,
)
from reranker.heuristics.keyword import KeywordMatchAdapter
from reranker.persistence import load_safe
from reranker.strategies.hybrid import HybridFusionReranker


def test_hybrid_reranks_relevant_doc_first() -> None:
    queries = [
        "python dataclass default factory",
        "python dataclass default factory",
        "bm25 exact match retrieval",
        "bm25 exact match retrieval",
    ]
    docs = [
        "Use field(default_factory=list) to avoid shared mutable defaults in dataclasses.",
        "Ocean currents shape weather systems across large regions.",
        "BM25 rewards exact term overlap between query and document.",
        "A cross encoder can be expensive at inference time.",
    ]
    labels = [1, 0, 1, 0]
    reranker = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit(
        queries, docs, docs, labels
    )
    ranked = reranker.rerank(
        "python dataclass default factory",
        [
            "Ocean currents shape weather systems across large regions.",
            "Use field(default_factory=list) to avoid shared mutable defaults in dataclasses.",
        ],
    )
    assert "default_factory" in ranked[0].doc


def test_hybrid_save_and_load(tmp_path: Path) -> None:
    """Test model persistence and loading."""
    queries = ["python test", "java test"]
    docs = ["python programming", "java development"]
    labels = [1, 0]

    reranker = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit(
        queries, docs, docs, labels
    )

    # Save model
    model_path = tmp_path / "hybrid_model.pkl"
    reranker.save(model_path)
    assert (tmp_path / "hybrid_model.meta.json").exists()
    assert (tmp_path / "hybrid_model.weights.joblib").exists()

    # Load model
    loaded_reranker = HybridFusionReranker.load(model_path, adapters=[KeywordMatchAdapter()])

    # Verify it works
    ranked = loaded_reranker.rerank("python test", ["java development", "python programming"])
    assert "python" in ranked[0].doc.lower()


def test_hybrid_pickle_round_trip_preserves_artifact_metadata(tmp_path: Path) -> None:
    queries = ["python test", "java test"]
    docs = ["python programming", "java development"]
    labels = [1, 0]

    reranker = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit(
        queries, docs, docs, labels
    )
    model_path = tmp_path / "hybrid_model.pkl"
    reranker.save(model_path)
    meta, weights = load_safe(model_path, expected_type="hybrid_reranker")

    assert meta["artifact_type"] == "hybrid_reranker"
    assert meta["format"] == "safe-joblib"
    assert "token_overlap_ratio" in meta["feature_names"]


def test_hybrid_empty_document_list() -> None:
    """Test handling of empty document list."""
    queries = ["test query"]
    docs = ["test document"]
    labels = [1]

    reranker = HybridFusionReranker().fit(queries, docs, docs, labels)
    ranked = reranker.rerank("test query", [])

    assert len(ranked) == 0


def test_hybrid_single_document() -> None:
    """Test handling of single document."""
    queries = ["test query"]
    docs = ["test document"]
    labels = [1]

    reranker = HybridFusionReranker().fit(queries, docs, docs, labels)
    ranked = reranker.rerank("test query", ["test document"])

    assert len(ranked) == 1
    assert ranked[0].rank == 1


def test_hybrid_adapter_feature_names() -> None:
    """Test that adapter features are included in feature names."""
    adapter = KeywordMatchAdapter()
    queries = ["python test"]
    docs = ["python programming"]
    labels = [1]

    reranker = HybridFusionReranker(adapters=[adapter]).fit(queries, docs, docs, labels)

    assert "keyword_hit_rate" in reranker.feature_names_


def test_hybrid_unfitted_rerank_returns_blended_scores() -> None:
    """Test that unfitted reranker falls back to blended scores without crashing."""
    reranker = HybridFusionReranker()
    ranked = reranker.rerank(
        "python test",
        [
            "python programming is great",
            "java development is good",
        ],
    )
    assert len(ranked) == 2
    for doc in ranked:
        assert doc.score >= 0.0
    assert ranked[0].doc == "python programming is great"
    assert ranked[0].score >= ranked[1].score


def test_hybrid_registers_adapter_features_across_all_docs() -> None:
    class PerDocAdapter:
        def compute(self, query: str, doc: str) -> dict[str, float]:
            return {f"doc::{doc}": 1.0}

    reranker = HybridFusionReranker(adapters=[PerDocAdapter()])
    features = reranker._build_features("python test", ["doc1", "doc2"])

    assert "doc::doc1" in reranker.feature_names_
    assert "doc::doc2" in reranker.feature_names_

    doc1_idx = reranker.feature_names_.index("doc::doc1")
    doc2_idx = reranker.feature_names_.index("doc::doc2")

    np.testing.assert_array_equal(features[:, doc1_idx], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(features[:, doc2_idx], np.array([0.0, 1.0], dtype=np.float32))


def test_hybrid_fit_pointwise_trains_meta_router_when_enabled(monkeypatch) -> None:
    reset_settings_cache()
    clear_settings_override()
    apply_settings_override(
        settings_from_dict(
            {
                "hybrid": {"weighting_mode": "meta_router"},
                "meta_router": {"enabled": True},
            }
        )
    )

    reranker = HybridFusionReranker(adapters=[KeywordMatchAdapter()])
    monkeypatch.setattr(
        reranker,
        "_auto_label_queries",
        lambda queries, docs, scores: [0, 1],
    )

    try:
        reranker.fit_pointwise(
            queries=["query one", "query two"],
            docs=["doc alpha", "doc beta"],
            scores=[0.1, 0.9],
        )
    finally:
        clear_settings_override()
        reset_settings_cache()

    assert reranker._router is not None
    assert reranker._router.is_fitted is True


def test_hybrid_score_uses_router_weights_in_meta_router_mode() -> None:
    class StubRouter:
        is_fitted = True

        def __init__(self) -> None:
            self.calls = 0

        def get_weights(self, query: str) -> dict[str, float]:
            del query
            self.calls += 1
            return {
                "sem_score": 0.0,
                "bm25_score": 0.0,
                "token_overlap_ratio": 0.0,
                "query_coverage_ratio": 0.0,
                "shared_token_char_sum": 0.0,
                "exact_phrase_match": 1.0,
                "keyword_hit_rate": 0.0,
            }

    reset_settings_cache()
    clear_settings_override()
    apply_settings_override(
        settings_from_dict(
            {
                "hybrid": {"weighting_mode": "meta_router"},
                "meta_router": {"enabled": True},
            }
        )
    )

    reranker = HybridFusionReranker()
    reranker._router = StubRouter()

    try:
        scores = reranker.score("exact phrase", ["exact phrase in doc", "unrelated text"])
    finally:
        clear_settings_override()
        reset_settings_cache()

    assert reranker._router.calls == 1
    assert scores[0] > scores[1]


def test_hybrid_score_uses_model_predictions_in_learned_mode() -> None:
    class StubModel:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.asarray([0.2, 0.8], dtype=np.float32)

    reset_settings_cache()
    clear_settings_override()
    apply_settings_override(settings_from_dict({"hybrid": {"weighting_mode": "learned"}}))

    reranker = HybridFusionReranker()
    reranker.model = StubModel()
    reranker.is_fitted = True

    try:
        scores = reranker.score("python", ["python guide", "java guide"])
    finally:
        clear_settings_override()
        reset_settings_cache()

    np.testing.assert_allclose(scores, np.array([0.2, 0.8], dtype=np.float32))


def test_hybrid_auto_label_queries_can_emit_balanced_category(monkeypatch) -> None:
    class StubBM25:
        def __init__(self, tokenize_fn=None) -> None:
            del tokenize_fn

        def fit(self, docs: list[str]) -> None:
            del docs

        def score(self, query: str) -> np.ndarray:
            del query
            return np.array([1.0, 0.2], dtype=np.float32)

    class StubEmbedder:
        def encode(self, texts: list[str]) -> np.ndarray:
            del texts
            return np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        def similarity(self, left: np.ndarray, right: np.ndarray) -> float:
            del left, right
            return 1.0

        def tokenize(self, text: str) -> list[str]:
            return text.split()

    reset_settings_cache()
    clear_settings_override()
    apply_settings_override(settings_from_dict({"meta_router": {"n_categories": 3}}))
    monkeypatch.setattr("reranker.lexical.BM25Engine", StubBM25)

    reranker = HybridFusionReranker(embedder=StubEmbedder())

    try:
        categories = reranker._auto_label_queries(
            queries=["query one", "query one"],
            docs=["doc alpha", "doc beta"],
            scores=[0.9, 0.1],
        )
    finally:
        clear_settings_override()
        reset_settings_cache()

    assert categories == [2, 2]
