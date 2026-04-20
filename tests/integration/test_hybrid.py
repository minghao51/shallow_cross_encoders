from pathlib import Path

import numpy as np

from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter
from reranker.utils import load_pickle


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
    assert model_path.exists()

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
    payload = load_pickle(model_path)

    assert payload["artifact_type"] == "hybrid_reranker"
    assert payload["format"] == "pickle"
    assert "token_overlap_ratio" in payload["feature_names"]


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
