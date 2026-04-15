from pathlib import Path

import pytest

from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter
from reranker.utils import load_pickle


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_hybrid_empty_document_list() -> None:
    """Test handling of empty document list."""
    queries = ["test query"]
    docs = ["test document"]
    labels = [1]

    reranker = HybridFusionReranker().fit(queries, docs, docs, labels)
    ranked = reranker.rerank("test query", [])

    assert len(ranked) == 0


@pytest.mark.unit
def test_hybrid_single_document() -> None:
    """Test handling of single document."""
    queries = ["test query"]
    docs = ["test document"]
    labels = [1]

    reranker = HybridFusionReranker().fit(queries, docs, docs, labels)
    ranked = reranker.rerank("test query", ["test document"])

    assert len(ranked) == 1
    assert ranked[0].rank == 1


@pytest.mark.unit
def test_hybrid_adapter_feature_names() -> None:
    """Test that adapter features are included in feature names."""
    adapter = KeywordMatchAdapter()
    queries = ["python test"]
    docs = ["python programming"]
    labels = [1]

    reranker = HybridFusionReranker(adapters=[adapter]).fit(queries, docs, docs, labels)

    # Check that adapter features are included
    assert "keyword_hit_rate" in reranker.feature_names_
