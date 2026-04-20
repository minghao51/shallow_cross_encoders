from pathlib import Path

from reranker.strategies.distilled import DistilledPairwiseRanker


def test_pairwise_ranker_prefers_better_document() -> None:
    queries = ["bm25 exact term match"] * 4
    doc_as = [
        "BM25 is a lexical ranking function that values exact term overlap.",
        "Dense vectors can capture semantic similarity.",
        "BM25 exact term overlap improves retrieval for acronyms.",
        "Lexical retrieval focuses on token matches.",
    ]
    doc_bs = [
        "Ocean currents influence temperature across coastlines.",
        "Football teams rotate players during tournaments.",
        "Sunlight intensity changes throughout the day.",
        "Cloud cover affects evening visibility.",
    ]
    labels = [1, 0, 1, 1]
    ranker = DistilledPairwiseRanker().fit(queries, doc_as, doc_bs, labels)
    ranked = ranker.rerank(
        "bm25 exact term match",
        [
            "Ocean currents influence temperature across coastlines.",
            "BM25 exact term overlap improves retrieval for acronyms.",
            "Cloud cover affects evening visibility.",
        ],
    )
    assert "BM25" in ranked[0].doc


def test_pairwise_ranker_uses_scalable_merge_path_for_larger_lists() -> None:
    queries = ["bm25 exact term match"] * 4
    doc_as = [
        "BM25 is a lexical ranking function that values exact term overlap.",
        "Dense vectors can capture semantic similarity.",
        "BM25 exact term overlap improves retrieval for acronyms.",
        "Lexical retrieval focuses on token matches.",
    ]
    doc_bs = [
        "Ocean currents influence temperature across coastlines.",
        "Football teams rotate players during tournaments.",
        "Sunlight intensity changes throughout the day.",
        "Cloud cover affects evening visibility.",
    ]
    labels = [1, 0, 1, 1]
    ranker = DistilledPairwiseRanker().fit(queries, doc_as, doc_bs, labels)
    ranker.full_tournament_max_docs = 2

    ranked = ranker.rerank(
        "bm25 exact term match",
        [
            "Ocean currents influence temperature across coastlines.",
            "BM25 exact term overlap improves retrieval for acronyms.",
            "Lexical retrieval focuses on token matches.",
        ],
    )

    assert "BM25" in ranked[0].doc


def test_pairwise_compare_method() -> None:
    """Test the compare method directly."""
    queries = ["test query"] * 2
    doc_as = ["relevant document about testing"]
    doc_bs = ["irrelevant document"]
    labels = [1, 1]

    ranker = DistilledPairwiseRanker().fit(queries, doc_as, doc_bs, labels)

    # Compare two documents
    prob = ranker.compare("test query", "relevant document about testing", "irrelevant document")

    # Should prefer doc_a (relevant document)
    assert prob >= 0.5


def test_pairwise_save_and_load(tmp_path: Path) -> None:
    """Test model persistence and loading."""
    queries = ["test"] * 2
    doc_as = ["relevant"]
    doc_bs = ["irrelevant"]
    labels = [1, 0]

    ranker = DistilledPairwiseRanker().fit(queries, doc_as, doc_bs, labels)

    # Save model
    model_path = tmp_path / "pairwise_model.pkl"
    ranker.save(model_path)
    assert model_path.exists()

    # Load model
    loaded_ranker = DistilledPairwiseRanker.load(model_path)

    # Verify it works
    ranked = loaded_ranker.rerank("test", ["irrelevant", "relevant"])
    assert len(ranked) == 2


def test_pairwise_tournament_scoring() -> None:
    """Test tournament-style scoring with multiple documents."""
    queries = ["test"] * 3
    doc_as = ["document A", "document B", "document C"]
    doc_bs = ["document D", "document E", "document F"]
    labels = [1, 1, 0]

    ranker = DistilledPairwiseRanker().fit(queries, doc_as, doc_bs, labels)

    # Rerank multiple documents
    docs = ["document D", "document A", "document E", "document B"]
    ranked = ranker.rerank("test", docs)

    # All documents should be ranked
    assert len(ranked) == 4
    # Ranks should be 1-4
    assert [r.rank for r in ranked] == [1, 2, 3, 4]


def test_pairwise_empty_document_list() -> None:
    """Test handling of empty document list."""
    queries = ["test"]
    doc_as = ["doc a"]
    doc_bs = ["doc b"]
    labels = [1]

    ranker = DistilledPairwiseRanker().fit(queries, doc_as, doc_bs, labels)
    ranked = ranker.rerank("test", [])

    assert len(ranked) == 0


def test_pairwise_single_document() -> None:
    """Test handling of single document."""
    queries = ["test"]
    doc_as = ["doc a"]
    doc_bs = ["doc b"]
    labels = [1]

    ranker = DistilledPairwiseRanker().fit(queries, doc_as, doc_bs, labels)
    ranked = ranker.rerank("test", ["single document"])

    assert len(ranked) == 1
    assert ranked[0].rank == 1
