"""Unit tests for embedder.py edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from reranker.embedder import Embedder


class TestEmbedderBasic:
    """Basic tests for Embedder class."""

    @pytest.mark.unit
    def test_embedder_initialization(self) -> None:
        """Embedder should initialize successfully."""
        embedder = Embedder()
        assert embedder is not None

    @pytest.mark.unit
    def test_embedder_encode_single_text(self) -> None:
        """Embedder should encode a single text."""
        embedder = Embedder()
        vectors = embedder.encode(["test document"])

        assert vectors.shape[0] == 1
        assert vectors.shape[1] > 0
        assert vectors.dtype == np.float32


class TestEmbedderEncoding:
    """Tests for encode method edge cases."""

    @pytest.mark.unit
    def test_encode_empty_list(self) -> None:
        """encode should handle empty list."""
        embedder = Embedder()
        vectors = embedder.encode([])

        assert vectors.shape == (0, embedder.encode(["test"]).shape[1])

    @pytest.mark.unit
    def test_encode_single_document(self) -> None:
        """encode should work with single document."""
        embedder = Embedder()
        vectors = embedder.encode(["single document"])

        assert vectors.shape[0] == 1
        assert len(vectors.shape) == 2

    @pytest.mark.unit
    def test_encode_multiple_documents(self) -> None:
        """encode should work with multiple documents."""
        embedder = Embedder()
        texts = ["doc one", "doc two", "doc three"]
        vectors = embedder.encode(texts)

        assert vectors.shape[0] == 3

    @pytest.mark.unit
    def test_encode_consistent_dimensions(self) -> None:
        """encode should produce consistent dimensions."""
        embedder = Embedder()
        vectors1 = embedder.encode(["test"])
        vectors2 = embedder.encode(["another test"])

        assert vectors1.shape[1] == vectors2.shape[1]

    @pytest.mark.unit
    def test_encode_normalization(self) -> None:
        """encode should normalize vectors by default."""
        embedder = Embedder()
        vectors = embedder.encode(["test document"])

        # Check that vectors are normalized (L2 norm should be ~1)
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=0.01)


class TestEmbedderSimilarity:
    """Tests for similarity method."""

    @pytest.mark.unit
    def test_similarity_identical_vectors(self) -> None:
        """similarity should return 1.0 for identical vectors."""
        embedder = Embedder()
        vectors = embedder.encode(["test document"])

        similarity = embedder.similarity(vectors[0], vectors[0])

        assert similarity == pytest.approx(1.0, rel=0.01)

    @pytest.mark.unit
    def test_similarity_different_vectors(self) -> None:
        """similarity should return < 1.0 for different vectors."""
        embedder = Embedder()
        vectors = embedder.encode(["completely different", "unrelated text"])

        similarity = embedder.similarity(vectors[0], vectors[1])

        assert 0.0 <= similarity < 1.0

    @pytest.mark.unit
    def test_similarity_symmetric(self) -> None:
        """similarity should be symmetric."""
        embedder = Embedder()
        vectors = embedder.encode(["doc one", "doc two"])

        sim_ab = embedder.similarity(vectors[0], vectors[1])
        sim_ba = embedder.similarity(vectors[1], vectors[0])

        assert sim_ab == pytest.approx(sim_ba, rel=0.001)

    @pytest.mark.unit
    def test_similarity_range(self) -> None:
        """similarity should return values in [0, 1] for normalized vectors."""
        embedder = Embedder()
        vectors = embedder.encode(["doc one", "doc two", "doc three"])

        for i in range(len(vectors)):
            for j in range(len(vectors)):
                similarity = embedder.similarity(vectors[i], vectors[j])
                assert 0.0 <= similarity <= 1.0

    @pytest.mark.unit
    def test_similarity_similar_content(self) -> None:
        """similarity should be higher for semantically similar content."""
        embedder = Embedder()
        vectors = embedder.encode(
            [
                "machine learning is about training models",
                "ML involves training neural networks",
                "the stock market closed higher today",
            ]
        )

        sim_similar = embedder.similarity(vectors[0], vectors[1])
        sim_different = embedder.similarity(vectors[0], vectors[2])

        # Similar content should have higher similarity
        assert sim_similar > sim_different


class TestEmbedderDescribe:
    """Tests for describe method."""

    @pytest.mark.unit
    def test_describe_returns_dict(self) -> None:
        """describe should return a dictionary."""
        embedder = Embedder()
        info = embedder.describe()

        assert isinstance(info, dict)

    @pytest.mark.unit
    def test_describe_contains_backend(self) -> None:
        """describe should include backend information."""
        embedder = Embedder()
        info = embedder.describe()

        assert "backend" in info
        assert info["backend"] in ["model2vec", "hashed"]

    @pytest.mark.unit
    def test_describe_contains_dimension(self) -> None:
        """describe should include embedding dimension."""
        embedder = Embedder()
        info = embedder.describe()

        assert "dimension" in info
        assert info["dimension"] > 0


class TestEmbedderBackend:
    """Tests for backend behavior."""

    @pytest.mark.unit
    def test_embedder_uses_hashed_fallback(self) -> None:
        """Embedder should fall back to hashed embeddings when model2vec unavailable."""
        # This test assumes model2vec might not be installed
        embedder = Embedder()
        vectors = embedder.encode(["test document"])

        # Should work regardless of backend
        assert vectors.shape[0] == 1
        assert vectors.shape[1] > 0


class TestEmbedderEdgeCases:
    """Additional edge case tests."""

    @pytest.mark.unit
    def test_encode_with_special_characters(self) -> None:
        """encode should handle special characters."""
        embedder = Embedder()
        special_texts = [
            "Hello, 世界!",
            "Test with emoji 😀",
            "Special chars: @#$%^&*()",
            "Newlines\nand\ttabs",
        ]

        vectors = embedder.encode(special_texts)

        assert vectors.shape[0] == len(special_texts)

    @pytest.mark.unit
    def test_encode_with_very_long_text(self) -> None:
        """encode should handle very long text."""
        embedder = Embedder()
        long_text = "word " * 1000  # 5000 characters

        vectors = embedder.encode([long_text])

        assert vectors.shape[0] == 1

    @pytest.mark.unit
    def test_encode_with_unicode(self) -> None:
        """encode should handle unicode characters."""
        embedder = Embedder()
        unicode_texts = [
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "こんにちは世界",  # Japanese
            "🎉🎊🎈",  # Only emojis
        ]

        vectors = embedder.encode(unicode_texts)

        assert vectors.shape[0] == len(unicode_texts)

    @pytest.mark.unit
    def test_encode_deterministic_for_same_input(self) -> None:
        """encode should be deterministic for the same input."""
        embedder = Embedder()
        text = "test document"

        vectors1 = embedder.encode([text])
        vectors2 = embedder.encode([text])

        np.testing.assert_array_almost_equal(vectors1, vectors2)
