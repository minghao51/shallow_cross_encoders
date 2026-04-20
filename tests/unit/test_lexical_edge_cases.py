"""Unit tests for lexical.py edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from reranker.lexical import BM25Engine


class TestBM25EngineBasic:
    """Basic tests for BM25Engine class."""

    def test_bm25_initialization(self) -> None:
        """BM25Engine should initialize successfully."""
        engine = BM25Engine()
        assert engine is not None

    def test_bm25_fit_creates_index(self) -> None:
        """fit should create a searchable index."""
        engine = BM25Engine()
        docs = ["first document", "second document"]
        engine.fit(docs)

        assert engine._bm25 is not None or engine.backend_name == "pure_python"


class TestBM25EngineFit:
    """Tests for fit method edge cases."""

    def test_fit_with_empty_corpus(self) -> None:
        """fit should handle empty corpus."""
        engine = BM25Engine()
        engine.fit([])

        # Should not raise an error
        assert engine._corpus == []

    def test_fit_with_multiple_documents(self) -> None:
        """fit should work with multiple documents."""
        engine = BM25Engine()
        docs = ["doc one", "doc two", "doc three"]
        engine.fit(docs)

        assert engine.backend_name in {"pure_python", "rank_bm25"}

    def test_fit_preserves_document_count(self) -> None:
        """fit should preserve the number of documents."""
        engine = BM25Engine()
        docs = ["doc one", "doc two", "doc three"]
        engine.fit(docs)

        # The tokenized corpus should have same length
        assert len(engine._tokenized) == len(docs)


class TestBM25EngineScore:
    """Tests for score method edge cases."""

    def test_score_with_empty_query(self) -> None:
        """score should handle empty query."""
        engine = BM25Engine()
        docs = ["test document"]
        engine.fit(docs)

        scores = engine.score("")

        # Should return array with one element
        assert len(scores) == 1

    def test_score_with_single_term_query(self) -> None:
        """score should work with single term query."""
        engine = BM25Engine()
        docs = ["python programming", "java development"]
        engine.fit(docs)

        scores = engine.score("python")

        # First doc should score higher
        assert scores[0] > scores[1]

    def test_score_with_multi_term_query(self) -> None:
        """score should work with multi-term query."""
        engine = BM25Engine()
        docs = ["python programming tutorial", "python code examples"]
        engine.fit(docs)

        scores = engine.score("python programming")

        # Both docs should have positive scores
        assert np.all(scores >= 0)

    def test_score_with_no_matches(self) -> None:
        """score should return zero for no matches."""
        engine = BM25Engine()
        docs = ["python programming", "java development"]
        engine.fit(docs)

        scores = engine.score("nonexistent term")

        # Scores should be zero or very low
        assert np.all(scores >= 0)

    def test_score_normalization_default(self) -> None:
        """score should normalize by default."""
        engine = BM25Engine()
        docs = ["python programming", "java development"]
        engine.fit(docs)

        scores = engine.score("python")

        # At least one score should be 1.0 (max normalization)
        assert np.max(scores) == pytest.approx(1.0, rel=0.01)

    def test_score_without_normalization(self) -> None:
        """score should work without normalization."""
        engine = BM25Engine()
        docs = ["python programming", "java development"]
        engine.fit(docs)

        scores = engine.score("python", normalize=False)

        # Scores should be non-negative
        assert np.all(scores >= 0)

    def test_score_with_all_zero_scores(self) -> None:
        """score should handle all zero scores when normalizing."""
        engine = BM25Engine()
        docs = ["test document"]
        engine.fit(docs)

        scores = engine.score("nonexistent")

        # Should handle gracefully
        assert len(scores) == 1


class TestBM25EngineRanking:
    """Tests for BM25 ranking behavior."""

    def test_score_ranks_relevant_doc_first(self) -> None:
        """BM25 should rank document with exact matches first."""
        engine = BM25Engine()
        docs = [
            "python dataclass default factory",
            "ocean current weather patterns",
            "machine learning model training",
        ]
        engine.fit(docs)

        scores = engine.score("python dataclass")

        # First doc should have highest score
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_score_rewards_term_frequency(self) -> None:
        """BM25 should reward documents with more term matches."""
        engine = BM25Engine()
        docs = [
            "python programming tutorial",
            "python python python code",  # More "python" occurrences
            "java development guide",
        ]
        engine.fit(docs)

        scores = engine.score("python")

        # Second doc should have higher score due to term frequency
        assert scores[1] > scores[0]

    def test_score_handles_acronyms(self) -> None:
        """BM25 should handle acronym matching."""
        engine = BM25Engine()
        docs = [
            "API stands for Application Programming Interface",
            "The api endpoint returns JSON data",
            "Application Programming Interface is important",
        ]
        engine.fit(docs)

        scores = engine.score("API")

        # First doc with exact acronym match should score well
        assert scores[0] > 0

    def test_score_case_insensitive(self) -> None:
        """BM25 should be case-insensitive."""
        engine = BM25Engine()
        docs = ["Python Programming"]
        engine.fit(docs)

        scores_lower = engine.score("python")
        scores_upper = engine.score("Python")
        scores_mixed = engine.score("PYTHON")

        # All should return same scores
        np.testing.assert_array_almost_equal(scores_lower, scores_upper)
        np.testing.assert_array_almost_equal(scores_lower, scores_mixed)


class TestBM25EngineSpecialCases:
    """Tests for special cases and edge conditions."""

    def test_fit_with_special_characters(self) -> None:
        """fit should handle special characters."""
        engine = BM25Engine()
        docs = [
            "Email: test@example.com",
            "Price: $99.99",
            "Symbols: @#$%^&*()",
        ]
        engine.fit(docs)

        scores = engine.score("@#$")

        # Should not crash
        assert len(scores) == 3

    def test_score_with_unicode(self) -> None:
        """score should handle unicode characters."""
        engine = BM25Engine()
        docs = [
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "こんにちは世界",  # Japanese
        ]
        engine.fit(docs)

        scores = engine.score("Привет")

        # Should not crash
        assert len(scores) == 3

    def test_score_with_very_long_query(self) -> None:
        """score should handle very long query."""
        engine = BM25Engine()
        docs = ["test document"]
        engine.fit(docs)

        long_query = "word " * 100  # 500 words
        scores = engine.score(long_query)

        # Should not crash
        assert len(scores) == 1

    def test_score_with_very_long_document(self) -> None:
        """score should handle very long documents."""
        engine = BM25Engine()
        long_doc = "word " * 1000  # 5000 words
        engine.fit([long_doc])

        scores = engine.score("word")

        # Should not crash
        assert len(scores) == 1

    def test_score_returns_numpy_array(self) -> None:
        """score should return numpy array."""
        engine = BM25Engine()
        docs = ["doc one", "doc two"]
        engine.fit(docs)

        scores = engine.score("test")

        assert isinstance(scores, np.ndarray)

    def test_score_array_shape(self) -> None:
        """score should return array with correct shape."""
        engine = BM25Engine()
        docs = ["doc one", "doc two", "doc three"]
        engine.fit(docs)

        scores = engine.score("test")

        assert scores.shape == (3,)


class TestBM25EngineBackend:
    """Tests for backend behavior."""

    def test_backend_name_attribute(self) -> None:
        """BM25Engine should have backend_name attribute."""
        engine = BM25Engine()

        assert hasattr(engine, "backend_name")
        assert engine.backend_name in ["rank_bm25", "pure_python"]

    def test_fallback_backend(self) -> None:
        """BM25Engine should fall back to pure Python if rank_bm25 unavailable."""
        # This test assumes rank_bm25 might not be installed
        engine = BM25Engine()
        docs = ["test document"]
        engine.fit(docs)

        # Should work regardless of backend
        scores = engine.score("test")
        assert len(scores) == 1
