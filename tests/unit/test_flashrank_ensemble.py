"""Tests for FlashRankEnsemble wrapper."""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from reranker.strategies.flashrank_ensemble import FlashRankEnsemble


class TestFlashRankEnsemble:
    def test_init_empty_models_raises_valueerror(self):
        """Verify ValueError is raised when models list is empty."""
        with pytest.raises(ValueError, match="models list cannot be empty"):
            FlashRankEnsemble(models=[])

    def test_init_with_models(self):
        """Verify FlashRankEnsemble initializes with model list."""
        models = ["ms-marco-TinyBERT-L-2-v2", "ms-marco-MiniLM-L-12-v2"]
        ensemble = FlashRankEnsemble(models=models)
        assert ensemble.models == models

    def test_score_batch_returns_averaged_scores(self, mocker):
        """Mock flashrank.Ranker, simulate different scores from two teachers, verify averaging."""
        # Create mock flashrank module and Ranker class
        mock_flashrank = Mock()
        mock_ranker_class = Mock()

        # Create two mock rankers with different scores
        mock_ranker_1 = MagicMock()
        mock_ranker_2 = MagicMock()

        # Teacher 1 returns scores [0.8, 0.6]
        mock_ranker_1.rerank.return_value = [
            {"id": "0", "text": "doc1", "score": 0.8},
            {"id": "1", "text": "doc2", "score": 0.6},
        ]

        # Teacher 2 returns scores [0.7, 0.5]
        mock_ranker_2.rerank.return_value = [
            {"id": "0", "text": "doc1", "score": 0.7},
            {"id": "1", "text": "doc2", "score": 0.5},
        ]

        # Configure the mock to return different instances
        mock_ranker_class.side_effect = [mock_ranker_1, mock_ranker_2]
        mock_flashrank.Ranker = mock_ranker_class
        mock_flashrank.RerankRequest = mocker.Mock()

        # Inject mock into sys.modules
        mocker.patch.dict("sys.modules", {"flashrank": mock_flashrank})

        # Create ensemble with two models
        ensemble = FlashRankEnsemble(models=["model1", "model2"])

        # Score documents
        query = "test query"
        docs = ["doc1", "doc2"]
        scores = ensemble.score_batch(query, docs)

        # Verify averaging: (0.8+0.7)/2=0.75, (0.6+0.5)/2=0.55
        expected = np.array([0.75, 0.55], dtype=np.float32)
        np.testing.assert_array_almost_equal(scores, expected, decimal=5)

        # Verify both rankers were called
        assert mock_ranker_1.rerank.called
        assert mock_ranker_2.rerank.called

    def test_score_batch_single_model(self, mocker):
        """Single model returns its scores directly."""
        # Create mock flashrank module and Ranker class
        mock_flashrank = Mock()
        mock_ranker_class = Mock()

        # Create mock ranker
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {"id": "0", "text": "doc1", "score": 0.8},
            {"id": "1", "text": "doc2", "score": 0.6},
        ]

        mock_ranker_class.return_value = mock_ranker
        mock_flashrank.Ranker = mock_ranker_class
        mock_flashrank.RerankRequest = mocker.Mock()

        # Inject mock into sys.modules
        mocker.patch.dict("sys.modules", {"flashrank": mock_flashrank})

        # Create ensemble with single model
        ensemble = FlashRankEnsemble(models=["model1"])

        # Score documents
        query = "test query"
        docs = ["doc1", "doc2"]
        scores = ensemble.score_batch(query, docs)

        # Verify scores are returned directly
        expected = np.array([0.8, 0.6], dtype=np.float32)
        np.testing.assert_array_almost_equal(scores, expected, decimal=5)

    def test_score_batch_empty_docs(self):
        """Empty docs list returns empty array."""
        ensemble = FlashRankEnsemble(models=["model1"])
        scores = ensemble.score_batch("test query", [])

        expected = np.zeros(0, dtype=np.float32)
        np.testing.assert_array_equal(scores, expected)

    def test_score_batch_flashrank_not_available(self, mocker):
        """Test ImportError when flashrank is not available."""
        # Create mock flashrank module that raises ImportError
        mock_flashrank = Mock()
        mock_flashrank.Ranker = Mock(side_effect=ImportError("flashrank not installed"))

        # Inject mock into sys.modules
        mocker.patch.dict("sys.modules", {"flashrank": mock_flashrank})

        # Creating ensemble should work (lazy import)
        ensemble = FlashRankEnsemble(models=["model1"])

        # But scoring should raise ImportError
        with pytest.raises(ImportError, match="flashrank"):
            ensemble.score_batch("test query", ["doc1"])
