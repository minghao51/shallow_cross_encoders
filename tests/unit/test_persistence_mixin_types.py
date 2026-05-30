"""Tests for persistence_mixin.py and types.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reranker.persistence_mixin import MAX_DOCS, MAX_QUERY_LENGTH, SaveableReranker
from reranker.protocols import NotFittedError
from reranker.types import RankedDoc


class TestRankedDoc:
    def test_ranked_doc_creation(self) -> None:
        doc = RankedDoc(doc="test doc", score=0.85, rank=1)
        assert doc.doc == "test doc"
        assert doc.score == 0.85
        assert doc.rank == 1
        assert doc.metadata == {}

    def test_ranked_doc_with_metadata(self) -> None:
        doc = RankedDoc(doc="test", score=0.5, rank=2, metadata={"strategy": "hybrid"})
        assert doc.metadata["strategy"] == "hybrid"

    def test_ranked_doc_equality(self) -> None:
        doc_a = RankedDoc(doc="same", score=0.9, rank=1)
        doc_b = RankedDoc(doc="same", score=0.9, rank=1)
        assert doc_a == doc_b


class TestSaveableRerankerValidation:
    def test_validate_inputs_passes_for_normal_input(self) -> None:
        SaveableReranker._validate_inputs("short query", ["doc1", "doc2"])

    def test_validate_inputs_raises_on_long_query(self) -> None:
        with pytest.raises(ValueError, match="Query exceeds max length"):
            SaveableReranker._validate_inputs("x" * (MAX_QUERY_LENGTH + 1), ["doc1"])

    def test_validate_inputs_raises_on_too_many_docs(self) -> None:
        with pytest.raises(ValueError, match="Too many documents"):
            SaveableReranker._validate_inputs("query", ["d"] * (MAX_DOCS + 1))

    def test_validate_inputs_accepts_max_bounds(self) -> None:
        SaveableReranker._validate_inputs("x" * MAX_QUERY_LENGTH, ["d"] * 100)


class TestSaveableRerankerRequireFitted:
    def test_require_fitted_raises_when_not_fitted(self) -> None:
        obj = MagicMock(spec=SaveableReranker, is_fitted=False)
        with pytest.raises(NotFittedError, match="is not fitted"):
            SaveableReranker._require_fitted(obj, "TestStrategy")

    def test_require_fitted_passes_when_fitted(self) -> None:
        obj = MagicMock(spec=SaveableReranker, is_fitted=True)
        SaveableReranker._require_fitted(obj, "TestStrategy")


class TestSaveableRerankerDefaults:
    def test_default_artifact_type_is_empty_string(self) -> None:
        class MinimalReranker(SaveableReranker):
            pass

        assert MinimalReranker._artifact_type == ""

    def test_save_metadata_default_returns_model_name(self) -> None:
        class MinimalReranker(SaveableReranker):
            pass

        obj = MinimalReranker()
        meta = obj._save_metadata()
        assert "embedder_model_name" in meta
