"""Tests for BM25Engine incremental update() and remove() methods."""

from __future__ import annotations

import numpy as np
import pytest

from reranker.lexical import BM25Engine


class TestBM25Update:
    def test_update_adds_documents(self) -> None:
        engine = BM25Engine()
        engine.fit(["doc one"])
        engine.update(["doc two", "doc three"])
        assert len(engine._corpus) == 3
        assert len(engine._tokenized) == 3

    def test_update_empty_list_noop(self) -> None:
        engine = BM25Engine()
        engine.fit(["doc one"])
        engine.update([])
        assert len(engine._corpus) == 1

    def test_update_on_empty_engine(self) -> None:
        engine = BM25Engine()
        engine.update(["hello world"])
        assert len(engine._corpus) == 1
        scores = engine.score("hello")
        assert scores.shape == (1,)
        assert scores[0] > 0

    def test_update_scores_match_full_rebuild(self) -> None:
        corpus = ["python data analysis", "java web development", "machine learning models"]
        engine_inc = BM25Engine()
        engine_inc.fit(corpus[:1])
        engine_inc.update(corpus[1:])

        engine_full = BM25Engine()
        engine_full.fit(corpus)

        query = "python machine"
        np.testing.assert_array_almost_equal(
            engine_inc.score(query, normalize=False),
            engine_full.score(query, normalize=False),
        )

    def test_update_preserves_relevance_order(self) -> None:
        engine = BM25Engine()
        engine.fit(["java web development"])
        engine.update(["python data analysis", "python machine learning"])

        scores = engine.score("python")
        assert scores[1] > scores[0]
        assert scores[2] > scores[0]

    def test_update_avgdl_correct(self) -> None:
        engine = BM25Engine()
        engine.fit(["a b c"])
        assert engine._avgdl == pytest.approx(3.0)
        engine.update(["d e"])
        assert engine._avgdl == pytest.approx((3 + 2) / 2)

    def test_update_doc_freqs_accumulated(self) -> None:
        engine = BM25Engine()
        engine.fit(["alpha beta"])
        assert engine._doc_freqs["alpha"] == 1
        engine.update(["alpha gamma"])
        assert engine._doc_freqs["alpha"] == 2
        assert engine._doc_freqs["gamma"] == 1


class TestBM25Remove:
    def test_remove_single_document(self) -> None:
        engine = BM25Engine()
        engine.fit(["doc one", "doc two", "doc three"])
        engine.remove([1])
        assert len(engine._corpus) == 2
        assert engine._corpus == ["doc one", "doc three"]

    def test_remove_multiple_documents(self) -> None:
        engine = BM25Engine()
        engine.fit(["a", "b", "c", "d"])
        engine.remove([0, 2])
        assert engine._corpus == ["b", "d"]

    def test_remove_empty_list_noop(self) -> None:
        engine = BM25Engine()
        engine.fit(["doc one"])
        engine.remove([])
        assert len(engine._corpus) == 1

    def test_remove_out_of_range_raises(self) -> None:
        engine = BM25Engine()
        engine.fit(["doc one"])
        with pytest.raises(IndexError, match="out of range"):
            engine.remove([5])

    def test_remove_negative_index_raises(self) -> None:
        engine = BM25Engine()
        engine.fit(["doc one"])
        with pytest.raises(IndexError):
            engine.remove([-1])

    def test_remove_scores_match_full_rebuild(self) -> None:
        corpus = ["python data analysis", "java web development", "machine learning models"]
        engine_inc = BM25Engine()
        engine_inc.fit(corpus)
        engine_inc.remove([1])

        engine_full = BM25Engine()
        engine_full.fit(["python data analysis", "machine learning models"])

        query = "python machine"
        np.testing.assert_array_almost_equal(
            engine_inc.score(query, normalize=False),
            engine_full.score(query, normalize=False),
        )

    def test_remove_updates_doc_freqs(self) -> None:
        engine = BM25Engine()
        engine.fit(["unique alpha", "shared beta", "shared beta"])
        assert engine._doc_freqs["unique"] == 1
        engine.remove([0])
        assert engine._doc_freqs["unique"] == 0
        assert engine._doc_freqs["shared"] == 2

    def test_remove_updates_avgdl(self) -> None:
        engine = BM25Engine()
        engine.fit(["a b c", "d e"])
        engine.remove([0])
        assert engine._avgdl == pytest.approx(2.0)

    def test_remove_all_documents(self) -> None:
        engine = BM25Engine()
        engine.fit(["doc one", "doc two"])
        engine.remove([0, 1])
        assert engine._corpus == []
        scores = engine.score("test")
        assert len(scores) == 0


class TestBM25UpdateRemoveRoundtrip:
    def test_update_then_remove_matches_original(self) -> None:
        base = ["alpha beta", "gamma delta"]
        engine = BM25Engine()
        engine.fit(base)
        engine.update(["epsilon zeta"])
        assert len(engine._corpus) == 3
        engine.remove([2])
        assert engine._corpus == ["alpha beta", "gamma delta"]

        engine_ref = BM25Engine()
        engine_ref.fit(["alpha beta", "gamma delta"])
        query = "alpha gamma"
        np.testing.assert_array_almost_equal(
            engine.score(query, normalize=False),
            engine_ref.score(query, normalize=False),
        )

    def test_multiple_updates_then_remove(self) -> None:
        engine = BM25Engine()
        engine.fit(["doc a"])
        engine.update(["doc b"])
        engine.update(["doc c"])
        engine.remove([1])
        assert engine._corpus == ["doc a", "doc c"]
        scores = engine.score("doc")
        assert len(scores) == 2
