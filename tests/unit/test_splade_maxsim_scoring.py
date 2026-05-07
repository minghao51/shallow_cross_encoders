"""Regression tests for SPLADE _maxsim_score using dot product (C-3)."""

import pytest

from reranker.strategies.splade import SPLADEReranker


class TestMaxsimScoreDotProduct:
    def test_dot_product_not_min(self) -> None:
        reranker = SPLADEReranker.__new__(SPLADEReranker)
        query_terms = {"machine": 0.8, "learning": 0.6}
        doc_terms = {"machine": 0.5, "learning": 0.9}
        score = reranker._maxsim_score(query_terms, doc_terms)
        expected = 0.8 * 0.5 + 0.6 * 0.9
        assert score == pytest.approx(expected)

    def test_min_would_overestimate(self) -> None:
        reranker = SPLADEReranker.__new__(SPLADEReranker)
        query_terms = {"neural": 0.9}
        doc_terms = {"neural": 0.9}
        dot_score = reranker._maxsim_score(query_terms, doc_terms)
        assert dot_score == pytest.approx(0.81)
        min_score = min(0.9, 0.9)
        assert dot_score <= min_score

    def test_no_overlap_zero_score(self) -> None:
        reranker = SPLADEReranker.__new__(SPLADEReranker)
        query_terms = {"cat": 0.5}
        doc_terms = {"dog": 0.5}
        assert reranker._maxsim_score(query_terms, doc_terms) == 0.0

    def test_empty_terms(self) -> None:
        reranker = SPLADEReranker.__new__(SPLADEReranker)
        assert reranker._maxsim_score({}, {"a": 0.5}) == 0.0
        assert reranker._maxsim_score({"a": 0.5}, {}) == 0.0
        assert reranker._maxsim_score({}, {}) == 0.0

    def test_single_term_overlap(self) -> None:
        reranker = SPLADEReranker.__new__(SPLADEReranker)
        score = reranker._maxsim_score({"search": 0.7}, {"search": 0.3})
        assert score == pytest.approx(0.21)

    def test_partial_overlap(self) -> None:
        reranker = SPLADEReranker.__new__(SPLADEReranker)
        query_terms = {"information": 0.8, "retrieval": 0.6, "model": 0.4}
        doc_terms = {"information": 0.5, "system": 0.3}
        score = reranker._maxsim_score(query_terms, doc_terms)
        assert score == pytest.approx(0.8 * 0.5)
