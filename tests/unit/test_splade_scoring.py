"""Regression test for C-3: SPLADE scoring uses dot product, not min."""

from reranker.strategies.splade import SPLADEReranker


def test_maxsim_score_uses_dot_product():
    reranker = SPLADEReranker.__new__(SPLADEReranker)
    query_terms = {"ml": 2.0, "ai": 3.0}
    doc_terms = {"ml": 1.5, "ai": 0.5, "nlp": 4.0}

    score = reranker._maxsim_score(query_terms, doc_terms)

    assert score == 2.0 * 1.5 + 3.0 * 0.5


def test_maxsim_score_with_no_overlap():
    reranker = SPLADEReranker.__new__(SPLADEReranker)
    query_terms = {"x": 1.0}
    doc_terms = {"y": 1.0}

    score = reranker._maxsim_score(query_terms, doc_terms)
    assert score == 0.0


def test_unfitted_splade_rerank_raises():
    reranker = SPLADEReranker()
    import pytest

    with pytest.raises(RuntimeError, match="not fitted"):
        reranker.rerank("query", ["doc1"])
