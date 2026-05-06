"""Regression test for C-5: active distillation populates preferences."""

from reranker.data.active_distill import ActiveDistiller


def test_derive_preferences_from_labeled_pairs():
    pairs = [
        {"query": "q1", "doc": "a", "score": 3},
        {"query": "q1", "doc": "b", "score": 1},
        {"query": "q2", "doc": "c", "score": 2},
        {"query": "q2", "doc": "d", "score": 0},
    ]
    prefs = ActiveDistiller._derive_preferences(pairs)

    assert len(prefs) == 2
    assert prefs[0]["query"] == "q1"
    assert prefs[0]["doc_a"] == "a"
    assert prefs[0]["doc_b"] == "b"
    assert prefs[0]["preferred"] == "A"
    assert prefs[0]["confidence"] == 0.95
    assert prefs[1]["preferred"] == "A"
    assert prefs[1]["doc_a"] == "c"
    assert prefs[1]["doc_b"] == "d"


def test_derive_preferences_empty():
    assert ActiveDistiller._derive_preferences([]) == []


def test_derive_preferences_single_doc_per_query():
    pairs = [
        {"query": "q1", "doc": "a", "score": 1},
    ]
    prefs = ActiveDistiller._derive_preferences(pairs)
    assert prefs == []


def test_derive_preferences_generates_multiple_pairs_for_larger_query_group():
    pairs = [
        {"query": "q1", "doc": "best", "score": 3},
        {"query": "q1", "doc": "middle", "score": 2},
        {"query": "q1", "doc": "worst", "score": 0},
    ]
    prefs = ActiveDistiller._derive_preferences(pairs)
    assert len(prefs) == 3
    assert {p["preferred"] for p in prefs} == {"A"}
    pairs_set = {(p["doc_a"], p["doc_b"]) for p in prefs}
    assert ("best", "worst") in pairs_set
    assert ("best", "middle") in pairs_set
    assert ("middle", "worst") in pairs_set


def test_derive_preferences_preserves_required_fields():
    pairs = [
        {"query": "q1", "doc": "a", "score": 1},
        {"query": "q1", "doc": "b", "score": 0},
    ]
    prefs = ActiveDistiller._derive_preferences(pairs)
    assert prefs
    required = {"query", "doc_a", "doc_b", "preferred", "confidence", "generation_mode"}
    for pref in prefs:
        assert required.issubset(pref.keys())
