"""Regression tests for active_distill._derive_preferences (C-5)."""

from reranker.data.active_distill import ActiveDistiller


class TestDerivePreferences:
    def test_returns_preferences_for_multi_score_pairs(self) -> None:
        pairs = [
            {"query": "python tutorial", "doc": "python basics guide", "score": 2},
            {"query": "python tutorial", "doc": "java for beginners", "score": 0},
            {"query": "python tutorial", "doc": "python advanced tips", "score": 1},
        ]
        prefs = ActiveDistiller._derive_preferences(pairs)
        assert len(prefs) == 3

    def test_high_vs_low_preference(self) -> None:
        pairs = [
            {"query": "q1", "doc": "good_doc", "score": 2},
            {"query": "q1", "doc": "bad_doc", "score": 0},
        ]
        prefs = ActiveDistiller._derive_preferences(pairs)
        assert len(prefs) == 1
        assert prefs[0]["doc_a"] == "good_doc"
        assert prefs[0]["doc_b"] == "bad_doc"
        assert prefs[0]["preferred"] == "A"

    def test_three_score_levels_produce_three_preferences(self) -> None:
        pairs = [
            {"query": "q", "doc": "high", "score": 2},
            {"query": "q", "doc": "mid", "score": 1},
            {"query": "q", "doc": "low", "score": 0},
        ]
        prefs = ActiveDistiller._derive_preferences(pairs)
        assert len(prefs) == 3
        doc_a_set = {p["doc_a"] for p in prefs}
        assert "high" in doc_a_set
        assert "mid" in doc_a_set

    def test_empty_pairs_returns_empty(self) -> None:
        assert ActiveDistiller._derive_preferences([]) == []

    def test_single_pair_returns_empty(self) -> None:
        pairs = [{"query": "q", "doc": "d", "score": 1}]
        assert ActiveDistiller._derive_preferences(pairs) == []

    def test_generation_mode_set(self) -> None:
        pairs = [
            {"query": "q", "doc": "a", "score": 1},
            {"query": "q", "doc": "b", "score": 0},
        ]
        prefs = ActiveDistiller._derive_preferences(pairs)
        assert all(p["generation_mode"] == "active_distill" for p in prefs)

    def test_multiple_queries_independent(self) -> None:
        pairs = [
            {"query": "q1", "doc": "a1", "score": 1},
            {"query": "q1", "doc": "b1", "score": 0},
            {"query": "q2", "doc": "a2", "score": 1},
            {"query": "q2", "doc": "b2", "score": 0},
        ]
        prefs = ActiveDistiller._derive_preferences(pairs)
        assert len(prefs) == 2
        queries = {p["query"] for p in prefs}
        assert queries == {"q1", "q2"}
