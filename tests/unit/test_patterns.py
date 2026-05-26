"""Unit tests for strategies/patterns.py — regex patterns for consistency engine."""

from __future__ import annotations

import re

from reranker.strategies.patterns import (
    ENTITY_PATTERN,
    SENTENCE_SPLIT_PATTERN,
    STRUCTURED_PATTERN_HINTS,
    STRUCTURED_PATTERNS,
    VALUE_PATTERN,
)


class TestModuleExports:
    """Tests that module exports exist and have correct types."""

    def test_entity_pattern_is_str(self) -> None:
        assert isinstance(ENTITY_PATTERN, str)
        assert len(ENTITY_PATTERN) > 0

    def test_value_pattern_is_str(self) -> None:
        assert isinstance(VALUE_PATTERN, str)
        assert len(VALUE_PATTERN) > 0

    def test_sentence_split_is_compiled(self) -> None:
        assert isinstance(SENTENCE_SPLIT_PATTERN, re.Pattern)

    def test_structured_patterns_is_tuple(self) -> None:
        assert isinstance(STRUCTURED_PATTERNS, tuple)

    def test_structured_patterns_all_compiled(self) -> None:
        for p in STRUCTURED_PATTERNS:
            assert isinstance(p, re.Pattern)

    def test_structured_pattern_hints_is_tuple(self) -> None:
        assert isinstance(STRUCTURED_PATTERN_HINTS, tuple)

    def test_hints_match_patterns_count(self) -> None:
        assert len(STRUCTURED_PATTERN_HINTS) == len(STRUCTURED_PATTERNS)


class TestSentenceSplitPattern:
    """Tests for SENTENCE_SPLIT_PATTERN."""

    def test_splits_on_period_space(self) -> None:
        parts = SENTENCE_SPLIT_PATTERN.split("First sentence. Second sentence.")
        assert "First sentence." in parts
        assert "Second sentence." in parts

    def test_splits_on_semicolon(self) -> None:
        parts = SENTENCE_SPLIT_PATTERN.split("Part one; part two")
        assert "Part one" in parts
        assert "part two" in parts

    def test_splits_on_exclamation(self) -> None:
        parts = SENTENCE_SPLIT_PATTERN.split("Stop! Go away.")
        assert "Stop!" in parts

    def test_splits_on_question_mark(self) -> None:
        parts = SENTENCE_SPLIT_PATTERN.split("Is this real? Yes it is.")
        assert "Is this real?" in parts


class TestStructuredPatterns:
    """Tests for STRUCTURED_PATTERNS matching."""

    def test_entity_attribute_value_match(self) -> None:
        text = "ModelX reports accuracy as 95%"
        matched = False
        for p in STRUCTURED_PATTERNS:
            m = p.search(text)
            if m and "entity" in m.groupdict() and m.group("entity"):
                matched = True
                break
        assert matched

    def test_entity_possessive_match(self) -> None:
        text = "ModelX's accuracy is 95%"
        matched = False
        for p in STRUCTURED_PATTERNS:
            m = p.search(text)
            if m and "entity" in m.groupdict() and m.group("entity"):
                matched = True
                break
        assert matched

    def test_has_attribute_of_match(self) -> None:
        text = "ModelX has a latency of 7ms"
        matched = False
        for p in STRUCTURED_PATTERNS:
            m = p.search(text)
            if m and "entity" in m.groupdict() and m.group("entity"):
                matched = True
                break
        assert matched

    def test_no_match_on_gibberish(self) -> None:
        text = "xyzzy nothing to see here"
        for p in STRUCTURED_PATTERNS:
            assert p.search(text) is None

    def test_all_patterns_have_named_groups(self) -> None:
        for p in STRUCTURED_PATTERNS:
            names = set(p.groupindex.keys())
            assert "entity" in names or "value" in names


class TestPatternHints:
    """Tests for STRUCTURED_PATTERN_HINTS."""

    def test_each_hint_is_tuple_of_strings(self) -> None:
        for hints in STRUCTURED_PATTERN_HINTS:
            assert isinstance(hints, tuple)
            for h in hints:
                assert isinstance(h, str)

    def test_hints_are_lowercase(self) -> None:
        for hints in STRUCTURED_PATTERN_HINTS:
            for h in hints:
                assert h == h.lower() or h in (":", "=", " - ")
