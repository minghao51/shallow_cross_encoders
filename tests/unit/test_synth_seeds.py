"""Unit tests for data/synth/_seeds.py — default seed data for pair generation."""

from __future__ import annotations

from reranker.data.synth._seeds import DEFAULT_PAIR_SEEDS

REQUIRED_KEYS = {"query", "domain", "variations", "positive", "negative"}


class TestModuleExports:
    """Tests that module exports exist and have correct types."""

    def test_default_pair_seeds_is_list(self) -> None:
        assert isinstance(DEFAULT_PAIR_SEEDS, list)

    def test_default_pair_seeds_not_empty(self) -> None:
        assert len(DEFAULT_PAIR_SEEDS) > 0


class TestSeedStructure:
    """Tests for individual seed structure."""

    def test_all_seeds_have_required_keys(self) -> None:
        for seed in DEFAULT_PAIR_SEEDS:
            assert REQUIRED_KEYS.issubset(seed.keys()), f"Missing keys in: {seed.get('query', '?')}"

    def test_all_seeds_have_no_extra_keys(self) -> None:
        for seed in DEFAULT_PAIR_SEEDS:
            assert seed.keys() == REQUIRED_KEYS, f"Unexpected keys in: {seed['query']}"

    def test_query_is_nonempty_string(self) -> None:
        for seed in DEFAULT_PAIR_SEEDS:
            assert isinstance(seed["query"], str) and len(seed["query"]) > 0

    def test_domain_is_nonempty_string(self) -> None:
        for seed in DEFAULT_PAIR_SEEDS:
            assert isinstance(seed["domain"], str) and len(seed["domain"]) > 0

    def test_positive_is_nonempty_string(self) -> None:
        for seed in DEFAULT_PAIR_SEEDS:
            assert isinstance(seed["positive"], str) and len(seed["positive"]) > 0

    def test_negative_is_nonempty_string(self) -> None:
        for seed in DEFAULT_PAIR_SEEDS:
            assert isinstance(seed["negative"], str) and len(seed["negative"]) > 0

    def test_variations_is_nonempty_list(self) -> None:
        for seed in DEFAULT_PAIR_SEEDS:
            assert isinstance(seed["variations"], list) and len(seed["variations"]) > 0

    def test_variations_are_strings(self) -> None:
        for seed in DEFAULT_PAIR_SEEDS:
            for v in seed["variations"]:
                assert isinstance(v, str) and len(v) > 0


class TestSeedUniqueness:
    """Tests for seed data uniqueness."""

    def test_no_duplicate_queries(self) -> None:
        queries = [s["query"] for s in DEFAULT_PAIR_SEEDS]
        assert len(queries) == len(set(queries))

    def test_no_duplicate_positive_texts(self) -> None:
        positives = [s["positive"] for s in DEFAULT_PAIR_SEEDS]
        assert len(positives) == len(set(positives))


class TestSeedDomains:
    """Tests for domain coverage."""

    def test_at_least_three_domains(self) -> None:
        domains = {s["domain"] for s in DEFAULT_PAIR_SEEDS}
        assert len(domains) >= 3

    def test_domain_names_are_snake_case(self) -> None:
        for seed in DEFAULT_PAIR_SEEDS:
            domain = seed["domain"]
            assert domain == domain.lower()
            assert " " not in domain
