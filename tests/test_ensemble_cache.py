"""Tests for EnsembleLabelCache."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reranker.data.ensemble_cache import EnsembleLabelCache


@pytest.mark.unit
class TestEnsembleLabelCache:
    def test_get_hash_deterministic(self, tmp_path):
        """Same inputs produce same 16-char hash."""
        cache = EnsembleLabelCache(cache_dir=tmp_path)

        hash1 = cache.get_hash("dataset", ["teacher1", "teacher2"])
        hash2 = cache.get_hash("dataset", ["teacher1", "teacher2"])

        assert hash1 == hash2
        assert len(hash1) == 16

    def test_get_hash_different_inputs(self, tmp_path):
        """Different inputs produce different hashes."""
        cache = EnsembleLabelCache(cache_dir=tmp_path)

        hash1 = cache.get_hash("dataset", ["teacher1", "teacher2"])
        hash2 = cache.get_hash("dataset", ["teacher1", "teacher3"])
        hash3 = cache.get_hash("other_dataset", ["teacher1", "teacher2"])

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3
        assert all(len(h) == 16 for h in [hash1, hash2, hash3])

    def test_load_or_generate_miss(self, tmp_path):
        """Cache miss calls generator, creates cache file."""
        cache = EnsembleLabelCache(cache_dir=tmp_path)

        generator_fn = MagicMock(return_value={"labels": [1, 2, 3]})
        labels = cache.load_or_generate("dataset", ["teacher1"], generator_fn)

        assert labels == {"labels": [1, 2, 3]}
        assert generator_fn.called

        # Verify cache file was created
        hash_key = cache.get_hash("dataset", ["teacher1"])
        cache_file = tmp_path / f"{hash_key}.json"
        assert cache_file.exists()

    def test_load_or_generate_hit(self, tmp_path):
        """Cache hit loads from file, doesn't call generator."""
        cache = EnsembleLabelCache(cache_dir=tmp_path)

        # Pre-populate cache
        from reranker.utils import write_json

        hash_key = cache.get_hash("dataset", ["teacher1"])
        cache_file = tmp_path / f"{hash_key}.json"
        expected_labels = {"labels": [4, 5, 6]}
        write_json(cache_file, expected_labels)

        # Load from cache
        generator_fn = MagicMock(return_value={"labels": [1, 2, 3]})
        labels = cache.load_or_generate("dataset", ["teacher1"], generator_fn)

        assert labels == expected_labels
        assert not generator_fn.called

    def test_load_or_generate_force(self, tmp_path):
        """force_regenerate bypasses cache."""
        cache = EnsembleLabelCache(cache_dir=tmp_path)

        # Pre-populate cache
        from reranker.utils import write_json

        hash_key = cache.get_hash("dataset", ["teacher1"])
        cache_file = tmp_path / f"{hash_key}.json"
        old_labels = {"labels": [4, 5, 6]}
        write_json(cache_file, old_labels)

        # Force regeneration
        generator_fn = MagicMock(return_value={"labels": [1, 2, 3]})
        labels = cache.load_or_generate(
            "dataset", ["teacher1"], generator_fn, force_regenerate=True
        )

        assert labels == {"labels": [1, 2, 3]}
        assert generator_fn.called
