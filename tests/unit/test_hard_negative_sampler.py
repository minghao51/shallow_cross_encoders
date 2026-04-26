"""Tests for hard negative sampler."""

from pathlib import Path

import pytest


def test_prepare_benchmark_data_invalid_ratio():
    """Test prepare_benchmark_data_with_hard_negatives validates ratio."""
    from reranker.data.hard_negative_sampler import prepare_benchmark_data_with_hard_negatives

    dataset = {
        "corpus": {"doc1": {"_id": "doc1", "title": "", "text": "Test doc"}},
        "queries": {"q1": "Test query"},
        "qrels": {"q1": {"doc1": 2}},
    }

    # Test ratio > 1.0
    with pytest.raises(ValueError, match="hard_negative_ratio must be between 0.0 and 1.0"):
        prepare_benchmark_data_with_hard_negatives(
            dataset, hard_negative_ratio=1.5, num_queries=1, docs_per_query=10
        )

    # Test ratio < 0.0
    with pytest.raises(ValueError, match="hard_negative_ratio must be between 0.0 and 1.0"):
        prepare_benchmark_data_with_hard_negatives(
            dataset, hard_negative_ratio=-0.1, num_queries=1, docs_per_query=10
        )


def test_bm25_index_cache_init():
    """Test BM25IndexCache initialization."""
    from reranker.data.hard_negative_sampler import BM25IndexCache

    cache = BM25IndexCache(Path("/tmp/test_cache"))
    assert cache.cache_dir == Path("/tmp/test_cache")


def test_bm25_index_cache_get_cache_key():
    """Test BM25IndexCache generates consistent keys."""
    from reranker.data.hard_negative_sampler import BM25IndexCache

    cache = BM25IndexCache(Path("/tmp/test_cache"))

    corpus = ["doc1", "doc2", "doc3"]
    key1 = cache._get_cache_key(corpus)
    key2 = cache._get_cache_key(corpus)

    # Same corpus should generate same key
    assert key1 == key2
    assert len(key1) == 16  # 16-character hex string

    # Different corpus should generate different key
    different_corpus = ["doc1", "doc2", "different"]
    key3 = cache._get_cache_key(different_corpus)
    assert key3 != key1


def test_prepare_benchmark_data_basic():
    """Test prepare_benchmark_data_with_hard_negatives basic functionality."""
    from reranker.data.hard_negative_sampler import prepare_benchmark_data_with_hard_negatives

    # Just verify the function exists and is callable
    assert callable(prepare_benchmark_data_with_hard_negatives)
