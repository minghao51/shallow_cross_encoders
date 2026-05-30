from scripts.benchmark_quantization import estimate_cache_hits


def test_estimate_cache_hits_counts_duplicates():
    stats = estimate_cache_hits(["a", "b", "a", "c", "b"])
    assert stats["total_texts"] == 5
    assert stats["unique_texts"] == 3
    assert stats["estimated_cache_hits"] == 2
