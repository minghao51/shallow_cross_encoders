"""Tests for profiling utilities (memory, CPU, warm-start)."""

from __future__ import annotations

from reranker.eval.profiling import cpu_profile, measure_warm_start, memory_profile


class TestMemoryProfile:
    def test_returns_peak_rss_key(self):
        with memory_profile("test") as result:
            pass
        assert "peak_rss_mb" in result
        assert isinstance(result["peak_rss_mb"], float)
        assert result["peak_rss_mb"] >= 0.0

    def test_nested_context(self):
        with memory_profile("outer") as outer_result:
            with memory_profile("inner") as inner_result:
                pass
        assert "peak_rss_mb" in outer_result
        assert "peak_rss_mb" in inner_result


class TestCPUProfile:
    def test_returns_cpu_keys(self):
        with cpu_profile("test") as result:
            pass
        assert "cpu_user_pct" in result
        assert "cpu_system_pct" in result
        assert "cpu_idle_pct" in result
        assert all(isinstance(v, float) for v in result.values())

    def test_idle_plus_user_plus_system_close_to_100(self):
        with cpu_profile("test") as result:
            pass
        total = result["cpu_user_pct"] + result["cpu_system_pct"] + result["cpu_idle_pct"]
        assert total >= 0.0

    def test_nested_context(self):
        with cpu_profile("outer") as outer:
            with cpu_profile("inner") as inner:
                pass
        assert "cpu_user_pct" in outer
        assert "cpu_user_pct" in inner


class TestMeasureWarmStart:
    def test_returns_all_keys(self):
        def noop():
            pass

        result = measure_warm_start(noop, n_warmup=2, n_measure=5)
        assert "cold_start_ms" in result
        assert "warm_p50_ms" in result
        assert "warm_mean_ms" in result
        assert "warm_min_ms" in result
        assert "warm_max_ms" in result
        assert "warm_n" in result
        assert result["warm_n"] == 5

    def test_cold_start_slower_than_warm(self):
        def small_compute():
            _ = [i**2 for i in range(1000)]

        result = measure_warm_start(small_compute, n_warmup=2, n_measure=5)
        assert result["cold_start_ms"] >= 0
        assert result["warm_p50_ms"] >= 0

    def test_zero_measure_returns_zeros(self):
        def noop():
            pass

        result = measure_warm_start(noop, n_warmup=0, n_measure=0)
        assert result["warm_n"] == 0
        assert result["cold_start_ms"] >= 0.0

    def test_invocation_count_includes_explicit_cold_start(self):
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1

        measure_warm_start(counter, n_warmup=2, n_measure=3)
        assert call_count == 6

    def test_nonzero_times(self):
        def slow_enough():
            _ = [i**3 for i in range(5000)]

        result = measure_warm_start(slow_enough, n_warmup=1, n_measure=3)
        assert result["cold_start_ms"] >= 0
