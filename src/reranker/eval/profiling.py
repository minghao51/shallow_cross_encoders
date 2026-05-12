"""Memory and CPU profiling utilities for benchmark runs.

Provides optional profiling context managers that gracefully degrade
when optional dependencies (memory_profiler, psutil) are unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def memory_profile(label: str = "") -> Iterator[dict[str, float]]:
    """Context manager that tracks peak RSS memory usage.

    Requires ``memory_profiler`` (optional). If unavailable, returns 0.0.

    Yields a dict updated with ``peak_rss_mb`` on exit.
    """
    result: dict[str, float] = {"peak_rss_mb": 0.0}
    try:
        import memory_profiler  # type: ignore[import-untyped]
    except ImportError:
        yield result
        return

    try:
        before = memory_profiler.memory_usage(-1, interval=0.01, timeout=1)
        yield result
        after = memory_profiler.memory_usage(-1, interval=0.01, timeout=1)
    except Exception:
        logger.debug(
            "memory_profile fallback to zero metrics",
            extra={"label": label},
            exc_info=True,
        )
        yield result
        return

    max_before = max(before) if before else 0.0
    max_after = max(after) if after else 0.0
    result["peak_rss_mb"] = max(max_after - max_before, 0.0)
    return


@contextmanager
def cpu_profile(label: str = "") -> Iterator[dict[str, float]]:
    """Context manager that tracks CPU utilization breakdown.

    Requires ``psutil`` (optional). If unavailable, returns 0.0 for all fields.

    Yields a dict updated with ``cpu_user_pct``, ``cpu_system_pct``, ``cpu_idle_pct``.
    """
    result: dict[str, float] = {
        "cpu_user_pct": 0.0,
        "cpu_system_pct": 0.0,
        "cpu_idle_pct": 0.0,
    }
    try:
        import psutil  # type: ignore[import-untyped]
    except ImportError:
        yield result
        return

    try:
        proc = psutil.Process(os.getpid())
        before = proc.cpu_times()
        start = time.perf_counter()
        yield result
        elapsed = time.perf_counter() - start
        after = proc.cpu_times()
    except Exception:
        logger.debug(
            "cpu_profile fallback to zero metrics",
            extra={"label": label},
            exc_info=True,
        )
        yield result
        return

    user_delta = after.user - before.user
    system_delta = after.system - before.system
    total_delta = user_delta + system_delta

    if elapsed > 0 and total_delta > 0:
        result["cpu_user_pct"] = (user_delta / elapsed) * 100.0
        result["cpu_system_pct"] = (system_delta / elapsed) * 100.0
        result["cpu_idle_pct"] = max(0.0, 100.0 - result["cpu_user_pct"] - result["cpu_system_pct"])

    return


def measure_warm_start(
    warmup_fn: Any,
    n_warmup: int = 3,
    n_measure: int = 10,
) -> dict[str, float]:
    """Measure cold-start vs warm latency by running a callable repeatedly.

    The callable is invoked with no arguments. The first call is considered
    the "cold start". Subsequent calls are tracked for p50 warm latency.

    Args:
        warmup_fn: Zero-argument callable to measure.
        n_warmup: Number of warmup iterations (thrown away).
        n_measure: Number of measured iterations.

    Returns:
        Dict with ``cold_start_ms``, ``warm_p50_ms``, ``warm_mean_ms``.
    """
    import statistics

    cold_start_begin = time.perf_counter()
    warmup_fn()
    cold_start_ms = (time.perf_counter() - cold_start_begin) * 1000

    for _ in range(n_warmup):
        warmup_fn()

    samples: list[float] = []
    for _ in range(n_measure):
        start = time.perf_counter()
        warmup_fn()
        elapsed = (time.perf_counter() - start) * 1000
        samples.append(elapsed)

    ordered = sorted(samples)
    return {
        "cold_start_ms": cold_start_ms,
        "warm_p50_ms": float(statistics.median(ordered)) if ordered else 0.0,
        "warm_mean_ms": float(statistics.fmean(ordered)) if ordered else 0.0,
        "warm_min_ms": ordered[0] if ordered else 0.0,
        "warm_max_ms": ordered[-1] if ordered else 0.0,
        "warm_n": len(samples),
    }
