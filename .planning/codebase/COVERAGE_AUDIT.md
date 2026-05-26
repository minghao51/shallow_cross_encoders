# Coverage Audit Report

**Date:** 2026-05-25
**Command:** `uv run pytest --cov=reranker --cov-report=term-missing -m "not llm and not slow" --tb=no -q`
**Tests run:** 692 passed, 1 skipped, 17 deselected (4m23s)

## Global Coverage

| Metric | Value |
|--------|-------|
| **Total statements** | 5,091 |
| **Missing coverage** | 911 |
| **Global coverage** | **82.11%** |
| **fail_under threshold** | 85% |
| **Status** | FAIL (2.89pp below threshold) |

## Coverage Configuration

```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/__init__.py", "*/__main__.py"]

[tool.coverage.report]
fail_under = 85
exclude_lines = ["pragma: no cover", "def __repr__", "raise NotImplementedError",
                 "if TYPE_CHECKING:", "if __name__ == .__main__.:"]

# No per-package or per-file minimums configured.
```

## Files Below 85% Coverage (sorted lowest first)

| # | File | Stmts | Miss | Cover | Missing Lines |
|---|------|-------|------|-------|---------------|
| 1 | `cli/train.py` | 153 | 127 | **17%** | 29-40, 50-56, 65-83, 92-112, 123-140, 151-166, 180-223, 232-265, 275-292, 296-303, 313-321, 326-336 |
| 2 | `data/hard_negative_sampler.py` | 78 | 58 | **26%** | 69-84, 126-221 |
| 3 | `cli/eval.py` | 18 | 11 | **39%** | 31-43 |
| 4 | `data/beir_loader.py` | 122 | 68 | **44%** | 45-46, 60, 75, 86, 135-136, 141-143, 149-236 |
| 5 | `strategies/splade.py` | 77 | 43 | **44%** | 36-43, 49-64, 67-92, 103, 106-107, 110, 113, 117-124 |
| 6 | `strategies/flashrank_ensemble.py` | 110 | 46 | **58%** | 30, 33-34, 59-62, 86-95, 100-106, 123-124, 127-137, 140-147, 162-172 |
| 7 | `cli/benchmark.py` | 92 | 36 | **61%** | 18-47, 55-68, 75-83, 116 |
| 8 | `strategies/hybrid_persistence.py` | 75 | 26 | **65%** | 46, 53-64, 87-114, 137, 148 |
| 9 | `data/synth/generator/contradictions.py` | 54 | 16 | **70%** | 115-152 |
| 10 | `strategies/consistency.py` | 273 | 80 | **71%** | 116, 153, 157, 159, 185, 209, 215, 287, 296, 327-340, 354-372, 399-463, 466, 473, 477-478 |
| 11 | `data/synth/generator/preferences.py` | 77 | 20 | **74%** | 29-30, 83-121, 169, 172 |
| 12 | `data/synth/generator/pairs.py` | 78 | 17 | **78%** | 95, 102-137 |
| 13 | `eval/runner.py` | 207 | 46 | **78%** | 62, 74, 158, 186, 244, 272-282, 291-304, 313-331, 340-361, 378, 437 |
| 14 | `data/litellm_client.py` | 44 | 8 | **82%** | 22-27, 54, 98-99 |
| 15 | `eval/benchmark_utils.py` | 68 | 12 | **82%** | 149-152, 156-161, 165-167, 170-172 |
| 16 | `strategies/distilled.py` | 212 | 35 | **83%** | 104-105, 130-131, 146-151, 165, 175, 206-211, 226, 240-244, 250-251, 294-295, 343, 349, 354-355, 374-387 |

## Files at 0% Coverage

None. All source files under `src/reranker/` have at least partial test coverage.

## Files at 100% Coverage

`data/_expanded/contradictions.py`, `data/_expanded/helpers.py` (95% — close), `data/_expanded/pairs.py`, `data/_expanded/preferences.py`, `data/_expanded/seeds.py`, `data/_expanded/types.py`, `data/synth/_models.py`, `data/synth/_prompts.py`, `data/synth/_seeds.py`, `data/expanded.py`, `cli/generate.py`, `cli/serve.py`, `deps.py`, `eval/statistics.py`, `heuristics/keyword.py`, `heuristics/lsh.py`, `lexical.py`, `logging_config.py`, `protocols.py`, `strategies/patterns.py`, `strategies/pipeline.py`, `types.py`, `config.py` (97%)

## Analysis by Package

| Package | Files Below 85% | Worst File | Avg Coverage (approx) |
|---------|-----------------|------------|-----------------------|
| `cli/` | 3 of 6 | `train.py` (17%) | ~60% |
| `data/` | 5 of ~15 | `hard_negative_sampler.py` (26%) | ~75% |
| `strategies/` | 5 of 12 | `splade.py` (44%) | ~78% |
| `eval/` | 2 of 5 | `runner.py` (78%) | ~87% |

## Comparison with CONCERNS.md Audit Findings

| Concern (CONCERNS.md) | Coverage Reality |
|------------------------|-----------------|
| "Large files with unverified test coverage" in `benchmark.py`, `strategies/hybrid.py` | `hybrid.py` is at 95% (well-covered). `benchmark.py` (old name) may have been refactored into `cli/benchmark.py` at 61% and `eval/benchmark_utils.py` at 82% — both below threshold. |
| "Synthetic data generation is fragile" (`data/synth/`) | 3 of 8 synth generator files below 85%. `enhanced.py` at 48% is the weakest. |
| "Error handling paths not tested" | Confirmed: many `except` blocks in low-coverage files like `train.py` (17%), `beir_loader.py` (44%), and `consistency.py` (71%) are untested. |

## Recommendations

### 1. Do NOT raise `fail_under` — lower it temporarily or fix coverage

The project is at 82.11%, already below the 85% threshold. Either:
- **Option A (recommended):** Fix the worst offenders (see priority list below) to get above 85%
- **Option B:** Temporarily lower `fail_under` to 80 to unblock CI while coverage is improved incrementally

### 2. Priority files for test improvement

| Priority | File | Current | Why |
|----------|------|---------|-----|
| P0 | `cli/train.py` | 17% | Largest single gap (127 missed stmts). Entire CLI train workflow untested. |
| P0 | `data/hard_negative_sampler.py` | 26% | Core sampling logic untested (lines 126-221). |
| P1 | `data/beir_loader.py` | 44% | Data loading paths (lines 149-236) not exercised. |
| P1 | `strategies/splade.py` | 44% | Sparse vector strategy mostly untested. |
| P1 | `strategies/flashrank_ensemble.py` | 58% | Ensemble scoring paths untested. |
| P1 | `strategies/consistency.py` | 71% | High stmt count (273) — 80 missed lines is significant. |
| P2 | `cli/benchmark.py` | 61% | CLI command paths untested. |
| P2 | `strategies/hybrid_persistence.py` | 65% | Save/load paths untested. |
| P2 | `eval/runner.py` | 78% | Core eval orchestrator, many untested branches. |

### 3. Files to consider for `coverage.run.omit`

| File | Justification |
|------|---------------|
| `cli/train.py` | CLI-only orchestration; test via integration tests separately. Only if P0 testing is deferred. |
| `strategies/splade.py` | Requires GPU/Sparse model deps not available in CI. Consider omitting if deps remain optional. |

**Recommendation:** Do NOT omit these. Instead, add integration tests with mocked dependencies.

### 4. Add per-package coverage minimums

Recommended additions to `pyproject.toml`:

```toml
[tool.coverage.report]
exclude_also = []
fail_under = 85

# Consider per-module thresholds via coverage.py fail_under in CI:
# --cov-fail-under=85 for global, plus custom checks per package
```

pytest-cov does not natively support per-package `fail_under`. To enforce per-package minimums, use a custom CI script or the `coverage` JSON report with post-processing.

### 5. `__init__.py` / `__main__.py` omits

The current `omit` excludes `*/__init__.py` and `*/__main__.py`. This is appropriate — these files are typically re-exports or entry points and don't contain testable logic. No low-coverage modules are hidden by this omission.

## Impact Summary

Bringing just the top 3 files (`cli/train.py`, `data/hard_negative_sampler.py`, `cli/eval.py`) to 85% would recover approximately **~150 statements**, raising global coverage to roughly **~85%** and meeting the threshold.

---

*Coverage audit: 2026-05-25*
