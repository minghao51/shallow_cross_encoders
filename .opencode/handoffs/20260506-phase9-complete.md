# Handoff: Phase 9 Complete — Phase 10 Ready

## Current State

Phase 9 (Benchmarking Maturity) is fully delivered. All exit criteria met:

- **Statistical rigor**: `bootstrap_ci()`, `wilcoxon_signed_rank()`, `compare_strategies()` in `src/reranker/eval/statistics.py`. 19 tests.
- **Profiling**: `memory_profile`, `cpu_profile`, `measure_warm_start()` in `src/reranker/eval/profiling.py`. 8 tests. Opt-in via `--profile` flag.
- **Visualization**: `plot_pareto_frontier()`, `plot_radar()`, `generate_comparison_table()` in `src/reranker/eval/viz.py`. 11 tests. Auto-called from `save_results()`.
- **Sweep configs**: 9 YAML files in `benchmarks/configs/` (4 new: cascade, binary, pipeline, distilled).
- **Sweep runner**: `benchmarks/run_sweep.py` extended to support cascade/binary/pipeline/distilled variant builders.
- **CLI compare**: `reranker benchmark compare` subcommand with CI + Wilcoxon.
- **All 395 pre-existing tests pass** with zero regressions.

## Files Modified in Phase 9

```
src/reranker/eval/statistics.py    — NEW: bootstrap CI, Wilcoxon, compare_strategies
src/reranker/eval/profiling.py     — NEW: MemoryProfiler, CPUProfiler, warm-start
src/reranker/eval/viz.py           — NEW: Pareto plot, radar, comparison table
benchmarks/runner.py               — MODIFIED: CI integration, profiling, viz in save_results
benchmarks/run_sweep.py            — MODIFIED: cascade/binary/pipeline/distilled support
src/reranker/cli/benchmark.py      — MODIFIED: compare subcommand, --profile flag
benchmarks/configs/sweep_cascade.yaml   — NEW
benchmarks/configs/sweep_binary.yaml    — NEW
benchmarks/configs/sweep_pipeline.yaml  — NEW
benchmarks/configs/sweep_distilled.yaml — NEW
tests/unit/test_statistics.py      — NEW: 19 tests
tests/unit/test_profiling.py       — NEW: 8 tests
tests/unit/test_viz.py             — NEW: 11 tests
tests/unit/test_sweep_configs.py   — NEW: 10 tests
pyproject.toml                     — MODIFIED: profiling optional-deps group
justfile                           — MODIFIED: benchmark-viz, benchmark-compare targets
tests/conftest.py                  — MODIFIED: auto-mark benchmarks/ as slow
```

## Test Infrastructure Changes

- Default `pytest` now excludes both `llm` AND `slow` markers (`-m "not llm and not slow"`)
- `tests/benchmarks/` (7 tests) auto-marked as `unit` + `slow` via conftest
- New just targets: `test-quick`, `test-e2e`, `test-slow`, `test-full`
- `just test` = 551 tests (excludes llm + slow)
- `just test-full` = 558 tests (excludes llm only, includes slow benchmarks)
- `just test-slow` = 7 benchmark tests

## Next Phase: Phase 10 — Feature Expansion

Phase 10 has 5 major work items from the roadmap:

### Priority Order

1. **Structured Logging (10.4)** — Easiest, broadest impact
   - Replace all `print()` with `structlog` or `logging`
   - Add correlation IDs for request tracing
   - Configurable log levels per strategy
   - File: search all `*.py` for `print(` statements

2. **QueryExpansionReranker (10.1)** — Medium complexity, reusable infra
   - New first-class strategy wrapping template-based expansion
   - Configurable expansion count and template set
   - RRF fusion of original + expanded query results
   - File: new `strategies/query_expansion.py`

3. **Incremental BM25 (10.5)** — Standalone, well-scoped
   - `BM25Engine.update(docs)` for incremental index updates
   - `BM25Engine.remove(doc_ids)` for document deletion
   - Maintain index statistics across updates
   - File: `lexical.py` (modify `BM25Engine`)

4. **AutoReranker / Strategy Auto-Selection (10.2)** — Medium complexity
   - Lightweight classifier on benchmark metadata
   - `AutoReranker.from_config("config.yaml")` API
   - Fallback to HybridFusion when confidence low
   - File: new `strategies/auto_reranker.py`

5. **ONNX Export (10.3)** — Highest complexity, optional dep
   - Export StaticColBERT MaxSim as ONNX graph
   - Export BinaryQuantizedReranker scoring as ONNX graph
   - Optional: `onnxruntime` for accelerated inference
   - Files: `strategies/late_interaction.py`, `strategies/binary_reranker.py`

### Architecture Invariants

- `reranker.rerank(query, docs) -> list[RankedDoc]` must never change
- CPU-native only; GPU optional (ONNX is CPU-accelerated, not GPU)
- Drop-in compatibility: existing imports must not break

### Risk Notes

- ONNX export is fragile with numpy version changes — pin in tests
- Auto-selection classifier may overfit to benchmark datasets — hold out multiple
- Structured logging touches ~20+ files — systematic search needed

### Dependencies

Phase 10 depends on Phase 9 being stable (it is). Phases 10 + 11 are independent and could be parallelized across two agents.
