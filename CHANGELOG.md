# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## [Unreleased]

### Added

- **Phase 7 — Performance & Optimization**
  - Shared `EmbeddingCache` singleton with `cachetools.TTLCache`, keyed by `(text, model_name)`, configurable size and TTL (`src/reranker/embedding_cache.py`)
  - `BinaryQuantizedReranker` doc encoding cache (LRU via `OrderedDict`), invalidated on refit
  - `StaticColBERTReranker` pre-built `TokenIndex` reuse via `prebuilt_indices` kwarg
  - `StaticColBERTReranker.rerank_batch()` for batch query encoding and MaxSim matrix computation
  - `HybridFusionReranker.rerank_batch()` with batch embedding pre-computation
  - `int8` quantization mode (`quantize_int8` / `dequantize_int8`)
  - `float16` storage option (`quantize_float16` / `dequantize_float16`)
  - Quantization benchmark script comparing 5 modes (4bit, int8, float16, ternary, none)

### Changed

- **Phase 7 — Algorithmic fixes**
  - `DistilledPairwiseRanker._merge_rank` now uses `collections.deque` instead of `list.pop(0)` — O(n log n) instead of O(n^2 log n)
  - `StaticColBERT._compute_salience` vectorized with `np.unique` and numpy ops — no Python for-loop over tokens
  - 4-bit quantization batch-vectorized with numpy fancy indexing — no per-byte loop

### Fixed

- **Phase 6 — Correctness Bugs (C-1 through C-6)**
  - `PipelineReranker.run_pipeline()` tracks docs through stages via `RankedDoc` extraction instead of zip alignment (C-1)
  - `BinaryQuantizedReranker` raises `RuntimeError` on unfitted `rerank()` / `score()` instead of auto-fitting with all-1s labels (C-2)
  - `SPLADEReranker._maxsim_score()` uses proper dot product `qw * dt` instead of `min(qw, dt)` (C-3)
  - `HybridFusionReranker` raises `RuntimeError` on unfitted state for all weighting modes (C-4)
  - `active_distill.py` `_derive_preferences()` now fully populates `new_preferences` with high-vs-low, high-vs-mid, mid-vs-low records (C-5)
  - Removed duplicate `model_config = ConfigDict(extra="forbid")` in `data/synth/_models.py` (C-6)

- **Phase 6 — Lifecycle Consistency**
  - `is_fitted` attribute standardized across all 6 strategies (`False` in `__init__`, `True` after `fit()` or `load()`)
  - Auto-fit behavior removed from `BinaryQuantizedReranker`, `StaticColBERTReranker`, `SPLADEReranker`
  - `CascadeReranker` fallback_strategy now validates against `FallbackStrategy` enum
  - `CASCADE_SCORE_VARIANCE` moved to top-level `ConfidenceMetric` enum

- **Phase 6 — Architecture Cleanup**
  - `multi.py` imports `BaseReranker` from `protocols` instead of defining a local Protocol
  - `FlashRankEnsemble` now persistable with `save()` / `load()` methods
  - `SentenceTransformerWrapper` exported from `adapters/__init__.py`
  - `WeightingMode` enum validation in `HybridFusionReranker` accepts only `static`, `learned`, `meta_router`

### Infrastructure

- **Phase 8 — Developer Experience & CLI**
  - `typer`-based CLI with entry point `reranker` (subcommands: train, eval, benchmark, generate, doctor, serve)
  - `justfile` with 20+ tasks (test, lint, train, benchmark, docs, doctor)
  - Training scripts for cascade, meta_router, and splade strategies
  - `reranker doctor` command checking dependencies, models, and config
  - Getting Started guide (`docs/getting-started.md`)
  - `tests/integration/test_verification.py` replacing assert-based verification

- **Phase 9 — Benchmarking Maturity**
  - Bootstrap confidence intervals (`eval/statistics.py`)
  - Wilcoxon signed-rank test with CI-overlap detection
  - Memory and CPU profiling with graceful degradation (`eval/profiling.py`)
  - YAML sweep configs for cascade, binary, pipeline, distilled (plus active_distill, lsh, colbert, hybrid)
  - Pareto frontier, radar chart, and comparison table visualization (`eval/viz.py`)

- CI pipeline with 5 jobs: lint, typecheck, test (85% coverage), security, docs build
- `structlog` structured logging across 33+ call sites
- Security: MD5 replaced with SHA-256 in LSH, integrity checks on BEIR downloads
- MkDocs docs site with Material theme, Quarto notebook rendering, 60+ pages
