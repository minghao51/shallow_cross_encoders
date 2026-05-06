# Technical Roadmap: Next Evolution (2026 H2)

> **Objective:** Evolve the shallow cross encoders pipeline from a feature-complete research toolkit into a production-grade, developer-friendly reranking library — fixing known bugs, optimizing performance, adding a CLI, expanding benchmarks, and enabling ecosystem integration.
>
> **Horizon:** 6–12 months (June 2026 – December 2026)
> **Starting state:** All Phase 0–5 deliverables complete (see legacy roadmap). 12 strategies, synthetic data engine, BEIR evaluation, consistency engine all operational.
> **Architecture invariant:** `reranker.rerank(query: str, docs: list[str]) -> list[RankedDoc]` — no GPU, no inference server, pure Python.

---

## Guiding Principles

1. **Correctness before speed.** Every optimization must preserve existing behavior or explicitly change it with test coverage.
2. **Drop-in compatibility.** Existing `from reranker import ...` paths must not break. New APIs are additive.
3. **CPU-native, always.** No mandatory GPU or heavy runtime. Optional accelerations (ONNX, SIMD) are opt-in.
4. **Measurable gates.** Each phase has exit criteria backed by benchmarks or test metrics.

---

## Phase 6 — Stability & Correctness

> **Goal:** Fix all known correctness bugs and architectural inconsistencies. Establish a trustworthy foundation for optimization work.
> **Duration:** Month 1–2
> **Dependencies:** None (start immediately)

### 6.1 Correctness Bugs

| ID | Issue | File | Fix |
|----|-------|------|-----|
| C-1 | `PipelineReranker.run_pipeline()` — `zip(current_docs, passed)` may misalign when stages filter differently | `strategies/pipeline.py` | Track docs by index through stages; validate correspondence before zip |
| C-2 | `BinaryQuantizedReranker` auto-fits with all-1s labels → always triggers `DummyClassifier` | `strategies/binary_reranker.py` | Raise `RuntimeError` if `rerank()` called before `fit()` instead of silent degenerate fit |
| C-3 | `SPLADEReranker._maxsim_score()` uses `min()` instead of dot product for sparse scoring | `strategies/splade.py` | Replace `min(qw, dt)` with `qw * dt` for proper sparse dot product |
| C-4 | `HybridFusionReranker` returns zero-initialized scores when unfitted + `weighting_mode="learned"` | `strategies/hybrid.py` | Raise `RuntimeError` or fall back to static weighting with warning |
| C-5 | `active_distill.py` — `new_preferences` always returned empty, never populated | `data/active_distill.py` | Implement preference generation path (already exists in `synth/generator/preferences.py`) |
| C-6 | Duplicate `model_config = ConfigDict(extra="forbid")` in `_models.py` | `data/synth/_models.py:87-88` | Remove duplicate line |

### 6.2 Lifecycle Consistency

- **`is_fitted` attribute:** Standardize across all strategies. Set to `False` in `__init__`, `True` after `fit()` or successful `load()`. Remove dead `is_fitted` from `CascadeReranker` or wire it properly.
- **Auto-fit removal:** Remove auto-fit-on-rerank behavior from `BinaryQuantizedReranker`, `StaticColBERTReranker`, `SPLADEReranker`. Replace with explicit error: `"Model not fitted. Call fit() or load() first."`
- **`CascadeReranker.fallback_strategy`** validation: Accept only valid enum values, raise on unknown strings.

### 6.3 Architecture Cleanup

- Deduplicate `multi.py` internal `Reranker` Protocol → import `protocols.BaseReranker`
- Make `FlashRankEnsemble` persistable: add `save()` / `load()` with teacher model config serialization
- Export `SentenceTransformerWrapper` from `adapters/__init__.py`
- Add `weighting_mode` enum validation in `HybridFusionReranker` (accept only `static`, `learned`, `meta_router`)
- Replace `CASCADE_SCORE_VARIANCE` inline import with top-level import

### Phase 6 Exit Criteria

- [ ] All 6 correctness bugs fixed with regression tests
- [ ] No strategy auto-fits with degenerate labels — all raise on unfitted rerank
- [ ] `is_fitted` consistently managed across all strategies
- [ ] `FlashRankEnsemble.save()` / `.load()` roundtrip works
- [ ] All existing tests still pass (no regressions)

---

## Phase 7 — Performance & Optimization

> **Goal:** Eliminate algorithmic bottlenecks. Reduce latency by 30–50% on common workloads.
> **Duration:** Month 2–3
> **Dependencies:** Phase 6 complete (correctness bugs fixed)

### 7.1 Algorithmic Fixes

| ID | Bottleneck | File | Fix | Expected Gain |
|----|-----------|------|-----|---------------|
| P-1 | `DistilledPairwiseRanker._merge_rank` uses `list.pop(0)` → O(n^2 log n) | `strategies/distilled.py` | Replace with `collections.deque` or index-based traversal | 10–50x for n>50 |
| P-2 | `StaticColBERT._compute_salience` O(n^2) inner loop | `strategies/late_interaction.py` | Vectorized lookup via numpy fancy indexing | 5–10x for large token sets |
| P-3 | 4-bit quantization per-byte loop | `quantization.py` | Batch vectorize with numpy | 3–5x for high-dim vectors |

### 7.2 Caching

- **Doc embedding cache in `BinaryQuantizedReranker`:** Cache binary-encoded docs between `score()` calls. Invalidate on new `fit()`.
- **Shared embedding cache across strategies:** Introduce `EmbeddingCache` singleton (backed by `cachetools.TTLCache`) that all strategies share. Keyed by `(text, model_name)`. Configurable size and TTL.
- **StaticColBERT `TokenIndex` reuse:** Allow pre-built token indices to be passed into `rerank()` instead of rebuilding per call.

### 7.3 Batch Operations

- **`StaticColBERTReranker.rerank_batch(queries, docs)`:** Encode all queries in one batch, compute MaxSim matrix in one operation.
- **`HybridFusionReranker.rerank_batch(queries, docs_list)`:** Batch embed all queries + docs, construct feature matrices in parallel.

### 7.4 Quantization Enhancements

- Add `int8` quantization mode (midpoint between 4-bit and float32)
- Add `float16` storage option for GPU-optional paths
- Benchmark compression ratio vs quality tradeoff for each mode

### Phase 7 Exit Criteria

- [ ] `DistilledPairwiseRanker._merge_rank` is O(n log n) with deque
- [ ] `StaticColBERT._compute_salience` uses vectorized numpy
- [ ] Shared embedding cache reduces redundant encoding by 50%+ in multi-strategy benchmarks
- [ ] `rerank_batch` available on ColBERT and Hybrid strategies
- [ ] End-to-end latency improved by 30%+ on 100-doc benchmark

---

## Phase 8 — Developer Experience & CLI

> **Goal:** Replace 16 ad-hoc scripts with a unified CLI. Make the library self-documenting and easy to adopt.
> **Duration:** Month 3–5
> **Dependencies:** Phase 6 complete (stable API surface)

### 8.1 CLI Framework

Add `typer`-based CLI with `[project.scripts]` entry point:

```
reranker train <strategy> [--dataset PATH] [--output PATH] [--config YAML]
reranker eval <strategy> [--dataset PATH] [--metrics ndcg,map,mrr] [--split test]
reranker benchmark [--config YAML] [--quick] [--output PATH]
reranker generate <pairs|preferences|contradictions> [--count N] [--output PATH]
reranker serve [--host 0.0.0.0] [--port 8000]   # Phase 11 preview
```

Each subcommand delegates to existing logic in `scripts/` and `benchmarks/`. Scripts are not removed — they remain as programmatic entry points.

### 8.2 Task Runner

Add `justfile` (or `Makefile`) with common tasks:

```
just train-all          # Train all strategies sequentially
just benchmark-quick    # Run synthetic benchmark
just benchmark-full     # Run full sweep
just test               # Run pytest
just lint               # Run ruff + mypy
just docs-serve         # MkDocs dev server
just generate-data      # Generate all synthetic datasets
```

### 8.3 Missing Training Scripts

Add dedicated training scripts for:

- `scripts/train_cascade.py` — trains primary + fallback + confidence thresholds
- `scripts/train_meta_router.py` — trains query classifier on labeled data
- `scripts/train_splade.py` — fits SPLADE index from corpus

### 8.4 Verification & Validation

- Replace `verify_enhanced_strategies.py` assert-based verification with pytest `test_verification.py` in `tests/integration/`
- Add `reranker doctor` CLI command that checks dependency availability, model presence, and config validity

### 8.5 Documentation

- Write **Getting Started** guide (`docs/getting-started.md`): install, first rerank, train a model, run a benchmark
- Write **Benchmark Configuration** guide: YAML sweep format, available parameters, interpreting results
- Add inline `--help` examples to all CLI subcommands

### Phase 8 Exit Criteria

- [ ] `reranker train hybrid` works end-to-end (replaces `uv run scripts/train_hybrid.py`)
- [ ] `reranker benchmark --quick` produces same results as `uv run benchmarks/run.py synthetic`
- [ ] `justfile` covers all common development tasks
- [ ] Getting Started guide enables a new user to rerank documents in under 10 minutes
- [ ] All training scripts exist for all strategies

---

## Phase 9 — Benchmarking Maturity

> **Goal:** Produce benchmarks that are statistically rigorous, multi-dataset, and comprehensive enough to drive optimization decisions.
> **Duration:** Month 4–6
> **Dependencies:** Phase 7 complete (optimized strategies)

### 9.1 Dataset Expansion

- Expand synthetic test set from 10 to 500+ test pairs with balanced label distribution
- Add BEIR multi-dataset benchmarking: TREC-COVID, NFCorpus, SciDocs, FiQA-QA, ArguAna
- Create golden test dataset versioned in `data/benchmarks/` with reproducibility seed

### 9.2 Statistical Rigor

- Add bootstrap confidence intervals (1000 resamples) for all ranking metrics
- Add paired t-test / Wilcoxon signed-rank test between strategy pairs
- Report metric ± 95% CI in all benchmark outputs
- Flag results where CI overlaps (statistically indistinguishable)

### 9.3 Profiling

- Add `memory_profiler` integration to `BenchmarkRunner`: peak RSS, total allocation per strategy
- Add CPU utilization tracking (user/system/idle breakdown)
- Report latency cold-start vs warm (first call vs p50)

### 9.4 Sweep Coverage

Add YAML sweep configs for:

- `sweep_cascade.yaml` — confidence thresholds × primary/fallback combos
- `sweep_binary.yaml` — hamming-only vs bilinear, quantization modes
- `sweep_pipeline.yaml` — stage orderings, top-k per stage
- `sweep_distilled.yaml` — loss types × tournament sizes

### 9.5 Visualization

- Generate latency–accuracy Pareto frontier plots
- Add per-strategy radar charts (NDCG, MAP, latency, memory, model size)
- Auto-generate benchmark comparison table for docs

### Phase 9 Exit Criteria

- [ ] All benchmarks run on 5+ BEIR datasets with statistical significance
- [ ] 95% CIs computed for NDCG@10 on all strategies
- [ ] Memory profiling output included in benchmark results
- [ ] Sweep configs exist for all strategies
- [ ] Pareto frontier plot auto-generated from benchmark results

---

## Phase 10 — Feature Expansion

> **Goal:** Add high-value features that expand the library's capability without compromising CPU-native simplicity.
> **Duration:** Month 5–8
> **Dependencies:** Phase 8 (CLI) and Phase 9 (benchmarks) complete

### 10.1 Query Expansion Strategy

- Implement `QueryExpansionReranker` as a first-class strategy
- Uses template-based expansion (already in `synth/generator/enhanced.py`) at inference
- Configurable expansion count and template set
- RRF fusion of original + expanded query results

### 10.2 Strategy Auto-Selection

- Train a lightweight classifier on benchmark metadata (query length, doc count, domain) → optimal strategy
- Expose as `AutoReranker` that wraps strategy selection:

```python
reranker = AutoReranker.from_config("config.yaml")
results = reranker.rerank(query, docs)  # picks best strategy automatically
```

- Fallback to `HybridFusion` when classifier confidence is low

### 10.3 ONNX Export

- Export `StaticColBERT` MaxSim computation as ONNX graph
- Export `BinaryQuantizedReranker` scoring as ONNX graph
- Optional dependency: `onnxruntime` for accelerated inference
- Benchmark ONNX vs pure-Python latency

### 10.4 Structured Logging

- Replace all `print()` statements with `structlog` or `logging` with structured fields
- Add correlation IDs for request tracing
- Configurable log levels per strategy

### 10.5 Incremental BM25

- Add `BM25Engine.update(docs)` for incremental index updates without full rebuild
- Add `BM25Engine.remove(doc_ids)` for document deletion
- Maintain index statistics across updates

### Phase 10 Exit Criteria

- [ ] `QueryExpansionReranker` achieves NDCG within 2pts of non-expanded on nfcorpus
- [ ] `AutoReranker` picks optimal strategy >80% of the time on held-out queries
- [ ] ONNX export produces working models with verified outputs
- [ ] All `print()` replaced with structured logging
- [ ] `BM25Engine.update()` is 10x faster than full `fit()` rebuild

---

## Phase 11 — Ecosystem & Integration

> **Goal:** Make the library embeddable in production systems and external frameworks.
> **Duration:** Month 8–12
> **Dependencies:** Phase 10 (feature-complete library)

### 11.1 REST API Server

FastAPI wrapper with standard endpoints:

```
POST /rerank          — {query, docs, strategy?} → RankedDoc[]
POST /train           — {strategy, dataset} → training job
GET  /health          — dependency check + model status
GET  /strategies      — list available strategies + is_fitted
```

- Dockerfile for containerized deployment
- Batch endpoint for multi-query workloads

### 11.2 Framework Adapters

- **LangChain:** `ShallowCrossEncoderReranker(BaseReranker)` compatible with LangChain's retriever pipeline
- **LlamaIndex:** `ShallowRerankPostprocessor` implementing LlamaIndex's node postprocessor interface

### 11.3 Embedding Index Persistence

- Pre-built embedding index for large corpora (avoid re-encoding on every load)
- Index format: compressed numpy arrays + metadata (model name, dimension, checksum)
- Incremental index updates (add/remove docs without full rebuild)

### 11.4 WASM-Compatible Subset

- Identify pure-numpy subset that works in browser via Pyodide
- WASM compatibility check skill validates notebook compatibility
- Ship a minimal `reranker-lite` package for browser-side reranking

### Phase 11 Exit Criteria

- [ ] REST API serves `/rerank` at <10ms p99 for 50 docs
- [ ] LangChain adapter works with standard retriever pipeline
- [ ] Pre-built index loads 10x faster than re-encoding
- [ ] WASM subset runs in Pyodide with core strategies functional

---

## Cross-Cutting Concerns

### Testing

- Maintain 85% coverage floor for all new code
- Every Phase 6 bug fix includes a regression test
- Every Phase 7 optimization includes a benchmark regression test
- Phase 8 CLI tests cover all subcommands

### Documentation

- Update `ARCHITECTURE.md` after Phase 6 (lifecycle changes) and Phase 10 (new strategies)
- Each phase delivers updated API reference
- MkDocs site rebuilt with new guides as they're written

### Changelog

- Maintain `CHANGELOG.md` with entries per phase
- Follow Conventional Commits for all changes
- Tag releases at phase boundaries

---

## Milestone Summary

| Phase | Deliverable | Duration | Depends On | Status |
|-------|------------|----------|------------|--------|
| **6** | Stability & Correctness | Month 1–2 | — | Planned |
| **7** | Performance & Optimization | Month 2–3 | Phase 6 | Planned |
| **8** | Developer Experience & CLI | Month 3–5 | Phase 6 | Planned |
| **9** | Benchmarking Maturity | Month 4–6 | Phase 7 | Planned |
| **10** | Feature Expansion | Month 5–8 | Phases 8, 9 | Planned |
| **11** | Ecosystem & Integration | Month 8–12 | Phase 10 | Planned |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ONNX export breaks with numpy updates | Medium | Medium | Pin numpy version in ONNX tests; test against numpy nightly |
| BEIR datasets unavailable or format changes | Low | Medium | Cache datasets locally; add SHA256 verification |
| CLI framework adds heavy dependency | Low | Low | `typer` is lightweight; keep optional via extras |
| Auto-selection classifier overfits to benchmark datasets | Medium | Medium | Hold out multiple datasets; report CI on accuracy |
| Breaking changes in model2vec API | Low | High | Pin minimum version; test against latest on CI |
