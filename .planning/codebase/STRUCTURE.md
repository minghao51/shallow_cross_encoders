# Codebase Structure

**Analysis Date:** 2026-04-23

## Directory Layout

```
shallow_cross_encoders/
├── src/reranker/              # Main package: all core logic
│   ├── strategies/            # Reranking strategy implementations
│   ├── data/                  # Data loading and synthetic generation
│   ├── eval/                  # Evaluation, metrics, CLI
│   ├── heuristics/            # Heuristic algorithms (LSH, keyword)
│   ├── benchmark.py           # Benchmarking utilities
│   ├── config.py              # Pydantic settings (17+ classes)
│   ├── embedder.py            # model2vec embedding wrapper
│   ├── lexical.py             # BM25 lexical search
│   ├── protocols.py           # Base contracts (BaseReranker, etc.)
│   └── utils.py               # Shared utilities (pickle, JSON)
├── scripts/                   # Training, benchmarking, data generation
├── tests/                     # Tests (unit/integration/e2e)
├── data/                      # Data storage (raw, processed, models)
├── docs/                      # Documentation and methodology
├── notebooks/                 # Jupyter notebooks for exploration
├── benchmarks/                # Benchmark results storage
├── pyproject.toml             # Project config, dependencies, pytest setup
└── README.md                  # Quickstart and usage guide
```

## Directory Purposes

**src/reranker/strategies/:**
- Purpose: All reranking algorithm implementations
- Contains: 11+ strategy files (hybrid.py, distilled.py, late_interaction.py, binary_reranker.py, splade.py, cascade.py, pipeline.py, multi.py, flashrank_ensemble.py, meta_router.py, patterns.py, consistency.py)
- Key files: `hybrid.py`, `distilled.py`, `late_interaction.py`, `cascade.py`

**src/reranker/data/:**
- Purpose: Dataset loaders and synthetic data generation
- Contains: BEIR loaders, OpenRouter/LiteLLM clients, generators (pairs, preferences, contradictions)
- Key files: `client.py`, `litellm_client.py`, `custom_beir.py`

**src/reranker/eval/:**
- Purpose: Evaluation framework and metrics
- Contains: NDCG, MAP, Recall metrics, runner, CLI entry point
- Key files: `metrics.py`, `runner.py`, `__main__.py`

**scripts/:**
- Purpose: Training, benchmarking, data generation entry points
- Contains: train_*.py, benchmark_*.py, generate_*.py, distill_ensemble_to_hybrid.py
- Key files: `train_hybrid.py`, `benchmark_beir.py`, `distill_ensemble_to_hybrid.py`

**tests/:**
- Purpose: Test suite organized by type
- Contains: unit/, integration/, e2e/ subdirectories
- Key files: Test files following `test_*.py` naming

**data/:**
- Purpose: Dataset storage and model artifacts
- Contains: raw/, processed/, models/, benchmarks/, logs/, trec-covid/, beir/
- Generated: Yes (models, benchmarks, logs)
- Committed: No (gitignore excludes most)

## Key File Locations

**Entry Points:**
- `src/reranker/eval/__main__.py`: CLI for evaluating strategies (`python -m reranker.eval`)
- `src/reranker/__init__.py`: Public package exports

**Configuration:**
- `src/reranker/config.py`: All Pydantic settings (HybridSettings, DistilledSettings, etc.)
- `pyproject.toml`: Dependencies, pytest config, ruff, mypy settings

**Core Logic:**
- `src/reranker/strategies/hybrid.py`: HybridFusionReranker (ML ensemble)
- `src/reranker/strategies/distilled.py`: DistilledPairwiseRanker (Logistic regression)
- `src/reranker/strategies/late_interaction.py`: StaticColBERTReranker (ColBERT)
- `src/reranker/protocols.py`: BaseReranker, TrainableReranker, HeuristicAdapter
- `src/reranker/embedder.py`: model2vec embedding wrapper
- `src/reranker/lexical.py`: BM25Engine

**Testing:**
- `tests/unit/`: Fast isolated unit tests
- `tests/integration/`: Tests loading local models or mocked services
- `tests/e2e/`: Full workflow end-to-end tests

## Naming Conventions

**Files:**
- Lowercase with underscores: `hybrid.py`, `train_hybrid.py`, `benchmark_beir.py`
- Test files: `test_*.py`
- Module init files: `__init__.py` in all packages

**Directories:**
- Lowercase: `strategies/`, `heuristics/`, `data/`
- Data directories: `raw/`, `processed/`, `models/`, `logs/`

**Classes:**
- CamelCase: `HybridFusionReranker`, `DistilledPairwiseRanker`, `StaticColBERTReranker`
- Settings: `HybridSettings`, `DistilledSettings`, `PathSettings`

**Functions:**
- snake_case: `rerank()`, `fit()`, `fit_pointwise()`, `save()`, `load()`

## Where to Add New Code

**New Feature:**
- Primary code: `src/reranker/strategies/[name].py`
- Tests: `tests/unit/test_[name].py`
- Training script: `scripts/train_[name].py` (if trainable)

**New Component/Module:**
- Implementation: `src/reranker/[module_name]/` (create new package)
- Exports: Add to `src/reranker/__init__.py` if public

**Utilities:**
- Shared helpers: `src/reranker/utils.py` (general utilities)

**New Strategy:**
- Implementation: `src/reranker/strategies/[strategy_name].py`
- Protocol compliance: Implement BaseReranker, optionally TrainableReranker, SaveableReranker
- Export: Add to `src/reranker/strategies/__init__.py`

## Special Directories

**data/:**
- Purpose: Dataset storage, model artifacts, logs, benchmarks
- Generated: Yes (models/, processed/, logs/, benchmarks/)
- Committed: No (gitignore excludes most subdirectories)

**.planning/:**
- Purpose: Planning and architecture documentation
- Generated: No (manual documentation)
- Committed: Yes

**docs/:**
- Purpose: Methodology docs, benchmark results, plans
- Generated: Some (benchmark results auto-generated)
- Committed: Yes

**.mypy_cache/, .pytest_cache/, .ruff_cache/, .benchmarks/:**
- Purpose: Tool cache directories
- Generated: Yes
- Committed: No (gitignore excludes)

---

*Structure analysis: 2026-04-23*
