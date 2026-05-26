# Code Style Guide

## File Organization

```
shallow_cross_encoders/
├── src/reranker/                    # Core package (src layout)
│   ├── __init__.py                  # Public API with lazy imports via _LAZY_ATTRS
│   ├── __main__.py                  # CLI entry: `python -m reranker`
│   ├── cli/                         # Typer CLI subcommands (train, eval, benchmark, generate, serve, doctor)
│   │   ├── __init__.py              # app = typer.Typer(), sub-app registration
│   │   ├── __main__.py              # app()
│   │   ├── train.py                 # reranker train <strategy>
│   │   ├── eval.py                  # reranker eval run <strategy>
│   │   ├── benchmark.py             # reranker benchmark run/compare
│   │   ├── generate.py              # reranker generate pairs/preferences/contradictions
│   │   ├── serve.py                 # API server
│   │   └── doctor.py                # Dependency/health checks
│   ├── config.py                    # Pydantic settings (19 classes), env override, YAML loading
│   ├── protocols.py                 # BaseReranker, TrainableReranker, HeuristicAdapter, SaveableReranker, RankedDoc
│   ├── embedder.py                  # model2vec wrapper with hashed fallback, EmbeddingCache
│   ├── lexical.py                   # BM25Engine
│   ├── persistence.py               # Structured joblib+JSON save/load (replaces ad-hoc pickle)
│   ├── quantization.py              # FP16/INT8 quantization utilities
│   ├── embedding_cache.py           # TTLCache wrapper, get_shared_cache singleton
│   ├── utils.py                     # JSON/JSONL/pickle I/O, RRF fusion, rank_docs
│   ├── deps.py                      # Optional dependency checkers (check_model2vec, check_xgboost, etc.)
│   ├── logging_config.py            # structlog configuration
│   ├── strategies/                  # Reranking algorithm implementations (one file per strategy)
│   │   ├── __init__.py              # Lazy imports (_LAZY_ATTRS pattern)
│   │   ├── hybrid.py                # HybridFusionReranker (XGBoost + sklearn ensemble)
│   │   ├── distilled.py             # DistilledPairwiseRanker (logistic regression)
│   │   ├── late_interaction.py      # StaticColBERTReranker (ColBERT-style)
│   │   ├── binary_reranker.py       # BinaryQuantizedReranker (Hamming distance)
│   │   ├── splade.py                # SPLADEReranker (sparse lexical)
│   │   ├── cascade.py               # CascadeReranker + CascadeConfig + ConfidenceMetric
│   │   ├── pipeline.py              # PipelineReranker, PipelineStage, PipelineResult
│   │   ├── consistency.py           # ConsistencyEngine, Claim, ClaimSet, Contradiction
│   │   ├── multi.py                 # MultiReranker, MultiRerankerConfig
│   │   ├── flashrank_ensemble.py    # FlashRankWrapper, FlashRankEnsemble, SentenceTransformerWrapper
│   │   ├── meta_router.py           # MetaRouter (query-type routing)
│   │   ├── patterns.py              # Shared pattern utilities
│   │   └── distilled.py             # DistilledPairwiseRanker
│   ├── data/                        # Data loading, synthetic generation, LLM client
│   │   ├── beir_loader.py           # BEIR dataset loading
│   │   ├── custom_beir.py           # Custom BEIR wrappers
│   │   ├── client.py                # HTTP client
│   │   ├── litellm_client.py        # LiteLLM multi-provider LLM client
│   │   ├── hard_negative_sampler.py # Hard negative sampling
│   │   ├── ensemble_cache.py        # Ensemble caching
│   │   ├── splits.py                # Train/val/test splitting
│   │   ├── expanded.py              # Expanded data loading
│   │   ├── active_distill.py        # Active distillation loop
│   │   ├── synth/                   # Synthetic data generation (LLM-based)
│   │   │   ├── __init__.py
│   │   │   ├── _models.py           # Pydantic models for synthetic data
│   │   │   ├── _prompts.py          # Prompt templates
│   │   │   ├── _seeds.py            # Seed data
│   │   │   ├── _generator.py        # Core generator
│   │   │   └── generator/           # Sub-generators (pairs, preferences, contradictions)
│   │   └── _expanded/               # Expanded pair/preference/contradiction generation
│   ├── eval/                        # Evaluation framework
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── metrics.py               # NDCG, DCG, MAP, Recall, LatencyTracker
│   │   ├── runner.py                # evaluate_strategy()
│   │   ├── benchmark_utils.py       # Benchmark helpers
│   │   ├── profiling.py             # Memory + CPU profiling
│   │   ├── statistics.py            # Statistical analysis
│   │   └── viz.py                   # Visualization utilities
│   └── heuristics/                  # Heuristic algorithms
│       ├── __init__.py
│       ├── keyword.py               # KeywordMatchAdapter
│       └── lsh.py                   # LSH near-duplicate detection
├── scripts/                         # One-off training/benchmark/data-gen entry points
│   ├── train_hybrid.py, train_distilled.py, train_cascade.py, ...
│   ├── benchmark_beir_multi.py, benchmark_beir_colbert.py, benchmark_quantization.py
│   ├── generate_pairs.py, generate_preferences.py, generate_contradictions.py, ...
│   └── materialize_demo_data.py, distill_ensemble_to_hybrid.py, verify_enhanced_strategies.py
├── benchmarks/                      # Benchmark orchestration
│   ├── run.py, run_sweep.py, runner.py, measure_roi.py
├── tests/
│   ├── conftest.py                  # Shared fixtures, auto-marking, mock factories
│   ├── unit/                        # Fast isolated tests
│   ├── integration/                 # Tests with real/mocked models
│   ├── e2e/                         # Full workflow end-to-end
│   └── benchmarks/                  # Performance/latency benchmarks
├── notebooks/                       # Marimo notebooks
│   └── 01_interactive_reranking.py
├── docs/                            # MkDocs documentation site
├── data/                            # Data artifacts (raw, processed, models, logs)
│   └── beir/                        # BEIR dataset cache
├── .planning/                       # Architecture/codebase documentation
├── .pre-commit-config.yaml          # Pre-commit hooks (ruff, mypy, bandit, pip-audit, mkdocs)
├── pyproject.toml                   # Dependencies, tool configs
├── justfile                         # Dev task runner (test, lint, typecheck, benchmark, docs)
└── mkdocs.yml                       # Documentation site config
```

## Naming Conventions

| Category | Convention | Real Examples |
|---|---|---|
| **Classes** | PascalCase | `HybridFusionReranker`, `DistilledPairwiseRanker`, `StaticColBERTReranker`, `BinaryQuantizedReranker`, `Embedder`, `BM25Engine`, `MetaRouter` |
| **Settings models** | `{Name}Settings` | `HybridSettings`, `DistilledSettings`, `PathSettings`, `EmbedderSettings`, `SyntheticDataSettings`, `CascadeSettings` |
| **Protocols** | PascalCase | `BaseReranker`, `TrainableReranker`, `HeuristicAdapter`, `SaveableReranker` |
| **Dataclasses** | PascalCase | `RankedDoc`, `PipelineStage`, `PipelineResult`, `Claim`, `ClaimSet`, `Contradiction` |
| **Enums** | PascalCase (class) + UPPER (members) | `WeightingMode.STATIC`, `FallbackStrategy.FLASHRANK`, `ConfidenceMetric.RECIPROCAL_RANK` |
| **Functions/methods** | snake_case | `rerank()`, `fit()`, `encode()`, `save()`, `load()`, `rank_docs()`, `ndcg_at_k()`, `reciprocal_rank_fusion()` |
| **Private functions** | `_` prefix snake_case | `_normalize_rows()`, `_make_model()`, `_compute_env_overrides()`, `_build_sub_settings()` |
| **Private attributes** | `_` prefix snake_case | `_backend`, `_encode_cache`, `_pipeline`, `_logger` |
| **Module-level constants** | UPPER_SNAKE_CASE | `ARTIFACT_VERSION = 1`, `SAFE_FORMAT_VERSION = 2`, `POTION_MODELS` |
| **Test functions** | `test_` prefix snake_case | `test_ndcg_perfect_ranking()`, `test_bm25_prefers_exact_match()` |
| **Test classes** | `Test` prefix PascalCase | `TestNDCG`, `TestRoot`, `TestTrain`, `TestLatencyTracker` |
| **CLI sub-apps** | `{name}_app` | `train_app`, `eval_app`, `benchmark_app`, `generate_app`, `serve_app`, `doctor_app` |
| **Source files** | snake_case.py | `hybrid.py`, `late_interaction.py`, `binary_reranker.py`, `flashrank_ensemble.py` |
| **Test files** | `test_` prefix | `test_hybrid.py`, `test_cascade_reranker.py`, `test_cli.py` |
| **Training scripts** | `train_` prefix | `train_hybrid.py`, `train_distilled.py`, `train_binary_reranker.py` |
| **Enum member values** | lowercase snake_case | `"static"`, `"learned"`, `"meta_router"`, `"flashrank"`, `"reciprocal_rank"` |

## Code Patterns

### Pydantic Settings (`src/reranker/config.py`)
All configuration is frozen Pydantic models with env-var overrides. A root `Settings` model composes 19 sub-settings. Each sub-setting class uses `model_config = ConfigDict(frozen=True)`. Access via `get_settings()` singleton. Env var pattern: `RERANKER_<SECTION>_<KEY>`.

```python
class EmbedderSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_name: str = "minishlab/potion-base-32M"
    dimension: int = 256
```

### Protocols + Base Class (`src/reranker/protocols.py`)
`@runtime_checkable` Protocol defines the `rerank(query, docs) -> list[RankedDoc]` contract. `SaveableReranker` base class provides DRY save/load via structured joblib+JSON (not pickle). Each subclass provides its own `load()` classmethod.

```python
@runtime_checkable
class BaseReranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]: ...
```

### Dataclasses with `slots=True`
Performance-critical data holders use `@dataclass(slots=True)`. Common in protocols.py, patterns.py, metrics.py.

```python
@dataclass(slots=True)
class RankedDoc:
    doc: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Lazy Imports (`__init__.py`)
Both `src/reranker/__init__.py` and `src/reranker/strategies/__init__.py` use a `_LAZY_ATTRS` dict mapping name → `(module_path, attr_name)` with a custom `__getattr__` to defer imports until first access.

### Typer CLI (`src/reranker/cli/`)
Nested Typer apps: root `app` in `cli/__init__.py` with `add_typer()` for 6 subcommands. Tested via `typer.testing.CliRunner`. Sub-apps use `@app.command()` and `@app.callback()` decorators.

### structlog Logging (`src/reranker/logging_config.py`)
Configured at import time via `configure_logging()` in `cli/__init__.py`. Uses structlog processors (timestamps, log level, exception info) with ConsoleRenderer (dev) or JSONRenderer (production). Log level via `RERANKER_LOG_LEVEL` env var. Module-level loggers: `logger = structlog.get_logger(__name__)`.

### Optional Dependency Guards (`src/reranker/deps.py`)
Each optional dependency has a checker returning `(module_or_None, DepStatus)`. Callers use `if module is not None:` to branch, with graceful fallbacks (e.g., hashed embeddings when model2vec unavailable, sklearn fallback when xgboost unavailable).

### Persistence Layer (`src/reranker/persistence.py`)
Structured save/load replaces ad-hoc pickle: weights in `.weights.joblib`, metadata in `.meta.json`. Backward-compatible with legacy pickle (emits security warning). Usage: `save_safe(path, artifact_type, metadata, weights)` / `try_load_safe_or_warn(path, expected_type, legacy_loader)`.

### Marimo Notebooks (`notebooks/01_interactive_reranking.py`)
Interactive exploration notebooks using marimo `@app.cell` decorators. Declare dependencies in `# /// script` header block. Exported to docs via CI.

### Pluralized Fixture Names (`tests/conftest.py`)
Global fixtures are named as plural nouns: `sample_queries`, `sample_docs`, `sample_relevance_scores`, `sample_binary_relevances`, `sample_query_doc_pairs`, `sample_contradiction_docs`. Mock fixtures prefixed with `mock_` or `fake_`.

## Linting & Formatting

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]
```

**Pre-commit hooks** (`.pre-commit-config.yaml`):
1. `pre-commit-hooks` — trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-merge-conflict, check-added-large-files (10MB max)
2. `uv-pre-commit` — `uv-lock` validation
3. `ruff` — `--fix --exit-non-zero-on-fix` (excludes `_expanded/seeds.py`, `data/_expanded/`, `tests/`, `notebooks/`)
4. `ruff-format` — auto-format all Python
5. `bandit` — security linting (excludes `tests/`, `scripts/`)
6. `pip-audit` — dependency vulnerability scan (via `uv run`)
7. `mypy` — strict type checking (excludes `tests/`, `scripts/`, `benchmarks/`, `notebooks/`)
8. `mkdocs build --strict` — doc build validation

## Testing

**Framework:** pytest 8.2+ with plugins: pytest-benchmark, pytest-cov (85% min), pytest-mock.

**Auto-marking** (`conftest.py`): `pytest_collection_modifyitems` assigns `unit`/`integration`/`e2e` markers based on test directory path. Tests in `tests/benchmarks/` get both `unit` + `slow` markers.

**Markers** (registered in `pyproject.toml`):
| Marker | Purpose |
|---|---|
| `unit` | Fast, isolated unit tests |
| `integration` | Tests loading local models or mocked services |
| `e2e` | Full workflow end-to-end tests |
| `llm` | Tests making real LLM API calls (requires `OPENROUTER_API_KEY`) |
| `llm_mock` | Tests mocking LLM API calls |
| `slow` | Tests taking >1s |

**Default filter** (via `addopts`): `-m "not llm and not slow"` — skips LLM API and slow tests by default.

**Import mode:** `--import-mode=importlib` with `pythonpath = ["src", "scripts", "."]`.

**Output:** `--strict-markers -ra --durations=10`.

**Mocking patterns:** `monkeypatch` for dependency injection (`fake_model2vec`, `fake_rank_bm25`, `fake_xgboost` in conftest.py), `mock_httpx_post` for LLM API mocking.

**Fixture categories in conftest.py:**
- Common test data: `sample_queries`, `sample_docs`, `sample_relevance_scores`, `sample_binary_relevances`, `sample_query_doc_pairs`, `sample_contradiction_docs`
- Mock fixtures: `mock_embedding_dim`, `mock_embeddings`, `mock_llm_response`, `mock_llm_metadata`, `mock_preference_triplet`
- Temp directories: `temp_data_dir`, `temp_model_dir`
- Model mocks: `fake_model2vec`, `fake_rank_bm25`, `fake_xgboost`
- LLM mocks: `skip_llm_tests`, `mock_httpx_post`
- Autouse: `reset_http_clients` (runs before and after every test)

**Coverage:** source = `src/`, omit `*/__init__.py` and `*/__main__.py`, fail_under = 85.

## Commands

All commands run via `uv run`, defined in `justfile`:

| Command | Description |
|---|---|
| `uv sync` | Install all dependencies |
| `just test` | Run tests (default: skip llm+slow) |
| `just test-quick` | Unit only, exclude slow |
| `just test-unit` | Unit tests only |
| `just test-integration` | Integration tests only |
| `just test-cov` | With coverage report |
| `just lint` | `ruff check src/ tests/ scripts/` |
| `just lint-fix` | `ruff check --fix` |
| `just format-check` | `ruff format --check` |
| `just typecheck` | `mypy src/reranker/` |
| `just check` | lint + typecheck |
| `uv run pytest tests/ -x -q` | Quick test run |
| `uv run pytest --cov=src --cov-report=term-missing` | Full coverage |
| `uv run mkdocs serve` | Documentation dev server |
| `uv run pre-commit run --all-files` | Run all pre-commit hooks |
| `uv run reranker <command>` | CLI entry point |
