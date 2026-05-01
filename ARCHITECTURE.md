# Architecture

## Overview

Shallow Cross Encoders is a CPU-native reranking and consistency-checking pipeline. It combines static embeddings, lexical signals, and shallow ML models to achieve sub-millisecond latency without GPU dependency.

## Core Design Principles

1. **Offline-first:** All embeddings are static (model2vec). No PyTorch/TensorFlow runtime required for the fast path.
2. **Graceful degradation:** Every optional dependency (model2vec, rank-bm25, xgboost) has a pure-Python fallback.
3. **Pluggable strategies:** Rerankers implement `BaseReranker` and self-register in `strategies/__init__.py`.
4. **Safe persistence:** New models are saved as joblib + JSON metadata. Pickle is supported for backward compatibility but emits a security warning on load.

## Data Flow

```
Query + Docs
    |
    v
[Optional] PipelineReranker (multi-stage filter)
    |-- BM25 (lexical, fastest)
    |-- BinaryQuantizedReranker (bit vectors)
    |-- HybridFusionReranker (GBDT on engineered features)
    |-- StaticColBERTReranker (token-level MaxSim)
    |
    v
[Optional] CascadeReranker (confidence-based fallback to FlashRank)
    |
    v
RankedDoc list
```

## Module Map

| Module | Responsibility |
|--------|--------------|
| `reranker.config` | Pydantic settings with env-var overrides and contextvar-based test isolation |
| `reranker.embedder` | Static embedding wrapper with deterministic hashed fallback and TTL cache |
| `reranker.lexical` | BM25 engine with pure-Python fallback |
| `reranker.strategies.*` | Reranking implementations (see Strategy Registry below) |
| `reranker.persistence` | Safe model serialization (joblib + JSON) with pickle backward compatibility |
| `reranker.eval.runner` | Training/evaluation orchestration for each strategy |
| `reranker.eval.metrics` | NDCG, MAP, MRR, P@1, accuracy, latency tracking |
| `reranker.data.synth` | Synthetic dataset generation via OpenRouter LLM calls |
| `benchmarks/` | Consolidated benchmark suite (see Benchmarking below) |

## Strategy Registry

Strategies live in `reranker/strategies/` and are re-exported from `reranker/strategies/__init__.py`:

| Strategy | Speed | Quality | Distillable | Key Idea |
|----------|-------|---------|-------------|----------|
| `BM25Engine` | ~0.1ms | Baseline | N/A | Lexical term overlap |
| `BinaryQuantizedReranker` | ~0.09ms | Low | Limited | Binary embeddings + Hamming distance |
| `DistilledPairwiseRanker` | ~0.15ms | High | Yes | Logistic regression on pairwise features |
| `HybridFusionReranker` | ~0.45ms | High | Yes | GBDT on semantic + lexical + heuristic features |
| `StaticColBERTReranker` | ~2.00ms (50 docs) | High (0.92 NDCG@10 nfcorpus) | N/A | Token-level MaxSim with static vectors |
| `SPLADEReranker` | Varies | Medium | N/A | Sparse term-based scoring |
| `PipelineReranker` | Sum of stages | Varies | N/A | Cascade candidates through stages |
| `MultiReranker` | Sum of components | Varies | N/A | RRF fusion of multiple rankers |
| `CascadeReranker` | Avg of primary+fallback | High | N/A | Confidence-based routing |
| `ConsistencyEngine` | ~0.05ms/claim | N/A | N/A | Claim extraction + contradiction detection |
| `FlashRankEnsemble` | Varies | High | N/A | Multi-teacher ensemble (TinyBERT + MiniLM) |
| `MetaRouter` | ~0.01ms | N/A | N/A | Decision tree query-type routing for feature weights |

## Model Persistence

The `persistence` module replaces ad-hoc `pickle` dumps with a safe format:

- `<name>.meta.json` — artifact metadata (type, version, feature names)
- `<name>.weights.joblib` — model weights via `joblib`
- `<name>.pkl` — backward-compatible marker (legacy pickle loader still works)

`load_pickle()` now emits a `RuntimeWarning` to alert users to the security risk.

## Configuration

Settings are layered (highest priority wins):

1. Contextvar override (`apply_settings_override`) — used in tests
2. Environment variables
3. YAML file (`settings_from_yaml`)
4. Pydantic defaults in `reranker.config`

## Distillation Pipeline

```
BEIR / Custom dataset
    |
    v
FlashRankEnsemble (teacher) --soft labels--> HybridFusionReranker (student)
                                    |
                                    v
                            DistilledPairwiseRanker (student)
```

Active distillation (`ActiveDistiller`) mines contested/hard-negative examples and labels them via an LLM teacher when labeled data is unavailable.

## Testing Strategy

| Layer | Location | Scope |
|-------|----------|-------|
| Unit | `tests/unit/` | Pure functions, no I/O |
| Integration | `tests/integration/` | Local models, mocked services |
| E2E | `tests/e2e/` | Full pipeline, real LLM calls (gated by `llm` marker) |

## Benchmarking

All benchmark scripts are consolidated in `benchmarks/`:

```
benchmarks/
├── run.py          # Single CLI entry point (synthetic, sweep, roi, full)
├── runner.py       # BenchmarkRunner class with all 12 strategies
├── run_sweep.py    # YAML-driven config sweep runner
├── measure_roi.py  # ROI/cost analysis
├── configs/        # YAML sweep configurations
└── results/        # Output directory
```

Usage: `uv run benchmarks/run.py synthetic` or `uv run benchmarks/run.py full`

## Dependency Groups

- **Core:** numpy, scikit-learn, pydantic, httpx, tenacity, joblib
- **Runtime:** model2vec, rank-bm25, xgboost
- **FlashRank:** flashrank (teacher models)
- **SentenceTransformers:** torch, sentence-transformers (SPLADE, cross-encoders)
- **Dev:** pytest, ruff, mypy, pre-commit, bandit
