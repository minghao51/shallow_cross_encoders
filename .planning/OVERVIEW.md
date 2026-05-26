# Shallow Cross-Encoders

## Identity

CPU-native reranking library using static embeddings, lexical signals, and lightweight ML models (XGBoost, logistic regression) instead of deep cross-encoders. Supports heuristic scoring, late interaction (ColBERT-style), SPLADE sparse retrieval, binary quantization, distilled pairwise rankers, cascading pipelines with confidence-based fallback, consistency checking across document claims, and synthetic data generation via LLMs.

## Architecture

**Core abstraction:** `BaseReranker` protocol (`src/reranker/protocols.py:42`) — any ranker implements `rerank(query: str, docs: list[str]) -> list[RankedDoc]`. Strategies are polymorphic; pipelines, cascades, and multi-ensembles compose them.

**Data flow:**
1. **Entry point:** CLI via `reranker.cli:app` (`pyproject.toml:30`, `src/reranker/cli/__init__.py`) — subcommands: `train`, `eval`, `benchmark`, `generate`, `serve`, `doctor`.
2. **Config:** Pydantic `Settings` tree (`src/reranker/config.py:282`) — frozen nested models with `RERANKER_*` env-var overrides, YAML file loading, and context-local overrides.
3. **Embedding:** `Embedder` (`src/reranker/embedder.py:40`) — wraps model2vec with deterministic hashed fallback, shared TTLCache.
4. **Reranking strategies** (all in `src/reranker/strategies/`):
   - `HybridFusionReranker` — XGBoost/sklearn ensemble fusing semantic, BM25, and heuristic features
   - `DistilledPairwiseRanker` — logistic regression trained on synthetic LLM preference pairs
   - `StaticColBERTReranker` — ColBERT-style late interaction with token salience
   - `SPLADEReranker` — sparse expansion via SPLADE (naver/splade-cocondenser)
   - `BinaryQuantizedReranker` — binary quantization + Hamming + bilinear scoring
   - `FlashRankWrapper` / `FlashRankEnsemble` — wraps FlashRank cross-encoders
   - `MultiReranker` — RRF fusion of multiple rankers
   - `PipelineReranker` — cascades candidates through stages, each stage narrows top-k
   - `CascadeReranker` — confidence-thresholded fallback to secondary ranker
   - `ConsistencyEngine` — extracts structured claims, detects contradictions via embedding similarity
5. **Data generation:** LLM-driven (OpenRouter/litellm) synthetic pairs, preferences, contradictions (`src/reranker/data/synth/`)
6. **EVAL/Benchmark:** BEIR dataset loader (`src/reranker/data/beir_loader.py`), standard metrics (`src/reranker/eval/metrics.py`), profiling (`src/reranker/eval/profiling.py`)

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | >=3.11 |
| Package manager | uv | — |
| Build system | setuptools | >=68.0 |
| CLI framework | typer | >=0.15.0 |
| Config | pydantic | >=2.8.0 |
| Logging | structlog | >=25.0.0 |
| Embeddings | model2vec | >=0.7.0 |
| ML models | scikit-learn | >=1.5.0 |
| Gradient boosting | xgboost | >=2.1.0 |
| Sparse retrieval | rank-bm25 | >=0.2.2 |
| LLM gateway | litellm | >=1.40.0 |
| HTTP client | httpx | >=0.27.0 |
| Numeric | numpy, scipy | >=1.26.0, >=1.13.0 |
| Cross-encoders | sentence-transformers | >=5.4.0 |
| Cross-encoders | flashrank | >=0.2.0 |
| Serialization | cloudpickle, joblib | >=3.1.0, >=1.4.0 |
| Serialization | pyyaml | >=6.0 |
| Retry | tenacity | >=9.0.0 |
| Caching | cachetools | >=5.5.0 |
| Dev tooling | pytest, ruff, mypy, pre-commit, marimo | (dev deps) |
| Documentation | mkdocs-material, mkdocstrings | (docs deps) |

## Infrastructure

- **CI:** GitHub Actions (`.github/workflows/ci.yml`) — lint (pre-commit), typecheck (mypy), test matrix (3.11/3.12/3.13), security (bandit, pip-audit), docs build (quarto + mkdocs)
- **Docs:** GitHub Actions (`.github/workflows/docs.yml`) — deploy to GitHub Pages
- **Pre-commit** (`.pre-commit-config.yaml`): trailing-whitespace, ruff (lint+format), bandit, pip-audit, mypy, mkdocs build
- **Justfile** (`justfile`): 30+ dev targets for test/lint/typecheck/benchmark/train/eval/docs
- **Coverage:** pytest-cov, fail-under 85%

## Integrations

| Integration | Type | What it's used for |
|---|---|---|
| OpenRouter / litellm | LLM API | Synthetic data generation (pair/preference/contradiction), active distillation labeling |
| BEIR | Benchmark dataset loader | Evaluating reranker quality on standard IR benchmarks |
| FlashRank | Cross-encoder library | FlashRankWrapper and FlashRankEnsemble strategies |
| model2vec | Static embeddings | Primary embedder backend for semantic features |
| SentenceTransformers | Cross-encoder embeddings | SentenceTransformerWrapper, distilled ranker teacher model |
| rank-bm25 | Lexical scoring | BM25 feature for hybrid fusion (pure-Python fallback) |
| XGBoost | Gradient boosting | HybridFusionReranker primary model (sklearn GBC fallback) |
| SPLADE (HuggingFace) | Sparse retrieval | SPLADEReranker sparse lexical expansion |
| scikit-learn | ML toolkit | Logistic regression (distilled), feature pipelines, metrics |
| Quarto + MkDocs | Documentation | Notebook rendering + docs site |

## Environment Variables

All defined in `.env.example`. Naming convention: `RERANKER_<SECTION>_<KEY>` maps to Pydantic settings (see `config.py:26-48`).

### LLM (OpenRouter)
| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | API key for LLM-based data generation |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Model for synthetic data generation |

### Embedder
| Variable | Default | Purpose |
|---|---|---|
| `RERANKER_EMBEDDER_MODEL` | `minishlab/potion-base-32M` | model2vec embedding model name |

### Synthetic Data Generation
| Variable | Default | Purpose |
|---|---|---|
| `RERANKER_SEED` | `42` | Random seed for reproducibility |
| `RERANKER_TEACHER_BATCH_SIZE` | `20` | Batch size for LLM teacher calls |
| `RERANKER_TEACHER_MAX_WORKERS` | `4` | Parallel worker threads |
| `RERANKER_PAIR_COUNT` | `60` | Number of synthetic relevance pairs |
| `RERANKER_PREFERENCE_COUNT` | `40` | Number of preference pairs |
| `RERANKER_CONTRADICTION_COUNT` | `20` | Number of contradiction pairs |
| `RERANKER_CONTROL_COUNT` | `8` | Control samples |
| `RERANKER_ROADMAP_*` | varied | Roadmap-scale generation counts |

### Strategy-specific
| Variable | Default | Purpose |
|---|---|---|
| `RERANKER_LATE_INTERACTION_TOP_K_TOKENS` | `128` | Max tokens per doc for ColBERT scoring |
| `RERANKER_LATE_INTERACTION_USE_SALIENCE` | `true` | Enable token salience weighting |
| `RERANKER_BINARY_RERANKER_HAMMING_TOP_K` | `500` | Top-K for Hamming pre-filter |
| `RERANKER_BINARY_RERANKER_BILINEAR_TOP_K` | `50` | Top-K for bilinear re-ranking |
| `RERANKER_DISTILLED_FULL_TOURNAMENT_MAX_DOCS` | `50` | Docs per tournament round |

### Pipeline
| Variable | Default | Purpose |
|---|---|---|
| `RERANKER_PIPELINE_DEFAULT_STAGE_TOP_K` | `200` | Default top-K per pipeline stage |

### Benchmark
| Variable | Default | Purpose |
|---|---|---|
| `RERANKER_BENCHMARK_CONSISTENCY_CLAIM_COUNT` | `1000` | Sample size for consistency engine benchmark |
| `RERANKER_BENCHMARK_CONSISTENCY_TARGET_MS_PER_1000_CLAIMS` | `50.0` | Latency target |
