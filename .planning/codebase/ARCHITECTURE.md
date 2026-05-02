# Architecture

**Analysis Date:** 2026-04-23

## Pattern Overview

**Overall:** Strategy-based plugin architecture with protocol-oriented design

**Key Characteristics:**
- Protocol-based contracts (BaseReranker, TrainableReranker) for extensibility
- Multiple interchangeable reranking strategies (Hybrid, Distilled, Late Interaction, Binary, SPLADE)
- Configuration-driven with Pydantic settings and environment variable overrides
- Composable patterns (Pipeline, Cascade, Multi) for complex workflows
- Teacher-student distillation pattern for fast local models

## Layers

**Configuration Layer:**
- Purpose: Centralized settings management with Pydantic models
- Location: `src/reranker/config.py`
- Contains: 17+ Pydantic settings classes (HybridSettings, DistilledSettings, etc.)
- Depends on: Environment variables, YAML config files
- Used by: All strategies and evaluation scripts

**Protocol Layer:**
- Purpose: Define contracts and interfaces
- Location: `src/reranker/protocols.py`
- Contains: BaseReranker, TrainableReranker, SaveableReranker, HeuristicAdapter, RankedDoc
- Depends on: Python typing/dataclasses
- Used by: All strategy implementations

**Strategy Layer:**
- Purpose: Core reranking algorithms
- Location: `src/reranker/strategies/`
- Contains: HybridFusionReranker, DistilledPairwiseRanker, StaticColBERTReranker, BinaryQuantizedReranker, SPLADEReranker, ConsistencyEngine, CascadeReranker, PipelineReranker
- Depends on: Protocols, config, embedder, lexical, heuristics
- Used by: Evaluation scripts, training scripts

**Data Layer:**
- Purpose: Synthetic data generation and dataset loading
- Location: `src/reranker/data/`
- Contains: OpenRouter/LiteLLM clients, BEIR dataset loaders, synthetic generators (pairs, preferences, contradictions)
- Depends on: External APIs (OpenRouter, LiteLLM), filesystem
- Used by: Training scripts

**Evaluation Layer:**
- Purpose: Benchmarking and metrics
- Location: `src/reranker/eval/`
- Contains: Runner, metrics (NDCG, MAP, Recall), __main__.py CLI
- Depends on: Strategies, datasets
- Used by: Scripts and CLI

## Data Flow

**Reranking Flow:**
1. User calls `strategy.rerank(query, docs)`
2. Strategy computes features (semantic, lexical, heuristics)
3. Model/ensemble scores documents
4. Results returned as `list[RankedDoc]` with scores and ranks

**Distillation Flow:**
1. FlashRank teacher ensemble scores query-doc pairs
2. Soft labels/pairwise preferences generated
3. Student model (Hybrid/Distilled) trained on labels
4. Student achieves 95-98% of teacher quality at 100x speed

**Training Flow:**
1. Data loaded from `data/raw/` or synthetic generation
2. Features extracted via embedder, lexical, heuristics
3. Model trained via `fit()` or `fit_pointwise()`
4. Model saved to `data/models/`

**State Management:**
- Immutable Pydantic settings (frozen config)
- Model state saved as pickle files with metadata
- No in-memory state persistence across runs

## Key Abstractions

**BaseReranker Protocol:**
- Purpose: Common interface for all ranking strategies
- Examples: `src/reranker/strategies/hybrid.py:89`, `src/reranker/strategies/distilled.py:47`
- Pattern: Protocol-based duck typing with runtime_checkable

**TrainableReranker Protocol:**
- Purpose: Training contract for ML-based rerankers
- Examples: `src/reranker/strategies/hybrid.py:89`, `src/reranker/strategies/distilled.py:47`
- Pattern: Pairwise learning (fit) and pointwise learning (fit_pointwise)

**HeuristicAdapter Protocol:**
- Purpose: Inject custom scalar features
- Examples: `src/reranker/strategies/hybrid.py:76`
- Pattern: Adapter pattern for extensible feature computation

**CascadeReranker:**
- Purpose: Confidence-based fallback from fast to accurate
- Examples: `src/reranker/strategies/cascade.py:12`
- Pattern: Strategy pattern with confidence metrics

**PipelineReranker:**
- Purpose: Multi-stage progressive filtering
- Examples: `src/reranker/strategies/pipeline.py:12`
- Pattern: Pipeline pattern with configurable stages

## Entry Points

**Training Scripts:**
- Location: `scripts/train_*.py`
- Triggers: Manual execution (`uv run python scripts/train_hybrid.py`)
- Responsibilities: Train models, save to `data/models/`

**Benchmarking Scripts:**
- Location: `scripts/benchmark_*.py`
- Triggers: Manual execution
- Responsibilities: Measure latency, quality (NDCG, MAP), compare strategies

**Evaluation CLI:**
- Location: `src/reranker/eval/__main__.py`
- Triggers: `uv run python -m reranker.eval --strategy <name>`
- Responsibilities: Evaluate strategies on datasets, output metrics

**Data Generation Scripts:**
- Location: `scripts/generate_*.py`, `scripts/distill_ensemble_to_hybrid.py`
- Triggers: Manual execution with OPENROUTER_API_KEY
- Responsibilities: Generate synthetic training data, distill teacher labels

## Error Handling

**Strategy:** Lightweight exception handling with explicit error messages

**Patterns:**
- ValueError for invalid inputs (ratios, thresholds)
- Missing dependencies handled via conditional imports (xgboost)
- Network errors in data clients (OpenRouter, LiteLLM) propagate to caller
- FileNotFoundError for missing model/data files

## Cross-Cutting Concerns

**Logging:** Minimal (print statements in CLI, no structured logging)

**Validation:** Pydantic validators in config (ratio checks, type conversion)

**Authentication:** API keys via environment variables (OPENROUTER_API_KEY, LITELLM_API_KEY)

**Testing:** pytest with markers (unit, integration, e2e, llm, slow)

**Caching:** model2vec embeddings cached, ensemble cache for FlashRank

---

*Architecture analysis: 2026-04-23*
