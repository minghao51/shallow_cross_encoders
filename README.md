# Shallow Cross Encoders

CPU-native reranking and consistency-checking pipeline inspired by the roadmap in
[`docs/technical-roadmap.md`](docs/technical-roadmap.md).

## What is included

- Hybrid semantic + lexical reranker with pluggable heuristic adapters
- Distilled pairwise ranker for local tournament ranking
- Static-ColBERT late interaction reranker (token-level MaxSim scoring)
- Binary quantized reranker (Hamming distance + bilinear refinement)
- Multi-stage pipeline for cascading rerankers
- Consistency engine for contradiction detection across extracted claims
- Synthetic data generation utilities with OpenRouter support and offline fallbacks
  - Hard negative mining, listwise preferences, query expansion
- Training, evaluation, benchmark, and ROI scripts

## Quickstart

Base install keeps the project offline-safe and usable without optional runtime packages:

```bash
uv sync
uv run python scripts/materialize_demo_data.py
uv run pytest
```

For the full roadmap path with real `model2vec`, `rank-bm25`, and `xgboost`, install the
runtime extras as well:

```bash
uv sync --extra dev --extra runtime
uv run pytest
uv run python scripts/train_hybrid.py
uv run python -m reranker.eval --strategy hybrid --split test
```

Teacher-backed data generation remains optional. Offline demo data still works with:

```bash
uv run python scripts/materialize_demo_data.py
```

To exercise the OpenRouter path with roadmap-scale targets:

```bash
export OPENROUTER_MODEL=openai/gpt-4o-mini  # optional; defaults to this model
OPENROUTER_API_KEY=... uv run python scripts/generate_pairs.py --teacher --count 2000
OPENROUTER_API_KEY=... uv run python scripts/generate_preferences.py --teacher --count 1500
OPENROUTER_API_KEY=... uv run python scripts/generate_contradictions.py --teacher --contradictions 500 --controls 200
```

Generation writes dataset metadata to `data/raw/manifest.json`, label summaries to
`data/processed/label_distribution_summary.json`, and API cost logs to
`data/logs/api_costs.jsonl`.

The generation scripts now refresh manifest and label-distribution metadata after each run, so
you can regenerate one dataset at a time without losing the aggregate summary.

Most fallback and default settings now live in [src/reranker/config.py](/Users/minghao/Desktop/personal/shallow_cross_encoders/src/reranker/config.py)
and can be overridden with environment variables such as `OPENROUTER_MODEL`,
`RERANKER_EMBEDDER_MODEL`, `RERANKER_SEED`, `RERANKER_DISTILLED_FULL_TOURNAMENT_MAX_DOCS`,
and the `RERANKER_*COUNT` values shown in `.env.example`.

## Runtime Validation

These commands exercise the main roadmap deliverables after data generation:

```bash
uv run python scripts/train_hybrid.py
uv run python scripts/train_distilled.py
uv run python scripts/train_late_interaction.py
uv run python scripts/train_binary_reranker.py
uv run python -m reranker.eval --strategy hybrid --split test
uv run python -m reranker.eval --strategy distilled --split test
uv run python -m reranker.eval --strategy late_interaction --split test
uv run python -m reranker.eval --strategy binary_reranker --split test
uv run python -m reranker.eval --strategy consistency --split test
uv run python scripts/measure_roi.py
uv run python scripts/benchmark_core.py
```

Hybrid training still writes the compatibility-safe pickle artifact by default through the eval
path, and can also save an XGBoost-native JSON artifact when the runtime backend is available.

## Dependency Policy

The project remains intentionally offline-first:

- Base install: required dependencies only, deterministic hashed embeddings, pure-Python BM25
  fallback, no external API dependency.
- Runtime extras: enable `model2vec`, `rank-bm25`, and `xgboost` for real latency and quality
  measurements.
- OpenRouter: still optional and used only for teacher-backed dataset generation, never for the
  inference path.

This keeps local development and testing safe by default while still supporting the full roadmap
validation flow when optional services are configured.

## Layout

The source package lives under `src/reranker`. Generated datasets, models, and logs are stored
under `data/`. The exploratory notebook for label-distribution and cost review lives in
`notebooks/00_data_exploration.ipynb`.
