# Shallow Cross Encoders

CPU-native reranking and consistency-checking toolkit focused on fast local inference, distillation, and reproducible evaluation.

See [architecture](docs/technical-roadmap.md) and [full docs](docs/index.md) for deep dives.

## What This Repo Covers

- Multiple reranking strategies: hybrid, distilled pairwise, late interaction, binary quantized, pipeline, and cascade
- Teacher-student distillation from FlashRank teachers into fast local models
- Synthetic data generation for pairs, preferences, and contradiction samples
- Evaluation and benchmarking for quality, latency, and statistical comparison
- Config-driven workflows via Pydantic settings and environment overrides

## Quickstart

```bash
uv sync --extra dev
uv run scripts/materialize_demo_data.py
uv run -m pytest
```

## Common Workflows

### Train local models

```bash
uv run scripts/train_hybrid.py
uv run scripts/train_distilled.py
uv run scripts/train_late_interaction.py
uv run scripts/train_binary_reranker.py
```

### Evaluate strategies

```bash
uv run -m reranker.eval --strategy hybrid --split test
uv run -m reranker.eval --strategy distilled --split test
uv run -m reranker.eval --strategy late_interaction --split test
uv run -m reranker.eval --strategy binary_reranker --split test
uv run -m reranker.eval --strategy consistency --split test
```

### Run benchmarks

```bash
uv run benchmarks/run.py synthetic --quick
uv run benchmarks/run.py full
uv run scripts/benchmark_beir_multi.py
```

### Distill labels from FlashRank teachers

```bash
uv sync
uv run scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise
```

Custom dataset:

```bash
uv run scripts/distill_ensemble_to_hybrid.py \
  --dataset custom \
  --custom-path data/custom_dataset.jsonl \
  --method pointwise
```

### Generate synthetic training data (OpenRouter)

```bash
OPENROUTER_API_KEY=... uv run scripts/generate_pairs.py --teacher --count 2000
OPENROUTER_API_KEY=... uv run scripts/generate_preferences.py --teacher --count 1500
OPENROUTER_API_KEY=... uv run scripts/generate_contradictions.py --teacher --count 500
```

## Benchmark Artifacts

`uv run reranker benchmark run` writes `benchmark_results.json` with aggregate metrics and per-query vectors.

Use paired comparisons with:

```bash
uv run reranker benchmark compare hybrid binary_reranker \
  --results benchmarks/results/benchmark_results.json \
  --metric ndcg@10
```

More details: [benchmark summary](docs/benchmarks/benchmark_summary.md), [comprehensive analysis](docs/benchmarks/20260426-comprehensive-benchmark-results.md).

## Production Patterns

- `CascadeReranker`: fast model first, fallback on low confidence
- `PipelineReranker`: progressive filtering for larger candidate sets
- Combined pipeline + cascade: higher efficiency with quality guardrails

Reference: [cascading strategy analysis](docs/benchmarks/20260415-benchmark-analysis.md#cascading-strategy-fast--smart).

## Environment Variables

- `OPENROUTER_API_KEY`: required for synthetic generation
- `OPENROUTER_MODEL`: optional model override (default `openai/gpt-4o-mini`)
- `RERANKER_PAIR_COUNT`: generated pair count (default `2000`)
- `RERANKER_PREFERENCE_COUNT`: generated preference count (default `1500`)
- `RERANKER_CONTRADICTION_COUNT`: generated contradiction count (default `500`)
- `RERANKER_CONTROL_COUNT`: generated control count (default `200`)

## Key References

- [Getting started](docs/getting-started.md)
- [Ensemble distillation guide](docs/guides/ensemble-distillation-guide.md)
- [API reference guide](docs/guides/api-reference.md)
- [Methodology docs](docs/methodology/)
- [Architecture overview](ARCHITECTURE.md)
