# Getting Started

Install, rerank your first documents, train a model, and run a benchmark — in under 10 minutes.

## Install

```bash
git clone <repo-url> && cd shallow_cross_encoders
uv sync --extra dev
```

## Quick Rerank (no training required)

```python
from reranker.strategies.hybrid import HybridFusionReranker
from reranker.heuristics.keyword import KeywordMatchAdapter

reranker = HybridFusionReranker(adapters=[KeywordMatchAdapter()])

# Train on a few labeled examples
queries = ["python tutorial"] * 4
docs    = ["Python beginner guide", "Java basics", "Python data science", "Cooking tips"]
scores  = [1.0, 0.0, 1.0, 0.0]

reranker.fit_pointwise(queries=queries, docs=docs, scores=scores)

# Rerank
results = reranker.rerank("python tutorial", [
    "Python advanced patterns",
    "Java streams",
    "Python web scraping",
    "Baking bread",
])
for r in results:
    print(f"  {r.rank}. {r.doc} (score={r.score:.3f})")
```

## Train with the CLI

Generate synthetic data and train all strategies:

```bash
# Generate training data
uv run reranker generate pairs --count 100
uv run reranker generate preferences --count 100

# Train individual strategies
uv run reranker train hybrid
uv run reranker train distilled
uv run reranker train binary
uv run reranker train late_interaction
uv run reranker train cascade
uv run reranker train meta_router
uv run reranker train splade
```

## Evaluate

```bash
# Evaluate a specific strategy
uv run reranker eval run hybrid --split test

# Evaluate with specific metrics
uv run reranker eval run hybrid --metrics ndcg,map,mrr
```

## Benchmark

```bash
# Quick synthetic benchmark
just benchmark-quick

# Full benchmark suite
just benchmark-full

# YAML sweep
just benchmark-sweep benchmarks/configs/sweep_hybrid.yaml
```

### Benchmark Artifact Schema

`reranker benchmark run` writes `benchmark_results.json` under the output directory.

Each result entry contains:

- `strategy`: strategy identifier (for example `hybrid`)
- `metrics`: aggregate metrics (for example `ndcg@10`, `mrr`, latency summaries)
- `per_query_metrics`: paired per-query vectors used for statistical comparison

Example shape:

```json
{
  "results": [
    {
      "strategy": "hybrid",
      "metrics": {
        "ndcg@10": 0.71,
        "mrr": 0.69
      },
      "per_query_metrics": {
        "per_query_ndcg@10": [0.81, 0.63, 0.74],
        "per_query_mrr": [1.0, 0.5, 0.5]
      }
    }
  ]
}
```

`reranker benchmark compare` now requires paired per-query vectors in `per_query_metrics`.
Aggregate-only artifacts are rejected for significance testing.

```bash
# Works: requires per_query_ndcg@10 for both strategies
uv run reranker benchmark compare hybrid binary_reranker \
  --results benchmarks/results/benchmark_results.json \
  --metric ndcg@10
```

## Use the Task Runner

All common tasks are available via `just`:

```bash
just test              # Run all tests
just test-unit         # Unit tests only
just lint              # Ruff linting
just typecheck         # mypy type checking
just check             # lint + typecheck
just train-all         # Train all strategies
just generate-data     # Generate all synthetic data
just doctor            # Check dependency health
```

## Programmatic API

All strategies implement the same interface:

```python
from reranker.protocols import RankedDoc

results: list[RankedDoc] = reranker.rerank(query="search query", docs=["doc1", "doc2", "doc3"])
# Each result has: .doc, .score, .rank, .metadata
```

### Save and Load

```python
reranker.save("data/models/my_model.pkl")
loaded = HybridFusionReranker.load("data/models/my_model.pkl")
```

## Available Strategies

| Strategy | Description | CLI Command |
|----------|-------------|-------------|
| **Hybrid Fusion** | GBDT on semantic + lexical features | `reranker train hybrid` |
| **Distilled Pairwise** | Pairwise tournament from teacher preferences | `reranker train distilled` |
| **Binary Quantized** | Hamming + bilinear scoring | `reranker train binary` |
| **Static ColBERT** | Late-interaction MaxSim | `reranker train late_interaction` |
| **Cascade** | Confidence-based fast/slow fallback | `reranker train cascade` |
| **Meta Router** | Query-type weight adaptation | `reranker train meta_router` |
| **SPLADE** | Sparse encoder sparse dot product | `reranker train splade` |
| **Pipeline** | Multi-stage progressive filtering | Use programmatically |
| **Multi (RRF)** | Reciprocal Rank Fusion ensemble | Use programmatically |

## Check System Health

```bash
uv run reranker doctor check
```

Reports dependency availability, embedder backend, and data/model directory status.

## Next Steps

- Read `ARCHITECTURE.md` for design details
- Run `just benchmark-quick` to see strategy comparisons
- Explore `notebooks/` for interactive examples
