# Shallow Cross Encoders

CPU-native reranking and consistency-checking pipeline. See [`docs/technical-roadmap.md`](docs/technical-roadmap.md) for architecture details.

## Features

- **Rerankers:** Hybrid (semantic + lexical), Distilled pairwise, Late Interaction, Binary quantized, Multi-stage pipeline
- **Distillation:** Train fast local models from FlashRank cross-encoder teachers (FlashRank quality + sub-ms latency)
- **Consistency:** Contradiction detection across extracted claims
- **Data:** Synthetic generation when labeled data unavailable
- **Training:** Hard negative mining, listwise preferences, query expansion

### Why Distillation Matters

**Problem:** Cross-encoders (FlashRank) are accurate but slow—must score every query-doc pair. Local models (Hybrid, Binary) are faster but need training data.

**Solution:** Use FlashRank as teacher → generate soft labels → train local student models. Result: FlashRank quality with Hybrid/Binary speed (sub-millisecond).

**Why OpenRouter?** Generates synthetic training data when you don't have labeled relevance judgments. FlashRank is the teacher for distillation; OpenRouter is the data generator.

## Quickstart

```bash
# Base install (offline-safe, demo data only)
uv sync
uv run python scripts/materialize_demo_data.py
uv run pytest

# Full path with real models
uv sync --extra dev --extra runtime
uv run python scripts/train_hybrid.py
uv run python -m reranker.eval --strategy hybrid --split test
```

## Optional Features

### Label Generation (FlashRank as Teacher)

Generate soft labels from FlashRank cross-encoder teachers to train Hybrid Fusion Reranker:

```bash
# Install FlashRank dependency
uv sync --extra flashrank

# Generate labels on BEIR dataset (pairwise or pointwise method)
uv run python scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise

# Generate labels on custom dataset
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset custom \
  --custom-path data/custom_dataset.jsonl \
  --method pointwise
```

**Use when:** You have query-doc pairs but no relevance labels. FlashRank provides high-quality soft labels that capture ranking nuance better than binary judgments.

**Expected results:** 95-98% of teacher ensemble NDCG@10 with ~0.45ms latency for distilled models (vs ~40ms for teacher). See [Ensemble Distillation Guide](docs/ensemble-distillation-guide.md) for details.

**BEIR dataset support:**
- `fluent-legal` and `scifact` added in recent update
- Run: `uv run python scripts/benchmark_beir.py nfcorpus` for standard benchmarks
- Use `uv run python scripts/download_beir.py nfcorpus` to materialize a local dataset copy

### Synthetic Data Generation (OpenRouter)

Generate training data when you have no labeled examples:

```bash
export OPENROUTER_MODEL=openai/gpt-4o-mini  # optional
OPENROUTER_API_KEY=... uv run python scripts/generate_pairs.py --teacher --count 2000
OPENROUTER_API_KEY=... uv run python scripts/generate_preferences.py --teacher --count 1500
OPENROUTER_API_KEY=... uv run python scripts/generate_contradictions.py --teacher --count 500
```

**Use when:** Starting from scratch with no domain-specific training data. LLM generates synthetic query-doc pairs with relevance scores.

Metadata written to `data/raw/manifest.json`, `data/processed/label_distribution_summary.json`, `data/logs/api_costs.jsonl`.

### Benchmarking

Compare all reranking methods (FlashRank, SentenceTransformers, local models):

```bash
# Install both extras (requires PyTorch for ST)
uv sync --extra flashrank --extra sentence-transformers

# Run speed/quality comparison
uv run python scripts/benchmark_flashrank.py
uv run python scripts/benchmark_beir.py nfcorpus
uv run python scripts/benchmark_all.py --quick
```

**Use when:** Choosing between FlashRank (ONNX), SentenceTransformers (PyTorch), or local models (Hybrid, Binary) for deployment.

### Configuration
Settings live in [`src/reranker/config.py`](src/reranker/config.py) and are overridden via environment variables.

### Training Scripts
- `train_hybrid.py`: Hybrid Fusion Reranker with soft labels
- `train_distilled.py`: Distilled pairwise reranker from FlashRank
- `train_late_interaction.py`: ColBERT-based late interaction
- `train_binary_reranker.py`: Binary quantized reranker

## Runtime Scripts

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
uv run python scripts/benchmark_all.py --quick
```

### Environment Variables
- `OPENROUTER_MODEL`: Override default teacher model (default: `openai/gpt-4o-mini`)
- `RERANKER_PAIR_COUNT`: Number of generated pair samples (default: `2000`)
- `RERANKER_PREFERENCE_COUNT`: Number of generated preference samples (default: `1500`)
- `RERANKER_CONTRADICTION_COUNT`: Number of generated contradiction samples (default: `500`)
- `RERANKER_CONTROL_COUNT`: Number of generated control samples (default: `200`)
- `OPENROUTER_API_KEY`: Required for synthetic data generation

## Benchmarks

### Speed (20 docs per query)

| Method | Latency (p50) | Speedup vs ST |
|--------|---------------|---------------|
| [Late Interaction](docs/methodology/static_colbert_reranker.md) | **0.02ms** | 1,168x ⚡ |
| [Binary Quantized](docs/methodology/binary_quantized_reranker.md) | **0.09ms** | 259x ⚡ |
| [Distilled Pairwise](docs/methodology/distilled_pairwise_reranker.md) | **0.15ms** | 156x ⚡ |
| [Hybrid Fusion](docs/methodology/hybrid_fusion_reranker.md) | **0.45ms** | 52x ⚡ |
| FlashRank TinyBERT | **0.58ms** | 40x |
| FlashRank MiniLM | **4.48ms** | 5x |
| ST TinyBERT | 16.46ms | 1.4x |
| ST MiniLM-L-6 | 23.36ms | baseline |

### Quality (BEIR nfcorpus, 50 queries)

| Method | NDCG@10 | Latency | Notes |
|--------|---------|---------|-------|
| **FlashRank MiniLM** | **0.3464** | 832ms | Best quality (teacher) |
| **FlashRank TinyBERT** | **0.331** | 40ms | Fast + good quality |
| Hybrid Fusion (distilled*) | ~0.34 | **0.45ms** | 95-98% of teacher, 100x faster |
| Distilled Pairwise (distilled*) | ~0.33 | **0.15ms** | Preference learning from FlashRank |
| Hybrid Fusion (non-distilled) | 0.320 | 54ms | Trained on demo data |
| Distilled Pairwise (non-distilled) | ~0.31 | 0.15ms | Trained on synthetic preferences |
| Late Interaction | 0.203 | **4ms** | Pre-indexed ColBERT |
| Binary Quantized | 0.151 | **10ms** | Ultra-fast hash-based |

\*Distilled methods use FlashRank as teacher. Run: `uv run python scripts/distill_ensemble_to_hybrid.py --dataset beir`

**Key findings:**
- **Distillation wins:** FlashRank quality (0.34) + local speed (0.15-0.45ms) = 100-1000x speedup
- **FlashRank (ONNX)** is 5-28x faster than SentenceTransformers (PyTorch)
- **Local ML methods** achieve sub-ms latency with competitive quality

**Distillation support:**
- ✅ **Hybrid Fusion** - Soft label regression (pointwise)
- ✅ **Distilled Pairwise** - Preference learning from FlashRank comparisons
- ⚠️ **Binary Quantized** - Limited (binary labels only)
- ❌ **Late Interaction** - Not applicable (pre-indexed)

**Production tip:** Use cascading strategy - distilled models for 70-90% of queries (fast path), FlashRank fallback for low-confidence cases (quality path). See [analysis](docs/20260415-benchmark-analysis.md#cascading-strategy-fast--smart) for details.

[→ Full analysis with BEIR results](docs/20260415-benchmark-analysis.md)

Run: `uv run python scripts/benchmark_flashrank.py` (speed) | `uv run python scripts/benchmark_beir.py nfcorpus` (quality)

## Production Patterns

### CascadeReranker: Confidence-Based Fallback

Balance speed and quality by using fast distilled models for common cases, falling back to FlashRank only when confidence is low.

```python
from reranker.strategies import CascadeReranker, CascadeConfig, HybridFusionReranker, ConfidenceMetric
from reranker.strategies.flashrank_ensemble import FlashRankEnsemble

# Fast distilled model (0.15-0.45ms)
primary = HybridFusionReranker()

# Slow but accurate teacher (4-40ms)
fallback = FlashRankEnsemble(models=["ms-marco-TinyBERT-L-2-v2"])

# Cascade with confidence threshold
cascade = CascadeReranker(
    primary,
    fallback,
    config=CascadeConfig(
        confidence_threshold=0.6,
        confidence_metric=ConfidenceMetric.MAX_SCORE
    )
)

# Use cascade
results = cascade.rerank("python tutorial", docs)

# Check stats
stats = cascade.get_stats()
print(f"Fallback rate: {stats['fallback_rate']:.1%}")
```

**Expected performance:**
- 70-90% queries: 0.15-0.45ms (distilled)
- 10-30% queries: 4-40ms (FlashRank)
- **Average: 1-5ms** vs 40-832ms (FlashRank alone)

**Confidence metrics:**
- `MAX_SCORE`: Highest relevance score (simple, default)
- `TOP_MARGIN`: Difference between top-2 scores (robustness)
- `SCORE_VARIANCE`: Variance of all scores (uncertainty)



### Choosing Between Patterns

| Pattern | When to Use | Benefit |
|---------|-------------|---------|
| **CascadeReranker** | Need quality guarantees with speed | Confidence-based routing, observability |
| **PipelineReranker** | Large candidate sets, need filtering | Reduces compute progressively |
| **Standalone** | Small candidate sets, simple use case | Easiest to deploy |
| **Combine** | Production systems with SLAs | Pipeline → Cascade for maximum efficiency |

