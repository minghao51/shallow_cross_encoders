# Cascade Reranker

Confidence-based cascading reranker that balances speed and quality by using fast distilled models for common cases and falling back to slower but more accurate models when confidence is low.

## Overview

**Problem:** Production systems need both speed (sub-millisecond latency) and quality (high NDCG@10). Fast distilled models may have lower quality on uncertain queries, while accurate cross-encoders are too slow for all queries.

**Solution:** CascadeReranker uses a fast distilled model (Hybrid Fusion, Distilled Pairwise) as the primary, falling back to a slower but more accurate model (FlashRank) only when confidence is low.

## Architecture

```
Query → Primary Reranker (Fast)
           ↓
         Compute Confidence
           ↓
    ┌──────┴──────┐
    ↓             ↓
High          Low
Confidence     Confidence
    ↓             ↓
Return       FlashRank (Slow)
Results       Results
```

## Configuration

### CascadeConfig

```python
from reranker.strategies import CascadeConfig, ConfidenceMetric

config = CascadeConfig(
    confidence_threshold=0.6,        # Threshold for triggering fallback
    confidence_metric=ConfidenceMetric.MAX_SCORE,  # How to compute confidence
    fallback_strategy="flashrank"    # When to use fallback
)
```

**Parameters:**
- `confidence_threshold`: Value below which fallback triggers (0-1, default 0.6)
- `confidence_metric`: How to compute confidence (see below)
- `fallback_strategy`: "flashrank" (conditional), "always" (benchmark), "never" (fast-only)

### Confidence Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `MAX_SCORE` | Highest relevance score | Simple, works across all models (default) |
| `TOP_MARGIN` | Difference between top-2 scores | Robustness indicator |
| `SCORE_VARIANCE` | Variance of all scores | Uncertainty indicator |

## Usage

### Basic Usage

```python
from reranker.strategies import CascadeReranker, CascadeConfig, HybridFusionReranker
from reranker.strategies.flashrank_ensemble import FlashRankEnsemble

# Train primary (fast distilled model)
primary = HybridFusionReranker()
primary.fit_pointwise(queries, docs, scores)

# Initialize fallback (slow but accurate)
fallback = FlashRankEnsemble(models=["ms-marco-TinyBERT-L-2-v2"])

# Create cascade
cascade = CascadeReranker(primary, fallback)

# Rerank with automatic fallback
results = cascade.rerank("python tutorial", docs)
```

### Advanced Configuration

```python
from reranker.strategies import ConfidenceMetric

# Use TOP_MARGIN metric for robustness
cascade = CascadeReranker(
    primary=primary,
    fallback=fallback,
    config=CascadeConfig(
        confidence_threshold=0.1,  # Lower threshold for margin
        confidence_metric=ConfidenceMetric.TOP_MARGIN
    )
)

results = cascade.rerank("python tutorial", docs)
```

### Observability

```python
# Check which path was taken
for result in results:
    print(f"Doc: {result.doc}")
    print(f"  Fallback used: {result.metadata['fallback_used']}")
    print(f"  Confidence: {result.metadata['confidence']:.3f}")
    print(f"  Metric: {result.metadata['metric']}")

# Get aggregate statistics
stats = cascade.get_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Fallback rate: {stats['fallback_rate']:.1%}")
print(f"Avg confidence: {stats['avg_confidence']:.3f}")
```

## Performance Characteristics

### Latency

| Scenario | Primary | Fallback | Expected Mix | Average Latency |
|----------|---------|----------|--------------|-----------------|
| High confidence | 0.15-0.45ms | N/A | 70-90% | 0.1-0.4ms |
| Low confidence | 0.15-0.45ms | 4-40ms | 10-30% | 0.4-12ms |
| **Overall** | - | - | 100% | **1-5ms** |

### Quality

- **High confidence path:** Maintains primary model quality
- **Low confidence path:** Matches fallback model quality (FlashRank: 0.33-0.346 NDCG@10)
- **Overall:** Weighted average based on confidence distribution

## Tuning Guidelines

### Setting confidence_threshold

1. **Start conservative** (0.6-0.7) - minimizes fallback usage
2. **Measure quality** - track NDCG@10 on validation set
3. **Adjust based on:**
   - **Fallback rate too high** (>30%) → Increase threshold
   - **Quality too low** → Decrease threshold
   - **Latency budget exceeded** → Increase threshold

### Choosing confidence_metric

| Metric | When to Use | Characteristics |
|--------|-------------|-----------------|
| `MAX_SCORE` | General purpose | Simple, correlates with absolute relevance |
| `TOP_MARGIN` | Need robustness | Indicates clear winner, less sensitive to score scale |
| `SCORE_VARIANCE` | Uncertain queries | High variance = model uncertainty |

### Example Tuning Process

```python
# 1. Start with default
cascade = CascadeReranker(primary, fallback)

# 2. Run on validation set
for query, docs in validation_set:
    cascade.rerank(query, docs)

# 3. Check stats
stats = cascade.get_stats()
print(f"Fallback rate: {stats['fallback_rate']:.1%}")

# 4. Adjust if needed
if stats['fallback_rate'] > 0.3:
    # Too much fallback, increase threshold
    cascade.config.confidence_threshold = 0.7
elif quality_metric < target_quality:
    # Quality too low, decrease threshold
    cascade.config.confidence_threshold = 0.5
```

## Best Practices

1. **Use distilled primary models** - Hybrid Fusion or Distilled Pairwise trained via FlashRank
2. **Set appropriate threshold** - Start at 0.6, tune based on validation metrics
3. **Monitor fallback rate** - Target 10-30% for optimal balance
4. **Track statistics** - Use `get_stats()` for observability
5. **Consider PIPELINE+CASCADE** - Use PipelineReranker as primary, CascadeReranker for final quality assurance

## Comparison to Other Patterns

| Pattern | Trigger | Benefit |
|---------|---------|---------|
| **CascadeReranker** | Confidence-based routing | Quality guarantees with speed |
| **PipelineReranker** | Sequential top-k filtering | Reduces compute progressively |
| **MultiReranker** | Ensemble fusion | Robustness via multiple models |

## Implementation Details

### Thread Safety

CascadeReranker maintains internal state for statistics tracking. Not thread-safe. Use separate instances per thread/process or reset stats between batches.

### Serialization

CascadeReranker is stateless except for statistics. To save/load:

```python
# Save components
primary.save("primary_model.pkl")
# fallback.save("fallback_model.pkl")  # if applicable

# Load and recreate
primary = HybridFusionReranker.load("primary_model.pkl")
fallback = FlashRankEnsemble(models=["ms-marco-TinyBERT-L-2-v2"])
cascade = CascadeReranker(primary, fallback)
```

## References

- [Hybrid Fusion Reranker](./hybrid_fusion_reranker.md) - Primary model options
- [Ensemble Distillation Guide](../guides/ensemble-distillation-guide.md) - Fallback teacher setup
