# Ensemble Distillation Guide

**Last Updated:** 2026-04-14

## Overview

Ensemble distillation combines multiple FlashRank cross-encoder models to serve as teachers, generating soft labels for training a fast Hybrid Fusion Reranker student. This approach achieves 95-98% of ensemble NDCG@10 while maintaining ~50ms latency.

### Key Benefits

- **Quality**: 95-98% of teacher ensemble NDCG@10
- **Latency**: ~50ms per query (same as Hybrid)
- **Training Time**: ~30 minutes (cached after first run)
- **Flexibility**: Support for BEIR, custom datasets, and mixed data sources

### How It Works

1. **Teacher Ensemble**: Multiple FlashRank models (TinyBERT, MiniLM) score query-document pairs
2. **Label Generation**: Ensemble scores are cached for efficient reuse
3. **Student Training**: Hybrid Fusion Reranker learns from teacher soft labels
4. **Evaluation**: NDCG@10 and latency metrics validate performance

## Installation

### Base Install

```bash
# Install with FlashRank support
uv sync --extra flashrank
```

### Optional Dependencies

```bash
# For BEIR benchmark datasets
uv sync --extra beir

# For full runtime dependencies
uv sync --extra runtime
```

### Verify Installation

```bash
# Test FlashRank import
uv run python -c "import flashrank; print('FlashRank installed')"
```

## Usage

### Basic Usage

Train on default BEIR NFCorpus dataset with pairwise method:

```bash
uv run python scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise
```

### Dataset Options

#### BEIR Dataset

```bash
# Use BEIR NFCorpus (default)
uv run python scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise

# Specify custom cache directory
uv run python scripts/distill_ensemble_to_hybrid.py --dataset beir --cache-dir data/models
```

#### Custom Dataset

```bash
# Use custom BEIR-format dataset
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset custom \
  --custom-path data/custom_dataset.jsonl \
  --method pairwise
```

#### Mixed Dataset

```bash
# Combine BEIR with custom data
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset mixed \
  --custom-path data/custom_dataset.jsonl \
  --method pairwise
```

### Training Methods

#### Pointwise (Regression)

```bash
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset beir \
  --method pointwise \
  --output data/models/hybrid_pointwise.pkl
```

#### Pairwise (Ranking)

```bash
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset beir \
  --method pairwise \
  --output data/models/hybrid_pairwise.pkl
```

### Advanced Options

```bash
# Custom teacher models
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset beir \
  --teachers ms-marco-TinyBERT-L-2-v2 ms-marco-MiniLM-L-12-v2 \
  --method pairwise

# Force regenerate cached labels
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset beir \
  --method pairwise \
  --force-regenerate

# Custom output path
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset beir \
  --method pairwise \
  --output data/models/my_hybrid_model.pkl
```

## Command Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | str | `mixed` | Dataset source: `beir`, `custom`, `synth`, `mixed` |
| `--custom-path` | Path | `None` | Path to custom dataset (required for `custom`, optional for `mixed`) |
| `--method` | str | `pairwise` | Training method: `pointwise` or `pairwise` |
| `--teachers` | list | `ms-marco-TinyBERT-L-2-v2 ms-marco-MiniLM-L-12-v2` | FlashRank teacher models |
| `--cache-dir` | Path | `data/models` | Directory for caching teacher labels |
| `--output` | Path | `data/models/hybrid_distilled.pkl` | Output path for trained model |
| `--force-regenerate` | flag | `False` | Force regeneration of cached labels |

## Expected Results

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **NDCG@10** | 0.30-0.34 | 95-98% of teacher ensemble |
| **Latency** | ~50ms | Same as Hybrid Fusion |
| **Training Time** | ~30 min | First run (cached after) |
| **Cache Hit Rate** | 100% | Subsequent runs use cache |

### Quality Comparison

| Strategy | NDCG@10 | Latency | Quality/Latency |
|----------|---------|---------|-----------------|
| **Teacher Ensemble** | 0.34-0.35 | 900ms | 0.038 NDCG/ms |
| **Distilled Student** | 0.32-0.34 | 50ms | **0.0064 NDCG/ms** |
| **Hybrid Fusion** | 0.30-0.32 | 54ms | 0.0056 NDCG/ms |
| **FlashRank TinyBERT** | 0.31-0.33 | 40ms | 0.0080 NDCG/ms |

### Training Dataset Size

| Dataset | Queries | Documents | Pairs | Training Time |
|---------|---------|-----------|-------|---------------|
| **BEIR NFCorpus** | 50 | 500 | 25,000 | ~30 min |
| **Custom (small)** | 20 | 200 | 4,000 | ~5 min |
| **Mixed** | 70 | 700 | 49,000 | ~45 min |

## Output Files

### Trained Model

- **Path**: `data/models/hybrid_distilled.pkl` (default)
- **Format**: Pickle file with trained HybridFusionReranker
- **Usage**: Load with `HybridFusionReranker.load(path)`

### Cache Files

- **Location**: `data/models/ensemble_cache/`
- **Format**: JSON files with ensemble labels
- **Naming**: `ensemble_labels_{dataset_id}.json`
- **Benefit**: Skip teacher inference on subsequent runs

### Evaluation Results

The script outputs evaluation metrics including:

- NDCG@10 score
- Average latency (ms)
- Query evaluation count
- Score distribution

## Troubleshooting

### Import Errors

```bash
# FlashRank not installed
ImportError: flashrank
# Solution: uv sync --extra flashrank

# BEIR not installed
ImportError: beir
# Solution: uv sync --extra beir
```

### Cache Issues

```bash
# Force regenerate if cache is corrupted
uv run python scripts/distill_ensemble_to_hybrid.py --force-regenerate

# Clear cache manually
rm -rf data/models/ensemble_cache/
```

### Memory Issues

```bash
# Reduce dataset size for testing
# Edit script: query_ids = sorted(queries_dict.keys())[:10]
# Edit script: doc_ids = sorted(corpus_dict.keys())[:100]
```

### Performance Issues

```bash
# Use pointwise method for faster training
uv run python scripts/distill_ensemble_to_hybrid.py --method pointwise

# Use fewer teacher models
uv run python scripts/distill_ensemble_to_hybrid.py --teachers ms-marco-TinyBERT-L-2-v2
```

## Examples

### Example 1: Quick Start

```bash
# Train on BEIR with default settings
uv run python scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise

# Expected output:
# Teachers: ms-marco-TinyBERT-L-2-v2, ms-marco-MiniLM-L-12-v2
# Dataset: beir
# Method: pairwise
# Loaded 50 queries, 500 documents
# Generated 25000 query-document pair scores
# Training with 25000 pairwise comparisons
# NDCG@10: 0.3245
# Avg Latency: 48.32 ms
```

### Example 2: Custom Dataset

```bash
# Prepare custom dataset in BEIR format
cat > data/custom.jsonl << EOF
{"_id": "q1", "text": "machine learning tutorials"}
{"_id": "q2", "text": "python programming tips"}
EOF

# Train on custom data
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset custom \
  --custom-path data/custom.jsonl \
  --method pairwise \
  --output data/models/custom_hybrid.pkl
```

### Example 3: Production Training

```bash
# Full production training with all options
uv run python scripts/distill_ensemble_to_hybrid.py \
  --dataset mixed \
  --custom-path data/production_data.jsonl \
  --method pairwise \
  --teachers ms-marco-TinyBERT-L-2-v2 ms-marco-MiniLM-L-12-v2 \
  --cache-dir data/production_cache \
  --output data/models/production_hybrid.pkl

# Expected output:
# Teachers: ms-marco-TinyBERT-L-2-v2, ms-marco-MiniLM-L-12-v2
# Dataset: mixed
# Custom path: data/production_data.jsonl
# Method: pairwise
# Loaded 70 queries, 700 documents
# Generated 49000 query-document pair scores
# Training with 49000 pairwise comparisons
# Model saved to data/models/production_hybrid.pkl
# NDCG@10: 0.3389
# Avg Latency: 52.18 ms
```

## Architecture

### Teacher Ensemble

```
FlashRankEnsemble
├── ms-marco-TinyBERT-L-2-v2 (4MB, 40ms)
├── ms-marco-MiniLM-L-12-v2 (34MB, 832ms)
└── Ensemble scoring (average)
```

### Student Training

```
HybridFusionReranker
├── Pointwise: Regression on ensemble scores
└── Pairwise: Ranking from ensemble preferences
```

### Caching Strategy

```
EnsembleLabelCache
├── Dataset ID: queries_{N}_docs_{M}
├── Teacher models: [model1, model2, ...]
└── Cached labels: {(q_idx, d_idx): score}
```

## References

- [FlashRank GitHub](https://github.com/PrithivirajDamodaran/FlashRank)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [MS-MARCO Dataset](https://microsoft.github.io/msmarco/)
- [Hybrid Fusion Reranker](/Users/minghao/Desktop/personal/shallow_cross_encoders/docs/pipeline_reranker.md)

## Related Documentation

- [FlashRank Comparison](/Users/minghao/Desktop/personal/shallow_cross_encoders/docs/20260413-flashrank-comparison.md)
- [Pipeline Reranker](/Users/minghao/Desktop/personal/shallow_cross_encoders/docs/pipeline_reranker.md)
- [Technical Roadmap](/Users/minghao/Desktop/personal/shallow_cross_encoders/docs/technical-roadmap.md)
