# Ensemble Distillation Design

**Date:** 2026-04-14
**Status:** Approved

## Overview

Implement ensemble distillation pipeline where FlashRank models (TinyBERT + MiniLM) serve as teachers to train Hybrid Fusion Reranker student. Goal: MiniLM-quality (0.3464 NDCG@10) with Hybrid-speed (50ms).

## CLI Interface

```bash
uv run scripts/distill_ensemble_to_hybrid.py [OPTIONS]

Options:
  --dataset {beir,synth,mixed,custom}  Data source (default: mixed)
  --custom-path PATH                    Path to custom BEIR-format dataset
  --method {pointwise,pairwise}         Training mode (default: pairwise)
  --force-regenerate                    Bypass cached labels
  --output PATH                         Model save location
  --teachers LIST                       FlashRank models (default: tinybert,minilm)
  --cache-dir PATH                      Label cache directory
```

## Architecture

### Pipeline Stages

1. **Label Generation** (cached)
   - FlashRankEnsemble scores query-doc pairs
   - Batch inference for speed
   - Ensemble = average of TinyBERT + MiniLM scores
   - Cache: `data/models/ensemble_labels_{dataset}_{hash}.json`

2. **Data Mixing**
   - BEIR: NFCorpus queries + hard negatives
   - Synthetic: LLM-generated preferences
   - Custom: User-provided BEIR-format
   - Combine with source tracking

3. **Training**
   - Pointwise: `fit(queries, docs, scores)` — regression to ensemble scores
   - Pairwise: Convert scores → preferences, `fit(queries, doc_as, doc_bs, labels)`

4. **Evaluation**
   - NDCG@10 comparison: Original vs Distilled
   - Latency benchmark (50ms target)
   - Quality retention metrics

### Data Flow

```
1. Load datasets
   ├─ BEIR: queries (3233), corpus docs, qrels
   ├─ Synth: generated query-doc pairs
   └─ Custom: BEIR-format JSON

2. Generate labels (cached)
   ├─ Check cache for existing ensemble_labels_{hash}.json
   ├─ If miss: FlashRankEnsemble.score_batch() → average → cache
   └─ Load cached

3. Mix data
   ├─ BEIR/Synth/Custom pairs + ensemble scores
   └─ Combine, shuffle, split train/val

4. Train Hybrid
   ├─ Pointwise: fit with ensemble scores
   └─ Pairwise: scores → pairwise preferences

5. Evaluate
   ├─ NDCG@10, latency
   └─ Save distilled model
```

## Core Components

### FlashRankEnsemble (new)

```python
# src/reranker/strategies/flashrank_ensemble.py
class FlashRankEnsemble:
    """Wrapper for FlashRank multi-teacher ensemble."""

    def __init__(self, models: list[str]):
        self.rankers = [flashrank.Ranker(m) for m in models]

    def score_batch(self, query: str, docs: list[str]) -> np.ndarray:
        scores = [r.rerank(query, docs) for r in self.rankers]
        return np.mean(scores, axis=0)
```

### EnsembleLabelCache (new)

```python
# src/reranker/data/ensemble_cache.py
class EnsembleLabelCache:
    """Persistent storage for teacher-generated labels."""

    def get_hash(self, dataset: str, teachers: list[str]) -> str:
        config = f"{dataset}:{','.join(teachers)}"
        return hashlib.sha256(config.encode()).hexdigest()[:16]

    def load_or_generate(self, dataset, teachers, generator_fn) -> dict:
        cache_key = self.get_hash(dataset, teachers)
        cached = self._load(cache_key)
        if cached and not force_regenerate:
            return cached
        labels = generator_fn()
        self._save(cache_key, labels)
        return labels
```

### Hybrid Training Extension

- Pointwise mode: Float ensemble scores, MSE loss
- Pairwise mode: Scores → preferences via comparison

## Error Handling

| Scenario | Action |
|----------|--------|
| FlashRank missing | Fallback to sentence-transformers |
| BEIR data missing | Auto-run download, prompt user |
| Cache corruption | Warn, regenerate labels |
| Empty labels | Dataset error, clear message |
| Training failure | XGBoost → sklearn fallback |

## Custom Dataset Format

```json
{
  "queries": {"q1": "query text", "q2": "another query"},
  "corpus": {"d1": {"text": "doc text"}, "d2": {"text": "doc 2"}},
  "qrels": {"q1": {"d1": 1, "d2": 0}, "q2": {"d1": 0, "d2": 1}}
}
```

**Validation:** Schema check, non-empty, binary/graded relevance.

## Expected Results

- Quality: 95-98% of ensemble NDCG@10 (~0.335-0.340)
- Latency: ~50ms (same as Hybrid)
- Training time: ~30 min (mostly teacher inference)
