# shallow-cross-encoders

CPU-native reranking and consistency pipeline using static embeddings, lexical signals, and shallow models.

## Features

- **Hybrid fusion reranking** combining lexical (BM25) and semantic (static embedding) signals
- **Distilled pairwise rankers** for lightweight, fast inference
- **Consistency engine** for detecting contradictions across documents
- **ColBERT-style late interaction** with static models
- **Meta-routing** to automatically select the best strategy per query
- **Synthetic data generation** pipelines for training and evaluation

## Quick Start

```bash
pip install shallow-cross-encoders
```

```python
from reranker import HybridFusionReranker, RankedDoc

reranker = HybridFusionReranker()
results = reranker.rank("your query", ["doc one", "doc two", "doc three"])
```

Head to the [API Reference](api/config.md) for detailed documentation.
