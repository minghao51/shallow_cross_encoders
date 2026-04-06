# Reranking Strategies

Overview of all ranking strategies in the shallow cross encoders pipeline.

## Strategy Catalog

| Strategy | Type | Latency | NDCG@10 | Best For |
|----------|------|---------|---------|----------|
| [BM25 Engine](methodology/bm25_engine.md) | Lexical | ~0.24ms | 0.665 | Fast baseline, large corpora |
| [Binary Quantized](methodology/binary_quantized_reranker.md) | Semantic (2-stage) | ~0.59ms | 0.848 | Best accuracy/latency trade-off |
| [Hybrid Fusion](methodology/hybrid_fusion_reranker.md) | GBDT + Heuristics | ~1.27ms | 0.801 | Production accuracy-critical |
| [Static ColBERT](methodology/static_colbert_reranker.md) | Late Interaction | ~0.50ms | 0.801 | Token-level precision |
| [Distilled Pairwise](methodology/distilled_pairwise_reranker.md) | Pairwise Tournament | ~0.17ms | — | Pairwise comparisons |
| [Consistency Engine](methodology/consistency_engine.md) | Claim Extraction | ~0.07ms | — | Contradiction detection |
| [Pipeline](methodology/pipeline_reranker.md) | Cascading | ~2.21ms | 0.528 | Multi-stage filtering |

## Shared Components

All strategies share these building blocks:

### Embedder
- **Model**: `minishlab/potion-base-8M` (configurable)
- **Dimension**: 256
- **Backend**: model2vec with deterministic hashed fallback
- **Purpose**: Converts text to dense vectors for all semantic strategies

### Protocols
- `BaseReranker`: Common `rerank(query, docs) -> list[RankedDoc]` contract
- `HeuristicAdapter`: Injects domain-specific scalar features
- `RankedDoc`: `(doc, score, rank, metadata)` dataclass

### Configuration
All hyperparameters are managed through `src/reranker/config.py` (Pydantic settings) and can be overridden via environment variables.

---

*See individual strategy documents for mathematical formulations, DAG components, and methodology.*
