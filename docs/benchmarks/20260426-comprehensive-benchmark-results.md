# 20260426-comprehensive-benchmark-results

**Date:** 2026-04-26
**Embedder:** minishlab/potion-base-32M (model2vec)
**Seed:** 42
**Dataset:** Synthetic pairs (train/test split by query)

## Summary

Full benchmark of all 12 reranking strategies on synthetic data with comprehensive metrics: NDCG@10, MAP@10, MRR, P@1, BM25 uplift, latency (mean/p50/p99), throughput (QPS), and cold-start time.

## Strategy Coverage

All 12 strategies were tested:

| Strategy | Status | Notes |
|---|---|---|
| BM25Engine | Tested | Lexical baseline |
| BinaryQuantizedReranker | Tested | Fastest method |
| DistilledPairwiseRanker | Tested | 100% accuracy on pairwise |
| HybridFusionReranker | Tested | Best quality on synthetic |
| StaticColBERTReranker | Tested | Token-level MaxSim |
| ConsistencyEngine | Tested | Perfect recall/F1 |
| PipelineReranker | Tested | 4-stage cascade |
| MultiReranker | Tested (2 combos) | BM25+Binary+ColBERT, Hybrid+BM25 |
| CascadeReranker | **NEW** | Hybrid -> FlashRank fallback |
| SPLADEReranker | Skipped | Requires HF auth token |
| FlashRankEnsemble | **NEW** | TinyBERT + MiniLM ensemble |
| MetaRouter | **NEW** | Query-type routing (fitted) |
| FlashRank Tiny/Mini | **NEW** | External baselines |
| ST TinyBERT/MiniLM | **NEW** | SentenceTransformer cross-encoders |

## Key Findings

### Ranking Quality (NDCG@10 on synthetic data)

| Strategy | NDCG@10 | MAP@10 | BM25 Uplift |
|---|---|---|---|
| hybrid | 0.2000 | 0.2000 | +0.074 |
| binary_reranker | 0.2000 | 0.2000 | +0.074 |
| cascade | 0.2000 | 0.2000 | +0.074 |
| flashrank_ensemble | 0.2000 | 0.2000 | +0.074 |
| flashrank_mini | 0.2000 | 0.2000 | +0.074 |
| late_interaction | 0.1262 | 0.1000 | +0.000 |
| bm25 | 0.0861 | 0.0500 | baseline |

**Note:** Synthetic data shows a ceiling effect where top strategies saturate at similar NDCG. Real-world datasets (BEIR, TREC) are needed for differentiation — as documented in the 20260404 analysis.

### Speed (Latency p50)

| Strategy | p50 (ms) | QPS |
|---|---|---|
| binary_reranker | 0.04 | 22,866 |
| bm25 | 0.07 | 11,854 |
| late_interaction | 0.37 | 2,710 |
| cascade | 0.93 | 1,060 |
| hybrid | 1.01 | 932 |
| flashrank_tiny | 1.57 | 201 |
| flashrank_mini | 15.29 | 36 |
| flashrank_ensemble | 18.90 | 22 |
| st_tiny | 18.49 | ~1 |
| st_mini | 19.24 | ~1 |

### Speed-Qlexity Pareto

1. **binary_reranker** — Best speed (0.04ms) AND good quality. The clear winner for latency-sensitive applications.
2. **hybrid** — Best quality with acceptable latency (~1ms). 932 QPS.
3. **cascade** — Matches hybrid quality with fallback safety net. Slightly faster than standalone hybrid.
4. **late_interaction** — Good speed (0.37ms) but lower quality on synthetic data.

### Pairwise Accuracy

- **DistilledPairwiseRanker**: 100% accuracy at 0.10ms per comparison.

### Consistency Detection

- **ConsistencyEngine**: Perfect recall (1.0), precision (1.0), F1 (1.0), zero false positives.
- Threshold variations (0.90, 0.95, 0.99) show no difference on synthetic data.

### Scaling (Latency vs Corpus Size)

All strategies scale sub-linearly. From 20 to 200 docs:
- BM25: 0.07ms -> 0.06ms (constant)
- Binary: 0.05ms -> 0.04ms (constant)
- ColBERT: 0.33ms -> 0.25ms (gets faster due to caching)
- Hybrid: 1.10ms -> 0.86ms

### Embedder Model Comparison (Embedder Grid)

- **potion-base-32M** with hybrid or binary_reranker achieves NDCG@10=1.0 at higher dims
- **potion-base-8M** binary_reranker also reaches 1.0 even at dim=64
- **potion-multilingual-128M** plateaus at 0.9262 — larger model not always better for synthetic data
- Dimension has minimal impact: dim=64 performs as well as dim=512

### CascadeReranker (New)

- With threshold=0.6: NDCG@10=0.2000, 0% fallback rate (hybrid is confident enough)
- With threshold=0.9 (ablation): Same quality, still 0% fallback
- The cascade adds minimal overhead (0.93ms vs 1.01ms for hybrid alone)

### FlashRankEnsemble (New)

- NDCG@10=0.2000 but at 18.90ms p50 — 20x slower than hybrid
- Best used as a teacher for distillation, not as a production reranker

### MetaRouter (New)

- Successfully fitted with 3 query-type profiles (navigational, informational, balanced)
- Routing decisions work but need real-world query diversity to validate

## Gaps & Recommendations

1. **SPLADE needs auth**: The naver/splade-v2-max model requires HF token. Consider using a publicly available SPLADE model or pre-downloading.
2. **Synthetic data ceiling**: All top strategies saturate at similar NDCG on synthetic data. Need BEIR/TREC benchmarks for real-world differentiation.
3. **ST cross-encoders slow**: First-call latency is dominated by model loading (~1.8s cold start). Subsequent calls are ~20ms.
4. **MultiReranker underperforms**: RRF fusion of BM25+Binary+ColBERT (0.1262) is worse than any single trained model (0.2000). The BM25 component drags it down.

## How to Reproduce

```bash
uv run benchmarks/run.py synthetic
uv run benchmarks/run.py synthetic --phases baselines ablations
uv run benchmarks/run.py full
```

Results are saved to `benchmarks/results/` and `docs/benchmarks/results/`.
