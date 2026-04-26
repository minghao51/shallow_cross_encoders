# Comprehensive Benchmark Results

Generated: 2026-04-26 23:43:44
Seed: 42
Embedder: minishlab/potion-base-32M
Quick mode: False
Test pairs: 10
Test preferences: 7
Test contradictions: 7

## Baseline Results (Ranking)

| Strategy | NDCG@10 | MAP@10 | MRR | P@1 | Latency (ms) | p50 | p99 | QPS | Cold-start (ms) | BM25 Uplift |
|----------|---------|--------|-----|-----|--------------|-----|-----|-----|-----------------|-------------|
| bm25 | 0.0861 | 0.0500 | 0.0500 | 0.0000 | 0.08 | 0.07 | 0.17 | 11854 | 0.0 | +0.0000 |
| hybrid | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 1.07 | 1.01 | 1.38 | 932 | 119.3 | +0.0738 |
| late_interaction | 0.1262 | 0.1000 | 0.1000 | 0.0000 | 0.37 | 0.37 | 0.43 | 2710 | 0.3 | +0.0000 |
| binary_reranker | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.04 | 0.04 | 0.07 | 22866 | 0.1 | +0.0738 |
| pipeline | 0.1262 | 0.1000 | 0.1000 | 0.0000 | 1.38 | 1.35 | 1.52 | 724 | 0.0 | +0.0401 |
| cascade | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.94 | 0.93 | 1.00 | 1060 | 0.0 | +0.0738 |
| flashrank_ensemble | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 46.09 | 18.90 | 141.10 | 22 | 0.0 | +0.0738 |
| flashrank_tiny | 0.1262 | 0.1000 | 0.1000 | 0.0000 | 4.97 | 1.57 | 17.76 | 201 | 0.0 | +0.0000 |
| flashrank_mini | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 27.96 | 15.29 | 77.59 | 36 | 0.0 | +0.0738 |
| st_tiny | 0.1262 | 0.1000 | 0.1000 | 0.0000 | 1844.66 | 18.49 | 8793.87 | 1 | 0.0 | +0.0000 |
| st_mini | 0.1262 | 0.1000 | 0.1000 | 0.0000 | 1917.45 | 19.24 | 9137.83 | 1 | 0.0 | +0.0000 |

## Distilled Pairwise Ranker

| Accuracy | Latency (ms) | p50 (ms) | p99 (ms) |
|----------|--------------|----------|----------|
| 1.0000 +/- 0.0000 | 0.10 | 0.10 | 0.14 |

## Consistency Engine

| Recall | Precision | F1 | FPR | Accuracy | Latency (ms) |
|--------|-----------|-----|-----|----------|--------------|
| 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.29 |

## MultiReranker (RRF Fusion)

| Experiment | Rerankers | NDCG@10 | MAP@10 | MRR | P@1 | Latency (ms) |
|------------|-----------|---------|--------|-----|-----|--------------|
| multi_bm25_binary_colbert | bm25, binary, colbert | 0.1262 | 0.1000 | 0.1000 | 0.0000 | 0.40 |
| multi_hybrid_bm25 | hybrid, bm25 | 0.1262 | 0.1000 | 0.1000 | 0.0000 | 1.14 |

## Ablation Studies

### hybrid

| Experiment | NDCG@10 | MAP@10 | Delta vs Baseline |
|------------|---------|--------|-------------------|
| hybrid_ablation_no_adapters | 0.2000 | 0.2000 | +0.0000 |

### late_interaction

| Experiment | NDCG@10 | MAP@10 | Delta vs Baseline |
|------------|---------|--------|-------------------|
| colbert_ablation_no_salience | 0.1262 | 0.1000 | +0.0000 |
| colbert_ablation_64_tokens | 0.1262 | 0.1000 | +0.0000 |

### binary_reranker

| Experiment | NDCG@10 | MAP@10 | Delta vs Baseline |
|------------|---------|--------|-------------------|
| binary_ablation_hamming_only | 0.2000 | 0.2000 | +0.0000 |
| binary_ablation_aggressive | 0.2000 | 0.2000 | +0.0000 |

### consistency

| Experiment | Recall | F1 | Delta Recall | Delta F1 |
|------------|--------|----|-------------|----------|
| consistency_ablation_relaxed | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| consistency_ablation_strict | 1.0000 | 1.0000 | +0.0000 | +0.0000 |

### cascade

| Experiment | NDCG@10 | Fallback Rate | Delta NDCG |
|------------|---------|---------------|------------|
| cascade_ablation_high_threshold | 0.2000 | 0.00% | +0.0000 |

## Scaling Results (Latency in ms)

| Corpus Size | BM25 (ms) | Hybrid (ms) | ColBERT (ms) | Binary (ms) |
|-------------|-----------|-------------|--------------|-------------|
| 20 | 0.07 | 1.10 | 0.33 | 0.05 |
| 50 | 0.07 | 0.92 | 0.30 | 0.05 |
| 100 | 0.06 | 0.86 | 0.28 | 0.04 |
| 200 | 0.06 | 0.86 | 0.25 | 0.04 |

## Embedder Model Comparison

| Model | Dim | Strategy | NDCG@10 | MAP@10 | MRR | P@1 | Latency (ms) |
|-------|-----|----------|---------|--------|-----|-----|--------------|
| potion-base-8M | 64 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.02 |
| potion-base-8M | 128 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.02 |
| potion-base-8M | 256 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.02 |
| potion-base-8M | 512 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.02 |
| potion-base-32M | 64 | hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.67 |
| potion-base-32M | 64 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.02 |
| potion-base-32M | 128 | hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.68 |
| potion-base-32M | 128 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.02 |
| potion-base-32M | 256 | hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.63 |
| potion-base-32M | 256 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.02 |
| potion-base-32M | 512 | hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.65 |
| potion-base-32M | 512 | binary_reranker | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.02 |
| potion-base-8M | 64 | hybrid | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.66 |
| potion-base-8M | 64 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.19 |
| potion-base-8M | 128 | hybrid | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.66 |
| potion-base-8M | 128 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.17 |
| potion-base-8M | 256 | hybrid | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.64 |
| potion-base-8M | 256 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.17 |
| potion-base-8M | 512 | hybrid | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.67 |
| potion-base-8M | 512 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.16 |
| potion-base-32M | 64 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.19 |
| potion-base-32M | 128 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.17 |
| potion-base-32M | 256 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.17 |
| potion-base-32M | 512 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.18 |
| potion-multilingual-128M | 64 | hybrid | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.68 |
| potion-multilingual-128M | 64 | binary_reranker | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.02 |
| potion-multilingual-128M | 64 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.18 |
| potion-multilingual-128M | 128 | hybrid | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.79 |
| potion-multilingual-128M | 128 | binary_reranker | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.02 |
| potion-multilingual-128M | 128 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.18 |
| potion-multilingual-128M | 256 | hybrid | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.73 |
| potion-multilingual-128M | 256 | binary_reranker | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.02 |
| potion-multilingual-128M | 256 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.18 |
| potion-multilingual-128M | 512 | hybrid | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.77 |
| potion-multilingual-128M | 512 | binary_reranker | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.02 |
| potion-multilingual-128M | 512 | late_interaction | 0.9262 | 0.9000 | 0.9000 | 0.8000 | 0.17 |

**Best NDCG@10**: minishlab/potion-base-8M (dim=64, strategy=binary_reranker) = 1.0000

## Key Findings

- **Best NDCG@10**: hybrid_baseline (0.2000)
- **Fastest**: binary_baseline (0.04ms)
- **Best BM25 Uplift**: hybrid_baseline (+0.0738)
