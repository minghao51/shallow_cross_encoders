"""BEIR nfcorpus benchmark for StaticColBERTReranker.

Usage:
    uv run scripts/benchmark_beir_colbert.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import structlog

from reranker.data.beir_loader import load_beir_simple
from reranker.embedder import Embedder
from reranker.eval.benchmark_utils import evaluate_reranker_on_rows
from reranker.strategies.hybrid import HybridFusionReranker
from reranker.strategies.late_interaction import StaticColBERTReranker

logger = structlog.get_logger(__name__)


def beir_to_rows(queries: dict, corpus: dict, qrels: dict) -> list[dict]:
    rows = []
    for qid, doc_rel_map in qrels.items():
        query = queries.get(qid, "")
        for docid, rel in doc_rel_map.items():
            doc_entry = corpus.get(docid, {})
            doc_text = doc_entry.get("text", "")
            rows.append(
                {
                    "query": query,
                    "doc": doc_text,
                    "score": rel,
                }
            )
    return rows


def main() -> None:
    logger.info("Loading BEIR nfcorpus...")
    queries, corpus, qrels = load_beir_simple("nfcorpus")
    logger.info(f"  Queries: {len(queries)}")
    logger.info(f"  Corpus:  {len(corpus)}")
    logger.info(f"  Qrels:   {len(qrels)}")

    rows = beir_to_rows(queries, corpus, qrels)
    logger.info(f"  Rows:    {len(rows)}")

    corpus_texts = [entry["text"] for entry in corpus.values()]
    logger.info(f"  Corpus texts: {len(corpus_texts)}")

    embedder = Embedder(model_name="minishlab/potion-base-8M")
    logger.info(f"  Embedder: {embedder.backend_name}, dim={embedder.dimension}")

    # --- StaticColBERTReranker ---
    logger.info("\n=== StaticColBERTReranker ===")
    colbert = StaticColBERTReranker(
        embedder=embedder,
        top_k_tokens=128,
        use_salience=True,
    )

    t0 = time.perf_counter()
    colbert.fit(corpus_texts)
    fit_time = time.perf_counter() - t0
    logger.info(f"  Fit time: {fit_time:.2f}s")

    colbert_metrics = evaluate_reranker_on_rows(rows, colbert)
    logger.info(f"  NDCG@10: {colbert_metrics['ndcg@10']:.4f}")
    logger.info(f"  MRR:     {colbert_metrics['mrr']:.4f}")
    logger.info(f"  P@1:     {colbert_metrics['p@1']:.4f}")
    p50 = colbert_metrics["latency_p50_ms"]
    p99 = colbert_metrics["latency_p99_ms"]
    logger.info(f"  Latency: {p50:.2f}ms p50, {p99:.2f}ms p99")

    # --- Quantized variant (4-bit) ---
    logger.info("\n=== StaticColBERTReranker (4-bit quantized) ===")
    colbert_4bit = StaticColBERTReranker(
        embedder=embedder,
        top_k_tokens=128,
        use_salience=True,
        quantization_mode="4bit",
    )
    colbert_4bit.fit(corpus_texts)
    colbert_4bit_metrics = evaluate_reranker_on_rows(rows, colbert_4bit)
    logger.info(f"  NDCG@10: {colbert_4bit_metrics['ndcg@10']:.4f}")
    logger.info(f"  MRR:     {colbert_4bit_metrics['mrr']:.4f}")
    logger.info(f"  P@1:     {colbert_4bit_metrics['p@1']:.4f}")
    p50 = colbert_4bit_metrics["latency_p50_ms"]
    p99 = colbert_4bit_metrics["latency_p99_ms"]
    logger.info(f"  Latency: {p50:.2f}ms p50, {p99:.2f}ms p99")

    # --- Quantized variant (ternary) ---
    logger.info("\n=== StaticColBERTReranker (ternary quantized) ===")
    colbert_tern = StaticColBERTReranker(
        embedder=embedder,
        top_k_tokens=128,
        use_salience=True,
        quantization_mode="ternary",
    )
    colbert_tern.fit(corpus_texts)
    colbert_tern_metrics = evaluate_reranker_on_rows(rows, colbert_tern)
    logger.info(f"  NDCG@10: {colbert_tern_metrics['ndcg@10']:.4f}")
    logger.info(f"  MRR:     {colbert_tern_metrics['mrr']:.4f}")
    logger.info(f"  P@1:     {colbert_tern_metrics['p@1']:.4f}")
    p50 = colbert_tern_metrics["latency_p50_ms"]
    p99 = colbert_tern_metrics["latency_p99_ms"]
    logger.info(f"  Latency: {p50:.2f}ms p50, {p99:.2f}ms p99")

    # --- HybridFusionReranker (reference) ---
    logger.info("\n=== HybridFusionReranker (reference) ===")
    hybrid = HybridFusionReranker(embedder=embedder)
    binary_labels = [1 if row["score"] >= 2 else 0 for row in rows]
    hybrid.fit_pointwise(
        queries=[str(r["query"]) for r in rows],
        docs=[str(r["doc"]) for r in rows],
        scores=[float(s) for s in binary_labels],
    )
    hybrid_metrics = evaluate_reranker_on_rows(rows, hybrid)
    logger.info(f"  NDCG@10: {hybrid_metrics['ndcg@10']:.4f}")
    logger.info(f"  MRR:     {hybrid_metrics['mrr']:.4f}")
    logger.info(f"  P@1:     {hybrid_metrics['p@1']:.4f}")
    p50 = hybrid_metrics["latency_p50_ms"]
    p99 = hybrid_metrics["latency_p99_ms"]
    logger.info(f"  Latency: {p50:.2f}ms p50, {p99:.2f}ms p99")

    # --- Index size comparison ---
    logger.info("\n=== Index Size Comparison ===")

    def estimate_mb(obj) -> float:
        import pickle

        return len(pickle.dumps(obj)) / (1024 * 1024)

    colbert_size = sum(entry.vectors.nbytes + (len(entry.tokens) * 50) for entry in colbert._index)
    logger.info(f"  ColBERT index:   {colbert_size / (1024 * 1024):.2f} MB")
    if colbert_4bit._index[0].quantized is not None:
        colbert_4bit_size = sum(
            entry.quantized.codes.nbytes if entry.quantized else entry.vectors.nbytes
            for entry in colbert_4bit._index
        )
        logger.info(f"  ColBERT 4-bit:   {colbert_4bit_size / (1024 * 1024):.2f} MB")
    if colbert_tern._index[0].quantized is not None:
        colbert_tern_size = sum(
            entry.quantized.codes.nbytes if entry.quantized else entry.vectors.nbytes
            for entry in colbert_tern._index
        )
        logger.info(f"  ColBERT ternary: {colbert_tern_size / (1024 * 1024):.2f} MB")

    hybrid_model = hybrid.model_backend
    if hasattr(hybrid_model, "get_booster"):
        import joblib

        hybrid_size = len(joblib.dumps(hybrid_model)) / (1024 * 1024)
    else:
        import pickle

        hybrid_size = len(pickle.dumps(hybrid_model)) / (1024 * 1024)
    logger.info(f"  Hybrid model:    {hybrid_size:.2f} MB")
    ratio = colbert_size / max(hybrid_size * (1024 * 1024), 1)
    logger.info(f"  Colbert / Hybrid ratio: {ratio:.2f}x")

    # --- Latency at 50 docs ---
    logger.info("\n=== Latency at 50 documents ===")
    sample_docs = corpus_texts[:50]
    sample_query = list(queries.values())[0]
    colbert_50 = StaticColBERTReranker(embedder=embedder, top_k_tokens=128, use_salience=True)
    colbert_50.fit(sample_docs)

    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        colbert_50.rerank(sample_query, sample_docs)
        latencies.append((time.perf_counter() - t0) * 1000)
    mean_lat = np.mean(latencies)
    p50_lat = np.median(latencies)
    p99_lat = np.percentile(latencies, 99)
    logger.info(f"  Mean: {mean_lat:.2f}ms, P50: {p50_lat:.2f}ms, P99: {p99_lat:.2f}ms")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    results = {
        "dataset": "nfcorpus",
        "embedder": "minishlab/potion-base-8M",
        "colbert": {
            "ndcg@10": round(colbert_metrics["ndcg@10"], 4),
            "mrr": round(colbert_metrics["mrr"], 4),
            "p@1": round(colbert_metrics["p@1"], 4),
            "latency_p50_ms": colbert_metrics["latency_p50_ms"],
            "latency_p99_ms": colbert_metrics["latency_p99_ms"],
        },
        "colbert_4bit": {
            "ndcg@10": round(colbert_4bit_metrics["ndcg@10"], 4),
            "mrr": round(colbert_4bit_metrics["mrr"], 4),
            "p@1": round(colbert_4bit_metrics["p@1"], 4),
            "latency_p50_ms": colbert_4bit_metrics["latency_p50_ms"],
            "latency_p99_ms": colbert_4bit_metrics["latency_p99_ms"],
        },
        "colbert_ternary": {
            "ndcg@10": round(colbert_tern_metrics["ndcg@10"], 4),
            "mrr": round(colbert_tern_metrics["mrr"], 4),
            "p@1": round(colbert_tern_metrics["p@1"], 4),
            "latency_p50_ms": colbert_tern_metrics["latency_p50_ms"],
            "latency_p99_ms": colbert_tern_metrics["latency_p99_ms"],
        },
        "hybrid_reference": {
            "ndcg@10": round(hybrid_metrics["ndcg@10"], 4),
            "mrr": round(hybrid_metrics["mrr"], 4),
            "p@1": round(hybrid_metrics["p@1"], 4),
            "latency_p50_ms": hybrid_metrics["latency_p50_ms"],
            "latency_p99_ms": hybrid_metrics["latency_p99_ms"],
        },
        "index_size_mb": {
            "colbert_float32": round(colbert_size / (1024 * 1024), 2),
            "hybrid_model": round(hybrid_size, 2),
        },
        "latency_50_docs_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(float(np.median(latencies)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
        },
        "exit_criteria": {
            "ndcg_gte_0.38": colbert_metrics["ndcg@10"] >= 0.38,
            "latency_50_docs_lt_10ms": float(np.percentile(latencies, 99)) < 10.0,
        },
    }

    print(json.dumps(results, indent=2))
    out_path = Path("docs/benchmarks/20260501-beir-nfcorpus-colbert-results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
