"""Run benchmark on TREC COVID dataset (small, real data).

Usage:
    uv run scripts/benchmarks/run_trec.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from reranker.adapters.flashrank_wrapper import FlashRankWrapper
from reranker.eval.benchmark_utils import evaluate_reranker_on_rows, train_strategies
from reranker.strategies.hybrid import KeywordMatchAdapter


def download_trec_covid(save_path: Path = Path("data/trec-covid")) -> dict[str, Any]:
    save_path.mkdir(parents=True, exist_ok=True)

    url = "https://raw.githubusercontent.com/davidwsmith/beir-trec-covid/main/test.json"

    print("Downloading TREC COVID test data...")
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()

    data = response.json()
    print(f"Loaded {len(data.get('corpus', []))} docs, {len(data.get('queries', {}))} queries")

    return {
        "corpus": {str(k): v for k, v in data.get("corpus", {}).items()},
        "queries": {str(k): v for k, v in data.get("queries", {}).items()},
        "qrels": {
            str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in data.get("qrels", {}).items()
        },
    }


def prepare_for_benchmark(
    dataset: dict[str, Any],
    top_k_docs: int = 50,
    max_queries: int = 30,
) -> list[dict[str, Any]]:
    from rank_bm25 import BM25Okapi

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]

    corpus_texts = [doc.get("text", doc.get("title", "")) for doc in corpus.values()]
    corpus_ids = list(corpus.keys())
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]

    bm25 = BM25Okapi(tokenized_corpus)

    rows = []
    query_ids = list(queries.keys())[:max_queries]

    for query_id in query_ids:
        query = queries[query_id]

        tokenized_query = query.lower().split()
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-top_k_docs:][::-1]

        relevant_docs = qrels.get(query_id, {})

        for doc_idx in top_indices:
            doc_id = corpus_ids[doc_idx]
            doc_text = corpus[doc_id].get("text", corpus[doc_id].get("title", ""))

            relevance = 1 if relevant_docs.get(doc_id, 0) > 0 else 0

            rows.append(
                {
                    "query": query,
                    "doc": doc_text[:1000],
                    "score": relevance,
                    "query_id": query_id,
                    "doc_id": doc_id,
                }
            )

    print(f"Prepared {len(rows)} pairs from {len(query_ids)} queries")
    return rows


def evaluate_benchmark(
    max_queries: int = 30,
    top_k_docs: int = 50,
) -> dict[str, Any]:
    print("\n=== Benchmark on TREC COVID ===")

    dataset = download_trec_covid()
    rows = prepare_for_benchmark(dataset, top_k_docs=top_k_docs, max_queries=max_queries)

    results: dict[str, Any] = {
        "dataset": "trec-covid",
        "num_queries": len(set(r["query"] for r in rows)),
        "num_pairs": len(rows),
        "strategies": {},
    }

    train_rows = rows[:200]
    eval_rows = rows

    print(f"Training on {len(train_rows)} samples")

    print("\nTraining current strategies...")
    strategies_config = {
        "hybrid": {"adapters": [KeywordMatchAdapter()]},
        "binary_reranker": {},
    }
    trained = train_strategies(train_rows, strategies_config)

    all_strategies: list[tuple[str, Any]] = [
        ("flashrank_tiny", FlashRankWrapper("ms-marco-TinyBERT-L-2-v2")),
        ("flashrank_mini", FlashRankWrapper("ms-marco-MiniLM-L-12-v2")),
        ("hybrid", trained["hybrid"]),
        ("binary_reranker", trained["binary_reranker"]),
    ]

    for name, reranker in all_strategies:
        print(f"\nEvaluating {name}...")
        metrics = evaluate_reranker_on_rows(eval_rows, reranker)
        results["strategies"][name] = metrics
        print(
            f"{name}: NDCG@10={metrics['ndcg@10']:.4f}, P@1={metrics['p@1']:.4f}, "
            f"latency={metrics['latency_p50_ms']:.2f}ms"
        )

    print("\n=== Relative Performance ===")
    baseline = results["strategies"].get("hybrid", {}).get("ndcg@10", 0.0)
    if baseline > 0:
        for name, metrics in results["strategies"].items():
            ndcg = metrics.get("ndcg@10", 0.0)
            if ndcg > 0:
                uplift = ((ndcg - baseline) / baseline) * 100
                results["strategies"][name]["ndcg_uplift_vs_hybrid_pct"] = round(uplift, 2)
                print(f"{name}: {uplift:+.1f}% vs hybrid baseline")

    return results


if __name__ == "__main__":
    results = evaluate_benchmark(max_queries=30, top_k_docs=50)

    print("\n=== Full Results ===")
    print(json.dumps(results, indent=2))

    out_path = Path("data/logs/trec_covid_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
