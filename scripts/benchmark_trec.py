"""Run benchmark on TREC COVID dataset (small, real data).

Downloads TREC COVID data directly without BEIR dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from reranker.eval.metrics import LatencyTracker, ndcg_at_k, precision_at_k, reciprocal_rank
from reranker.strategies.binary_reranker import BinaryQuantizedReranker
from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter


def download_trec_covid(save_path: Path = Path("data/trec-covid")) -> dict[str, Any]:
    """Download TREC COVID dataset (small subset)."""
    save_path.mkdir(parents=True, exist_ok=True)

    # Use pre-processed subset from GitHub
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
    """Prepare TREC COVID data for benchmark."""
    from rank_bm25 import BM25Okapi

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]

    # Prepare corpus
    corpus_texts = [doc.get("text", doc.get("title", "")) for doc in corpus.values()]
    corpus_ids = list(corpus.keys())
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]

    # Build BM25
    bm25 = BM25Okapi(tokenized_corpus)

    rows = []
    query_ids = list(queries.keys())[:max_queries]

    for query_id in query_ids:
        query = queries[query_id]

        # Get top-k docs
        tokenized_query = query.lower().split()
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-top_k_docs:][::-1]

        relevant_docs = qrels.get(query_id, {})

        for doc_idx in top_indices:
            doc_id = corpus_ids[doc_idx]
            doc_text = corpus[doc_id].get("text", corpus[doc_id].get("title", ""))

            # TREC COVID uses binary relevance (0 or 1)
            relevance = 1 if relevant_docs.get(doc_id, 0) > 0 else 0

            rows.append(
                {
                    "query": query,
                    "doc": doc_text[:1000],  # Truncate for speed
                    "score": relevance,
                    "query_id": query_id,
                    "doc_id": doc_id,
                }
            )

    print(f"Prepared {len(rows)} pairs from {len(query_ids)} queries")
    return rows


class FlashRankWrapper:
    """Wrapper for FlashRank."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        from flashrank import Ranker

        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, docs: list[str]) -> list[dict[str, Any]]:
        from flashrank import RerankRequest

        passages = [{"id": str(i), "text": doc} for i, doc in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        ranked = []
        for result in results:
            idx = int(result["id"])
            ranked.append(
                {
                    "doc": docs[idx],
                    "score": float(result.get("score", 0.0)),
                    "rank": 0,
                }
            )

        for rank, doc in enumerate(ranked, start=1):
            doc["rank"] = rank

        return ranked


def evaluate_benchmark(
    max_queries: int = 30,
    top_k_docs: int = 50,
) -> dict[str, Any]:
    """Run benchmark on TREC COVID."""
    print("\n=== Benchmark on TREC COVID ===")

    # Download data
    dataset = download_trec_covid()

    # Prepare rows
    rows = prepare_for_benchmark(dataset, top_k_docs=top_k_docs, max_queries=max_queries)

    results = {
        "dataset": "trec-covid",
        "num_queries": len(set(r["query"] for r in rows)),
        "num_pairs": len(rows),
        "strategies": {},
    }

    # Train on subset
    train_rows = rows[:200]
    eval_rows = rows

    print(f"Training on {len(train_rows)} samples")

    # Train strategies
    print("\nTraining current strategies...")
    binary_labels = [1 if int(row["score"]) > 0 else 0 for row in train_rows]

    hybrid = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit(
        queries=[str(row["query"]) for row in train_rows],
        docs=[str(row["doc"]) for row in train_rows],
        labels=binary_labels,
    )

    binary = BinaryQuantizedReranker().fit(
        queries=[str(row["query"]) for row in train_rows],
        docs=[str(row["doc"]) for row in train_rows],
        labels=binary_labels,
    )

    # Evaluate
    for name, reranker_cls, model_name in [
        ("flashrank_tiny", FlashRankWrapper, "ms-marco-TinyBERT-L-2-v2"),
        ("flashrank_mini", FlashRankWrapper, "ms-marco-MiniLM-L-12-v2"),
        ("hybrid", lambda: hybrid, None),
        ("binary_reranker", lambda: binary, None),
    ]:
        print(f"\nEvaluating {name}...")

        if model_name:
            reranker = reranker_cls(model_name)
        else:
            reranker = reranker_cls()

        metrics = _evaluate(reranker, eval_rows)
        results["strategies"][name] = metrics

        print(
            f"{name}: NDCG@10={metrics['ndcg@10']:.4f}, P@1={metrics['p@1']:.4f}, "
            f"latency={metrics['latency_p50_ms']:.2f}ms"
        )

    # Compute relative performance
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


def _evaluate(reranker: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    """Evaluate reranker."""
    latency = LatencyTracker()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query"]), []).append(row)

    ndcgs: list[float] = []
    mrrs: list[float] = []
    p1s: list[float] = []

    for query, items in grouped.items():
        docs = [str(item["doc"]) for item in items]

        with latency.measure():
            ranked = reranker.rerank(query, docs)

        doc_to_relevance = {str(item["doc"]): int(item["score"]) for item in items}
        relevances = [float(doc_to_relevance.get(r["doc"], 0)) for r in ranked]
        binary = [1 if rel > 0 else 0 for rel in relevances]

        ndcgs.append(ndcg_at_k(relevances, 10))
        mrrs.append(reciprocal_rank(binary))
        p1s.append(precision_at_k(binary, 1))

    summary = latency.summary()

    return {
        "ndcg@10": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "mrr": round(float(np.mean(mrrs)) if mrrs else 0.0, 4),
        "p@1": round(float(np.mean(p1s)) if p1s else 0.0, 4),
        "latency_p50_ms": round(float(summary["p50"]), 4),
        "latency_p99_ms": round(float(summary["p99"]), 4),
        "queries_evaluated": len(grouped),
    }


if __name__ == "__main__":
    results = evaluate_benchmark(max_queries=30, top_k_docs=50)

    print("\n=== Full Results ===")
    print(json.dumps(results, indent=2))

    # Save
    out_path = Path("data/logs/trec_covid_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
