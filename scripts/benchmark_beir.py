"""Run benchmark on real BEIR dataset.

Downloads a small BEIR dataset and compares FlashRank with current strategies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reranker.eval.metrics import LatencyTracker, ndcg_at_k, precision_at_k, reciprocal_rank

# Try importing beir, handle gracefully
try:
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    BEIR_AVAILABLE = True
except ImportError:
    BEIR_AVAILABLE = False


def download_beir_dataset(
    dataset_name: str = "scifact",
    save_path: Path | None = None,
) -> dict[str, Any]:
    """Download BEIR dataset.

    Args:
        dataset_name: One of: scifact, nfcorpus, trec-covid, fiqa
        save_path: Where to save dataset

    Returns:
        Dict with corpus, queries, and qrels
    """
    if not BEIR_AVAILABLE:
        raise RuntimeError(
            "BEIR not available. Install: pip install beir (requires torch, pyyaml, etc)"
        )

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    save_path = save_path or Path("data/beir") / dataset_name

    print(f"Downloading {dataset_name} from {url}...")
    data_path = util.download_and_unzip(url, save_path)

    # Load dataset
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "data_path": data_path,
    }


def prepare_beir_for_benchmark(
    dataset: dict[str, Any],
    top_k_docs: int = 50,
    max_queries: int = 50,
) -> list[dict[str, Any]]:
    """Prepare BEIR data for benchmark format.

    For each query, get top-k docs from BM25, create query-doc pairs.
    """
    from rank_bm25 import BM25Okapi

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]

    # Prepare corpus for BM25
    corpus_texts = [doc["text"] for doc in corpus.values()]
    corpus_ids = list(corpus.keys())
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # For each query, get top-k docs
    rows = []
    query_ids = list(queries.keys())[:max_queries]

    for query_id in query_ids:
        query = queries[query_id]

        # Get top-k docs via BM25
        tokenized_query = query.lower().split()
        doc_scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-top_k_docs:][::-1]

        # Get relevance judgments
        relevant_docs = qrels.get(query_id, {})

        for doc_idx in top_indices:
            doc_id = corpus_ids[doc_idx]
            doc_text = corpus[doc_id]["text"]

            # Get relevance score (BEIR uses graded relevance: 0, 1, 2, 3)
            relevance = relevant_docs.get(doc_id, 0)

            rows.append(
                {
                    "query": query,
                    "doc": doc_text,
                    "score": relevance,
                    "query_id": query_id,
                    "doc_id": doc_id,
                }
            )

    print(f"Prepared {len(rows)} query-doc pairs from {len(query_ids)} queries")
    return rows


class FlashRankWrapper:
    """Wrapper for FlashRank to match our reranker protocol."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2", max_length: int = 512):
        try:
            from flashrank import Ranker

            self.ranker = Ranker(model_name=model_name, max_length=max_length)
        except ImportError as err:
            raise ImportError(
                "FlashRank not installed. Install with: uv sync --extra flashrank"
            ) from err

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


def evaluate_on_beir(
    dataset_name: str = "scifact",
    max_queries: int = 50,
    top_k_docs: int = 50,
) -> dict[str, Any]:
    """Run full benchmark on BEIR dataset."""
    print(f"\n=== Benchmark on BEIR {dataset_name} ===")

    # Download dataset
    dataset = download_beir_dataset(dataset_name)

    # Prepare data
    rows = prepare_beir_for_benchmark(dataset, top_k_docs=top_k_docs, max_queries=max_queries)

    # Import current strategies
    from reranker.strategies.binary_reranker import BinaryQuantizedReranker
    from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter

    results = {
        "dataset": dataset_name,
        "num_queries": len(set(r["query"] for r in rows)),
        "num_pairs": len(rows),
        "strategies": {},
    }

    # Import and train on first batch only (for speed)
    train_rows = rows[:200]
    eval_rows = rows

    print(f"Training on {len(train_rows)} samples, evaluating on {len(eval_rows)}")

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

    # Evaluate each strategy
    for name, reranker in [
        ("flashrank_tiny", FlashRankWrapper(model_name="ms-marco-TinyBERT-L-2-v2")),
        ("flashrank_mini", FlashRankWrapper(model_name="ms-marco-MiniLM-L-12-v2")),
        ("hybrid", hybrid),
        ("binary_reranker", binary),
    ]:
        print(f"\nEvaluating {name}...")

        metrics = _evaluate_on_beir_rows(reranker, eval_rows)
        results["strategies"][name] = metrics

        print(
            f"{name}: NDCG@10={metrics['ndcg@10']:.4f}, P@1={metrics['p@1']:.4f}, "
            f"latency={metrics['latency_p50_ms']:.2f}ms"
        )

    # Calculate improvements
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


def _evaluate_on_beir_rows(
    reranker: Any,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Evaluate reranker on BEIR rows."""
    latency = LatencyTracker()

    # Group by query
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

        # Get relevance scores
        doc_to_relevance = {str(item["doc"]): int(item["score"]) for item in items}
        relevances = [float(doc_to_relevance.get(r["doc"], 0)) for r in ranked]
        binary = [1 if rel > 0 else 0 for rel in relevances]

        ndcgs.append(ndcg_at_k(relevances, 10))
        mrrs.append(reciprocal_rank(binary))
        p1s.append(precision_at_k(binary, 1))

    summary = latency.summary()

    return {
        "ndcg@10": round(np.mean(ndcgs), 4) if ndcgs else 0.0,
        "mrr": round(np.mean(mrrs), 4) if mrrs else 0.0,
        "p@1": round(np.mean(p1s), 4) if p1s else 0.0,
        "latency_p50_ms": round(summary["p50"], 4),
        "latency_p99_ms": round(summary["p99"], 4),
        "queries_evaluated": len(grouped),
    }


if __name__ == "__main__":
    import sys

    # Default: scifact (small, scientific fact verification)
    dataset = sys.argv[1] if len(sys.argv) > 1 else "scifact"

    results = evaluate_on_beir(
        dataset_name=dataset,
        max_queries=50,
        top_k_docs=50,
    )

    print("\n=== Full Results ===")
    print(json.dumps(results, indent=2))

    # Save results
    out_path = Path(f"data/logs/beir_benchmark_{dataset}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
