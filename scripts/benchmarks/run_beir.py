"""Run benchmark on real BEIR dataset.

Usage:
    uv run scripts/benchmarks/run_beir.py
    uv run scripts/benchmarks/run_beir.py scifact
    uv run scripts/benchmarks/run_beir.py nfcorpus
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reranker.adapters.flashrank_wrapper import FlashRankWrapper
from reranker.eval.benchmark_utils import evaluate_reranker_on_rows, train_strategies
from reranker.strategies.hybrid import KeywordMatchAdapter

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
    if not BEIR_AVAILABLE:
        raise RuntimeError(
            "BEIR not available. Install: pip install beir (requires torch, pyyaml, etc)"
        )

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    save_path = save_path or Path("data/beir") / dataset_name

    print(f"Downloading {dataset_name} from {url}...")
    data_path = util.download_and_unzip(url, save_path)

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
    from rank_bm25 import BM25Okapi

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]

    corpus_texts = [doc["text"] for doc in corpus.values()]
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
            doc_text = corpus[doc_id]["text"]

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


def evaluate_on_beir(
    dataset_name: str = "scifact",
    max_queries: int = 50,
    top_k_docs: int = 50,
) -> dict[str, Any]:
    print(f"\n=== Benchmark on BEIR {dataset_name} ===")

    dataset = download_beir_dataset(dataset_name)
    rows = prepare_beir_for_benchmark(dataset, top_k_docs=top_k_docs, max_queries=max_queries)

    results = {
        "dataset": dataset_name,
        "num_queries": len(set(r["query"] for r in rows)),
        "num_pairs": len(rows),
        "strategies": {},
    }

    train_rows = rows[:200]
    eval_rows = rows

    print(f"Training on {len(train_rows)} samples, evaluating on {len(eval_rows)}")

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
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "scifact"

    results = evaluate_on_beir(
        dataset_name=dataset,
        max_queries=50,
        top_k_docs=50,
    )

    print("\n=== Full Results ===")
    print(json.dumps(results, indent=2))

    out_path = Path(f"data/logs/beir_benchmark_{dataset}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
