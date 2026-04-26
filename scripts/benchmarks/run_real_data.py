"""Run benchmark on real data using MS-MARCO dev subset.

Usage:
    uv run scripts/benchmarks/run_real_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reranker.adapters.flashrank_wrapper import FlashRankWrapper
from reranker.eval.benchmark_utils import evaluate_reranker_on_rows, train_strategies
from reranker.strategies.hybrid import KeywordMatchAdapter


def load_msmarco_sample() -> dict[str, Any]:
    sample_path = Path("data/benchmarks/msmarco_dev_sample.json")
    if not sample_path.exists():
        raise FileNotFoundError(
            f"MS-MARCO sample not found at {sample_path}. "
            "Run scripts/materialize_demo_data.py to create it."
        )

    with open(sample_path) as f:
        return json.load(f)


def prepare_benchmark_data() -> list[dict[str, Any]]:
    sample_data = load_msmarco_sample()
    rows = []

    for query in sample_data["queries"]:
        relevant_docs = sample_data["qrels"].get(query, {})

        for passage in sample_data["passages"]:
            doc_id = passage["id"]
            relevance = relevant_docs.get(doc_id, 0)

            rows.append(
                {
                    "query": query,
                    "doc": passage["text"],
                    "score": relevance,
                    "query_id": query,
                    "doc_id": doc_id,
                }
            )

    print(
        f"Prepared {len(rows)} query-doc pairs from {len(sample_data['queries'])} queries"
    )
    return rows


def evaluate_benchmark() -> dict[str, Any]:
    print("\n=== Benchmark on MS-MARCO Style Real Data ===")

    rows = prepare_benchmark_data()

    results = {
        "dataset": "msmarco_style",
        "num_queries": len(set(r["query"] for r in rows)),
        "num_pairs": len(rows),
        "strategies": {},
    }

    split_idx = min(100, max(len(rows) - 1, 1))
    train_rows = rows[:split_idx]
    eval_rows = rows[split_idx:]
    if not eval_rows:
        eval_rows = rows[-1:]

    print(f"Training on {len(train_rows)} samples")

    print("\nTraining current strategies...")
    strategies_config = {
        "hybrid": {"adapters": [KeywordMatchAdapter()]},
        "binary_reranker": {},
    }
    trained_strategies = train_strategies(train_rows, strategies_config)

    print("\nEvaluating strategies...")

    print("Evaluating flashrank_tiny...")
    results["strategies"]["flashrank_tiny"] = evaluate_reranker_on_rows(
        eval_rows, FlashRankWrapper("ms-marco-TinyBERT-L-2-v2")
    )
    print(
        f"flashrank_tiny: NDCG@10={results['strategies']['flashrank_tiny']['ndcg@10']:.4f}"
    )

    print("Evaluating flashrank_mini...")
    results["strategies"]["flashrank_mini"] = evaluate_reranker_on_rows(
        eval_rows, FlashRankWrapper("ms-marco-MiniLM-L-12-v2")
    )
    print(
        f"flashrank_mini: NDCG@10={results['strategies']['flashrank_mini']['ndcg@10']:.4f}"
    )

    print("Evaluating hybrid...")
    results["strategies"]["hybrid"] = evaluate_reranker_on_rows(
        eval_rows, trained_strategies["hybrid"]
    )
    print(f"hybrid: NDCG@10={results['strategies']['hybrid']['ndcg@10']:.4f}")

    print("Evaluating binary_reranker...")
    results["strategies"]["binary_reranker"] = evaluate_reranker_on_rows(
        eval_rows, trained_strategies["binary_reranker"]
    )
    print(
        f"binary_reranker: NDCG@10={results['strategies']['binary_reranker']['ndcg@10']:.4f}"
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
    results = evaluate_benchmark()

    print("\n=== Full Results ===")
    print(json.dumps(results, indent=2))

    out_path = Path("data/logs/msmarco_style_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
