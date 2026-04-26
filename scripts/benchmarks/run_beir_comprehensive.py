"""Run comprehensive BEIR benchmark with hard negatives.

Usage:
    uv run scripts/benchmarks/run_beir_comprehensive.py
    uv run scripts/benchmarks/run_beir_comprehensive.py nfcorpus
    uv run scripts/benchmarks/run_beir_comprehensive.py nfcorpus 30
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reranker.adapters.flashrank_wrapper import FlashRankWrapper
from reranker.data.beir_loader import load_beir_comprehensive
from reranker.data.hard_negative_sampler import prepare_benchmark_data_with_hard_negatives
from reranker.eval.benchmark_utils import evaluate_reranker_on_rows, train_strategies
from reranker.strategies.hybrid import KeywordMatchAdapter


def evaluate_all_strategies(
    dataset_name: str = "nfcorpus",
    num_queries: int = 50,
    docs_per_query: int = 100,
) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f"BEIR Benchmark: {dataset_name}")
    print(f"{'=' * 60}")

    dataset_path = Path(f"data/beir/{dataset_name}")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run download_beir.py first."
        )

    dataset = load_beir_comprehensive(dataset_path)

    rows = prepare_benchmark_data_with_hard_negatives(
        dataset,
        num_queries=num_queries,
        docs_per_query=docs_per_query,
        cache_dir=Path("data/cache/bm25"),
    )

    results: dict[str, Any] = {
        "dataset": dataset_name,
        "num_queries": len(set(r["query_id"] for r in rows)),
        "num_pairs": len(rows),
        "relevant_pairs": sum(1 for r in rows if r["score"] > 0),
        "strategies": {},
    }

    train_size = int(len(rows) * 0.7)
    train_rows = rows[:train_size]
    eval_rows = rows[train_size:]

    print(f"\nTraining on {len(train_rows)} samples, evaluating on {len(eval_rows)}")

    print("\n" + "=" * 60)
    print("Training strategies...")
    print("=" * 60)

    strategies_config = {
        "hybrid": {"adapters": [KeywordMatchAdapter()]},
        "binary_reranker": {},
        "late_interaction": {},
    }
    trained_strategies = train_strategies(train_rows, strategies_config)

    print("\n" + "=" * 60)
    print("Evaluating strategies...")
    print("=" * 60)

    strategies = [
        ("flashrank_tiny", FlashRankWrapper("ms-marco-TinyBERT-L-2-v2")),
        ("flashrank_mini", FlashRankWrapper("ms-marco-MiniLM-L-12-v2")),
        ("hybrid", trained_strategies["hybrid"]),
        ("binary_reranker", trained_strategies["binary_reranker"]),
        ("late_interaction", trained_strategies["late_interaction"]),
    ]

    for name, reranker in strategies:
        print(f"\nEvaluating {name}...")
        try:
            metrics = evaluate_reranker_on_rows(eval_rows, reranker)
            results["strategies"][name] = metrics
            print(
                f"  NDCG@10={metrics['ndcg@10']:.4f}, P@1={metrics['p@1']:.4f}, "
                f"latency={metrics['latency_p50_ms']:.2f}ms"
            )
        except Exception as e:
            print(f"  Error: {e}")
            results["strategies"][name] = {"error": str(e)}

    print("\n" + "=" * 60)
    print("Relative Performance (vs Hybrid baseline)")
    print("=" * 60)

    baseline = results["strategies"].get("hybrid", {}).get("ndcg@10", 0.0)
    if baseline > 0:
        for name, metrics in results["strategies"].items():
            ndcg = metrics.get("ndcg@10", 0.0)
            if ndcg > 0:
                uplift = ((ndcg - baseline) / baseline) * 100
                results["strategies"][name]["ndcg_uplift_vs_hybrid_pct"] = round(
                    uplift, 2
                )
                marker = "+" if uplift > 5 else ("-" if uplift < -5 else "~")
                print(f"  [{marker}] {name}: {uplift:+.1f}% vs hybrid")

    return results


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    num_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    results = evaluate_all_strategies(
        dataset_name=dataset,
        num_queries=num_queries,
        docs_per_query=100,
    )

    out_path = Path(f"data/logs/beir_{dataset}_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 60}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Strategy':<20} {'NDCG@10':<10} {'P@1':<10} {'Latency (ms)':<15}")
    print("-" * 60)
    for name, metrics in results["strategies"].items():
        if "error" not in metrics:
            print(
                f"{name:<20} {metrics['ndcg@10']:<10.4f} {metrics['p@1']:<10.4f} "
                f"{metrics['latency_p50_ms']:<15.2f}"
            )
