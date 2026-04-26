"""Benchmark FlashRank vs current reranker strategies.

Usage:
    uv run scripts/benchmarks/run_flashrank.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reranker.adapters.flashrank_wrapper import FlashRankWrapper
from reranker.adapters.sentence_transformer_wrapper import SentenceTransformerWrapper
from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.eval.benchmark_utils import evaluate_reranker_on_rows, train_strategies
from reranker.strategies.hybrid import KeywordMatchAdapter
from reranker.utils import read_jsonl


def _split_ratios() -> tuple[float, float, float]:
    settings = get_settings()
    return (
        settings.eval.train_ratio,
        settings.eval.validation_ratio,
        settings.eval.test_ratio,
    )


def benchmark_all(
    data_root: Path | None = None,
    model_root: Path | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    data_root = data_root or settings.paths.processed_data_dir
    model_root = model_root or settings.paths.model_dir

    from reranker.eval.runner import _ensure_sample_data

    _ensure_sample_data(data_root)

    rows = read_jsonl(data_root / "pairs.jsonl")
    test_rows = partition_rows(
        rows,
        key_fn=lambda row: str(row["query"]),
        split="test",
        ratios=_split_ratios(),
    )
    if not test_rows:
        test_rows = rows

    unique_queries = len(set(row["query"] for row in test_rows))
    print(f"Testing on {len(test_rows)} samples ({unique_queries} unique queries)")

    results: dict[str, Any] = {
        "dataset_info": {
            "total_samples": len(test_rows),
            "unique_queries": len(set(r["query"] for r in test_rows)),
        },
        "strategies": {},
    }

    print("\n=== Evaluating FlashRank (TinyBERT) ===")
    flashrank_tiny = evaluate_reranker_on_rows(
        test_rows, FlashRankWrapper("ms-marco-TinyBERT-L-2-v2")
    )
    results["strategies"]["flashrank_tiny"] = flashrank_tiny
    print(
        f"FlashRank TinyBERT: "
        f"NDCG@10={flashrank_tiny.get('ndcg@10', 'N/A')}, "
        f"P@1={flashrank_tiny.get('p@1', 'N/A')}"
    )

    print("\n=== Evaluating FlashRank (MiniLM) ===")
    flashrank_mini = evaluate_reranker_on_rows(
        test_rows, FlashRankWrapper("ms-marco-MiniLM-L-12-v2")
    )
    results["strategies"]["flashrank_mini"] = flashrank_mini
    print(
        f"FlashRank MiniLM: "
        f"NDCG@10={flashrank_mini.get('ndcg@10', 'N/A')}, "
        f"P@1={flashrank_mini.get('p@1', 'N/A')}"
    )

    print("\n=== Evaluating SentenceTransformers (MiniLM-L-6) ===")
    st_mini = evaluate_reranker_on_rows(
        test_rows, SentenceTransformerWrapper("cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    results["strategies"]["st_mini"] = st_mini
    print(
        f"ST MiniLM-L-6: NDCG@10={st_mini.get('ndcg@10', 'N/A')}, "
        f"P@1={st_mini.get('p@1', 'N/A')}"
    )

    print("\n=== Evaluating SentenceTransformers (TinyBERT) ===")
    st_tiny = evaluate_reranker_on_rows(
        test_rows, SentenceTransformerWrapper("cross-encoder/ms-marco-TinyBERT-L-2")
    )
    results["strategies"]["st_tiny"] = st_tiny
    print(
        f"ST TinyBERT: NDCG@10={st_tiny.get('ndcg@10', 'N/A')}, "
        f"P@1={st_tiny.get('p@1', 'N/A')}"
    )

    print("\n=== Evaluating Current Strategies ===")
    strategies_config = {
        "hybrid": {"adapters": [KeywordMatchAdapter()]},
        "binary_reranker": {},
        "late_interaction": {},
    }
    trained = train_strategies(test_rows, strategies_config)
    for name, reranker in trained.items():
        metrics = evaluate_reranker_on_rows(test_rows, reranker)
        results["strategies"][name] = metrics
        print(f"{name}: NDCG@10={metrics.get('ndcg@10', 'N/A')}, P@1={metrics.get('p@1', 'N/A')}")

    print("\n=== Relative Performance ===")
    baseline_ndcg = results["strategies"].get("hybrid", {}).get("ndcg@10", 0.0)
    if baseline_ndcg > 0:
        for name, metrics in results["strategies"].items():
            ndcg = metrics.get("ndcg@10", 0.0)
            if ndcg > 0:
                uplift = ((ndcg - baseline_ndcg) / baseline_ndcg) * 100
                results["strategies"][name]["ndcg_uplift_vs_hybrid_pct"] = round(uplift, 2)
                print(f"{name}: {uplift:+.1f}% vs hybrid baseline")

    return results


if __name__ == "__main__":
    import json

    results = benchmark_all()
    print("\n=== Full Results ===")
    print(json.dumps(results, indent=2))
