"""Benchmark FlashRank vs current reranker strategies.

Compares accuracy and latency on the same test dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.eval.metrics import LatencyTracker, ndcg_at_k, precision_at_k, reciprocal_rank
from reranker.eval.runner import _ensure_sample_data
from reranker.strategies.binary_reranker import BinaryQuantizedReranker
from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter
from reranker.strategies.late_interaction import StaticColBERTReranker
from reranker.utils import read_jsonl


def _split_ratios() -> tuple[float, float, float]:
    settings = get_settings()
    return (
        settings.eval.train_ratio,
        settings.eval.validation_ratio,
        settings.eval.test_ratio,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


class FlashRankWrapper:
    """Wrapper for FlashRank to match our reranker protocol."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2", max_length: int = 512):
        try:
            from flashrank import Ranker

            self.ranker = Ranker(model_name=model_name, max_length=max_length)
            self.model_name = model_name
        except ImportError:
            raise ImportError("FlashRank not installed. Install with: uv sync --extra flashrank")

    def rerank(self, query: str, docs: list[str]) -> list[Any]:
        from flashrank import RerankRequest

        passages = [{"id": str(i), "text": doc} for i, doc in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        # Convert to our RankedDoc format
        ranked_docs = []
        for result in results:
            idx = int(result["id"])
            ranked_docs.append(
                {
                    "doc": docs[idx],
                    "score": float(result.get("score", 0.0)),
                    "rank": 0,  # Will be set below
                }
            )

        # Set ranks
        for rank, doc in enumerate(ranked_docs, start=1):
            doc["rank"] = rank

        return ranked_docs


class SentenceTransformerWrapper:
    """Wrapper for SentenceTransformer cross-encoders to match our reranker protocol."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(model_name)
            self.model_name = model_name
        except ImportError:
            raise ImportError(
                "SentenceTransformers not installed. Install with: pip install sentence-transformers"
            )

    def rerank(self, query: str, docs: list[str]) -> list[dict[str, Any]]:
        # Score all query-doc pairs
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)

        # Sort by score descending
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Build ranked results
        ranked_docs = []
        for rank, (idx, score) in enumerate(indexed_scores, start=1):
            ranked_docs.append(
                {
                    "doc": docs[idx],
                    "score": float(score),
                    "rank": rank,
                }
            )

        return ranked_docs


def evaluate_flashrank(
    rows: list[dict[str, Any]],
    model_name: str = "ms-marco-TinyBERT-L-2-v2",
) -> dict[str, float]:
    """Evaluate FlashRank on test data."""
    try:
        wrapper = FlashRankWrapper(model_name=model_name)
    except ImportError as e:
        return {"error": str(e)}

    latency = LatencyTracker()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query"]), []).append(row)

    ndcgs: list[float] = []
    mrrs: list[float] = []
    p1s: list[float] = []

    for query, items in grouped.items():
        docs = [str(item["doc"]) for item in items]

        try:
            with latency.measure():
                ranked = wrapper.rerank(query, docs)
        except Exception:
            # Skip this query on error
            continue

        label_map = {str(item["doc"]): int(item["score"]) for item in items}
        relevances: list[float] = [float(label_map[result["doc"]]) for result in ranked]
        binary = [1 if rel >= 2 else 0 for rel in relevances]
        ndcgs.append(ndcg_at_k(relevances, 10))
        mrrs.append(reciprocal_rank(binary))
        p1s.append(precision_at_k(binary, 1))

    summary = latency.summary()
    return {
        "ndcg@10": round(_mean(ndcgs), 4),
        "mrr": round(_mean(mrrs), 4),
        "p@1": round(_mean(p1s), 4),
        "latency_p50_ms": round(summary["p50"], 4),
        "latency_p99_ms": round(summary["p99"], 4),
        "queries_evaluated": len(grouped),
    }


def evaluate_sentencetransformer(
    rows: list[dict[str, Any]],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict[str, float]:
    """Evaluate SentenceTransformer cross-encoder on test data."""
    try:
        wrapper = SentenceTransformerWrapper(model_name=model_name)
    except ImportError as e:
        return {"error": str(e)}

    latency = LatencyTracker()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query"]), []).append(row)

    ndcgs: list[float] = []
    mrrs: list[float] = []
    p1s: list[float] = []

    for query, items in grouped.items():
        docs = [str(item["doc"]) for item in items]

        try:
            with latency.measure():
                ranked = wrapper.rerank(query, docs)
        except Exception:
            # Skip this query on error
            continue

        label_map = {str(item["doc"]): int(item["score"]) for item in items}
        relevances: list[float] = [float(label_map[result["doc"]]) for result in ranked]
        binary = [1 if rel >= 2 else 0 for rel in relevances]
        ndcgs.append(ndcg_at_k(relevances, 10))
        mrrs.append(reciprocal_rank(binary))
        p1s.append(precision_at_k(binary, 1))

    summary = latency.summary()
    return {
        "ndcg@10": round(_mean(ndcgs), 4),
        "mrr": round(_mean(mrrs), 4),
        "p@1": round(_mean(p1s), 4),
        "latency_p50_ms": round(summary["p50"], 4),
        "latency_p99_ms": round(summary["p99"], 4),
        "queries_evaluated": len(grouped),
    }


def evaluate_current_strategies(
    rows: list[dict[str, Any]],
    data_root: Path,
    model_root: Path,
) -> dict[str, dict[str, float]]:
    """Evaluate all current strategies on test data."""
    _ensure_sample_data(data_root)
    model_root.mkdir(parents=True, exist_ok=True)

    # Train/test split
    train_rows = partition_rows(
        rows,
        key_fn=lambda row: str(row["query"]),
        split="train",
        ratios=_split_ratios(),
    )
    eval_rows = partition_rows(
        rows,
        key_fn=lambda row: str(row["query"]),
        split="test",
        ratios=_split_ratios(),
    )
    if not eval_rows:
        eval_rows = rows

    results = {}

    # Hybrid Fusion Reranker
    print("Training HybridFusionReranker...")
    binary_labels = [1 if int(row["score"]) >= 2 else 0 for row in train_rows]
    hybrid = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit_pointwise(
        queries=[str(row["query"]) for row in train_rows],
        docs=[str(row["doc"]) for row in train_rows],
        scores=[float(l) for l in binary_labels],
    )
    results["hybrid"] = _evaluate_reranker(hybrid, eval_rows)

    # Distilled Pairwise Reranker
    print("Training DistilledPairwiseRanker...")
    pref_rows = read_jsonl(data_root / "preferences.jsonl")
    pref_train = partition_rows(
        pref_rows,
        key_fn=lambda row: str(row["query"]),
        split="train",
        ratios=_split_ratios(),
    )
    if pref_train:
        pref_labels = [1 if row["preferred"] == "A" else 0 for row in pref_train]
        distilled = DistilledPairwiseRanker().fit(
            queries=[str(row["query"]) for row in pref_train],
            doc_as=[str(row["doc_a"]) for row in pref_train],
            doc_bs=[str(row["doc_b"]) for row in pref_train],
            labels=pref_labels,
        )
        results["distilled"] = _evaluate_reranker(distilled, eval_rows)

    # Late Interaction Reranker
    print("Training StaticColBERTReranker...")
    unique_docs = list({str(row["doc"]) for row in train_rows})
    late_interaction = StaticColBERTReranker()
    late_interaction.fit(unique_docs)
    results["late_interaction"] = _evaluate_reranker(late_interaction, eval_rows)

    # Binary Reranker
    print("Training BinaryQuantizedReranker...")
    binary = BinaryQuantizedReranker().fit(
        queries=[str(row["query"]) for row in train_rows],
        docs=[str(row["doc"]) for row in train_rows],
        labels=binary_labels,
    )
    results["binary_reranker"] = _evaluate_reranker(binary, eval_rows)

    return results


def _evaluate_reranker(
    reranker: Any,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Evaluate a reranker on test data."""
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

        label_map = {str(item["doc"]): int(item["score"]) for item in items}
        relevances: list[float] = [float(r.score) for r in ranked]
        binary = [1 if rel >= 2 else 0 for rel in relevances]
        ndcgs.append(ndcg_at_k(relevances, 10))
        mrrs.append(reciprocal_rank(binary))
        p1s.append(precision_at_k(binary, 1))

    summary = latency.summary()
    return {
        "ndcg@10": round(_mean(ndcgs), 4),
        "mrr": round(_mean(mrrs), 4),
        "p@1": round(_mean(p1s), 4),
        "latency_p50_ms": round(summary["p50"], 4),
        "latency_p99_ms": round(summary["p99"], 4),
        "queries_evaluated": len(grouped),
    }


def benchmark_all(
    data_root: Path | None = None,
    model_root: Path | None = None,
) -> dict[str, Any]:
    """Run full benchmark comparing FlashRank with current strategies."""
    settings = get_settings()
    data_root = data_root or settings.paths.processed_data_dir
    model_root = model_root or settings.paths.model_dir

    # Ensure we have test data
    _ensure_sample_data(data_root)

    # Load test data
    rows = read_jsonl(data_root / "pairs.jsonl")
    test_rows = partition_rows(
        rows,
        key_fn=lambda row: str(row["query"]),
        split="test",
        ratios=_split_ratios(),
    )
    if not test_rows:
        test_rows = rows

    print(
        f"Testing on {len(test_rows)} samples ({len(set(r['query'] for r in test_rows))} unique queries)"
    )

    results = {
        "dataset_info": {
            "total_samples": len(test_rows),
            "unique_queries": len(set(r["query"] for r in test_rows)),
        },
        "strategies": {},
    }

    # Evaluate FlashRank (TinyBERT)
    print("\n=== Evaluating FlashRank (TinyBERT) ===")
    flashrank_tiny = evaluate_flashrank(test_rows, model_name="ms-marco-TinyBERT-L-2-v2")
    results["strategies"]["flashrank_tiny"] = flashrank_tiny
    print(
        f"FlashRank TinyBERT: NDCG@10={flashrank_tiny.get('ndcg@10', 'N/A')}, P@1={flashrank_tiny.get('p@1', 'N/A')}"
    )

    # Evaluate FlashRank (MiniLM - if available)
    print("\n=== Evaluating FlashRank (MiniLM) ===")
    flashrank_mini = evaluate_flashrank(test_rows, model_name="ms-marco-MiniLM-L-12-v2")
    results["strategies"]["flashrank_mini"] = flashrank_mini
    print(
        f"FlashRank MiniLM: NDCG@10={flashrank_mini.get('ndcg@10', 'N/A')}, P@1={flashrank_mini.get('p@1', 'N/A')}"
    )

    # Evaluate SentenceTransformers (MiniLM-L-6)
    print("\n=== Evaluating SentenceTransformers (MiniLM-L-6) ===")
    st_mini = evaluate_sentencetransformer(
        test_rows, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    results["strategies"]["st_mini"] = st_mini
    print(
        f"ST MiniLM-L-6: NDCG@10={st_mini.get('ndcg@10', 'N/A')}, P@1={st_mini.get('p@1', 'N/A')}"
    )

    # Evaluate SentenceTransformers (TinyBERT - if available)
    print("\n=== Evaluating SentenceTransformers (TinyBERT) ===")
    st_tiny = evaluate_sentencetransformer(
        test_rows, model_name="cross-encoder/ms-marco-TinyBERT-L-2"
    )
    results["strategies"]["st_tiny"] = st_tiny
    print(f"ST TinyBERT: NDCG@10={st_tiny.get('ndcg@10', 'N/A')}, P@1={st_tiny.get('p@1', 'N/A')}")

    # Evaluate current strategies
    print("\n=== Evaluating Current Strategies ===")
    current_results = evaluate_current_strategies(test_rows, data_root, model_root)
    for name, metrics in current_results.items():
        results["strategies"][name] = metrics
        print(f"{name}: NDCG@10={metrics.get('ndcg@10', 'N/A')}, P@1={metrics.get('p@1', 'N/A')}")

    # Calculate improvements
    print("\n=== Relative Performance ===")
    baseline_ndcg = current_results.get("hybrid", {}).get("ndcg@10", 0.0)
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
