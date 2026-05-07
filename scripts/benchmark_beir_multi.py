"""Multi-dataset BEIR benchmark sweep with statistical significance.

Downloads and evaluates strategies across multiple BEIR datasets,
computing bootstrap CIs and pairwise Wilcoxon tests.

Usage:
    uv run python scripts/benchmark_beir_multi.py
    uv run python scripts/benchmark_beir_multi.py --datasets nfcorpus trec-covid
    uv run python scripts/benchmark_beir_multi.py --output results/beir_multi.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import structlog

from reranker.data.beir_loader import load_beir_simple
from reranker.eval.benchmark_utils import evaluate_reranker_on_rows
from reranker.eval.statistics import bootstrap_ci, compare_strategies

logger = structlog.get_logger(__name__)

DATASETS = ["trec-covid", "nfcorpus", "scidocs", "fiqa-qa", "arguana"]

STRATEGY_NAMES = ["hybrid", "late_interaction", "binary_reranker"]


def beir_to_rows(queries: dict, corpus: dict, qrels: dict) -> list[dict]:
    rows = []
    for qid, doc_rel_map in qrels.items():
        query = queries.get(qid, "")
        if not query:
            continue
        for docid, rel in doc_rel_map.items():
            doc_entry = corpus.get(docid, {})
            doc_text = doc_entry.get("text", "")
            if not doc_text:
                continue
            rows.append({"query": query, "doc": doc_text, "score": rel})
    return rows


def beir_to_per_query_rows(queries: dict, corpus: dict, qrels: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for qid, doc_rel_map in qrels.items():
        query = queries.get(qid, "")
        if not query:
            continue
        query_rows = []
        for docid, rel in doc_rel_map.items():
            doc_entry = corpus.get(docid, {})
            doc_text = doc_entry.get("text", "")
            if not doc_text:
                continue
            query_rows.append({"query": query, "doc": doc_text, "score": rel})
        if query_rows:
            grouped[qid] = query_rows
    return grouped


def train_strategy(name: str, rows: list[dict]) -> object:
    from reranker.heuristics.keyword import KeywordMatchAdapter
    from reranker.strategies.binary_reranker import BinaryQuantizedReranker
    from reranker.strategies.hybrid import HybridFusionReranker
    from reranker.strategies.late_interaction import StaticColBERTReranker

    queries = [str(r["query"]) for r in rows]
    docs = [str(r["doc"]) for r in rows]
    labels = [1 if int(r["score"]) > 0 else 0 for r in rows]

    if name == "hybrid":
        reranker = HybridFusionReranker(adapters=[KeywordMatchAdapter()])
        reranker.fit_pointwise(queries, docs, [float(lbl) for lbl in labels], use_regression=True)
        return reranker
    if name == "binary_reranker":
        return BinaryQuantizedReranker().fit(queries, docs, labels)
    if name == "late_interaction":
        unique_docs = list(set(docs))
        reranker = StaticColBERTReranker()
        reranker.fit(unique_docs)
        return reranker

    raise ValueError(f"Unknown strategy: {name}")


def compute_per_query_ndcg(per_query: dict[str, list[dict]], reranker: object) -> list[float]:
    from reranker.eval.metrics import ndcg_at_k

    scores = []
    for _, items in per_query.items():
        query = items[0]["query"]
        docs = [str(it["doc"]) for it in items]
        try:
            ranked = reranker.rerank(query, docs)
        except Exception:
            continue
        doc_to_rel = {str(it["doc"]): int(it["score"]) for it in items}
        relevances = [float(doc_to_rel.get(r.doc, 0)) for r in ranked]
        if any(rel > 0 for rel in relevances):
            scores.append(ndcg_at_k(relevances, 10))
    return scores


def run_dataset(dataset_name: str) -> dict:
    logger.info("Loading dataset", dataset=dataset_name)
    try:
        queries, corpus, qrels = load_beir_simple(dataset_name)
    except Exception as e:
        logger.error("Failed to load dataset", dataset=dataset_name, error=str(e))
        return {"dataset": dataset_name, "error": str(e)}

    rows = beir_to_rows(queries, corpus, qrels)
    per_query = beir_to_per_query_rows(queries, corpus, qrels)
    logger.info(
        "Dataset loaded",
        dataset=dataset_name,
        queries=len(queries),
        corpus=len(corpus),
        qrels=len(qrels),
        eval_rows=len(rows),
    )

    if not rows:
        logger.warn("No eval rows", dataset=dataset_name)
        return {"dataset": dataset_name, "error": "no eval rows"}

    results: dict = {"dataset": dataset_name, "strategies": {}}
    per_query_scores: dict[str, list[float]] = {}

    for strat_name in STRATEGY_NAMES:
        logger.info("Training strategy", strategy=strat_name, dataset=dataset_name)
        t0 = time.perf_counter()
        try:
            reranker = train_strategy(strat_name, rows)
        except Exception as e:
            logger.error("Training failed", strategy=strat_name, error=str(e))
            continue
        train_time = time.perf_counter() - t0
        logger.info("Training complete", strategy=strat_name, time_s=f"{train_time:.2f}")

        metrics = evaluate_reranker_on_rows(rows, reranker)
        pq_scores = compute_per_query_ndcg(per_query, reranker)
        ci = bootstrap_ci(pq_scores) if len(pq_scores) >= 2 else (0.0, 0.0)

        results["strategies"][strat_name] = {
            "ndcg@10": metrics["ndcg@10"],
            "mrr": metrics["mrr"],
            "p@1": metrics["p@1"],
            "latency_p50_ms": metrics["latency_p50_ms"],
            "latency_p99_ms": metrics["latency_p99_ms"],
            "ndcg_ci_95": [round(ci[0], 4), round(ci[1], 4)],
            "queries_evaluated": metrics["queries_evaluated"],
        }
        per_query_scores[strat_name] = pq_scores

    if len(per_query_scores) >= 2:
        results["pairwise_comparisons"] = []
        names = list(per_query_scores.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                comp = compare_strategies(
                    names[i],
                    per_query_scores[names[i]],
                    names[j],
                    per_query_scores[names[j]],
                    metric_name="NDCG@10",
                )
                results["pairwise_comparisons"].append(comp)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-dataset BEIR benchmark sweep")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="BEIR datasets to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/beir-multi-dataset-results.json"),
        help="Output path for results JSON",
    )
    args = parser.parse_args()

    logger.info("Starting multi-dataset BEIR sweep", datasets=args.datasets)
    all_results = []

    for dataset_name in args.datasets:
        result = run_dataset(dataset_name)
        all_results.append(result)

    output = {
        "benchmark": "beir_multi_dataset",
        "strategies": STRATEGY_NAMES,
        "datasets": args.datasets,
        "results": all_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    logger.info("Results saved", path=str(args.output))

    print(f"\n{'=' * 70}")
    print(f"{'Dataset':<15} {'Strategy':<20} {'NDCG@10':>8} {'CI 95%':>18} {'P50 ms':>8}")
    print(f"{'=' * 70}")
    for ds_result in all_results:
        ds_name = ds_result["dataset"]
        if "error" in ds_result:
            print(f"{ds_name:<15} ERROR: {ds_result['error']}")
            continue
        for strat_name, metrics in ds_result.get("strategies", {}).items():
            ci = metrics.get("ndcg_ci_95", [0, 0])
            print(
                f"{ds_name:<15} {strat_name:<20} {metrics['ndcg@10']:>8.4f} "
                f"[{ci[0]:.4f}, {ci[1]:.4f}]{'':>6} {metrics['latency_p50_ms']:>8.2f}"
            )
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
