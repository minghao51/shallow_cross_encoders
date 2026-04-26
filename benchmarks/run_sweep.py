"""Benchmark sweep runner that evaluates config variants from YAML files.

Usage:
    uv run benchmarks/run_sweep.py --config benchmarks/configs/sweep_hybrid.yaml
    uv run benchmarks/run_sweep.py --config benchmarks/configs/full_sweep.yaml \
        --output benchmarks/results/full_sweep.json
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reranker.config import (
    apply_settings_override,
    clear_settings_override,
    get_settings,
    load_yaml_config,
    reset_settings_cache,
    settings_from_dict,
)
from reranker.embedder import Embedder
from reranker.eval.metrics import ndcg_at_k
from reranker.heuristics.lsh import LSHAdapter
from reranker.strategies.hybrid import HybridFusionReranker
from reranker.strategies.late_interaction import StaticColBERTReranker
from reranker.utils import read_jsonl


@dataclass
class SweepResult:
    variant_name: str
    configuration: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


def _build_reranker_for_variant(
    config_override: dict[str, Any],
    embedder: Embedder,
) -> HybridFusionReranker | None:
    lsh_cfg = config_override.get("lsh", {})

    adapters: list[Any] = []
    if lsh_cfg.get("enabled", False):
        adapters.append(
            LSHAdapter(
                ngram_size=lsh_cfg.get("ngram_size", 3),
                num_perm=lsh_cfg.get("num_perm", 128),
            )
        )

    return HybridFusionReranker(adapters=adapters, embedder=embedder)


def _evaluate_hybrid(
    reranker: HybridFusionReranker,
    pairs: list[dict[str, Any]],
) -> dict[str, float]:
    queries: list[str] = []
    docs: list[str] = []
    scores: list[float] = []
    for row in pairs:
        queries.append(row["query"])
        docs.append(row["doc"])
        scores.append(float(row.get("score", 0)))

    reranker.fit_pointwise(queries, docs, scores)

    query_groups: dict[str, list[tuple[str, float]]] = {}
    for row in pairs:
        q = row["query"]
        query_groups.setdefault(q, []).append((row["doc"], float(row.get("score", 0))))

    ndcg_scores: list[float] = []
    for query, group in query_groups.items():
        if len(group) < 2:
            continue
        group_docs = [d for d, _ in group]
        relevance = [s for _, s in group]
        results = reranker.rerank(query, group_docs)
        ranked_relevance = []
        doc_to_rel = dict(zip(group_docs, relevance, strict=False))
        for rd in results:
            ranked_relevance.append(doc_to_rel.get(rd.doc, 0.0))
        ndcg_scores.append(ndcg_at_k(ranked_relevance, k=min(10, len(ranked_relevance))))

    return {
        "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "n_queries": float(len(query_groups)),
    }


def _evaluate_colbert(
    config_override: dict[str, Any],
    pairs: list[dict[str, Any]],
    embedder: Embedder,
) -> dict[str, float]:
    li_cfg = config_override.get("late_interaction", {})
    colbert = StaticColBERTReranker(
        embedder=embedder,
        top_k_tokens=li_cfg.get("top_k_tokens", 128),
        use_salience=li_cfg.get("use_salience", True),
        quantization_mode=li_cfg.get("quantization", "none"),
    )

    all_docs: list[str] = list({row["doc"] for row in pairs})
    colbert.fit(all_docs)

    query_groups: dict[str, list[tuple[str, float]]] = {}
    for row in pairs:
        q = row["query"]
        query_groups.setdefault(q, []).append((row["doc"], float(row.get("score", 0))))

    ndcg_scores: list[float] = []
    for query, group in query_groups.items():
        if len(group) < 2:
            continue
        group_docs = [d for d, _ in group]
        relevance = [s for _, s in group]
        results = colbert.rerank(query, group_docs)
        ranked_relevance = []
        doc_to_rel = dict(zip(group_docs, relevance, strict=False))
        for rd in results:
            ranked_relevance.append(doc_to_rel.get(rd.doc, 0.0))
        ndcg_scores.append(ndcg_at_k(ranked_relevance, k=min(10, len(ranked_relevance))))

    return {
        "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "n_queries": float(len(query_groups)),
    }


def _measure_latency(
    reranker: HybridFusionReranker, query: str, docs: list[str], n_runs: int = 5
) -> float:
    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, docs)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return float(np.median(times))


def run_sweep(config_path: str | Path) -> list[SweepResult]:
    yaml_data = load_yaml_config(config_path)
    sweep_name = yaml_data.get("name", "unnamed_sweep")
    variants = yaml_data.get("variants", {})

    if not variants:
        print(f"No variants found in {config_path}")
        return []

    settings = get_settings()
    data_root = Path(settings.paths.raw_data_dir)
    pairs_file = data_root / "pairs.jsonl"

    if not pairs_file.exists():
        print(f"Warning: {pairs_file} not found. Run synthetic data generation first.")
        return []

    pairs = read_jsonl(pairs_file)
    print(f"Sweep: {sweep_name}")
    print(f"Loaded {len(pairs)} pairs, {len(variants)} variants\n")

    results: list[SweepResult] = []

    try:
        for variant_name, variant_config in variants.items():
            print(f"  Running variant: {variant_name}...")
            clear_settings_override()
            reset_settings_cache()
            apply_settings_override(settings_from_dict(variant_config))
            embedder = Embedder()

            reranker = _build_reranker_for_variant(variant_config, embedder)
            metrics: dict[str, float] = {}
            latency = 0.0

            if reranker is not None:
                metrics = _evaluate_hybrid(reranker, pairs)
                sample_query = pairs[0]["query"] if pairs else "test query"
                sample_docs = list({row["doc"] for row in pairs[:20]})
                if sample_docs:
                    latency = _measure_latency(reranker, sample_query, sample_docs)

            li_cfg = variant_config.get("late_interaction", {})
            if li_cfg.get("quantization", "none") != "none":
                colbert_metrics = _evaluate_colbert(variant_config, pairs, embedder)
                metrics["colbert_ndcg@10"] = colbert_metrics["ndcg@10"]
                metrics["colbert_n_queries"] = colbert_metrics["n_queries"]

            result = SweepResult(
                variant_name=variant_name,
                configuration=variant_config,
                metrics=metrics,
                latency_ms=latency,
            )
            results.append(result)
    finally:
        clear_settings_override()

    return results


def print_comparison_table(results: list[SweepResult]) -> None:
    if not results:
        return

    print("\n" + "=" * 80)
    print(f"{'Variant':<35} {'NDCG@10':>10} {'Latency(ms)':>12} {'Queries':>8}")
    print("-" * 80)

    best_ndcg = max(r.metrics.get("ndcg@10", 0) for r in results)
    best_latency = min(r.latency_ms for r in results if r.latency_ms > 0)

    for r in results:
        ndcg = r.metrics.get("ndcg@10", 0.0)
        n_queries = int(r.metrics.get("n_queries", 0))
        ndcg_marker = " *" if abs(ndcg - best_ndcg) < 0.001 else ""
        lat_marker = " *" if abs(r.latency_ms - best_latency) < 0.01 and r.latency_ms > 0 else ""
        print(
            f"{r.variant_name:<35} "
            f"{ndcg:>9.4f}{ndcg_marker} "
            f"{r.latency_ms:>10.2f}{lat_marker} "
            f"{n_queries:>8}"
        )

    print("=" * 80)
    print("  * = best in column")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sweep runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML sweep config (e.g., benchmarks/configs/sweep_hybrid.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )
    args = parser.parse_args()

    results = run_sweep(args.config)
    print_comparison_table(results)

    if args.output:
        import json

        output = [
            {
                "variant": r.variant_name,
                "metrics": r.metrics,
                "latency_ms": r.latency_ms,
                "configuration": r.configuration,
            }
            for r in results
        ]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
