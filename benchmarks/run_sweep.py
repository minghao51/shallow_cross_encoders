"""Benchmark sweep runner that evaluates config variants from YAML files.

Supports hybrid, colbert, cascade, binary, pipeline, and distilled strategy sweeps.

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
import structlog

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
from reranker.heuristics.keyword import KeywordMatchAdapter
from reranker.heuristics.lsh import LSHAdapter
from reranker.strategies.binary_reranker import BinaryQuantizedReranker
from reranker.strategies.cascade import CascadeConfig, CascadeReranker, ConfidenceMetric
from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.strategies.hybrid import HybridFusionReranker
from reranker.strategies.late_interaction import StaticColBERTReranker
from reranker.strategies.pipeline import PipelineReranker
from reranker.utils import read_jsonl

logger = structlog.get_logger(__name__)


@dataclass
class SweepResult:
    variant_name: str
    configuration: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


_ALLOWED_TOP_LEVEL_KEYS = {
    "embedder",
    "hybrid",
    "late_interaction",
    "lsh",
    "binary",
    "pipeline",
    "distilled",
    "cascade",
}
_ALLOWED_CASCADE_KEYS = {
    "confidence_threshold",
    "confidence_metric",
    "fallback_strategy",
    "primary_strategy",
}
_ALLOWED_BINARY_KEYS = {
    "hamming_top_k",
    "bilinear_top_k",
    "quantization",
    "use_binary_scoring",
}
_ALLOWED_PIPELINE_KEYS = {"stages", "top_ks"}
_ALLOWED_DISTILLED_KEYS = {"loss_type", "tournament_size", "C", "max_iter"}
_ALLOWED_LATE_INTERACTION_KEYS = {"top_k_tokens", "use_salience", "quantization"}
_CASCADE_METRIC_ALIASES = {
    "TOP_MARGIN": ConfidenceMetric.TOP_MARGIN,
    "top_margin": ConfidenceMetric.TOP_MARGIN,
    "MAX_SCORE": ConfidenceMetric.MAX_SCORE,
    "max_score": ConfidenceMetric.MAX_SCORE,
    "MAX_PROB": ConfidenceMetric.MAX_SCORE,
    "max_prob": ConfidenceMetric.MAX_SCORE,
    "SCORE_VARIANCE": ConfidenceMetric.SCORE_VARIANCE,
    "score_variance": ConfidenceMetric.SCORE_VARIANCE,
    "NORMALIZED_MAX": ConfidenceMetric.NORMALIZED_MAX,
    "normalized_max": ConfidenceMetric.NORMALIZED_MAX,
}
_DISTILLED_LOSS_ALIASES = {
    "pairwise": "pairwise",
    "logistic": "pairwise",
    "hinge": "pairwise",
    "squared_hinge": "pairwise",
    "listwise": "listwise",
    "lambdaloss": "lambdaloss",
}


def _detect_variant_type(config_override: dict[str, Any]) -> str:
    for key in ("cascade", "binary", "pipeline", "distilled", "hybrid", "late_interaction"):
        if key in config_override:
            return key
    return "hybrid"


def _evaluate_ranking_reranker(
    reranker: Any,
    pairs: list[dict[str, Any]],
    fit_fn: str | None = None,
) -> dict[str, float]:
    query_groups: dict[str, list[tuple[str, float]]] = {}
    for row in pairs:
        q = row["query"]
        query_groups.setdefault(q, []).append((row["doc"], float(row.get("score", 0))))

    ndcg_scores: list[float] = []
    latency_samples: list[float] = []
    for query, group in query_groups.items():
        if len(group) < 2:
            continue
        group_docs = [d for d, _ in group]
        relevance = [s for _, s in group]

        start = time.perf_counter()
        results = reranker.rerank(query, group_docs)
        latency_samples.append((time.perf_counter() - start) * 1000)

        ranked_relevance = []
        doc_to_rel = dict(zip(group_docs, relevance, strict=False))
        for rd in results:
            ranked_relevance.append(doc_to_rel.get(rd.doc, 0.0))
        ndcg_scores.append(ndcg_at_k(ranked_relevance, k=min(10, len(ranked_relevance))))

    return {
        "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "n_queries": float(len(ndcg_scores)),
        "latency_mean_ms": float(np.mean(latency_samples)) if latency_samples else 0.0,
    }


def _build_hybrid_for_variant(
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
    return _evaluate_ranking_reranker(reranker, pairs)


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
    return _evaluate_ranking_reranker(colbert, pairs)


def _build_cascade_for_variant(
    config_override: dict[str, Any],
    embedder: Embedder,
) -> CascadeReranker | None:
    from reranker.strategies.flashrank_ensemble import FlashRankEnsemble

    cascade_cfg = config_override.get("cascade", {})
    threshold = float(cascade_cfg.get("confidence_threshold", 0.6))
    metric_str = str(cascade_cfg.get("confidence_metric", "TOP_MARGIN"))
    metric = _CASCADE_METRIC_ALIASES.get(metric_str)
    if metric is None:
        valid = ", ".join(sorted(_CASCADE_METRIC_ALIASES.keys()))
        raise ValueError(
            f"Unsupported cascade confidence_metric '{metric_str}'. Valid values: {valid}"
        )

    primary = HybridFusionReranker(
        adapters=[KeywordMatchAdapter()],
        embedder=embedder,
    )
    fallback = FlashRankEnsemble(models=["ms-marco-TinyBERT-L-2-v2"])

    return CascadeReranker(
        primary=primary,
        fallback=fallback,
        config=CascadeConfig(confidence_threshold=threshold, confidence_metric=metric),
    )


def _evaluate_cascade(
    reranker: CascadeReranker,
    pairs: list[dict[str, Any]],
) -> dict[str, float]:
    queries: list[str] = []
    docs: list[str] = []
    scores: list[float] = []
    for row in pairs:
        queries.append(row["query"])
        docs.append(row["doc"])
        scores.append(float(row.get("score", 0)))

    reranker.primary.fit_pointwise(queries, docs, scores)  # type: ignore[attr-defined]
    base = _evaluate_ranking_reranker(reranker, pairs)
    try:
        stats = reranker.get_stats()
        base["fallback_rate"] = stats.get("fallback_rate", 0.0)
        base["avg_confidence"] = stats.get("avg_confidence", 0.0)
    except Exception:
        pass
    return base


def _build_binary_for_variant(
    config_override: dict[str, Any],
    embedder: Embedder,
) -> BinaryQuantizedReranker | None:
    binary_cfg = config_override.get("binary", {})
    quantization = str(binary_cfg.get("quantization", "4bit"))

    bk = binary_cfg.get("hamming_top_k", 500)
    bk2 = binary_cfg.get("bilinear_top_k", 50)

    # BinaryQuantizedReranker does not currently expose quantization modes in its API.
    # Keep sweep compatibility by validating accepted labels but using the supported ctor.
    if quantization not in {"4bit", "int8", "float16"}:
        raise ValueError(
            f"Unsupported binary quantization '{quantization}'. Valid values: 4bit, int8, float16"
        )
    reranker = BinaryQuantizedReranker(
        embedder=embedder,
        hamming_top_k=int(bk),
        bilinear_top_k=int(bk2),
    )
    return reranker


def _evaluate_binary(
    reranker: BinaryQuantizedReranker,
    pairs: list[dict[str, Any]],
) -> dict[str, float]:
    queries: list[str] = []
    docs: list[str] = []
    labels: list[int] = []
    for row in pairs:
        queries.append(row["query"])
        docs.append(row["doc"])
        labels.append(1 if float(row.get("score", 0)) >= 2 else 0)

    reranker.fit(queries, docs, labels)
    return _evaluate_ranking_reranker(reranker, pairs)


def _build_pipeline_for_variant(
    config_override: dict[str, Any],
    embedder: Embedder,
) -> PipelineReranker | None:
    pipeline_cfg = config_override.get("pipeline", {})
    stage_names = pipeline_cfg.get("stages", ["bm25"])
    top_ks = pipeline_cfg.get("top_ks", [200])

    pipeline = PipelineReranker()
    for stage_name, top_k in zip(stage_names, top_ks, strict=False):
        stage: Any = None
        if stage_name == "bm25":
            from reranker.lexical import BM25Engine

            stage = BM25Engine()
        elif stage_name == "hybrid":
            stage = HybridFusionReranker(
                adapters=[KeywordMatchAdapter()],
                embedder=embedder,
            )
        elif stage_name == "binary":
            stage = BinaryQuantizedReranker(embedder=embedder)
        elif stage_name == "colbert":
            stage = StaticColBERTReranker(embedder=embedder)
        else:
            logger.warning(f"  WARNING: Unknown pipeline stage '{stage_name}', skipping")
            continue
        pipeline.add_stage(stage_name, stage, top_k=int(top_k))
    return pipeline


def _evaluate_pipeline(
    reranker: PipelineReranker,
    pairs: list[dict[str, Any]],
) -> dict[str, float]:
    return _evaluate_ranking_reranker(reranker, pairs)


def _build_distilled_for_variant(
    config_override: dict[str, Any],
    embedder: Embedder,
) -> DistilledPairwiseRanker | None:
    distilled_cfg = config_override.get("distilled", {})
    raw_loss = str(distilled_cfg.get("loss_type", "pairwise"))
    mapped_loss = _DISTILLED_LOSS_ALIASES.get(raw_loss)
    if mapped_loss is None:
        valid = ", ".join(sorted(_DISTILLED_LOSS_ALIASES.keys()))
        raise ValueError(f"Unsupported distilled loss_type '{raw_loss}'. Valid values: {valid}")
    return DistilledPairwiseRanker(
        embedder=embedder,
        loss_type=mapped_loss,  # type: ignore[arg-type]
    )


def _validate_keys(config: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported keys in {context}: {unknown}. Allowed keys: {sorted(allowed)}"
        )


def _validate_variant(variant_name: str, variant_config: dict[str, Any]) -> None:
    _validate_keys(variant_config, _ALLOWED_TOP_LEVEL_KEYS, f"variant '{variant_name}'")
    if "cascade" in variant_config:
        cascade_cfg = variant_config["cascade"]
        if not isinstance(cascade_cfg, dict):
            raise ValueError(f"variant '{variant_name}' cascade config must be a mapping")
        _validate_keys(cascade_cfg, _ALLOWED_CASCADE_KEYS, f"variant '{variant_name}'.cascade")
        metric = str(cascade_cfg.get("confidence_metric", "TOP_MARGIN"))
        if metric not in _CASCADE_METRIC_ALIASES:
            valid = ", ".join(sorted(_CASCADE_METRIC_ALIASES.keys()))
            raise ValueError(
                f"variant '{variant_name}' has invalid cascade.confidence_metric '{metric}'. "
                f"Valid values: {valid}"
            )
    if "binary" in variant_config:
        binary_cfg = variant_config["binary"]
        if not isinstance(binary_cfg, dict):
            raise ValueError(f"variant '{variant_name}' binary config must be a mapping")
        _validate_keys(binary_cfg, _ALLOWED_BINARY_KEYS, f"variant '{variant_name}'.binary")
    if "pipeline" in variant_config:
        pipeline_cfg = variant_config["pipeline"]
        if not isinstance(pipeline_cfg, dict):
            raise ValueError(f"variant '{variant_name}' pipeline config must be a mapping")
        _validate_keys(pipeline_cfg, _ALLOWED_PIPELINE_KEYS, f"variant '{variant_name}'.pipeline")
    if "distilled" in variant_config:
        distilled_cfg = variant_config["distilled"]
        if not isinstance(distilled_cfg, dict):
            raise ValueError(f"variant '{variant_name}' distilled config must be a mapping")
        _validate_keys(
            distilled_cfg, _ALLOWED_DISTILLED_KEYS, f"variant '{variant_name}'.distilled"
        )
        raw_loss = str(distilled_cfg.get("loss_type", "pairwise"))
        if raw_loss not in _DISTILLED_LOSS_ALIASES:
            valid = ", ".join(sorted(_DISTILLED_LOSS_ALIASES.keys()))
            raise ValueError(
                f"variant '{variant_name}' has invalid distilled.loss_type '{raw_loss}'. "
                f"Valid values: {valid}"
            )
    if "late_interaction" in variant_config:
        li_cfg = variant_config["late_interaction"]
        if not isinstance(li_cfg, dict):
            raise ValueError(f"variant '{variant_name}' late_interaction config must be a mapping")
        _validate_keys(
            li_cfg,
            _ALLOWED_LATE_INTERACTION_KEYS,
            f"variant '{variant_name}'.late_interaction",
        )


def _evaluate_distilled(
    reranker: DistilledPairwiseRanker,
    preferences: list[dict[str, Any]],
) -> dict[str, float]:
    queries: list[str] = []
    doc_as: list[str] = []
    doc_bs: list[str] = []
    labels: list[int] = []
    for row in preferences:
        queries.append(str(row.get("query", "")))
        doc_as.append(str(row.get("doc_a", "")))
        doc_bs.append(str(row.get("doc_b", "")))
        labels.append(1 if row.get("preferred") == "A" else 0)

    if not queries:
        return {"accuracy": 0.0, "n_comparisons": 0.0}

    reranker.fit(queries, doc_as, doc_bs, labels)

    accuracies: list[float] = []
    latency_samples: list[float] = []
    for query, doc_a, doc_b, actual in zip(queries, doc_as, doc_bs, labels, strict=False):
        start = time.perf_counter()
        score = reranker.compare(query, doc_a, doc_b)
        latency_samples.append((time.perf_counter() - start) * 1000)
        pred = 1 if score > 0.5 else 0
        accuracies.append(1.0 if pred == actual else 0.0)

    return {
        "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "n_comparisons": float(len(accuracies)),
        "latency_mean_ms": float(np.mean(latency_samples)) if latency_samples else 0.0,
    }


def _measure_latency_generic(reranker: Any, query: str, docs: list[str], n_runs: int = 5) -> float:
    if not docs:
        return 0.0
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
        logger.info(f"No variants found in {config_path}")
        return []

    settings = get_settings()
    data_root = Path(settings.paths.raw_data_dir)
    pairs_file = data_root / "pairs.jsonl"
    prefs_file = data_root / "preferences.jsonl"

    if not pairs_file.exists():
        logger.info(f"Warning: {pairs_file} not found. Run synthetic data generation first.")
        return []

    pairs = read_jsonl(pairs_file)
    preferences: list[dict[str, Any]] = []
    if prefs_file.exists():
        preferences = read_jsonl(prefs_file)

    logger.info(f"Sweep: {sweep_name}")
    n_variants = len(variants)
    logger.info(
        f"Loaded {len(pairs)} pairs, {len(preferences)} preferences, {n_variants} variants\n"
    )

    results: list[SweepResult] = []

    try:
        for variant_name, variant_config in variants.items():
            logger.info(f"  Running variant: {variant_name}...")
            _validate_variant(variant_name, variant_config)
            clear_settings_override()
            reset_settings_cache()
            apply_settings_override(settings_from_dict(variant_config))
            embedder = Embedder()

            variant_type = _detect_variant_type(variant_config)
            metrics: dict[str, float] = {}
            latency = 0.0

            reranker: Any = None
            if variant_type == "cascade":
                reranker = _build_cascade_for_variant(variant_config, embedder)
                if reranker is not None:
                    metrics = _evaluate_cascade(reranker, pairs)
                    sample_query = pairs[0]["query"] if pairs else "test query"
                    sample_docs = list({row["doc"] for row in pairs[:20]})
                    if sample_docs:
                        latency = _measure_latency_generic(reranker, sample_query, sample_docs)

            elif variant_type == "binary":
                reranker = _build_binary_for_variant(variant_config, embedder)
                if reranker is not None:
                    metrics = _evaluate_binary(reranker, pairs)
                    sample_query = pairs[0]["query"] if pairs else "test query"
                    sample_docs = list({row["doc"] for row in pairs[:20]})
                    if sample_docs:
                        latency = _measure_latency_generic(reranker, sample_query, sample_docs)

            elif variant_type == "pipeline":
                reranker = _build_pipeline_for_variant(variant_config, embedder)
                if reranker is not None:
                    metrics = _evaluate_pipeline(reranker, pairs)
                    sample_query = pairs[0]["query"] if pairs else "test query"
                    sample_docs = list({row["doc"] for row in pairs[:20]})
                    if sample_docs:
                        latency = _measure_latency_generic(reranker, sample_query, sample_docs)

            elif variant_type == "distilled":
                reranker = _build_distilled_for_variant(variant_config, embedder)
                if reranker is not None and preferences:
                    metrics = _evaluate_distilled(reranker, preferences)

            else:
                reranker = _build_hybrid_for_variant(variant_config, embedder)
                if reranker is not None:
                    metrics = _evaluate_hybrid(reranker, pairs)
                    sample_query = pairs[0]["query"] if pairs else "test query"
                    sample_docs = list({row["doc"] for row in pairs[:20]})
                    if sample_docs:
                        latency = _measure_latency_generic(reranker, sample_query, sample_docs)

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

    has_accuracy = any("accuracy" in r.metrics for r in results)

    if has_accuracy:
        logger.info("\n" + "=" * 80)
        logger.info(f"{'Variant':<35} {'Accuracy':>10} {'Latency(ms)':>12} {'Comparisons':>10}")
        logger.info("-" * 80)
        best_acc = max(r.metrics.get("accuracy", 0) for r in results)
        for r in results:
            acc = r.metrics.get("accuracy", 0.0)
            n_comp = int(r.metrics.get("n_comparisons", 0))
            marker = " *" if abs(acc - best_acc) < 0.001 else ""
            lat = r.metrics.get("latency_mean_ms", r.latency_ms)
            logger.info(f"{r.variant_name:<35} {acc:>9.4f}{marker} {lat:>10.2f} {n_comp:>10}")
    else:
        logger.info("\n" + "=" * 80)
        logger.info(f"{'Variant':<35} {'NDCG@10':>10} {'Latency(ms)':>12} {'Queries':>8}")
        logger.info("-" * 80)
        best_ndcg = max(r.metrics.get("ndcg@10", 0) for r in results)
        best_latency = min(r.latency_ms for r in results if r.latency_ms > 0)
        for r in results:
            ndcg = r.metrics.get("ndcg@10", 0.0)
            n_queries = int(r.metrics.get("n_queries", 0))
            ndcg_marker = " *" if abs(ndcg - best_ndcg) < 0.001 else ""
            lat_marker = (
                " *" if abs(r.latency_ms - best_latency) < 0.01 and r.latency_ms > 0 else ""
            )
            logger.info(
                f"{r.variant_name:<35} "
                f"{ndcg:>9.4f}{ndcg_marker} "
                f"{r.latency_ms:>10.2f}{lat_marker} "
                f"{n_queries:>8}"
            )

    logger.info("=" * 80)
    logger.info("  * = best in column")


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
        logger.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
