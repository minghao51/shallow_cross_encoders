"""Shared benchmarking utilities.

This module provides common evaluation and training functions used across
different benchmarking scripts to reduce code duplication.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from reranker.eval.metrics import LatencyTracker, ndcg_at_k, precision_at_k, reciprocal_rank


def evaluate_reranker_on_rows(
    rows: list[dict[str, Any]], reranker: Any
) -> dict[str, float]:
    """Evaluate a reranker on benchmark rows.

    This function groups rows by query, runs the reranker for each query,
    and computes standard metrics (NDCG@10, MRR, P@1, latency).

    Args:
        rows: List of benchmark rows with keys:
              - query: Query text
              - doc: Document text
              - score: Relevance score (higher = more relevant)
        reranker: Reranker instance with rerank(query, docs) method.
                  Can be any reranker (FlashRankWrapper, HybridFusionReranker, etc.)

    Returns:
        Dictionary with metrics:
        - ndcg@10: NDCG@10 score
        - mrr: Mean Reciprocal Rank
        - p@1: Precision at 1
        - latency_p50_ms: 50th percentile latency in ms
        - latency_p99_ms: 99th percentile latency in ms
        - queries_evaluated: Number of queries processed

    Raises:
        ValueError: If rows is empty or reranker is None.
    """
    if not rows:
        raise ValueError("rows cannot be empty")

    if reranker is None:
        raise ValueError("reranker cannot be None")

    latency = LatencyTracker()

    # Group by query
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        query = str(row.get("query", ""))
        grouped.setdefault(query, []).append(row)

    ndcgs: list[float] = []
    mrrs: list[float] = []
    p1s: list[float] = []

    for items in grouped.values():
        query = items[0].get("query", "")
        docs = [str(item.get("doc", "")) for item in items]

        # Measure latency and rerank
        with latency.measure():
            ranked = reranker.rerank(query, docs)

        # Get relevance scores
        doc_to_relevance = {
            str(item.get("doc", "")): int(item.get("score", 0)) for item in items
        }

        # Handle both dict (FlashRank) and RankedDoc (current) formats
        relevances = []
        for r in ranked:
            if isinstance(r, dict):
                doc_text = r.get("doc", "")
            else:
                doc_text = r.doc
            relevances.append(float(doc_to_relevance.get(doc_text, 0)))

        binary = [1 if rel > 0 else 0 for rel in relevances]

        # Compute metrics only if we have at least one relevant doc
        if sum(binary) > 0:
            ndcgs.append(ndcg_at_k(relevances, 10))
            mrrs.append(reciprocal_rank(binary))
            p1s.append(precision_at_k(binary, 1))

    summary = latency.summary()

    return {
        "ndcg@10": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
        "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "p@1": round(float(np.mean(p1s)), 4) if p1s else 0.0,
        "latency_p50_ms": round(float(summary["p50"]), 4),
        "latency_p99_ms": round(float(summary["p99"]), 4),
        "queries_evaluated": len(grouped),
    }


def train_strategies(
    train_rows: list[dict[str, Any]],
    strategies_config: dict[str, Any],
) -> dict[str, Any]:
    """Train multiple strategies from configuration.

    This function creates and trains strategies based on a configuration
    dictionary, returning a mapping of strategy names to trained instances.

    Args:
        train_rows: Training data with keys:
                   - query: Query text
                   - doc: Document text
                   - score: Relevance label
        strategies_config: Configuration dict with keys:
                          - hybrid: dict with config for HybridFusionReranker
                          - binary_reranker: dict with config for BinaryQuantizedReranker
                          - late_interaction: dict with config for StaticColBERTReranker
                          - etc.

    Returns:
        Dictionary mapping strategy names to trained reranker instances.

    Raises:
        ValueError: If train_rows is empty or strategies_config is invalid.
        ImportError: If required dependencies are missing.
    """
    if not train_rows:
        raise ValueError("train_rows cannot be empty")

    if not strategies_config:
        raise ValueError("strategies_config cannot be empty")

    queries = [str(row.get("query", "")) for row in train_rows]
    docs = [str(row.get("doc", "")) for row in train_rows]
    labels = [1 if int(row.get("score", 0)) > 0 else 0 for row in train_rows]

    strategies: dict[str, Any] = {}

    # Train Hybrid Fusion (use pointwise for simple classification)
    if "hybrid" in strategies_config:
        from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter

        config = strategies_config["hybrid"]
        adapters = config.get("adapters", [KeywordMatchAdapter()])
        hybrid = HybridFusionReranker(adapters=adapters)
        # Use fit_pointwise for regression-style training with binary labels
        binary_labels = [float(lbl) for lbl in labels]
        hybrid.fit_pointwise(queries, docs, binary_labels, use_regression=True)
        strategies["hybrid"] = hybrid

    # Train Binary Reranker
    if "binary_reranker" in strategies_config:
        from reranker.strategies.binary_reranker import BinaryQuantizedReranker

        binary = BinaryQuantizedReranker().fit(queries, docs, labels)
        strategies["binary_reranker"] = binary

    # Train Late Interaction
    if "late_interaction" in strategies_config:
        from reranker.strategies.late_interaction import StaticColBERTReranker

        unique_docs = list(set(docs))
        late_interaction = StaticColBERTReranker()
        late_interaction.fit(unique_docs)
        strategies["late_interaction"] = late_interaction

    # FlashRank models (don't need training, just instantiate)
    if "flashrank_tiny" in strategies_config:
        from reranker.adapters.flashrank_wrapper import FlashRankWrapper

        strategies["flashrank_tiny"] = FlashRankWrapper("ms-marco-TinyBERT-L-2-v2")

    if "flashrank_mini" in strategies_config:
        from reranker.adapters.flashrank_wrapper import FlashRankWrapper

        strategies["flashrank_mini"] = FlashRankWrapper("ms-marco-MiniLM-L-12-v2")

    return strategies
