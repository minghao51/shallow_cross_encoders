"""Ensemble distillation pipeline functions for training Hybrid student models.

Extracted from scripts/distill_ensemble_to_hybrid.py to enable reuse and testing.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from reranker.config import get_settings
from reranker.data.beir_loader import load_beir_simple
from reranker.data.ensemble_cache import EnsembleLabelCache
from reranker.eval.metrics import ndcg_at_k
from reranker.strategies.flashrank_ensemble import FlashRankEnsemble
from reranker.strategies.hybrid import HybridFusionReranker
from reranker.utils import read_jsonl

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = structlog.get_logger(__name__)


def _load_synth_data() -> tuple[
    dict[str, str], dict[str, dict[str, Any]], dict[str, dict[str, int]]
]:
    data_root = Path(get_settings().paths.raw_data_dir)
    pairs_path = data_root / "pairs.jsonl"
    rows = read_jsonl(pairs_path)
    if not rows:
        raise ValueError(
            f"Synthetic dataset is empty or missing at {pairs_path}. "
            "Run: uv run scripts/materialize_demo_data.py"
        )

    queries: dict[str, str] = {}
    corpus: dict[str, dict[str, Any]] = {}
    qrels: dict[str, dict[str, int]] = {}
    query_id_by_text: dict[str, str] = {}
    doc_id_by_text: dict[str, str] = {}

    for row in rows:
        query_text = str(row.get("query", "")).strip()
        doc_text = str(row.get("doc", "")).strip()
        if not query_text or not doc_text:
            continue

        qid = query_id_by_text.setdefault(query_text, f"q{len(query_id_by_text)}")
        did = doc_id_by_text.setdefault(doc_text, f"d{len(doc_id_by_text)}")

        queries[qid] = query_text
        corpus.setdefault(did, {"_id": did, "title": "", "text": doc_text})
        qrels.setdefault(qid, {})[did] = int(float(row.get("score", 0)))

    if not queries or not corpus or not qrels:
        raise ValueError(f"Synthetic dataset at {pairs_path} did not yield usable qrels.")

    return queries, corpus, qrels


def load_training_data(
    dataset: str, custom_path: Path | None = None
) -> tuple[dict[str, str], dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    """Load training data based on dataset choice.

    Args:
        dataset: Dataset type ('beir', 'custom', 'synth', 'mixed')
        custom_path: Path to custom dataset file (required for 'custom' dataset)

    Returns:
        Tuple of (queries_dict, corpus_dict, qrels_dict) where:
            - queries_dict: Mapping from query_id to query text
            - corpus_dict: Mapping from doc_id to doc dict with _id, title, text
            - qrels_dict: Mapping from query_id to {doc_id: relevance_score}
    """
    if dataset == "beir":
        return load_beir_simple()
    elif dataset == "custom":
        if not custom_path:
            raise ValueError("--custom-path required for custom dataset")
        from reranker.data.custom_beir import load_custom_beir

        data = load_custom_beir(custom_path)
        return data["queries"], data["corpus"], data["qrels"]
    elif dataset == "synth":
        return _load_synth_data()
    else:  # mixed
        beir_queries, beir_corpus, beir_qrels = load_beir_simple()

        if custom_path:
            from reranker.data.custom_beir import load_custom_beir

            custom_data = load_custom_beir(custom_path)

            for qid, query in custom_data["queries"].items():
                new_qid = f"custom_{qid}"
                beir_queries[new_qid] = query

            for did, doc in custom_data["corpus"].items():
                new_did = f"custom_{did}"
                beir_corpus[new_did] = doc

            for qid, doc_rels in custom_data["qrels"].items():
                new_qid = f"custom_{qid}"
                beir_qrels[new_qid] = {f"custom_{did}": score for did, score in doc_rels.items()}

        return beir_queries, beir_corpus, beir_qrels


def generate_ensemble_labels(
    ensemble: FlashRankEnsemble,
    queries: list[str],
    corpus_docs: list[str],
    qrels: dict[str, dict[str, int]],
    cache: EnsembleLabelCache,
    force_regenerate: bool = False,
) -> dict[tuple[int, int], float]:
    del qrels

    def generator_fn() -> dict[tuple[int, int], float]:
        logger.info(f"Generating labels for {len(queries)} queries...")
        labels = {}
        query_iterator = tqdm(queries) if tqdm else queries
        for q_idx, query in enumerate(query_iterator):
            if q_idx % 100 == 0 and tqdm is None:
                logger.info(f"  Processing query {q_idx}/{len(queries)}")
            scores = ensemble.score_batch(query, corpus_docs)
            for d_idx, score in enumerate(scores):
                labels[(q_idx, d_idx)] = float(score)
        return labels

    dataset_id = f"queries_{len(queries)}_docs_{len(corpus_docs)}"
    cached_labels = cache.load_or_generate(
        dataset_id, ensemble.models, generator_fn, force_regenerate
    )
    return cached_labels


def train_hybrid_pointwise(
    queries: list[str],
    corpus_docs: list[str],
    labels: dict[tuple[int, int], float],
    output_path: Path,
) -> None:
    """Train HybridFusionReranker using pointwise (regression) method.

    Args:
        queries: List of query texts.
        corpus_docs: List of document texts.
        labels: Dict mapping (query_idx, doc_idx) -> ensemble_score.
        output_path: Path to save trained model.
    """
    logger.info("\nTraining HybridFusionReranker with pointwise method...")

    train_queries = []
    train_docs = []
    train_scores = []

    for (q_idx, d_idx), score in labels.items():
        if q_idx < len(queries) and d_idx < len(corpus_docs):
            train_queries.append(queries[q_idx])
            train_docs.append(corpus_docs[d_idx])
            train_scores.append(score)

    if not train_queries:
        raise ValueError("No valid training samples generated from labels")
    logger.info(f"Training samples: {len(train_queries)}")
    logger.info(f"Score range: [{min(train_scores):.4f}, {max(train_scores):.4f}]")

    hybrid = HybridFusionReranker()
    hybrid.fit_pointwise(train_queries, train_docs, train_scores)

    if not hybrid.is_fitted:
        raise RuntimeError("Model training failed - is_fitted is False")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid.save(output_path)
    logger.info(f"Model saved to {output_path}")


def train_hybrid_pairwise(
    queries: list[str],
    corpus_docs: list[str],
    labels: dict[tuple[int, int], float],
    output_path: Path,
) -> None:
    """Train HybridFusionReranker using pairwise (ranking) method.

    Args:
        queries: List of query texts.
        corpus_docs: List of document texts.
        labels: Dict mapping (query_idx, doc_idx) -> ensemble_score.
        output_path: Path to save trained model.
    """
    logger.info("\nTraining HybridFusionReranker with pairwise method...")

    query_labels: defaultdict[int, dict[int, float]] = defaultdict(dict)
    for (q_idx, d_idx), score in labels.items():
        if q_idx < len(queries) and d_idx < len(corpus_docs):
            query_labels[q_idx][d_idx] = score

    train_queries = []
    train_doc_as = []
    train_doc_bs = []
    train_labels = []

    total_pairs = 0
    skipped_equal = 0
    max_pairs_per_query = 1000

    for q_idx, doc_scores in query_labels.items():
        doc_indices = sorted(doc_scores.keys())

        pairs_generated = 0
        for i_idx in doc_indices:
            if pairs_generated >= max_pairs_per_query:
                break
            for j_idx in doc_indices:
                if i_idx >= j_idx:
                    continue
                if pairs_generated >= max_pairs_per_query:
                    break

                score_a = doc_scores[i_idx]
                score_b = doc_scores[j_idx]

                if abs(score_a - score_b) < 1e-9:
                    skipped_equal += 1
                    continue

                label = 1 if score_a > score_b else 0

                train_queries.append(queries[q_idx])
                train_doc_as.append(corpus_docs[i_idx])
                train_doc_bs.append(corpus_docs[j_idx])
                train_labels.append(label)
                total_pairs += 1
                pairs_generated += 1

        if pairs_generated >= max_pairs_per_query:
            logger.info(
                f"  Query {q_idx}: limited to {max_pairs_per_query} pairs "
                f"(out of {len(doc_indices) * (len(doc_indices) - 1) // 2} possible)"
            )

    if not train_queries:
        raise ValueError(
            "No valid pairwise training samples generated. Check if labels have score variations."
        )

    logger.info(f"Generated {total_pairs} pairwise comparisons")
    if skipped_equal > 0:
        logger.info(f"Skipped {skipped_equal} pairs with equal scores")

    label_1_count = sum(1 for label in train_labels if label == 1)
    label_0_count = len(train_labels) - label_1_count
    logger.info(
        f"Label distribution: {label_1_count} pairs prefer doc_a, "
        f"{label_0_count} pairs prefer doc_b"
    )

    hybrid = HybridFusionReranker()
    logger.info(f"Training with {len(train_queries)} pairwise comparisons")
    hybrid.fit(train_queries, train_doc_as, train_doc_bs, train_labels)

    if not hybrid.is_fitted:
        raise RuntimeError("Model training failed - is_fitted is False")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid.save(output_path)
    logger.info(f"Model saved to {output_path}")


def evaluate_hybrid(
    hybrid: HybridFusionReranker,
    queries: dict,
    docs: dict,
    qrels: dict,
    top_k: int = 10,
) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION METRICS")
    logger.info("=" * 60)

    queries_with_qrels = [qid for qid in queries.keys() if qid in qrels and qrels[qid]]

    num_eval_queries = min(100, len(queries_with_qrels))
    query_ids_eval = sorted(queries_with_qrels)[:num_eval_queries]

    ndcg_scores: list[float] = []
    latencies: list[float] = []

    logger.info(f"\n[Evaluation] Testing on {num_eval_queries} queries...")

    for qid in query_ids_eval:
        query = queries[qid]
        doc_ids_for_query = list(qrels[qid].keys())
        if not doc_ids_for_query:
            continue

        docs_list = [
            f"{docs[did]['title']} {docs[did]['text']}".strip()
            for did in doc_ids_for_query
            if did in docs
        ]
        if not docs_list:
            continue

        start_time = time.perf_counter()
        ranked = hybrid.rerank(query, docs_list)
        end_time = time.perf_counter()

        latencies.append((end_time - start_time) * 1000)

        doc_id_to_idx = {did: i for i, did in enumerate(doc_ids_for_query) if did in docs}
        relevances = [0.0] * len(ranked)
        for rank_pos, ranked_doc in enumerate(ranked):
            for did, idx in doc_id_to_idx.items():
                if docs_list[idx] == ranked_doc.doc:
                    relevances[rank_pos] = float(qrels[qid].get(did, 0))
                    break

        ndcg_scores.append(ndcg_at_k(relevances, k=top_k))

    avg_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0

    logger.info(f"  NDCG@{top_k}: {avg_ndcg:.4f}")
    logger.info(f"  Avg Latency: {avg_latency_ms:.2f} ms")
    logger.info(f"  Queries evaluated: {len(ndcg_scores)}")

    return {
        "ndcg_at_10": avg_ndcg,
        "avg_latency_ms": avg_latency_ms,
        "num_queries": len(ndcg_scores),
    }
