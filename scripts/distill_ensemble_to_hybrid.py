"""Ensemble distillation: Train Hybrid student from FlashRank teachers.

This script implements knowledge distillation where multiple FlashRank
cross-encoder models (TinyBERT, MiniLM) serve as teachers to generate
soft labels for training a fast Hybrid Fusion Reranker student.

Expected quality: 95-98% of ensemble NDCG@10
Expected latency: ~50ms (same as Hybrid)
Training time: ~30 min (cached after first run)

Example:
    uv run scripts/distill_ensemble_to_hybrid.py --dataset beir --method pairwise
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

# Optional tqdm import for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from reranker.data.beir_loader import load_beir_simple
from reranker.data.ensemble_cache import EnsembleLabelCache
from reranker.eval.metrics import ndcg_at_k
from reranker.strategies.flashrank_ensemble import FlashRankEnsemble
from reranker.strategies.hybrid import HybridFusionReranker


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for ensemble distillation.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Distill knowledge from FlashRank ensemble into Hybrid student model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["beir", "synth", "mixed", "custom"],
        default="mixed",
        help="Dataset source for distillation (default: mixed)",
    )
    parser.add_argument(
        "--custom-path",
        type=Path,
        default=None,
        help="Path to custom dataset JSONL file (required if dataset=custom, "
        "optional for dataset=mixed)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["pointwise", "pairwise"],
        default="pairwise",
        help="Training method: pointwise or pairwise (default: pairwise)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of cached teacher labels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/models/hybrid_distilled.pkl"),
        help="Output path for distilled model (default: data/models/hybrid_distilled.pkl)",
    )
    parser.add_argument(
        "--teachers",
        type=str,
        nargs="+",
        default=["ms-marco-TinyBERT-L-2-v2", "ms-marco-MiniLM-L-12-v2"],
        help="List of FlashRank teacher model names (default: ms-marco-TinyBERT-L-2-v2 "
        "ms-marco-MiniLM-L-12-v2)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory for caching teacher labels (default: data/models)",
    )
    args = parser.parse_args()

    # Validate custom dataset requirement
    if args.dataset == "custom" and not args.custom_path:
        parser.error("--custom-path is required when --dataset=custom")

    return args


def load_training_data(dataset: str, custom_path: Path | None = None) -> tuple:
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
        raise NotImplementedError("Synthetic data loading pending")
    else:  # mixed
        beir_queries, beir_corpus, beir_qrels = load_beir_simple()

        # If custom_path provided, combine BEIR with custom data
        if custom_path:
            from reranker.data.custom_beir import load_custom_beir

            custom_data = load_custom_beir(custom_path)

            # Combine queries
            for qid, query in custom_data["queries"].items():
                new_qid = f"custom_{qid}"
                beir_queries[new_qid] = query

            # Combine corpus
            for did, doc in custom_data["corpus"].items():
                new_did = f"custom_{did}"
                beir_corpus[new_did] = doc

            # Combine qrels
            for qid, doc_rels in custom_data["qrels"].items():
                new_qid = f"custom_{qid}"
                beir_qrels[new_qid] = {f"custom_{did}": score for did, score in doc_rels.items()}

        return beir_queries, beir_corpus, beir_qrels


def generate_ensemble_labels(
    ensemble: FlashRankEnsemble,
    queries: list[str],
    corpus_docs: list[str],
    qrels: dict,  # query_id -> {doc_id: relevance}
    cache: EnsembleLabelCache,
    force_regenerate: bool = False,
) -> dict:
    def generator_fn():
        print(f"Generating labels for {len(queries)} queries...")
        labels = {}
        query_iterator = tqdm(queries) if tqdm else queries
        for q_idx, query in enumerate(query_iterator):
            if q_idx % 100 == 0 and tqdm is None:
                print(f"  Processing query {q_idx}/{len(queries)}")
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
    labels: dict[tuple[int, int], float],  # (query_idx, doc_idx) -> ensemble_score
    output_path: Path,
) -> None:
    """Train HybridFusionReranker using pointwise (regression) method.

    Args:
        queries: List of query texts.
        corpus_docs: List of document texts.
        labels: Dict mapping (query_idx, doc_idx) -> ensemble_score.
        output_path: Path to save trained model.
    """
    print("\nTraining HybridFusionReranker with pointwise method...")

    # Flatten labels dict to lists
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
    print(f"Training samples: {len(train_queries)}")
    print(f"Score range: [{min(train_scores):.4f}, {max(train_scores):.4f}]")

    # Create and train model
    hybrid = HybridFusionReranker()
    hybrid.fit(train_queries, train_docs, train_docs, train_scores)

    if not hybrid.is_fitted:
        raise RuntimeError("Model training failed - is_fitted is False")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid.save(output_path)
    print(f"Model saved to {output_path}")


def train_hybrid_pairwise(
    queries: list[str],
    corpus_docs: list[str],
    labels: dict[tuple[int, int], float],  # (query_idx, doc_idx) -> ensemble_score
    output_path: Path,
) -> None:
    """Train HybridFusionReranker using pairwise (ranking) method.

    Args:
        queries: List of query texts.
        corpus_docs: List of document texts.
        labels: Dict mapping (query_idx, doc_idx) -> ensemble_score.
        output_path: Path to save trained model.
    """
    print("\nTraining HybridFusionReranker with pairwise method...")

    # Group labels by query: query_labels[q_idx] = {d_idx: score}
    query_labels: defaultdict[int, dict[int, float]] = defaultdict(dict)
    for (q_idx, d_idx), score in labels.items():
        if q_idx < len(queries) and d_idx < len(corpus_docs):
            query_labels[q_idx][d_idx] = score

    # Generate pairwise comparisons
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
            print(
                f"  Query {q_idx}: limited to {max_pairs_per_query} pairs "
                f"(out of {len(doc_indices) * (len(doc_indices) - 1) // 2} possible)"
            )

    if not train_queries:
        raise ValueError(
            "No valid pairwise training samples generated. Check if labels have score variations."
        )

    print(f"Generated {total_pairs} pairwise comparisons")
    if skipped_equal > 0:
        print(f"Skipped {skipped_equal} pairs with equal scores")

    label_1_count = sum(1 for label in train_labels if label == 1)
    label_0_count = len(train_labels) - label_1_count
    print(
        f"Label distribution: {label_1_count} pairs prefer doc_a, "
        f"{label_0_count} pairs prefer doc_b"
    )

    hybrid = HybridFusionReranker()
    print(f"Training with {len(train_queries)} pairwise comparisons")
    hybrid.fit(train_queries, train_doc_as, train_doc_bs, train_labels)

    if not hybrid.is_fitted:
        raise RuntimeError("Model training failed - is_fitted is False")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid.save(output_path)
    print(f"Model saved to {output_path}")


def evaluate_hybrid(
    hybrid: HybridFusionReranker,
    queries: dict,
    docs: dict,
    qrels: dict,
    top_k: int = 10,
) -> dict:
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    queries_with_qrels = [qid for qid in queries.keys() if qid in qrels and qrels[qid]]

    num_eval_queries = min(100, len(queries_with_qrels))
    query_ids_eval = sorted(queries_with_qrels)[:num_eval_queries]

    ndcg_scores: list[float] = []
    latencies: list[float] = []

    print(f"\n[Evaluation] Testing on {num_eval_queries} queries...")

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

    print(f"  NDCG@{top_k}: {avg_ndcg:.4f}")
    print(f"  Avg Latency: {avg_latency_ms:.2f} ms")
    print(f"  Queries evaluated: {len(ndcg_scores)}")

    return {
        "ndcg_at_10": avg_ndcg,
        "avg_latency_ms": avg_latency_ms,
        "num_queries": len(ndcg_scores),
    }


def main() -> None:
    """Main entry point for ensemble distillation pipeline."""
    try:
        args = parse_args()
        # Print configuration
        print(f"Teachers: {', '.join(args.teachers)}")
        print(f"Dataset: {args.dataset}")
        if args.custom_path:
            print(f"Custom path: {args.custom_path}")
        print(f"Method: {args.method}")
        print(f"Force regenerate: {args.force_regenerate}")
        print(f"Output: {args.output}")
        print(f"Cache dir: {args.cache_dir}")

        # Initialize ensemble and cache
        ensemble = FlashRankEnsemble(args.teachers)
        cache = EnsembleLabelCache(args.cache_dir)

        # Load dataset using dispatcher
        print(f"\nLoading {args.dataset} dataset...")
        queries_dict, corpus_dict, qrels = load_training_data(args.dataset, args.custom_path)

        # Convert to lists for indexing and filter for testing
        query_ids = sorted(queries_dict.keys())[:50]  # Use subset for testing
        doc_ids = sorted(corpus_dict.keys())[:500]

        queries = [queries_dict[qid] for qid in query_ids]
        corpus_docs = [
            f"{corpus_dict[did]['title']} {corpus_dict[did]['text']}".strip() for did in doc_ids
        ]

        # Filter qrels to only include selected queries and docs
        filtered_qrels = {}
        for qid in query_ids:
            if qid in qrels:
                filtered_qrels[qid] = {
                    did: score for did, score in qrels[qid].items() if did in doc_ids
                }

        print(f"Loaded {len(queries)} queries, {len(corpus_docs)} documents")

        # Generate ensemble labels
        print("\nGenerating ensemble teacher labels...")
        labels = generate_ensemble_labels(
            ensemble=ensemble,
            queries=queries,
            corpus_docs=corpus_docs,
            qrels=filtered_qrels,
            cache=cache,
            force_regenerate=args.force_regenerate,
        )

        print(f"Generated {len(labels)} query-document pair scores")

        # Train Hybrid based on method
        if args.method == "pointwise":
            train_hybrid_pointwise(
                queries=queries,
                corpus_docs=corpus_docs,
                labels=labels,
                output_path=args.output,
            )
        elif args.method == "pairwise":
            train_hybrid_pairwise(
                queries=queries,
                corpus_docs=corpus_docs,
                labels=labels,
                output_path=args.output,
            )
        else:
            print(f"\nMethod '{args.method}' not yet implemented.")
            return

        # Load trained model and evaluate
        print("\n" + "=" * 60)
        print("LOADING TRAINED MODEL FOR EVALUATION")
        print("=" * 60)
        hybrid = HybridFusionReranker.load(args.output)
        print(f"Model loaded from {args.output}")

        # Run evaluation
        eval_results = evaluate_hybrid(
            hybrid=hybrid,
            queries=queries_dict,
            docs=corpus_dict,
            qrels=qrels,
            top_k=10,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"NDCG@10:              {eval_results['ndcg_at_10']:.4f}")
        print(f"Avg Latency:          {eval_results['avg_latency_ms']:.2f} ms")
        print(f"Queries Evaluated:    {eval_results['num_queries']}")
        print("=" * 60)

    except ImportError as e:
        print(f"ImportError: {e}")
        error_msg = str(e).lower()
        if "flashrank" in error_msg:
            print("Install: uv pip install flashrank")
        elif "beir" in error_msg:
            print("Install: uv pip install beir --no-deps && uv pip install rank-bm25 pyyaml")
        else:
            print("Install dependencies: uv sync --extra flashrank")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
