"""Ensemble distillation: FlashRank teachers train Hybrid student."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from reranker.config import get_settings
from reranker.data.ensemble_cache import EnsembleLabelCache
from reranker.strategies.flashrank_ensemble import FlashRankEnsemble
from reranker.strategies.hybrid import HybridFusionReranker

if TYPE_CHECKING:
    from collections.abc import Mapping


def load_beir_data(dataset_name: str = "nfcorpus") -> tuple[dict, dict, dict]:
    """Load BEIR dataset for distillation.

    Args:
        dataset_name: Name of BEIR dataset (default: "nfcorpus")

    Returns:
        Tuple of (queries, corpus, qrels) where:
            - queries: Mapping from query_id to query text
            - corpus: Mapping from doc_id to doc dict with _id, title, text
            - qrels: Mapping from query_id to {doc_id: relevance_score}
    """
    try:
        from beir import util
    except ImportError as e:
        raise ImportError("BEIR not installed. Run: uv pip install beir") from e

    beir_dir = Path("data/beir") / dataset_name

    if not beir_dir.exists():
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        print(f"Downloading {dataset_name} from {url}")
        util.download_and_unload(url, str(beir_dir.parent))

    # Parse corpus.jsonl
    corpus_file = beir_dir / "corpus.jsonl"
    corpus = {}
    with open(corpus_file) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "_id": doc["_id"],
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }

    # Parse queries.jsonl
    queries_file = beir_dir / "queries.jsonl"
    queries = {}
    with open(queries_file) as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]

    # Parse qrels/test.tsv
    qrels_file = beir_dir / "qrels" / "test.tsv"
    qrels_dict: defaultdict[str, dict[str, int]] = defaultdict(dict)
    with open(qrels_file) as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                qrels_dict[query_id][doc_id] = score

    qrels = dict(qrels_dict)
    return queries, corpus, qrels


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
        help="Path to custom dataset JSONL file (required if dataset=custom)",
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
        help="List of FlashRank teacher model names (default: ms-marco-TinyBERT-L-2-v2 ms-marco-MiniLM-L-12-v2)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/models"),
        help="Directory for caching teacher labels (default: data/models)",
    )
    return parser.parse_args()


def generate_ensemble_labels(
    ensemble: FlashRankEnsemble,
    queries: list[str],
    corpus_docs: list[str],
    qrels: dict,  # query_id -> {doc_id: relevance}
    cache: EnsembleLabelCache,
    force_regenerate: bool = False,
) -> dict:
    """Generate or load cached ensemble labels.

    Args:
        ensemble: FlashRank ensemble for teacher scoring.
        queries: List of query texts.
        corpus_docs: List of document texts.
        qrels: Query relevance judgments (unused but kept for API consistency).
        cache: Cache instance for persistent storage.
        force_regenerate: If True, bypass cache and regenerate.

    Returns:
        Dict mapping (query_idx, doc_idx) -> ensemble_score
    """

    def generator_fn():
        print(f"Generating labels for {len(queries)} queries...")
        labels = {}
        for q_idx, query in enumerate(queries):
            if q_idx % 100 == 0:
                print(f"  Processing query {q_idx}/{len(queries)}")
            scores = ensemble.score_batch(query, corpus_docs)
            for d_idx, score in enumerate(scores):
                labels[(q_idx, d_idx)] = float(score)
        return labels

    dataset_id = f"queries_{len(queries)}_docs_{len(corpus_docs)}"
    cached_labels = cache.load_or_generate(dataset_id, ensemble.models, generator_fn, force_regenerate)
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
    hybrid.fit_pointwise(train_queries, train_docs, train_scores)

    if not hybrid.is_fitted:
        raise RuntimeError("Model training failed - is_fitted is False")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid.save(output_path)
    print(f"Model saved to {output_path}")


def train_hybrid_pairwise(
    queries: list[str],
    corpus_docs: list[str],
    labels: dict,  # (query_idx, doc_idx) -> score
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
    max_pairs_per_query = 1000  # Limit to avoid combinatorial explosion

    for q_idx, doc_scores in query_labels.items():
        doc_indices = sorted(doc_scores.keys())

        # Generate all pairs (i, j) where i < j
        pairs_generated = 0
        for i_idx in doc_indices:
            for j_idx in doc_indices:
                if i_idx < j_idx:
                    # Limit pairs per query to avoid excessive computation
                    if pairs_generated >= max_pairs_per_query:
                        break

                    score_a = doc_scores[i_idx]
                    score_b = doc_scores[j_idx]

                    # Skip pairs with equal scores (no preference)
                    if score_a == score_b:
                        skipped_equal += 1
                        continue

                    # Label 1 if doc_a is preferred (higher score), else 0
                    label = 1 if score_a > score_b else 0

                    train_queries.append(queries[q_idx])
                    train_doc_as.append(corpus_docs[i_idx])
                    train_doc_bs.append(corpus_docs[j_idx])
                    train_labels.append(label)
                    total_pairs += 1
                    pairs_generated += 1

            # Break outer loop if we've hit the limit
            if pairs_generated >= max_pairs_per_query:
                print(f"  Query {q_idx}: limited to {max_pairs_per_query} pairs (out of {len(doc_indices) * (len(doc_indices) - 1) // 2} possible)")
                break

    if not train_queries:
        raise ValueError("No valid pairwise training samples generated. Check if labels have score variations.")

    print(f"Generated {total_pairs} pairwise comparisons")
    if skipped_equal > 0:
        print(f"Skipped {skipped_equal} pairs with equal scores")

    # Count label distribution
    label_1_count = sum(1 for label in train_labels if label == 1)
    label_0_count = len(train_labels) - label_1_count
    print(f"Label distribution: {label_1_count} pairs prefer doc_a, {label_0_count} pairs prefer doc_b")

    # Create and train model
    # Note: HybridFusionReranker.fit() expects (query, doc) pairs and binary labels
    # For pairwise, we concatenate the two docs to create a single representation
    hybrid = HybridFusionReranker()

    # Since fit() expects single doc per query, we need to adapt the pairwise input
    # We'll create a combined representation for pairwise comparison
    combined_queries = []
    combined_docs = []
    combined_labels = []

    for query, doc_a, doc_b, label in zip(train_queries, train_doc_as, train_doc_bs, train_labels, strict=False):
        # For each pair, we create two training examples:
        # 1. (query, doc_a) with label indicating preference
        # 2. (query, doc_b) with opposite label
        combined_queries.extend([query, query])
        combined_docs.extend([doc_a, doc_b])
        # For doc_a: use original label, for doc_b: use inverse
        combined_labels.extend([label, 1 - label])

    print(f"Training with {len(combined_queries)} samples (pairwise expansion)")
    hybrid.fit(combined_queries, combined_docs, combined_labels)

    if not hybrid.is_fitted:
        raise RuntimeError("Model training failed - is_fitted is False")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid.save(output_path)
    print(f"Model saved to {output_path}")


def main() -> None:
    """Main entry point for ensemble distillation pipeline."""
    args = parse_args()
    settings = get_settings()

    # Validate custom dataset requirement
    if args.dataset == "custom" and not args.custom_path:
        raise ValueError(
            "--custom-path is required when --dataset=custom. "
            "Provide path to your custom dataset JSONL file."
        )

    # Print configuration
    print(f"Teachers: {', '.join(args.teachers)}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == "custom":
        print(f"Custom path: {args.custom_path}")
    print(f"Method: {args.method}")
    print(f"Force regenerate: {args.force_regenerate}")
    print(f"Output: {args.output}")
    print(f"Cache dir: {args.cache_dir}")

    # Initialize ensemble and cache
    ensemble = FlashRankEnsemble(args.teachers)
    cache = EnsembleLabelCache(args.cache_dir)

    # Load dataset and generate labels for BEIR or mixed datasets
    if args.dataset in ("beir", "mixed"):
        print("\nLoading BEIR dataset...")
        queries_dict, corpus_dict, qrels = load_beir_data()

        # Convert to lists for indexing
        query_ids = sorted(queries_dict.keys())
        doc_ids = sorted(corpus_dict.keys())

        # Use subset for testing
        query_ids = query_ids[:50]
        doc_ids = doc_ids[:500]

        queries = [queries_dict[qid] for qid in query_ids]
        corpus_docs = [
            f"{corpus_dict[did]['title']} {corpus_dict[did]['text']}".strip()
            for did in doc_ids
        ]

        # Filter qrels to only include selected queries and docs
        filtered_qrels = {}
        for qid in query_ids:
            if qid in qrels:
                filtered_qrels[qid] = {
                    did: score
                    for did, score in qrels[qid].items()
                    if did in doc_ids
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
    else:
        print(f"\nDataset '{args.dataset}' not yet supported.")
        print("Supported datasets: beir, mixed")


if __name__ == "__main__":
    main()
