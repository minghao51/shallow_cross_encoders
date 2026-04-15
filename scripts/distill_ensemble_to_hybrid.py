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
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Optional tqdm import for progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from reranker.config import get_settings
from reranker.data.ensemble_cache import EnsembleLabelCache
from reranker.strategies.flashrank_ensemble import FlashRankEnsemble
from reranker.strategies.hybrid import HybridFusionReranker



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
        return load_beir_data()
    elif dataset == "custom":
        if not custom_path:
            raise ValueError("--custom-path required for custom dataset")
        from reranker.data.custom_beir import load_custom_beir
        data = load_custom_beir(custom_path)
        return data["queries"], data["corpus"], data["qrels"]
    elif dataset == "synth":
        raise NotImplementedError("Synthetic data loading pending")
    else:  # mixed
        beir_queries, beir_corpus, beir_qrels = load_beir_data()

        # If custom_path provided, combine BEIR with custom data
        if custom_path:
            from reranker.data.custom_beir import load_custom_beir
            custom_data = load_custom_beir(custom_path)

            # Merge datasets - use offset IDs to avoid collisions
            offset = len(beir_queries)

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
                beir_qrels[new_qid] = {
                    f"custom_{did}": score
                    for did, score in doc_rels.items()
                }

        return beir_queries, beir_corpus, beir_qrels


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
        help="Path to custom dataset JSONL file (required if dataset=custom, optional for dataset=mixed)",
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
    args = parser.parse_args()

    # Validate custom dataset requirement
    if args.dataset == "custom" and not args.custom_path:
        parser.error("--custom-path is required when --dataset=custom")

    return args


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
        query_iterator = tqdm(queries) if tqdm else queries
        for q_idx, query in enumerate(query_iterator):
            if q_idx % 100 == 0 and tqdm is None:
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
    labels: dict[tuple[int, int], float],  # (query_idx, doc_idx) -> score
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
                    if abs(score_a - score_b) < 1e-9:
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

    # Create and train model with true pairwise data
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
    """Evaluate hybrid model performance with NDCG@10 and latency benchmarking.

    Args:
        hybrid: Trained HybridFusionReranker model.
        queries: Mapping from query_id to query text.
        docs: Mapping from doc_id to doc dict with _id, title, text.
        qrels: Mapping from query_id to {doc_id: relevance_score}.
        top_k: Number of top documents to consider for NDCG.

    Returns:
        Dict with ndcg_at_10, avg_latency_ms, num_queries
    """
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)

    # Get queries that have qrels for evaluation
    queries_with_qrels = [qid for qid in queries.keys() if qid in qrels and qrels[qid]]

    # Latency benchmark: test on first 100 queries that have qrels
    num_latency_queries = min(100, len(queries_with_qrels))
    query_ids_latency = sorted(queries_with_qrels)[:num_latency_queries]
    latencies = []

    print(f"\n[Latency Benchmark] Testing on {num_latency_queries} queries...")

    for qid in query_ids_latency:
        query = queries[qid]
        docs_list = [
            f"{docs[did]['title']} {docs[did]['text']}".strip()
            for did in sorted(docs.keys())[:100]  # Limit to 100 docs for speed
        ]

        start_time = time.perf_counter()
        _ = hybrid.rerank(query, docs_list)
        end_time = time.perf_counter()

        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    avg_latency_ms = np.mean(latencies)
    std_latency_ms = np.std(latencies)

    print(f"  Average latency: {avg_latency_ms:.2f} ms")
    print(f"  Std deviation:  {std_latency_ms:.2f} ms")
    print(f"  Min latency:     {min(latencies):.2f} ms")
    print(f"  Max latency:     {max(latencies):.2f} ms")

    # NDCG@10 calculation on first 50 queries with qrels
    num_ndcg_queries = min(50, len(queries_with_qrels))
    query_ids_ndcg = sorted(queries_with_qrels)[:num_ndcg_queries]

    print(f"\n[NDCG@10] Calculating on {num_ndcg_queries} queries...")

    ndcg_scores = []
    evaluated_queries = 0

    for qid in query_ids_ndcg:

        query = queries[qid]
        relevant_docs = qrels[qid]

        # Get candidate documents - include relevant docs + random sample up to 200
        relevant_doc_ids = set(relevant_docs.keys())
        all_doc_ids = set(docs.keys())

        # Ensure all relevant docs are included
        candidate_doc_ids = list(relevant_doc_ids)

        # Add random docs to reach 200 total candidates
        remaining_docs = all_doc_ids - relevant_doc_ids
        if len(remaining_docs) > 0:
            additional_needed = min(200 - len(candidate_doc_ids), len(remaining_docs))
            candidate_doc_ids.extend(sorted(remaining_docs)[:additional_needed])

        docs_list = [
            f"{docs[did]['title']} {docs[did]['text']}".strip()
            for did in candidate_doc_ids
        ]

        # Create mapping from doc text to original ID
        doc_to_id = {doc_text: doc_id for doc_id, doc_text in zip(candidate_doc_ids, docs_list, strict=False)}

        # Rerank
        ranked_results = hybrid.rerank(query, docs_list)
        # Map ranked results back to IDs using doc content
        ranked_doc_ids = [doc_to_id[result.doc] for result in ranked_results if result.doc in doc_to_id]

        # Calculate DCG@10
        dcg = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1):
            relevance = relevant_docs.get(doc_id, 0)
            dcg += relevance / np.log2(rank + 1)

        # Calculate IDCG@10 (ideal ranking)
        ideal_relevances = sorted(
            [relevant_docs.get(did, 0) for did in candidate_doc_ids],
            reverse=True
        )[:top_k]
        idcg = sum(
            rel / np.log2(rank + 1)
            for rank, rel in enumerate(ideal_relevances, start=1)
        )

        # NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
        evaluated_queries += 1

    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0

    print(f"  NDCG@10: {avg_ndcg:.4f}")
    print(f"  Queries evaluated: {evaluated_queries}")

    return {
        "ndcg_at_10": avg_ndcg,
        "avg_latency_ms": avg_latency_ms,
        "num_queries": evaluated_queries,
    }


def main() -> None:
    """Main entry point for ensemble distillation pipeline."""
    try:
        args = parse_args()
        settings = get_settings()

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
