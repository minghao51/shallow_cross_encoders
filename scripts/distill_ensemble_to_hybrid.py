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
import traceback
from pathlib import Path

import structlog

from reranker.data.distill_pipeline import (
    evaluate_hybrid,
    generate_ensemble_labels,
    load_training_data,
    train_hybrid_pairwise,
    train_hybrid_pointwise,
)
from reranker.data.ensemble_cache import EnsembleLabelCache
from reranker.strategies.flashrank_ensemble import FlashRankEnsemble
from reranker.strategies.hybrid import HybridFusionReranker

logger = structlog.get_logger(__name__)


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

    if args.dataset == "custom" and not args.custom_path:
        parser.error("--custom-path is required when --dataset=custom")

    return args


def main() -> None:
    """Main entry point for ensemble distillation pipeline."""
    try:
        args = parse_args()

        logger.info(f"Teachers: {', '.join(args.teachers)}")
        logger.info(f"Dataset: {args.dataset}")
        if args.custom_path:
            logger.info(f"Custom path: {args.custom_path}")
        logger.info(f"Method: {args.method}")
        logger.info(f"Force regenerate: {args.force_regenerate}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Cache dir: {args.cache_dir}")

        ensemble = FlashRankEnsemble(args.teachers)
        cache = EnsembleLabelCache(args.cache_dir)

        logger.info(f"\nLoading {args.dataset} dataset...")
        queries_dict, corpus_dict, qrels = load_training_data(args.dataset, args.custom_path)

        query_ids = sorted(queries_dict.keys())[:50]
        doc_ids = sorted(corpus_dict.keys())[:500]

        queries = [queries_dict[qid] for qid in query_ids]
        corpus_docs = [
            f"{corpus_dict[did]['title']} {corpus_dict[did]['text']}".strip() for did in doc_ids
        ]

        filtered_qrels = {}
        for qid in query_ids:
            if qid in qrels:
                filtered_qrels[qid] = {
                    did: score for did, score in qrels[qid].items() if did in doc_ids
                }

        logger.info(f"Loaded {len(queries)} queries, {len(corpus_docs)} documents")

        logger.info("\nGenerating ensemble teacher labels...")
        labels = generate_ensemble_labels(
            ensemble=ensemble,
            queries=queries,
            corpus_docs=corpus_docs,
            qrels=filtered_qrels,
            cache=cache,
            force_regenerate=args.force_regenerate,
        )

        logger.info(f"Generated {len(labels)} query-document pair scores")

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
            logger.info(f"\nMethod '{args.method}' not yet implemented.")
            return

        logger.info("\n" + "=" * 60)
        logger.info("LOADING TRAINED MODEL FOR EVALUATION")
        logger.info("=" * 60)
        hybrid = HybridFusionReranker.load(args.output)
        logger.info(f"Model loaded from {args.output}")

        eval_results = evaluate_hybrid(
            hybrid=hybrid,
            queries=queries_dict,
            docs=corpus_dict,
            qrels=qrels,
            top_k=10,
        )

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"NDCG@10:              {eval_results['ndcg_at_10']:.4f}")
        logger.info(f"Avg Latency:          {eval_results['avg_latency_ms']:.2f} ms")
        logger.info(f"Queries Evaluated:    {eval_results['num_queries']}")
        logger.info("=" * 60)

    except ImportError as e:
        logger.info(f"ImportError: {e}")
        error_msg = str(e).lower()
        if "flashrank" in error_msg:
            logger.info("Install: uv pip install flashrank")
        elif "beir" in error_msg:
            logger.info("Install: uv pip install beir --no-deps && uv pip install rank-bm25 pyyaml")
        else:
            logger.info("Install dependencies: uv sync --extra flashrank")
        sys.exit(1)
    except Exception as e:
        logger.info(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
