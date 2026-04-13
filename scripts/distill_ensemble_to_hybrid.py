"""Ensemble distillation: FlashRank teachers train Hybrid student."""

from __future__ import annotations

import argparse
from pathlib import Path

from reranker.config import get_settings
from reranker.data.ensemble_cache import EnsembleLabelCache
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

    # Pipeline structure complete
    print("\nPipeline structure complete. Implementation continues...")


if __name__ == "__main__":
    main()
