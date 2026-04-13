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


def load_beir_data(dataset_name: str = "nfcorpus") -> tuple[Mapping[str, str], Mapping[str, dict], Mapping[str, dict]]:
    """Load BEIR dataset for distillation.

    Args:
        dataset_name: Name of BEIR dataset (default: "nfcorpus")

    Returns:
        Tuple of (queries, corpus, qrels) where:
            - queries: Mapping from query_id to query text
            - corpus: Mapping from doc_id to doc dict with _id, title, text
            - qrels: Mapping from query_id to {doc_id: relevance_score}
    """
    beir_dir = Path("data/beir") / dataset_name

    # Check if data exists
    if not beir_dir.exists():
        raise FileNotFoundError(
            f"BEIR dataset not found at {beir_dir}. "
            f"Download it first: uv run scripts/download_beir.py {dataset_name}"
        )

    # Parse corpus.jsonl
    corpus_file = beir_dir / "corpus.jsonl"
    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

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
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    queries = {}
    with open(queries_file) as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]

    # Parse qrels/test.tsv
    qrels_file = beir_dir / "qrels" / "test.tsv"
    if not qrels_file.exists():
        raise FileNotFoundError(f"Qrels file not found: {qrels_file}")

    qrels_dict = defaultdict(dict)
    with open(qrels_file) as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                if score > 0:
                    qrels_dict[query_id][doc_id] = score

    qrels = dict(qrels_dict)

    print(f"Loaded {dataset_name}: {len(queries)} queries, {len(corpus)} docs, {len(qrels)} qrels")
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
