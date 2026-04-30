"""CLI entrypoint for evaluating reranking strategies."""

from __future__ import annotations

import argparse
from pathlib import Path

from reranker.config import get_settings
from reranker.eval.runner import evaluate_strategy


def main() -> None:
    """CLI entrypoint: parse args and run evaluation for a given strategy."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate a reranking strategy.")
    parser.add_argument(
        "--strategy",
        choices=[
            "hybrid",
            "distilled",
            "consistency",
            "late_interaction",
            "binary_reranker",
            "splade",
            "multi",
        ],
        required=True,
    )
    parser.add_argument("--split", default=settings.eval.default_split)
    parser.add_argument("--data-root", default=str(settings.paths.raw_data_dir))
    parser.add_argument("--model-root", default=str(settings.paths.model_dir))
    args = parser.parse_args()

    report = evaluate_strategy(
        strategy=args.strategy,
        split=args.split,
        data_root=Path(args.data_root),
        model_root=Path(args.model_root),
    )
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
