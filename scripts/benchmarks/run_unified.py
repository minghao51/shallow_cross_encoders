"""CLI wrapper for the unified benchmark runner.

Usage:
    uv run scripts/benchmarks/run_unified.py
    uv run scripts/benchmarks/run_unified.py --phases baselines ablations
    uv run scripts/benchmarks/run_unified.py --quick
    uv run scripts/benchmarks/run_unified.py --output-dir .benchmarks/
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from reranker.benchmark import BenchmarkRunner
from reranker.config import get_settings


def main():
    parser = argparse.ArgumentParser(description="Unified benchmark for all reranking strategies")
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["baselines", "ablations", "scaling", "embedder-comparison"],
        default=["baselines", "ablations", "scaling", "embedder-comparison"],
        help="Which phases to run (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller sample sizes for faster runs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/benchmark_results",
        help="Where to save results (default: docs/benchmark_results)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--embedder-model",
        type=str,
        default=None,
        help="Override embedder model name (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="transformers")
    warnings.filterwarnings("ignore", module="torch")

    settings = get_settings()
    data_root = Path(args.data_root) if args.data_root else Path(settings.paths.raw_data_dir)
    model_root = Path(settings.paths.model_dir)
    output_dir = Path(args.output_dir)

    runner = BenchmarkRunner(
        data_root=data_root,
        model_root=model_root,
        seed=args.seed,
        embedder_model=args.embedder_model,
        quick=args.quick,
    )

    print(f"Embedder model: {runner.embedder_model_name}")
    print(f"Quick mode: {runner.quick}")
    print(f"Phases: {args.phases}")
    print()

    if "baselines" in args.phases:
        runner.run_baselines()

    if "ablations" in args.phases:
        runner.run_ablations()

    if "scaling" in args.phases:
        runner.run_scaling()

    if "embedder-comparison" in args.phases:
        runner.run_embedder_comparison()

    runner.save_results(output_dir)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
