# ruff: noqa: E501
"""Single entry point for all benchmarks.

Usage:
    uv run benchmarks/run.py synthetic \
        [--phases baselines ablations scaling embedder-comparison] [--quick]
    uv run benchmarks/run.py sweep \
        --config benchmarks/configs/sweep_hybrid.yaml
    uv run benchmarks/run.py roi
    uv run benchmarks/run.py full [--quick]
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BENCHMARKS_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
RESULTS_DIR = BENCHMARKS_DIR / "results"

SWEEP_CONFIGS = [
    "benchmarks/configs/sweep_hybrid.yaml",
    "benchmarks/configs/sweep_colbert.yaml",
    "benchmarks/configs/sweep_lsh.yaml",
    "benchmarks/configs/sweep_active_distill.yaml",
    "benchmarks/configs/full_sweep.yaml",
]


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)  # nosec B603
    elapsed = time.perf_counter() - start
    ok = result.returncode == 0
    status = "OK" if ok else "FAILED"
    print(f"\n  [{status}] {label} ({elapsed:.1f}s)")
    return ok


def cmd_synthetic(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", module="transformers")
    warnings.filterwarnings("ignore", module="torch")

    from benchmarks.runner import BenchmarkRunner
    from reranker.config import get_settings

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


def cmd_sweep(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = args.config
    name = Path(config).stem
    output = args.output or str(RESULTS_DIR / f"{name}.json")
    _run(
        ["uv", "run", "benchmarks/run_sweep.py", "--config", config, "--output", output],
        f"Sweep: {name}",
    )


def cmd_roi(args: argparse.Namespace) -> None:
    _run(["uv", "run", "benchmarks/measure_roi.py"], "ROI measurement")


def cmd_full(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    print(f"\n{'#' * 70}")
    print("  FULL BENCHMARK SWEEP")
    print(f"{'#' * 70}")

    print("\n\n===== PHASE 1: YAML SWEEPS =====\n")
    for config_path in SWEEP_CONFIGS:
        full_path = PROJECT_ROOT / config_path
        if not full_path.exists():
            print(f"  Skipping {config_path} (not found)")
            continue
        name = Path(config_path).stem
        output = str(RESULTS_DIR / f"{name}.json")
        ok = _run(
            ["uv", "run", "benchmarks/run_sweep.py", "--config", config_path, "--output", output],
            f"Sweep: {name}",
        )
        results[f"sweep/{name}"] = "ok" if ok else "failed"

    print("\n\n===== PHASE 2: COMPREHENSIVE SYNTHETIC =====\n")
    synthetic_cmd = [
        "uv",
        "run",
        "benchmarks/run.py",
        "synthetic",
        "--output-dir",
        str(RESULTS_DIR),
    ]
    if args.quick:
        synthetic_cmd.append("--quick")
    ok = _run(synthetic_cmd, "Comprehensive synthetic benchmark")
    results["synthetic"] = "ok" if ok else "failed"

    print("\n\n===== PHASE 3: ROI =====\n")
    ok = _run(["uv", "run", "benchmarks/measure_roi.py"], "ROI measurement")
    results["roi"] = "ok" if ok else "failed"

    print(f"\n\n{'#' * 70}")
    print("  FULL SWEEP COMPLETE - SUMMARY")
    print(f"{'#' * 70}\n")
    for name, status in results.items():
        marker = "+" if status == "ok" else "x"
        print(f"  [{marker}] {name}: {status}")

    n_ok = sum(1 for s in results.values() if s == "ok")
    print(f"\n  {n_ok}/{len(results)} passed")

    summary_path = RESULTS_DIR / "run_all_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified benchmark entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("synthetic", help="Run comprehensive synthetic benchmark")
    p.add_argument(
        "--phases",
        nargs="+",
        default=["baselines", "ablations", "scaling", "embedder-comparison"],
        choices=["baselines", "ablations", "scaling", "embedder-comparison"],
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--output-dir", type=str, default=str(BENCHMARKS_DIR / "results"))
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--embedder-model", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_synthetic)

    p = subparsers.add_parser("sweep", help="Run YAML sweep config")
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_sweep)

    p = subparsers.add_parser("roi", help="Run ROI measurement")
    p.set_defaults(func=cmd_roi)

    p = subparsers.add_parser("full", help="Run ALL benchmarks in sequence")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
