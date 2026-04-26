"""Single entry point for all benchmarks.

Usage:
    uv run scripts/benchmarks/run_all.py sweep --config benchmarks/full_sweep.yaml
    uv run scripts/benchmarks/run_all.py unified --phases baselines ablations
    uv run scripts/benchmarks/run_all.py beir --dataset scifact
    uv run scripts/benchmarks/run_all.py trec
    uv run scripts/benchmarks/run_all.py flashrank
    uv run scripts/benchmarks/run_all.py real-data
    uv run scripts/benchmarks/run_all.py roi
    uv run scripts/benchmarks/run_all.py full
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARKS_DIR.parent.parent
RESULTS_DIR = PROJECT_ROOT / ".benchmarks"

SWEEP_CONFIGS = [
    "benchmarks/sweep_hybrid.yaml",
    "benchmarks/sweep_colbert.yaml",
    "benchmarks/sweep_lsh.yaml",
    "benchmarks/sweep_active_distill.yaml",
    "benchmarks/full_sweep.yaml",
]


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - start
    ok = result.returncode == 0
    status = "OK" if ok else "FAILED"
    print(f"\n  [{status}] {label} ({elapsed:.1f}s)")
    return ok


def cmd_sweep(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = args.config
    name = Path(config).stem
    output = args.output or str(RESULTS_DIR / f"{name}.json")
    _run(
        ["uv", "run", "scripts/benchmarks/run_sweep.py", "--config", config, "--output", output],
        f"Sweep: {name}",
    )


def cmd_unified(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", "scripts/benchmarks/run_unified.py", "--output-dir", str(RESULTS_DIR)]
    if args.phases:
        cmd.extend(["--phases"] + args.phases)
    if args.quick:
        cmd.append("--quick")
    _run(cmd, "Unified benchmark")


def cmd_beir(args) -> None:
    _run(
        ["uv", "run", "scripts/benchmarks/run_beir.py", args.dataset],
        f"BEIR benchmark: {args.dataset}",
    )


def cmd_trec(args) -> None:
    _run(["uv", "run", "scripts/benchmarks/run_trec.py"], "TREC COVID benchmark")


def cmd_flashrank(args) -> None:
    _run(["uv", "run", "scripts/benchmarks/run_flashrank.py"], "FlashRank benchmark")


def cmd_real_data(args) -> None:
    _run(["uv", "run", "scripts/benchmarks/run_real_data.py"], "Real data benchmark (MS-MARCO)")


def cmd_roi(args) -> None:
    _run(["uv", "run", "scripts/benchmarks/measure_roi.py"], "ROI measurement")


def cmd_full(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}

    print(f"\n{'#' * 70}")
    print("  FULL BENCHMARK SWEEP")
    print(f"{'#' * 70}")

    # Phase 1: YAML sweeps
    print("\n\n===== PHASE 1: YAML SWEEPS =====\n")
    for config_path in SWEEP_CONFIGS:
        full_path = PROJECT_ROOT / config_path
        if not full_path.exists():
            print(f"  Skipping {config_path} (not found)")
            continue
        name = Path(config_path).stem
        output = str(RESULTS_DIR / f"{name}.json")
        ok = _run(
            [
                "uv", "run", "scripts/benchmarks/run_sweep.py",
                "--config", config_path, "--output", output,
            ],
            f"Sweep: {name}",
        )
        results[f"sweep/{name}"] = "ok" if ok else "failed"

    # Phase 2: Unified benchmark
    print("\n\n===== PHASE 2: UNIFIED BENCHMARK =====\n")
    ok = _run(
        ["uv", "run", "scripts/benchmarks/run_unified.py", "--output-dir", str(RESULTS_DIR)],
        "Unified benchmark",
    )
    results["unified"] = "ok" if ok else "failed"

    # Phase 3: External baselines
    print("\n\n===== PHASE 3: EXTERNAL BASELINES =====\n")
    for label, cmd in [
        ("TREC COVID", ["uv", "run", "scripts/benchmarks/run_trec.py"]),
        ("BEIR scifact", ["uv", "run", "scripts/benchmarks/run_beir.py", "scifact"]),
        ("FlashRank", ["uv", "run", "scripts/benchmarks/run_flashrank.py"]),
    ]:
        ok = _run(cmd, label)
        results[f"external/{label}"] = "ok" if ok else "failed"

    # Phase 4: ROI
    print("\n\n===== PHASE 4: ROI =====\n")
    ok = _run(["uv", "run", "scripts/benchmarks/measure_roi.py"], "ROI measurement")
    results["roi"] = "ok" if ok else "failed"

    # Summary
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

    # sweep
    p = subparsers.add_parser("sweep", help="Run YAML sweep config")
    p.add_argument("--config", required=True, help="Path to YAML sweep config")
    p.add_argument("--output", default=None, help="Output JSON path")
    p.set_defaults(func=cmd_sweep)

    # unified
    p = subparsers.add_parser(
        "unified", help="Run unified benchmark (baselines, ablations, scaling)"
    )
    p.add_argument("--phases", nargs="+", default=None)
    p.add_argument("--quick", action="store_true")
    p.set_defaults(func=cmd_unified)

    # beir
    p = subparsers.add_parser("beir", help="Run BEIR dataset benchmark")
    p.add_argument("--dataset", default="scifact")
    p.set_defaults(func=cmd_beir)

    # trec
    p = subparsers.add_parser("trec", help="Run TREC COVID benchmark")
    p.set_defaults(func=cmd_trec)

    # flashrank
    p = subparsers.add_parser("flashrank", help="Run FlashRank comparison benchmark")
    p.set_defaults(func=cmd_flashrank)

    # real-data
    p = subparsers.add_parser("real-data", help="Run MS-MARCO real data benchmark")
    p.set_defaults(func=cmd_real_data)

    # roi
    p = subparsers.add_parser("roi", help="Run ROI measurement")
    p.set_defaults(func=cmd_roi)

    # full
    p = subparsers.add_parser("full", help="Run ALL benchmarks in sequence")
    p.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
