# ruff: noqa: B008
from __future__ import annotations

from pathlib import Path

import typer

benchmark_app = typer.Typer(help="Run benchmarks.", no_args_is_help=True)


@benchmark_app.command("run")
def benchmark_run(
    quick: bool = typer.Option(False, "--quick", help="Run a reduced benchmark suite."),
    output: Path | None = typer.Option(None, "--output", help="Directory to save results."),
    config: Path | None = typer.Option(None, "--config", help="YAML sweep config path."),
    profile: bool = typer.Option(False, "--profile", help="Enable memory/CPU profiling."),
) -> None:
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    from benchmarks.runner import BenchmarkRunner
    from reranker.config import get_settings

    settings = get_settings()
    output_dir = output or Path("benchmarks/results")
    runner = BenchmarkRunner(
        data_root=Path(settings.paths.raw_data_dir),
        model_root=Path(settings.paths.model_dir),
        quick=quick,
        profiling_enabled=profile,
    )

    if config is not None:
        from benchmarks.run_sweep import print_comparison_table, run_sweep

        results = run_sweep(str(config))
        print_comparison_table(results)
        return

    runner.run_baselines()
    if not quick:
        runner.run_ablations()
        runner.run_scaling()
        runner.run_embedder_comparison()
    runner.save_results(output_dir)
    typer.echo(f"Results saved to {output_dir}")


@benchmark_app.command("sweep")
def benchmark_sweep(
    config: Path = typer.Option(..., "--config", help="Path to YAML sweep config."),
    output: Path | None = typer.Option(None, "--output", help="Path to save results JSON."),
) -> None:
    from benchmarks.run_sweep import print_comparison_table, run_sweep

    results = run_sweep(str(config))
    print_comparison_table(results)
    if output is not None:
        import json

        data = [
            {"variant": r.variant_name, "metrics": r.metrics, "latency_ms": r.latency_ms}
            for r in results
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2))
        typer.echo(f"Results saved to {output}")


@benchmark_app.command("full")
def benchmark_full(
    quick: bool = typer.Option(False, "--quick", help="Skip slow benchmarks."),
) -> None:
    import sys

    from benchmarks.run import main as run_main

    args = ["benchmarks/run.py", "full"]
    if quick:
        args.append("--quick")
    sys.argv = args
    run_main()


@benchmark_app.command("compare")
def benchmark_compare(
    strategy_a: str = typer.Argument(..., help="First strategy name."),
    strategy_b: str = typer.Argument(..., help="Second strategy name."),
    results_file: Path = typer.Option(
        ..., "--results", help="Path to benchmark_results.json from a previous run."
    ),
    metric: str = typer.Option("ndcg@10", "--metric", help="Metric key to compare."),
) -> None:
    """Compare two strategies with bootstrap CIs and Wilcoxon signed-rank test."""
    import json

    from reranker.eval.statistics import compare_strategies

    if not results_file.exists():
        typer.echo(f"Error: results file not found: {results_file}", err=True)
        raise typer.Exit(1)

    with open(results_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    scores_a: list[float] = []
    scores_b: list[float] = []
    per_query_metric_key = f"per_query_{metric}"

    for r in results:
        strat = r.get("strategy", "")
        per_query_metrics = r.get("per_query_metrics", {})
        if not isinstance(per_query_metrics, dict):
            per_query_metrics = {}
        values = per_query_metrics.get(per_query_metric_key)
        if not isinstance(values, list) or not values:
            continue
        if strat == strategy_a:
            scores_a.extend(float(v) for v in values)
        if strat == strategy_b:
            scores_b.extend(float(v) for v in values)

    if not scores_a:
        typer.echo(
            f"Error: no paired per-query metric '{per_query_metric_key}' found for strategy "
            f"'{strategy_a}'.",
            err=True,
        )
        raise typer.Exit(1)
    if not scores_b:
        typer.echo(
            f"Error: no paired per-query metric '{per_query_metric_key}' found for strategy "
            f"'{strategy_b}'.",
            err=True,
        )
        raise typer.Exit(1)
    if len(scores_a) != len(scores_b):
        typer.echo(
            "Error: paired per-query metric vectors have different lengths "
            f"({len(scores_a)} vs {len(scores_b)}).",
            err=True,
        )
        raise typer.Exit(1)

    comparison = compare_strategies(
        name_a=strategy_a,
        per_query_a=scores_a,
        name_b=strategy_b,
        per_query_b=scores_b,
        metric_name=metric,
    )

    typer.echo(f"\n  Comparison: {strategy_a} vs {strategy_b}")
    typer.echo(f"  Metric:     {metric}")
    typer.echo(f"  {strategy_a} mean:  {comparison[f'{strategy_a}_mean']:.4f}")
    ci_a = comparison[f"{strategy_a}_ci_95"]
    typer.echo(f"  {strategy_a} 95% CI: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    typer.echo(f"  {strategy_b} mean:  {comparison[f'{strategy_b}_mean']:.4f}")
    ci_b = comparison[f"{strategy_b}_ci_95"]
    typer.echo(f"  {strategy_b} 95% CI: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
    typer.echo(f"  Delta ({strategy_a} - {strategy_b}): {comparison['delta']:+.4f}")
    typer.echo(f"  CI overlap:    {comparison['ci_overlap']}")
    typer.echo(f"  Wilcoxon p:    {comparison['wilcoxon_p_value']:.4f}")
    typer.echo(f"  Significant:   {comparison['significant_at_005']}")
