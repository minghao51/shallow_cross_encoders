# ruff: noqa: B008
from __future__ import annotations

from pathlib import Path

import typer

eval_app = typer.Typer(help="Evaluate a reranking strategy.", no_args_is_help=True)

_STRATEGIES = (
    "hybrid",
    "distilled",
    "consistency",
    "late_interaction",
    "binary_reranker",
    "splade",
    "multi",
)


@eval_app.command("run")
def eval_run(
    strategy: str = typer.Argument(help=f"Strategy to evaluate: {', '.join(_STRATEGIES)}."),
    split: str = typer.Option("test", "--split", help="Data split: train, validation, or test."),
    data_root: Path | None = typer.Option(None, "--dataset", help="Path to data root directory."),
    model_root: Path | None = typer.Option(None, "--model-root", help="Path to model directory."),
    metrics: str = typer.Option(
        "ndcg,map,mrr", "--metrics", help="Comma-separated metrics to report."
    ),
) -> None:
    from reranker.config import get_settings
    from reranker.eval.runner import evaluate_strategy

    settings = get_settings()
    dr = data_root or Path(settings.paths.raw_data_dir)
    mr = model_root or Path(settings.paths.model_dir)

    report = evaluate_strategy(strategy=strategy, split=split, data_root=dr, model_root=mr)
    requested = [m.strip() for m in metrics.split(",")]
    for key, value in report.items():
        metric_prefix = key.split("_")[0] if "_" in key else key
        if any(m in key or m in metric_prefix for m in requested):
            typer.echo(f"{key}: {value}")
