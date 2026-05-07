# ruff: noqa: B008
from __future__ import annotations

from pathlib import Path

import typer

generate_app = typer.Typer(help="Generate synthetic training data.", no_args_is_help=True)


@generate_app.command("pairs")
def generate_pairs(
    count: int = typer.Option(100, "--count", "-n", help="Number of pairs to generate."),
    output: Path | None = typer.Option(None, "--output", help="Output file path."),
    seed: int = typer.Option(42, "--seed", help="Reproducibility seed."),
    teacher: bool = typer.Option(False, "--teacher", help="Use LLM teacher for generation."),
) -> None:
    from reranker.config import get_settings
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.utils import write_jsonl

    generator = SyntheticDataGenerator(seed=seed)
    pairs = generator.generate_pairs(target_count=count, use_teacher=teacher)
    settings = get_settings()
    out = output or settings.paths.raw_data_dir / "pairs.jsonl"
    write_jsonl(out, pairs)
    typer.echo(f"Generated {len(pairs)} pairs -> {out}")


@generate_app.command("preferences")
def generate_preferences(
    count: int = typer.Option(100, "--count", "-n", help="Number of preferences to generate."),
    output: Path | None = typer.Option(None, "--output", help="Output file path."),
    seed: int = typer.Option(42, "--seed", help="Reproducibility seed."),
    teacher: bool = typer.Option(False, "--teacher", help="Use LLM teacher for generation."),
) -> None:
    from reranker.config import get_settings
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.utils import write_jsonl

    generator = SyntheticDataGenerator(seed=seed)
    pairs = generator.generate_pairs(target_count=count, use_teacher=teacher)
    prefs = generator.generate_preferences(pairs, target_count=count, use_teacher=teacher)
    settings = get_settings()
    out = output or settings.paths.raw_data_dir / "preferences.jsonl"
    write_jsonl(out, prefs)
    typer.echo(f"Generated {len(prefs)} preferences -> {out}")


@generate_app.command("contradictions")
def generate_contradictions(
    count: int = typer.Option(50, "--count", "-n", help="Number of contradictions to generate."),
    output: Path | None = typer.Option(None, "--output", help="Output file path."),
    seed: int = typer.Option(42, "--seed", help="Reproducibility seed."),
    teacher: bool = typer.Option(False, "--teacher", help="Use LLM teacher for generation."),
) -> None:
    from reranker.config import get_settings
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.utils import write_jsonl

    generator = SyntheticDataGenerator(seed=seed)
    contradictions = generator.generate_contradictions(
        contradiction_count=count, use_teacher=teacher
    )
    settings = get_settings()
    out = output or settings.paths.raw_data_dir / "contradictions.jsonl"
    write_jsonl(out, contradictions)
    typer.echo(f"Generated {len(contradictions)} contradictions -> {out}")
