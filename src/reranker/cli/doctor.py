from __future__ import annotations

import typer

doctor_app = typer.Typer(help="Diagnose dependency and configuration issues.", no_args_is_help=True)


@doctor_app.command("check")
def doctor_check() -> None:
    from importlib.util import find_spec

    from reranker.config import get_settings
    from reranker.deps import check_model2vec, check_xgboost

    typer.echo(" reranker doctor")
    typer.echo("=" * 50)

    checks = [
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("pydantic", "pydantic"),
        ("model2vec", "model2vec"),
        ("rank_bm25", "rank_bm25"),
        ("xgboost", "xgboost"),
        ("sentence-transformers", "sentence_transformers"),
        ("cachetools", "cachetools"),
        ("typer", "typer"),
    ]

    ok = True
    for name, module in checks:
        available = find_spec(module) is not None
        status = "OK" if available else "MISSING"
        if not available:
            ok = False
        typer.echo(f"  [{status:^7s}] {name}")

    typer.echo("")
    _, m2v_status = check_model2vec()
    typer.echo(f"  Embedder backend: {m2v_status.backend}")
    _, xgb_status = check_xgboost()
    typer.echo(f"  GBDT backend: {xgb_status.backend}")

    settings = get_settings()
    data_dir = settings.paths.raw_data_dir
    model_dir = settings.paths.model_dir
    typer.echo("")
    typer.echo(f"  Data dir: {data_dir} ({'exists' if data_dir.exists() else 'missing'})")
    typer.echo(f"  Model dir: {model_dir} ({'exists' if model_dir.exists() else 'missing'})")

    if data_dir.exists():
        pairs = data_dir / "pairs.jsonl"
        prefs = data_dir / "preferences.jsonl"
        typer.echo(f"  pairs.jsonl: {'exists' if pairs.exists() else 'missing'}")
        typer.echo(f"  preferences.jsonl: {'exists' if prefs.exists() else 'missing'}")

    typer.echo("")
    if ok:
        typer.echo("  All dependencies available.")
    else:
        typer.echo("  Some dependencies missing. Install with: uv sync", err=True)
