from __future__ import annotations

import typer

serve_app = typer.Typer(
    help="Serve rerankers via REST API (Phase 11 preview).", no_args_is_help=True
)


@serve_app.command("start")
def serve_start(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
) -> None:
    typer.echo(
        f"REST API server is planned for Phase 11. "
        f"Configured for {host}:{port}. "
        f"Use 'uv run python -m reranker.eval' for CLI evaluation in the meantime.",
        err=True,
    )
    raise SystemExit(1)
