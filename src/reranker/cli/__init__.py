from __future__ import annotations

import typer

from reranker.cli.benchmark import benchmark_app
from reranker.cli.doctor import doctor_app
from reranker.cli.eval import eval_app
from reranker.cli.generate import generate_app
from reranker.cli.serve import serve_app
from reranker.cli.train import train_app
from reranker.logging_config import configure_logging

configure_logging()

app = typer.Typer(
    name="reranker",
    help="CPU-native reranking library — train, evaluate, benchmark, and serve rerankers.",
    no_args_is_help=True,
)

app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(benchmark_app, name="benchmark")
app.add_typer(generate_app, name="generate")
app.add_typer(serve_app, name="serve")
app.add_typer(doctor_app, name="doctor")
