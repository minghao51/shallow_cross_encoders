# ruff: noqa: B008
from __future__ import annotations

from pathlib import Path

import typer

train_app = typer.Typer(help="Train a reranking strategy.", no_args_is_help=True)

_STRATEGIES = {
    "hybrid",
    "distilled",
    "binary",
    "late_interaction",
    "cascade",
    "meta_router",
    "splade",
}


@train_app.command("hybrid")
def train_hybrid(
    dataset: Path | None = typer.Option(None, "--dataset", help="Path to data root directory."),
    output: Path | None = typer.Option(None, "--output", help="Path to save trained model."),
    config: Path | None = typer.Option(None, "--config", help="YAML config override."),
) -> None:
    from reranker.config import get_settings
    from reranker.data.splits import partition_rows
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.heuristics.keyword import KeywordMatchAdapter
    from reranker.strategies.hybrid import HybridFusionReranker
    from reranker.utils import read_jsonl

    _apply_config(config)
    settings = get_settings()
    data_root = dataset or Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "pairs.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)

    rows = read_jsonl(data_root / "pairs.jsonl")
    ratios = (settings.eval.train_ratio, settings.eval.validation_ratio, settings.eval.test_ratio)
    train_rows = partition_rows(
        rows, key_fn=lambda row: str(row["query"]), split="train", ratios=ratios
    )
    if len({1 if row["score"] >= 2 else 0 for row in train_rows}) < 2:
        train_rows = rows
    queries = [row["query"] for row in train_rows]
    docs = [row["doc"] for row in train_rows]
    scores = [float(row.get("score", 0)) for row in train_rows]
    reranker = HybridFusionReranker(adapters=[KeywordMatchAdapter()])
    reranker.fit_pointwise(queries=queries, docs=docs, scores=scores)
    model_suffix = ".json" if reranker.model_backend == "xgboost" else ".pkl"
    model_path = output or Path(settings.paths.model_dir / f"hybrid_reranker{model_suffix}")
    reranker.save(model_path)
    _print_train_summary("hybrid", model_path, len(train_rows), data_root, settings.paths.model_dir)


@train_app.command("distilled")
def train_distilled(
    dataset: Path | None = typer.Option(None, "--dataset", help="Path to data root directory."),
    output: Path | None = typer.Option(None, "--output", help="Path to save trained model."),
    config: Path | None = typer.Option(None, "--config", help="YAML config override."),
) -> None:
    from reranker.config import get_settings
    from reranker.data.splits import partition_rows
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.strategies.distilled import DistilledPairwiseRanker
    from reranker.utils import read_jsonl

    _apply_config(config)
    settings = get_settings()
    data_root = dataset or Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "preferences.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)

    rows = read_jsonl(data_root / "preferences.jsonl")
    ratios = (settings.eval.train_ratio, settings.eval.validation_ratio, settings.eval.test_ratio)
    train_rows = partition_rows(
        rows, key_fn=lambda row: str(row["query"]), split="train", ratios=ratios
    )
    if len({1 if row["preferred"] == "A" else 0 for row in train_rows}) < 2:
        train_rows = rows
    labels = [1 if row["preferred"] == "A" else 0 for row in train_rows]
    ranker = DistilledPairwiseRanker().fit(
        queries=[row["query"] for row in train_rows],
        doc_as=[row["doc_a"] for row in train_rows],
        doc_bs=[row["doc_b"] for row in train_rows],
        labels=labels,
    )
    model_path = output or Path(settings.paths.model_dir / "pairwise_ranker.pkl")
    ranker.save(model_path)
    _print_train_summary(
        "distilled", model_path, len(train_rows), data_root, settings.paths.model_dir
    )


@train_app.command("binary")
def train_binary(
    dataset: Path | None = typer.Option(None, "--dataset", help="Path to data root directory."),
    output: Path | None = typer.Option(None, "--output", help="Path to save trained model."),
    config: Path | None = typer.Option(None, "--config", help="YAML config override."),
) -> None:
    from reranker.config import get_settings
    from reranker.data.splits import partition_rows
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.strategies.binary_reranker import BinaryQuantizedReranker
    from reranker.utils import read_jsonl

    _apply_config(config)
    settings = get_settings()
    data_root = dataset or Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "pairs.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)

    rows = read_jsonl(data_root / "pairs.jsonl")
    ratios = (settings.eval.train_ratio, settings.eval.validation_ratio, settings.eval.test_ratio)
    train_rows = partition_rows(
        rows, key_fn=lambda row: str(row["query"]), split="train", ratios=ratios
    )
    if len({1 if row["score"] >= 2 else 0 for row in train_rows}) < 2:
        train_rows = rows
    labels = [1 if row["score"] >= 2 else 0 for row in train_rows]
    reranker = BinaryQuantizedReranker().fit(
        queries=[row["query"] for row in train_rows],
        docs=[row["doc"] for row in train_rows],
        labels=labels,
    )
    model_path = output or Path(settings.paths.model_dir / "binary_reranker.pkl")
    reranker.save(model_path)
    _print_train_summary(
        "binary_reranker", model_path, len(train_rows), data_root, settings.paths.model_dir
    )


@train_app.command("late_interaction")
def train_late_interaction(
    dataset: Path | None = typer.Option(None, "--dataset", help="Path to data root directory."),
    output: Path | None = typer.Option(None, "--output", help="Path to save trained model."),
    config: Path | None = typer.Option(None, "--config", help="YAML config override."),
) -> None:
    from reranker.config import get_settings
    from reranker.data.splits import partition_rows
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.strategies.late_interaction import StaticColBERTReranker
    from reranker.utils import read_jsonl

    _apply_config(config)
    settings = get_settings()
    data_root = dataset or Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "pairs.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)

    rows = read_jsonl(data_root / "pairs.jsonl")
    ratios = (settings.eval.train_ratio, settings.eval.validation_ratio, settings.eval.test_ratio)
    train_rows = partition_rows(
        rows, key_fn=lambda row: str(row["query"]), split="train", ratios=ratios
    )
    if len({1 if row["score"] >= 2 else 0 for row in train_rows}) < 2:
        train_rows = rows

    unique_docs = list({row["doc"] for row in train_rows})
    reranker = StaticColBERTReranker()
    reranker.fit(unique_docs)
    model_path = output or Path(settings.paths.model_dir / "late_interaction_reranker.pkl")
    reranker.save(model_path)
    _print_train_summary(
        "late_interaction", model_path, len(unique_docs), data_root, settings.paths.model_dir
    )


@train_app.command("cascade")
def train_cascade(
    dataset: Path | None = typer.Option(None, "--dataset", help="Path to data root directory."),
    output: Path | None = typer.Option(None, "--output", help="Path to save trained model."),
    config: Path | None = typer.Option(None, "--config", help="YAML config override."),
    confidence_threshold: float = typer.Option(
        0.6, "--threshold", help="Cascade confidence threshold."
    ),
) -> None:
    from reranker.config import get_settings
    from reranker.data.splits import partition_rows
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.heuristics.keyword import KeywordMatchAdapter
    from reranker.strategies.cascade import CascadeConfig, CascadeReranker
    from reranker.strategies.hybrid import HybridFusionReranker
    from reranker.utils import read_jsonl

    _apply_config(config)
    settings = get_settings()
    data_root = dataset or Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "pairs.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)

    rows = read_jsonl(data_root / "pairs.jsonl")
    ratios = (settings.eval.train_ratio, settings.eval.validation_ratio, settings.eval.test_ratio)
    train_rows = partition_rows(
        rows, key_fn=lambda row: str(row["query"]), split="train", ratios=ratios
    )
    if len({1 if row["score"] >= 2 else 0 for row in train_rows}) < 2:
        train_rows = rows
    queries = [row["query"] for row in train_rows]
    docs = [row["doc"] for row in train_rows]
    scores = [float(row.get("score", 0)) for row in train_rows]

    primary = HybridFusionReranker(adapters=[KeywordMatchAdapter()])
    primary.fit_pointwise(queries=queries, docs=docs, scores=scores)
    fallback = HybridFusionReranker(adapters=[KeywordMatchAdapter()])
    fallback.fit_pointwise(queries=queries, docs=docs, scores=scores)

    cascade_config = CascadeConfig(confidence_threshold=confidence_threshold)
    CascadeReranker(primary=primary, fallback=fallback, config=cascade_config)

    model_path = output or Path(settings.paths.model_dir / "cascade_reranker.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path = model_path.parent / "cascade_primary.pkl"
    fallback_path = model_path.parent / "cascade_fallback.pkl"
    primary.save(primary_path)
    fallback.save(fallback_path)

    from reranker.persistence import save_safe

    save_safe(
        model_path,
        artifact_type="cascade_reranker",
        metadata={
            "confidence_threshold": confidence_threshold,
            "primary_model": str(primary_path),
            "fallback_model": str(fallback_path),
        },
        weights={},
    )
    typer.echo(f"saved_model={model_path}")
    typer.echo(f"train_rows={len(train_rows)}")


@train_app.command("meta_router")
def train_meta_router(
    dataset: Path | None = typer.Option(None, "--dataset", help="Path to data root directory."),
    output: Path | None = typer.Option(None, "--output", help="Path to save trained model."),
    config: Path | None = typer.Option(None, "--config", help="YAML config override."),
) -> None:
    from reranker.config import get_settings
    from reranker.data.splits import partition_rows
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.persistence import save_safe
    from reranker.strategies.meta_router import MetaRouter
    from reranker.utils import read_jsonl

    _apply_config(config)
    settings = get_settings()
    data_root = dataset or Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "pairs.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)

    rows = read_jsonl(data_root / "pairs.jsonl")
    ratios = (settings.eval.train_ratio, settings.eval.validation_ratio, settings.eval.test_ratio)
    train_rows = partition_rows(
        rows, key_fn=lambda row: str(row["query"]), split="train", ratios=ratios
    )
    if len(train_rows) < 2:
        train_rows = rows

    queries = [str(row["query"]) for row in train_rows]
    categories = _auto_label_meta_router_categories(train_rows)
    router = MetaRouter()
    router.fit(queries, categories)

    model_path = output or Path(settings.paths.model_dir / "meta_router.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_safe(
        model_path,
        artifact_type="meta_router",
        metadata={
            "embedder_model_name": router.embedder.model_name,
            "n_categories": router.n_categories,
            "min_samples_leaf": router.min_samples_leaf,
        },
        weights={
            "model": router.model,
            "is_fitted": router.is_fitted,
        },
    )
    typer.echo(f"saved_model={model_path}")
    typer.echo(f"train_rows={len(train_rows)}")
    typer.echo(f"categories={len(set(categories))}")


@train_app.command("splade")
def train_splade(
    dataset: Path | None = typer.Option(None, "--dataset", help="Path to data root directory."),
    output: Path | None = typer.Option(None, "--output", help="Path to save trained model."),
    config: Path | None = typer.Option(None, "--config", help="YAML config override."),
    top_k_terms: int = typer.Option(128, "--top-k", help="Number of top terms per doc."),
) -> None:
    from reranker.config import get_settings
    from reranker.data.splits import partition_rows
    from reranker.data.synth import SyntheticDataGenerator
    from reranker.strategies.splade import SPLADEReranker
    from reranker.utils import read_jsonl

    _apply_config(config)
    settings = get_settings()
    data_root = dataset or Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "pairs.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)

    rows = read_jsonl(data_root / "pairs.jsonl")
    ratios = (settings.eval.train_ratio, settings.eval.validation_ratio, settings.eval.test_ratio)
    train_rows = partition_rows(
        rows, key_fn=lambda row: str(row["query"]), split="train", ratios=ratios
    )
    if not train_rows:
        train_rows = rows

    unique_docs = list({row["doc"] for row in train_rows})
    splade = SPLADEReranker(top_k_terms=top_k_terms)
    splade.fit(unique_docs)

    model_path = output or Path(settings.paths.model_dir / "splade_reranker.pkl")
    splade.save(model_path)
    typer.echo(f"saved_model={model_path}")
    typer.echo(f"indexed_docs={len(unique_docs)}")


def _apply_config(config: Path | None) -> None:
    if config is None:
        return
    import yaml

    from reranker.config import apply_settings_override, settings_from_dict

    data = yaml.safe_load(config.read_text())
    apply_settings_override(settings_from_dict(data))


def _print_train_summary(
    strategy: str,
    model_path: Path,
    train_count: int,
    data_root: Path,
    model_root: Path,
) -> None:
    from reranker.eval.runner import evaluate_strategy

    typer.echo(f"saved_model={model_path}")
    typer.echo(f"train_rows={train_count}")
    try:
        report = evaluate_strategy(strategy, "test", data_root, model_root)
        typer.echo(f"test_ndcg@10={report['ndcg@10']:.4f}")
    except Exception as exc:
        typer.echo(f"evaluation_skipped: {exc}", err=True)


def _auto_label_meta_router_categories(rows: list[dict[str, object]]) -> list[int]:
    """Derive stable pseudo-categories for MetaRouter training from query shape."""
    categories: list[int] = []
    for row in rows:
        query = str(row.get("query", ""))
        token_count = len(query.split())
        if token_count <= 2:
            categories.append(0)
        elif token_count <= 6:
            categories.append(1)
        else:
            categories.append(2)
    return categories
