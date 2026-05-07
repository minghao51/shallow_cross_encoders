from __future__ import annotations

from pathlib import Path

import structlog

from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.data.synth import SyntheticDataGenerator
from reranker.heuristics.keyword import KeywordMatchAdapter
from reranker.persistence import save_safe
from reranker.strategies.cascade import CascadeConfig, CascadeReranker
from reranker.strategies.hybrid import HybridFusionReranker
from reranker.utils import read_jsonl

logger = structlog.get_logger(__name__)


def main() -> None:
    settings = get_settings()
    data_root = Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "pairs.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)

    rows = read_jsonl(data_root / "pairs.jsonl")
    ratios = (
        settings.eval.train_ratio,
        settings.eval.validation_ratio,
        settings.eval.test_ratio,
    )
    train_rows = partition_rows(
        rows,
        key_fn=lambda row: str(row["query"]),
        split="train",
        ratios=ratios,
    )
    if len({1 if row["score"] >= 2 else 0 for row in train_rows}) < 2:
        train_rows = rows
    labels = [1 if row["score"] >= 2 else 0 for row in train_rows]

    primary = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit(
        queries=[row["query"] for row in train_rows],
        docs=[row["doc"] for row in train_rows],
        labels=labels,
    )
    fallback = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit(
        queries=[row["query"] for row in train_rows],
        docs=[row["doc"] for row in train_rows],
        labels=labels,
    )

    cascade_config = CascadeConfig(confidence_threshold=0.6)
    cascade = CascadeReranker(primary=primary, fallback=fallback, config=cascade_config)

    model_dir = Path(settings.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "cascade_reranker.pkl"
    primary_path = model_dir / "cascade_primary.pkl"
    fallback_path = model_dir / "cascade_fallback.pkl"
    primary.save(primary_path)
    fallback.save(fallback_path)
    save_safe(
        model_path,
        artifact_type="cascade_reranker",
        metadata={
            "confidence_threshold": cascade_config.confidence_threshold,
            "primary_model": str(primary_path),
            "fallback_model": str(fallback_path),
        },
        weights={},
    )
    print(f"saved_model={model_path}")
    print(f"train_rows={len(train_rows)}")
    print(f"confidence_threshold={cascade_config.confidence_threshold}")

    cascade.rerank("test query", ["doc a", "doc b", "doc c"])
    stats = cascade.get_stats()
    print(f"fallback_rate={stats['fallback_rate']}")


if __name__ == "__main__":
    main()
