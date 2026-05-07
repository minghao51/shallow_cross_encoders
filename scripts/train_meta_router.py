from __future__ import annotations

import pickle
from pathlib import Path

import structlog

from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.data.synth import SyntheticDataGenerator
from reranker.strategies.hybrid import HybridFusionReranker
from reranker.strategies.meta_router import MetaRouter
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
    if len(train_rows) < 2:
        train_rows = rows

    queries = [row["query"] for row in train_rows]
    docs = [row["doc"] for row in train_rows]
    scores = [float(row.get("score", 0)) for row in train_rows]

    reranker = HybridFusionReranker()
    categories = reranker._auto_label_queries(queries, docs, scores)
    router = MetaRouter()
    router.fit(queries, categories)

    model_dir = Path(settings.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "meta_router.pkl"
    model_path.write_bytes(pickle.dumps(router))
    print(f"saved_model={model_path}")
    print(f"train_rows={len(train_rows)}")
    print(f"categories={len(set(categories))}")


if __name__ == "__main__":
    main()
