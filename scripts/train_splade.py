from __future__ import annotations

from pathlib import Path

import structlog

from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.data.synth import SyntheticDataGenerator
from reranker.strategies.splade import SPLADEReranker
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
    if not train_rows:
        train_rows = rows

    unique_docs = list({row["doc"] for row in train_rows})
    splade = SPLADEReranker(top_k_terms=128)
    splade.fit(unique_docs)

    model_dir = Path(settings.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "splade_reranker.pkl"
    splade.save(model_path)
    print(f"saved_model={model_path}")
    print(f"indexed_docs={len(unique_docs)}")


if __name__ == "__main__":
    main()
