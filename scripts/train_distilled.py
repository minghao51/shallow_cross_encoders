from __future__ import annotations

from pathlib import Path

from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.data.synth import SyntheticDataGenerator
from reranker.eval.runner import evaluate_strategy
from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.utils import read_jsonl


def main() -> None:
    settings = get_settings()
    data_root = Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not (data_root / "preferences.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)

    rows = read_jsonl(data_root / "preferences.jsonl")
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
    if len({1 if row["preferred"] == "A" else 0 for row in train_rows}) < 2:
        train_rows = rows
    labels = [1 if row["preferred"] == "A" else 0 for row in train_rows]
    ranker = DistilledPairwiseRanker().fit(
        queries=[row["query"] for row in train_rows],
        doc_as=[row["doc_a"] for row in train_rows],
        doc_bs=[row["doc_b"] for row in train_rows],
        labels=labels,
    )
    model_path = Path(settings.paths.model_dir / "pairwise_ranker.pkl")
    ranker.save(model_path)
    validation_report = evaluate_strategy(
        "distilled",
        "validation",
        data_root,
        settings.paths.model_dir,
    )
    test_report = evaluate_strategy("distilled", "test", data_root, settings.paths.model_dir)
    print(f"saved_model={model_path}")
    print(f"train_rows={len(train_rows)}")
    print(f"validation_accuracy={validation_report['accuracy']}")
    print(f"test_accuracy={test_report['accuracy']}")


if __name__ == "__main__":
    main()
