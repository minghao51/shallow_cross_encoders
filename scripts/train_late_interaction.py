from __future__ import annotations

from pathlib import Path

from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.data.synth import SyntheticDataGenerator
from reranker.eval.runner import evaluate_strategy
from reranker.strategies.late_interaction import StaticColBERTReranker
from reranker.utils import read_jsonl


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

    unique_docs = list({row["doc"] for row in train_rows})
    reranker = StaticColBERTReranker()
    reranker.fit(unique_docs)

    model_path = Path(settings.paths.model_dir / "late_interaction_reranker.pkl")
    reranker.save(model_path)
    validation_report = evaluate_strategy(
        "late_interaction",
        "validation",
        data_root,
        settings.paths.model_dir,
    )
    test_report = evaluate_strategy(
        "late_interaction",
        "test",
        data_root,
        settings.paths.model_dir,
    )
    print(f"saved_model={model_path}")
    print(f"train_docs={len(unique_docs)}")
    print(f"validation_ndcg@10={validation_report['ndcg@10']}")
    print(f"validation_bm25_ndcg@10={validation_report['bm25_ndcg@10']}")
    print(f"validation_ndcg@10_uplift_vs_bm25={validation_report['ndcg@10_uplift_vs_bm25']}")
    print(f"test_ndcg@10={test_report['ndcg@10']}")
    print(f"test_bm25_ndcg@10={test_report['bm25_ndcg@10']}")
    print(f"test_ndcg@10_uplift_vs_bm25={test_report['ndcg@10_uplift_vs_bm25']}")


if __name__ == "__main__":
    main()
