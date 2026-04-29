from __future__ import annotations

import os
import sys
from pathlib import Path

from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.data.synth import SyntheticDataGenerator
from reranker.eval.runner import evaluate_strategy
from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.utils import read_jsonl


def _prompt_user(prompt: str) -> bool:
    while True:
        response = input(f"{prompt} [y/N] ").strip().lower()
        if response in ("y", "yes"):
            return True
        if response in ("", "n", "no"):
            return False
        print("Please answer 'y' or 'n'")


def _should_generate_synthetic_data(prompt: str) -> bool:
    override = os.environ.get("RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA")
    if override is not None:
        normalized = override.strip().lower()
        if normalized in ("1", "true", "y", "yes"):
            return True
        if normalized in ("0", "false", "n", "no", ""):
            return False
        raise ValueError(
            "RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA must be one of: 1, true, yes, y, 0, false, no, n"
        )
    if not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
        return False
    return _prompt_user(prompt)


def main() -> None:
    settings = get_settings()
    synth_settings = settings.synthetic_data
    data_root = Path(settings.paths.raw_data_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    pref_path = data_root / "preferences.jsonl"
    if not pref_path.exists():
        estimated_pairs = synth_settings.pair_count
        estimated_prefs = synth_settings.preference_count
        estimated_contradictions = synth_settings.contradiction_count + synth_settings.control_count
        total_records = estimated_pairs + estimated_prefs + estimated_contradictions

        cost_per_record = settings.roi.llm_cost_per_judgment_usd
        estimated_cost = total_records * cost_per_record

        print(f"WARNING: Synthetic data not found at {pref_path}")
        print(f"  Estimated records to generate: {total_records}")
        print(f"  - Pairs: {estimated_pairs}")
        print(f"  - Preferences: {estimated_prefs}")
        print(f"  - Contradictions/Controls: {estimated_contradictions}")
        print(f"  - Estimated cost: ${estimated_cost:.4f} (at ${cost_per_record}/record)")
        print()

        if not _should_generate_synthetic_data("Proceed with synthetic data generation?"):
            print("Aborted. Run with existing data or generate manually.")
            return

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
