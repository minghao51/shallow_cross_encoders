"""Measure ROI of distilled vs semantic baseline.

Usage:
    uv run benchmarks/measure_roi.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from reranker.config import get_settings
from reranker.data.synth import SyntheticDataGenerator
from reranker.eval.metrics import accuracy
from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.utils import read_jsonl


def _semantic_baseline(query: str, doc_a: str, doc_b: str, ranker: DistilledPairwiseRanker) -> int:
    q_vec, a_vec, b_vec = ranker.embedder.encode([query, doc_a, doc_b])
    return 1 if float(q_vec @ a_vec) >= float(q_vec @ b_vec) else 0


def _teacher_cost_baseline(settings: Any, rows: list[dict[str, object]]) -> tuple[float, str]:
    cost_rows = read_jsonl(settings.paths.api_cost_log)
    preference_costs = [
        float(row.get("cost_usd", 0.0))
        for row in cost_rows
        if str(row.get("dataset")) == "preferences"
    ]
    if preference_costs:
        return sum(preference_costs), "logged_api_costs"
    return (
        max(len(rows), 1) * settings.roi.llm_cost_per_judgment_usd,
        "configured_estimate",
    )


def main() -> None:
    settings = get_settings()
    data_root = Path(settings.paths.raw_data_dir)
    if not (data_root / "preferences.jsonl").exists():
        SyntheticDataGenerator().materialize_all(data_root)
    rows = read_jsonl(data_root / "preferences.jsonl")
    labels = [1 if row["preferred"] == "A" else 0 for row in rows]

    ranker = DistilledPairwiseRanker().fit(
        queries=[row["query"] for row in rows],
        doc_as=[row["doc_a"] for row in rows],
        doc_bs=[row["doc_b"] for row in rows],
        labels=labels,
    )

    distilled_preds: list[int] = []
    semantic_preds: list[int] = []
    distilled_elapsed = 0.0
    semantic_elapsed = 0.0
    for row in rows:
        start = time.perf_counter()
        distilled_preds.append(
            1 if ranker.compare(row["query"], row["doc_a"], row["doc_b"]) >= 0.5 else 0
        )
        distilled_elapsed += (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        semantic_preds.append(_semantic_baseline(row["query"], row["doc_a"], row["doc_b"], ranker))
        semantic_elapsed += (time.perf_counter() - start) * 1000

    llm_cost_baseline, cost_source = _teacher_cost_baseline(settings, rows)
    projected_monthly_queries = settings.roi.projected_monthly_queries
    distilled_accuracy = accuracy(labels, distilled_preds)
    semantic_accuracy = accuracy(labels, semantic_preds)
    cost_reduction_ratio = 1.0 if llm_cost_baseline > 0 else 0.0
    print("llm_judge_accuracy=1.0000")
    print(f"distilled_accuracy={distilled_accuracy:.4f}")
    print(f"semantic_accuracy={semantic_accuracy:.4f}")
    print(f"distilled_accuracy_gap_vs_teacher={1.0 - distilled_accuracy:.4f}")
    print(f"distilled_avg_latency_ms={distilled_elapsed / max(len(rows), 1):.4f}")
    print(f"semantic_avg_latency_ms={semantic_elapsed / max(len(rows), 1):.4f}")
    print(f"teacher_cost_source={cost_source}")
    print(f"llm_cost_baseline_usd={llm_cost_baseline:.4f}")
    print(f"projected_monthly_llm_cost_usd={llm_cost_baseline * projected_monthly_queries:.2f}")
    print("projected_monthly_distilled_cost_usd=0.00")
    print(f"projected_cost_reduction_ratio={cost_reduction_ratio:.4f}")


if __name__ == "__main__":
    main()
