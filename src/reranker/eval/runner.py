from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.data.synth import SyntheticDataGenerator
from reranker.eval.metrics import (
    LatencyTracker,
    accuracy,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
)
from reranker.lexical import BM25Engine
from reranker.protocols import BaseReranker
from reranker.strategies.binary_reranker import BinaryQuantizedReranker
from reranker.strategies.consistency import ConsistencyEngine
from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter
from reranker.strategies.late_interaction import StaticColBERTReranker
from reranker.strategies.multi import MultiReranker, MultiRerankerConfig
from reranker.strategies.splade import SPLADEReranker
from reranker.utils import read_jsonl


def _ensure_sample_data(data_root: Path) -> None:
    if (data_root / "pairs.jsonl").exists():
        return
    SyntheticDataGenerator().materialize_all(data_root, use_teacher=False)


def _split_ratios() -> tuple[float, float, float]:
    settings = get_settings()
    return (
        settings.eval.train_ratio,
        settings.eval.validation_ratio,
        settings.eval.test_ratio,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _hybrid_model_path(model_root: Path) -> Path:
    pickle_path = model_root / "hybrid_reranker.pkl"
    if pickle_path.exists():
        return pickle_path
    json_path = model_root / "hybrid_reranker.json"
    if json_path.exists():
        return json_path
    return pickle_path


def _ensure_train_rows(
    rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    label_fn: Callable[[dict[str, Any]], int],
) -> list[dict[str, Any]]:
    labels = {label_fn(row) for row in train_rows}
    if train_rows and len(labels) > 1:
        return train_rows
    return rows


def _metrics_for_rows(
    reranker: BaseReranker,
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    latency = LatencyTracker()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query"]), []).append(row)

    ndcgs: list[float] = []
    bm25_ndcgs: list[float] = []
    mrr_inputs: list[list[float]] = []
    map_inputs: list[list[float]] = []
    p1s: list[float] = []
    lexical = BM25Engine()
    for query, items in grouped.items():
        docs = [str(item["doc"]) for item in items]
        with latency.measure():
            ranked = reranker.rerank(query, docs)

        label_map = {str(item["doc"]): int(item["score"]) for item in items}
        relevances: list[float] = [float(label_map[result.doc]) for result in ranked]
        binary = [1 if rel >= 2 else 0 for rel in relevances]
        ndcgs.append(ndcg_at_k(relevances, 10))
        mrr_inputs.append(relevances)
        map_inputs.append(relevances)
        p1s.append(precision_at_k(binary, 1))

        lexical.fit(docs)
        baseline_scores = lexical.score(query)
        baseline_ranked = sorted(
            zip(docs, baseline_scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        baseline_relevances: list[float] = [float(label_map[doc]) for doc, _ in baseline_ranked]
        bm25_ndcgs.append(ndcg_at_k(baseline_relevances, 10))

    summary = latency.summary()
    hybrid_ndcg = _mean(ndcgs)
    bm25_ndcg = _mean(bm25_ndcgs)
    return {
        "ndcg@10": round(hybrid_ndcg, 4),
        "bm25_ndcg@10": round(bm25_ndcg, 4),
        "ndcg@10_uplift_vs_bm25": round(hybrid_ndcg - bm25_ndcg, 4),
        "mrr": round(mrr(mrr_inputs, k=10), 4),
        "map": round(mean_average_precision(map_inputs, k=10), 4),
        "p@1": round(_mean(p1s), 4),
        "latency_p50_ms": round(summary["p50"], 4),
        "latency_p99_ms": round(summary["p99"], 4),
    }


def evaluate_strategy(
    strategy: str,
    split: str,
    data_root: Path,
    model_root: Path,
) -> dict[str, float | str]:
    data_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    _ensure_sample_data(data_root)

    if strategy == "hybrid":
        rows = read_jsonl(data_root / "pairs.jsonl")
        train_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split="train",
            ratios=_split_ratios(),
        )
        train_rows = _ensure_train_rows(
            rows,
            train_rows,
            lambda row: 1 if cast(int, row["score"]) >= 2 else 0,
        )
        eval_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split=split,
            ratios=_split_ratios(),
        )
        if not eval_rows:
            eval_rows = rows
        binary_labels = [1 if cast(int, row["score"]) >= 2 else 0 for row in train_rows]
        model_path = _hybrid_model_path(model_root)
        if model_path.exists():
            reranker = HybridFusionReranker.load(model_path, adapters=[KeywordMatchAdapter()])
        else:
            reranker = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit_pointwise(
                queries=[str(row["query"]) for row in train_rows],
                docs=[str(row["doc"]) for row in train_rows],
                scores=[float(label) for label in binary_labels],
                use_regression=True,
            )
            model_path = model_root / "hybrid_reranker.pkl"
            reranker.save(model_path)
        return {
            "strategy": "hybrid",
            "split": split,
            **_metrics_for_rows(cast(BaseReranker, reranker), eval_rows),
        }

    if strategy == "distilled":
        rows = read_jsonl(data_root / "preferences.jsonl")
        train_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split="train",
            ratios=_split_ratios(),
        )
        train_rows = _ensure_train_rows(
            rows,
            train_rows,
            lambda row: 1 if row["preferred"] == "A" else 0,
        )
        eval_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split=split,
            ratios=_split_ratios(),
        )
        if not eval_rows:
            eval_rows = rows
        labels = [1 if row["preferred"] == "A" else 0 for row in train_rows]
        model_path = model_root / "pairwise_ranker.pkl"
        if model_path.exists():
            ranker = DistilledPairwiseRanker.load(model_path)
        else:
            ranker = DistilledPairwiseRanker().fit(
                queries=[str(row["query"]) for row in train_rows],
                doc_as=[str(row["doc_a"]) for row in train_rows],
                doc_bs=[str(row["doc_b"]) for row in train_rows],
                labels=labels,
            )
            ranker.save(model_path)
        latency = LatencyTracker()
        preds: list[int] = []
        eval_labels = [1 if row["preferred"] == "A" else 0 for row in eval_rows]
        for row in eval_rows:
            with latency.measure():
                prob = ranker.compare(str(row["query"]), str(row["doc_a"]), str(row["doc_b"]))
            preds.append(1 if prob >= 0.5 else 0)
        summary = latency.summary()
        return {
            "strategy": "distilled",
            "split": split,
            "accuracy": round(accuracy(eval_labels, preds), 4),
            "latency_p50_ms": round(summary["p50"], 4),
            "latency_p99_ms": round(summary["p99"], 4),
        }

    if strategy == "late_interaction":
        rows = read_jsonl(data_root / "pairs.jsonl")
        train_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split="train",
            ratios=_split_ratios(),
        )
        train_rows = _ensure_train_rows(
            rows,
            train_rows,
            lambda row: 1 if cast(int, row["score"]) >= 2 else 0,
        )
        eval_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split=split,
            ratios=_split_ratios(),
        )
        if not eval_rows:
            eval_rows = rows
        unique_docs = list({str(row["doc"]) for row in train_rows})
        model_path = model_root / "late_interaction_reranker.pkl"
        late_reranker: BaseReranker
        if model_path.exists():
            late_reranker = StaticColBERTReranker.load(model_path)
        else:
            late_reranker = StaticColBERTReranker()
            late_reranker.fit(unique_docs)
            late_reranker.save(model_path)
        return {
            "strategy": "late_interaction",
            "split": split,
            **_metrics_for_rows(late_reranker, eval_rows),
        }

    if strategy == "binary_reranker":
        rows = read_jsonl(data_root / "pairs.jsonl")
        train_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split="train",
            ratios=_split_ratios(),
        )
        train_rows = _ensure_train_rows(
            rows,
            train_rows,
            lambda row: 1 if cast(int, row["score"]) >= 2 else 0,
        )
        eval_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split=split,
            ratios=_split_ratios(),
        )
        if not eval_rows:
            eval_rows = rows
        labels = [1 if cast(int, row["score"]) >= 2 else 0 for row in train_rows]
        model_path = model_root / "binary_reranker.pkl"
        binary_reranker_inst = (
            BinaryQuantizedReranker.load(model_path) if model_path.exists() else None
        )
        if binary_reranker_inst is None:
            binary_reranker_inst = BinaryQuantizedReranker().fit(
                queries=[str(row["query"]) for row in train_rows],
                docs=[str(row["doc"]) for row in train_rows],
                labels=labels,
            )
            binary_reranker_inst.save(model_path)
        return {
            "strategy": "binary_reranker",
            "split": split,
            **_metrics_for_rows(binary_reranker_inst, eval_rows),
        }

    if strategy == "splade":
        rows = read_jsonl(data_root / "pairs.jsonl")
        eval_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split=split,
            ratios=_split_ratios(),
        )
        if not eval_rows:
            eval_rows = rows
        unique_docs = list({str(row["doc"]) for row in eval_rows})
        model_path = model_root / "splade_reranker.pkl"
        splade_reranker: BaseReranker
        if model_path.exists():
            splade_reranker = SPLADEReranker.load(model_path)
        else:
            splade_reranker = SPLADEReranker()
            splade_reranker.fit(unique_docs)
            splade_reranker.save(model_path)
        return {
            "strategy": "splade",
            "split": split,
            **_metrics_for_rows(splade_reranker, eval_rows),
        }

    if strategy == "multi":
        rows = read_jsonl(data_root / "pairs.jsonl")
        train_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split="train",
            ratios=_split_ratios(),
        )
        train_rows = _ensure_train_rows(
            rows,
            train_rows,
            lambda row: 1 if cast(int, row["score"]) >= 2 else 0,
        )
        eval_rows = partition_rows(
            rows,
            key_fn=lambda row: str(row["query"]),
            split=split,
            ratios=_split_ratios(),
        )
        if not eval_rows:
            eval_rows = rows

        unique_docs = list({str(row["doc"]) for row in train_rows})
        bm25_for_multi = BM25Engine()
        bm25_for_multi.fit(unique_docs)
        binary_reranker = BinaryQuantizedReranker().fit(
            queries=[str(row["query"]) for row in train_rows],
            docs=[str(row["doc"]) for row in train_rows],
            labels=[1 if cast(int, row["score"]) >= 2 else 0 for row in train_rows],
        )
        late_reranker = StaticColBERTReranker()
        late_reranker.fit(unique_docs)

        rerankers: list[tuple[str, Any]] = [  # type: ignore[type-arg]
            ("bm25", bm25_for_multi),
            ("binary", binary_reranker),
            ("late_interaction", late_reranker),
        ]
        multi_reranker = MultiReranker(
            rerankers=rerankers,
            config=MultiRerankerConfig(),
        )
        return {
            "strategy": "multi",
            "split": split,
            **_metrics_for_rows(cast(BaseReranker, multi_reranker), eval_rows),
        }

    rows = read_jsonl(data_root / "contradictions.jsonl")
    eval_rows = partition_rows(
        rows,
        key_fn=lambda row: str(row["subject"]),
        split=split,
        ratios=_split_ratios(),
    )
    if not eval_rows:
        eval_rows = rows
    engine = ConsistencyEngine()
    latency = LatencyTracker()
    y_true: list[int] = []
    y_pred: list[int] = []
    for idx, row in enumerate(eval_rows):
        docs = [str(row["doc_a"]), str(row["doc_b"])]
        with latency.measure():
            contradictions = engine.check(engine.extract_claims(docs, [f"{idx}_a", f"{idx}_b"]))
        y_true.append(1 if row.get("is_contradiction", True) else 0)
        y_pred.append(1 if contradictions else 0)
    summary = latency.summary()
    positives = [idx for idx, label in enumerate(y_true) if label == 1]
    negatives = [idx for idx, label in enumerate(y_true) if label == 0]
    recall = sum(y_pred[idx] for idx in positives) / max(len(positives), 1)
    false_positive_rate = sum(y_pred[idx] for idx in negatives) / max(len(negatives), 1)
    return {
        "strategy": "consistency",
        "split": split,
        "recall": round(recall, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "latency_p50_ms": round(summary["p50"], 4),
        "latency_p99_ms": round(summary["p99"], 4),
    }
