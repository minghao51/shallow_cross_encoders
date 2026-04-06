"""Benchmark runner for all reranking strategies.

Provides BenchmarkRunner, ExperimentResult, and related constants
for evaluating and comparing reranking strategies.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reranker.config import get_settings
from reranker.data.splits import partition_rows
from reranker.data.synth import SyntheticDataGenerator
from reranker.embedder import Embedder
from reranker.eval.metrics import LatencyTracker, ndcg_at_k, precision_at_k, reciprocal_rank
from reranker.lexical import BM25Engine
from reranker.strategies.binary_reranker import BinaryQuantizedReranker
from reranker.strategies.consistency import ConsistencyEngine
from reranker.strategies.distilled import DistilledPairwiseRanker
from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter
from reranker.strategies.late_interaction import StaticColBERTReranker
from reranker.strategies.multi import MultiReranker, MultiRerankerConfig
from reranker.strategies.pipeline import PipelineReranker
from reranker.utils import read_jsonl

POTION_MODELS = [
    "minishlab/potion-base-8M",
    "minishlab/potion-base-32M",
    "minishlab/potion-multilingual-128M",
]

DIMENSIONS = [64, 128, 256, 512]


@dataclass
class ExperimentResult:
    experiment_name: str
    strategy: str
    configuration: dict[str, Any]
    metrics: dict[str, float]
    latency_stats: dict[str, float] = field(default_factory=dict)
    ablation_info: dict[str, Any] = field(default_factory=dict)
    n_samples: int = 0
    embedder_model: str = ""


class BenchmarkRunner:
    def __init__(
        self,
        data_root: Path,
        model_root: Path,
        seed: int = 42,
        embedder_model: str | None = None,
        quick: bool = False,
    ):
        self.data_root = data_root
        self.model_root = model_root
        self.seed = seed
        self.quick = quick
        self.embedder_model_name = embedder_model or get_settings().embedder.model_name
        self.results: list[ExperimentResult] = []

        self.data_root.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)

        if not (data_root / "pairs.jsonl").exists():
            SyntheticDataGenerator(seed=seed).materialize_all(data_root)

        self.pairs = read_jsonl(data_root / "pairs.jsonl")
        self.preferences = read_jsonl(data_root / "preferences.jsonl")
        self.contradictions = read_jsonl(data_root / "contradictions.jsonl")

        self.train_pairs = partition_rows(
            self.pairs, key_fn=lambda r: str(r["query"]), split="train", seed=seed
        )
        self.test_pairs = partition_rows(
            self.pairs, key_fn=lambda r: str(r["query"]), split="test", seed=seed
        )
        self.train_prefs = partition_rows(
            self.preferences, key_fn=lambda r: str(r["query"]), split="train", seed=seed
        )
        self.test_prefs = partition_rows(
            self.preferences, key_fn=lambda r: str(r["query"]), split="test", seed=seed
        )
        self.train_contra = partition_rows(
            self.contradictions,
            key_fn=lambda r: str(r.get("subject", "")),
            split="train",
            seed=seed,
        )
        self.test_contra = partition_rows(
            self.contradictions, key_fn=lambda r: str(r.get("subject", "")), split="test", seed=seed
        )

        self.embedder = Embedder(model_name=self.embedder_model_name)
        self.bm25 = BM25Engine()

    def _evaluate_reranker(
        self, reranker, test_data: list[dict], strategy_name: str, n_docs: int = 20
    ) -> dict[str, float]:
        unique_queries = list(set(str(row["query"]) for row in test_data))
        n_queries = 5 if self.quick else 50
        unique_queries = unique_queries[:n_queries]

        ndcg_scores: list[float] = []
        mrr_scores: list[float] = []
        p1_scores: list[float] = []
        latencies: list[float] = []

        for query in unique_queries:
            query_docs = [str(row["doc"]) for row in test_data if str(row["query"]) == query]
            query_labels = [row["score"] for row in test_data if str(row["query"]) == query]

            if len(query_docs) < 5:
                other_docs = [str(row["doc"]) for row in test_data if str(row["query"]) != query]
                query_docs = query_docs + other_docs[: max(0, n_docs - len(query_docs))]
                query_labels = query_labels + [0] * max(0, n_docs - len(query_labels))

            query_docs = query_docs[:n_docs]
            query_labels = query_labels[:n_docs]

            if strategy_name == "late_interaction" and hasattr(reranker, "fit"):
                reranker.fit(query_docs)

            tracker = LatencyTracker()
            with tracker.measure():
                ranked = reranker.rerank(query, query_docs)

            latencies.append(tracker.summary()["mean"])

            if ranked:
                ranked_labels_map = {r.doc: i for i, r in enumerate(ranked)}
                ranked_labels = []
                for doc in query_docs:
                    if doc in ranked_labels_map:
                        ranked_labels.append(query_labels[query_docs.index(doc)])
                    else:
                        ranked_labels.append(0)

                binary_labels = [1 if label >= 2 else 0 for label in ranked_labels]

                ndcg_scores.append(ndcg_at_k(ranked_labels, k=10))
                mrr_scores.append(reciprocal_rank(binary_labels))
                p1_scores.append(precision_at_k(binary_labels, k=1))

        return {
            "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
            "ndcg@10_std": float(np.std(ndcg_scores)) if ndcg_scores else 0.0,
            "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            "mrr_std": float(np.std(mrr_scores)) if mrr_scores else 0.0,
            "p@1": float(np.mean(p1_scores)) if p1_scores else 0.0,
            "p@1_std": float(np.std(p1_scores)) if p1_scores else 0.0,
            "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
            "latency_std_ms": float(np.std(latencies)) if latencies else 0.0,
            "n_queries_evaluated": len(ndcg_scores),
        }

    def _evaluate_distilled(self, ranker, test_data: list[dict]) -> dict[str, float]:
        n_eval = 50 if self.quick else 200
        accuracies: list[float] = []
        latencies: list[float] = []

        for row in test_data[:n_eval]:
            query = str(row["query"])
            doc_a = str(row["doc_a"])
            doc_b = str(row["doc_b"])
            actual = row["preferred"]

            tracker = LatencyTracker()
            with tracker.measure():
                score = ranker.compare(query, doc_a, doc_b)
            latencies.append(tracker.summary()["mean"])

            pred = "A" if score > 0.5 else "B"
            accuracies.append(1.0 if pred == actual else 0.0)

        return {
            "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
            "accuracy_std": float(np.std(accuracies)) if accuracies else 0.0,
            "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
            "latency_std_ms": float(np.std(latencies)) if latencies else 0.0,
            "n_comparisons": len(accuracies),
        }

    def _evaluate_consistency(self, engine, test_data: list[dict]) -> dict[str, float]:
        n_eval = 25 if self.quick else 100
        latencies: list[float] = []
        tp = fp = tn = fn = 0

        for row in test_data[:n_eval]:
            doc_a = str(row["doc_a"])
            doc_b = str(row["doc_b"])
            is_contradiction = row.get("is_contradiction", False)

            tracker = LatencyTracker()
            with tracker.measure():
                claim_sets = engine.extract_claims([doc_a, doc_b], ["a", "b"])
                contradictions = engine.check(claim_sets)
            latencies.append(tracker.summary()["mean"])

            detected = len(contradictions) > 0

            if is_contradiction and detected:
                tp += 1
            elif is_contradiction and not detected:
                fn += 1
            elif not is_contradiction and detected:
                fp += 1
            else:
                tn += 1

        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)

        return {
            "recall": recall,
            "false_positive_rate": fpr,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
            "latency_std_ms": float(np.std(latencies)) if latencies else 0.0,
        }

    def _print_metrics(self, metrics: dict[str, float], strategy: str):
        if "ndcg@10" in metrics:
            print(f"  NDCG@10: {metrics['ndcg@10']:.4f} ± {metrics['ndcg@10_std']:.4f}")
            print(f"  MRR:     {metrics['mrr']:.4f} ± {metrics['mrr_std']:.4f}")
            print(f"  P@1:     {metrics['p@1']:.4f} ± {metrics['p@1_std']:.4f}")
            print(f"  Latency: {metrics['latency_mean_ms']:.2f}ms")
        elif "accuracy" in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}")
            print(f"  Latency:  {metrics['latency_mean_ms']:.2f}ms")
        elif "recall" in metrics:
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  FPR:    {metrics['false_positive_rate']:.4f}")
            print(f"  Latency: {metrics['latency_mean_ms']:.2f}ms")

    def run_baselines(self):
        print("=" * 80)
        print("PHASE 1: BASELINE EXPERIMENTS")
        print("=" * 80)

        all_docs = [str(row["doc"]) for row in self.train_pairs]
        self.bm25.fit(all_docs)

        print("\n--- BM25 Baseline ---")
        bm25_metrics = self._evaluate_reranker(self.bm25, self.test_pairs, "bm25")
        self.results.append(
            ExperimentResult(
                experiment_name="bm25_baseline",
                strategy="bm25",
                configuration={"backend": self.bm25.backend_name},
                metrics=bm25_metrics,
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(bm25_metrics, "bm25")

        print("\n--- Hybrid Fusion Reranker ---")
        hybrid = HybridFusionReranker(
            adapters=[KeywordMatchAdapter()],
            embedder=self.embedder,
            random_state=self.seed,
        )
        hybrid.fit(
            queries=[str(row["query"]) for row in self.train_pairs],
            docs=[str(row["doc"]) for row in self.train_pairs],
            labels=[1 if row["score"] >= 2 else 0 for row in self.train_pairs],
        )
        hybrid_metrics = self._evaluate_reranker(hybrid, self.test_pairs, "hybrid")
        self.results.append(
            ExperimentResult(
                experiment_name="hybrid_baseline",
                strategy="hybrid",
                configuration={
                    "backend": hybrid.model_backend,
                    "adapters": ["KeywordMatchAdapter"],
                    "n_estimators": 120,
                    "max_depth": 4,
                },
                metrics=hybrid_metrics,
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(hybrid_metrics, "hybrid")

        print("\n--- Distilled Pairwise Ranker ---")
        distilled = DistilledPairwiseRanker(embedder=self.embedder)
        distilled.fit(
            queries=[str(row["query"]) for row in self.train_prefs],
            doc_as=[str(row["doc_a"]) for row in self.train_prefs],
            doc_bs=[str(row["doc_b"]) for row in self.train_prefs],
            labels=[1 if row["preferred"] == "A" else 0 for row in self.train_prefs],
        )
        distilled_metrics = self._evaluate_distilled(distilled, self.test_prefs)
        self.results.append(
            ExperimentResult(
                experiment_name="distilled_baseline",
                strategy="distilled",
                configuration={"C": 1.0, "max_iter": 500},
                metrics=distilled_metrics,
                n_samples=len(self.test_prefs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(distilled_metrics, "distilled")

        print("\n--- Static ColBERT Reranker ---")
        colbert = StaticColBERTReranker(
            embedder=self.embedder,
            top_k_tokens=128,
            use_salience=True,
        )
        colbert.fit(docs=[str(row["doc"]) for row in self.pairs])
        colbert_metrics = self._evaluate_reranker(colbert, self.test_pairs, "late_interaction")
        self.results.append(
            ExperimentResult(
                experiment_name="colbert_baseline",
                strategy="late_interaction",
                configuration={"top_k_tokens": 128, "use_salience": True},
                metrics=colbert_metrics,
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(colbert_metrics, "late_interaction")

        print("\n--- Binary Quantized Reranker ---")
        binary = BinaryQuantizedReranker(
            embedder=self.embedder,
            hamming_top_k=500,
            bilinear_top_k=50,
            random_state=self.seed,
        )
        binary.fit(
            queries=[str(row["query"]) for row in self.train_pairs],
            docs=[str(row["doc"]) for row in self.train_pairs],
            labels=[1 if row["score"] >= 2 else 0 for row in self.train_pairs],
        )
        binary_metrics = self._evaluate_reranker(binary, self.test_pairs, "binary_reranker")
        self.results.append(
            ExperimentResult(
                experiment_name="binary_baseline",
                strategy="binary_reranker",
                configuration={"hamming_top_k": 500, "bilinear_top_k": 50},
                metrics=binary_metrics,
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(binary_metrics, "binary_reranker")

        print("\n--- Consistency Engine ---")
        consistency = ConsistencyEngine(
            sim_threshold=0.95,
            value_tolerance=0.01,
            embedder=self.embedder,
        )
        consistency_metrics = self._evaluate_consistency(consistency, self.test_contra)
        self.results.append(
            ExperimentResult(
                experiment_name="consistency_baseline",
                strategy="consistency",
                configuration={"sim_threshold": 0.95, "value_tolerance": 0.01},
                metrics=consistency_metrics,
                n_samples=len(self.test_contra),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(consistency_metrics, "consistency")

        print("\n--- Pipeline (BM25 → Binary → Hybrid → ColBERT) ---")
        pipeline = PipelineReranker()
        pipeline.add_stage("bm25", self.bm25, top_k=200)
        pipeline.add_stage("binary", binary, top_k=100)
        pipeline.add_stage("hybrid", hybrid, top_k=50)
        pipeline.add_stage("colbert", colbert, top_k=20)
        pipeline_metrics = self._evaluate_reranker(pipeline, self.test_pairs, "pipeline")
        self.results.append(
            ExperimentResult(
                experiment_name="pipeline_baseline",
                strategy="pipeline",
                configuration={
                    "stages": ["bm25", "binary", "hybrid", "colbert"],
                    "top_ks": [200, 100, 50, 20],
                },
                metrics=pipeline_metrics,
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(pipeline_metrics, "pipeline")

        print("\n--- MultiReranker (BM25 + Binary) ---")
        multi_bm25_binary = MultiReranker(
            rerankers=[("bm25", self.bm25), ("binary", binary)],
            config=MultiRerankerConfig(rrf_k=60),
        )
        multi_bm25_binary_metrics = self._evaluate_reranker(
            multi_bm25_binary, self.test_pairs, "multi"
        )
        self.results.append(
            ExperimentResult(
                experiment_name="multi_bm25_binary",
                strategy="multi",
                configuration={"rerankers": ["bm25", "binary"], "rrf_k": 60},
                metrics=multi_bm25_binary_metrics,
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(multi_bm25_binary_metrics, "multi")

        print("\n--- MultiReranker (Hybrid + BM25) ---")
        hybrid_no_adapters = HybridFusionReranker(
            adapters=[],
            embedder=self.embedder,
            random_state=self.seed,
        )
        hybrid_no_adapters.fit(
            queries=[str(row["query"]) for row in self.train_pairs],
            docs=[str(row["doc"]) for row in self.train_pairs],
            labels=[1 if row["score"] >= 2 else 0 for row in self.train_pairs],
        )
        multi_hybrid_bm25 = MultiReranker(
            rerankers=[("hybrid", hybrid_no_adapters), ("bm25", self.bm25)],
            config=MultiRerankerConfig(rrf_k=60),
        )
        multi_hybrid_bm25_metrics = self._evaluate_reranker(
            multi_hybrid_bm25, self.test_pairs, "multi"
        )
        self.results.append(
            ExperimentResult(
                experiment_name="multi_hybrid_bm25",
                strategy="multi",
                configuration={"rerankers": ["hybrid", "bm25"], "rrf_k": 60},
                metrics=multi_hybrid_bm25_metrics,
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(multi_hybrid_bm25_metrics, "multi")

        print("\n--- SPLADE ---")
        print("  SKIPPED: Requires pretrained SPLADE model (sentence-transformers).")
        print("  Install with: pip install sentence-transformers")
        self.results.append(
            ExperimentResult(
                experiment_name="splade_skipped",
                strategy="splade",
                configuration={"status": "skipped", "reason": "missing dependency"},
                metrics={"ndcg@10": 0.0, "mrr": 0.0, "p@1": 0.0, "latency_mean_ms": 0.0},
                n_samples=0,
                embedder_model=self.embedder_model_name,
            )
        )

    def run_ablations(self):
        print("\n" + "=" * 80)
        print("PHASE 2: ABLATION STUDIES")
        print("=" * 80)

        print("\n--- Hybrid: No Adapters ---")
        hybrid_no_adapters = HybridFusionReranker(
            adapters=[],
            embedder=self.embedder,
            random_state=self.seed,
        )
        hybrid_no_adapters.fit(
            queries=[str(row["query"]) for row in self.train_pairs],
            docs=[str(row["doc"]) for row in self.train_pairs],
            labels=[1 if row["score"] >= 2 else 0 for row in self.train_pairs],
        )
        metrics = self._evaluate_reranker(hybrid_no_adapters, self.test_pairs, "hybrid")
        self.results.append(
            ExperimentResult(
                experiment_name="hybrid_ablation_no_adapters",
                strategy="hybrid",
                configuration={"adapters": []},
                metrics=metrics,
                ablation_info={"removed": "KeywordMatchAdapter"},
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(metrics, "hybrid")

        print("\n--- ColBERT: No Salience ---")
        colbert_no_salience = StaticColBERTReranker(
            embedder=self.embedder,
            top_k_tokens=128,
            use_salience=False,
        )
        colbert_no_salience.fit(docs=[str(row["doc"]) for row in self.pairs])
        metrics = self._evaluate_reranker(colbert_no_salience, self.test_pairs, "late_interaction")
        self.results.append(
            ExperimentResult(
                experiment_name="colbert_ablation_no_salience",
                strategy="late_interaction",
                configuration={"top_k_tokens": 128, "use_salience": False},
                metrics=metrics,
                ablation_info={"removed": "TF-IDF salience weighting"},
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(metrics, "late_interaction")

        print("\n--- ColBERT: 64 Tokens ---")
        colbert_64 = StaticColBERTReranker(
            embedder=self.embedder,
            top_k_tokens=64,
            use_salience=True,
        )
        colbert_64.fit(docs=[str(row["doc"]) for row in self.pairs])
        metrics = self._evaluate_reranker(colbert_64, self.test_pairs, "late_interaction")
        self.results.append(
            ExperimentResult(
                experiment_name="colbert_ablation_64_tokens",
                strategy="late_interaction",
                configuration={"top_k_tokens": 64, "use_salience": True},
                metrics=metrics,
                ablation_info={"changed": "top_k_tokens: 128 → 64"},
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(metrics, "late_interaction")

        print("\n--- Binary: Hamming Only ---")
        binary_hamming = BinaryQuantizedReranker(
            embedder=self.embedder,
            hamming_top_k=999999,
            bilinear_top_k=0,
            random_state=self.seed,
        )
        binary_hamming.fit(
            queries=[str(row["query"]) for row in self.train_pairs],
            docs=[str(row["doc"]) for row in self.train_pairs],
            labels=[1 if row["score"] >= 2 else 0 for row in self.train_pairs],
        )
        metrics = self._evaluate_reranker(binary_hamming, self.test_pairs, "binary_reranker")
        self.results.append(
            ExperimentResult(
                experiment_name="binary_ablation_hamming_only",
                strategy="binary_reranker",
                configuration={"hamming_top_k": 999999, "bilinear_top_k": 0},
                metrics=metrics,
                ablation_info={"removed": "Bilinear refinement stage"},
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(metrics, "binary_reranker")

        print("\n--- Binary: Aggressive Pruning ---")
        binary_aggressive = BinaryQuantizedReranker(
            embedder=self.embedder,
            hamming_top_k=100,
            bilinear_top_k=10,
            random_state=self.seed,
        )
        binary_aggressive.fit(
            queries=[str(row["query"]) for row in self.train_pairs],
            docs=[str(row["doc"]) for row in self.train_pairs],
            labels=[1 if row["score"] >= 2 else 0 for row in self.train_pairs],
        )
        metrics = self._evaluate_reranker(binary_aggressive, self.test_pairs, "binary_reranker")
        self.results.append(
            ExperimentResult(
                experiment_name="binary_ablation_aggressive",
                strategy="binary_reranker",
                configuration={"hamming_top_k": 100, "bilinear_top_k": 10},
                metrics=metrics,
                ablation_info={"changed": "hamming_top_k: 500→100, bilinear_top_k: 50→10"},
                n_samples=len(self.test_pairs),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(metrics, "binary_reranker")

        print("\n--- Consistency: Relaxed Threshold (0.90) ---")
        consistency_relaxed = ConsistencyEngine(
            sim_threshold=0.90,
            value_tolerance=0.01,
            embedder=self.embedder,
        )
        metrics = self._evaluate_consistency(consistency_relaxed, self.test_contra)
        self.results.append(
            ExperimentResult(
                experiment_name="consistency_ablation_relaxed",
                strategy="consistency",
                configuration={"sim_threshold": 0.90, "value_tolerance": 0.01},
                metrics=metrics,
                ablation_info={"changed": "sim_threshold: 0.95 → 0.90"},
                n_samples=len(self.test_contra),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(metrics, "consistency")

        print("\n--- Consistency: Strict Threshold (0.99) ---")
        consistency_strict = ConsistencyEngine(
            sim_threshold=0.99,
            value_tolerance=0.01,
            embedder=self.embedder,
        )
        metrics = self._evaluate_consistency(consistency_strict, self.test_contra)
        self.results.append(
            ExperimentResult(
                experiment_name="consistency_ablation_strict",
                strategy="consistency",
                configuration={"sim_threshold": 0.99, "value_tolerance": 0.01},
                metrics=metrics,
                ablation_info={"changed": "sim_threshold: 0.95 → 0.99"},
                n_samples=len(self.test_contra),
                embedder_model=self.embedder_model_name,
            )
        )
        self._print_metrics(metrics, "consistency")

    def run_scaling(self):
        print("\n" + "=" * 80)
        print("PHASE 3: SCALING EXPERIMENTS")
        print("=" * 80)

        hybrid = HybridFusionReranker(
            adapters=[KeywordMatchAdapter()],
            embedder=self.embedder,
            random_state=self.seed,
        )
        hybrid.fit(
            queries=[str(row["query"]) for row in self.train_pairs],
            docs=[str(row["doc"]) for row in self.train_pairs],
            labels=[1 if row["score"] >= 2 else 0 for row in self.train_pairs],
        )

        colbert = StaticColBERTReranker(
            embedder=self.embedder,
            top_k_tokens=128,
            use_salience=True,
        )
        colbert.fit(docs=[str(row["doc"]) for row in self.pairs])

        binary = BinaryQuantizedReranker(
            embedder=self.embedder,
            hamming_top_k=500,
            bilinear_top_k=50,
            random_state=self.seed,
        )
        binary.fit(
            queries=[str(row["query"]) for row in self.train_pairs],
            docs=[str(row["doc"]) for row in self.train_pairs],
            labels=[1 if row["score"] >= 2 else 0 for row in self.train_pairs],
        )

        corpus_sizes = [20, 50, 100, 200]
        query = str(self.test_pairs[0]["query"])
        query_docs = [str(row["doc"]) for row in self.test_pairs if str(row["query"]) == query]
        if len(query_docs) < 200:
            other_docs = [str(row["doc"]) for row in self.test_pairs if str(row["query"]) != query]
            query_docs = query_docs + other_docs[: 200 - len(query_docs)]

        scaling_results: dict[str, dict[int, float]] = {
            "bm25": {},
            "hybrid": {},
            "colbert": {},
            "binary": {},
        }

        for n_docs in corpus_sizes:
            docs = query_docs[:n_docs]
            print(f"\n--- Corpus Size: {n_docs} docs ---")

            bm25_s = BM25Engine()
            bm25_s.fit(docs)
            start = time.perf_counter()
            bm25_s.rerank(query, docs)
            lat = (time.perf_counter() - start) * 1000
            scaling_results["bm25"][n_docs] = lat
            print(f"  bm25:     {lat:.2f}ms")

            start = time.perf_counter()
            hybrid.rerank(query, docs)
            lat = (time.perf_counter() - start) * 1000
            scaling_results["hybrid"][n_docs] = lat
            print(f"  hybrid:   {lat:.2f}ms")

            colbert_s = StaticColBERTReranker(
                embedder=self.embedder,
                top_k_tokens=128,
                use_salience=True,
            )
            colbert_s.fit(docs)
            start = time.perf_counter()
            colbert_s.rerank(query, docs)
            lat = (time.perf_counter() - start) * 1000
            scaling_results["colbert"][n_docs] = lat
            print(f"  colbert:  {lat:.2f}ms")

            binary_s = BinaryQuantizedReranker(
                embedder=self.embedder,
                hamming_top_k=500,
                bilinear_top_k=50,
                random_state=self.seed,
            )
            binary_s.fit([query] * len(docs), docs, [1] * len(docs))
            start = time.perf_counter()
            binary_s.rerank(query, docs)
            lat = (time.perf_counter() - start) * 1000
            scaling_results["binary"][n_docs] = lat
            print(f"  binary:   {lat:.2f}ms")

        self.results.append(
            ExperimentResult(
                experiment_name="scaling",
                strategy="scaling",
                configuration={"corpus_sizes": corpus_sizes},
                metrics={
                    f"{strat}_{size}ms": lat
                    for strat, sizes in scaling_results.items()
                    for size, lat in sizes.items()
                },
                n_samples=n_docs,
                embedder_model=self.embedder_model_name,
            )
        )

    def run_embedder_comparison(self):
        print("\n" + "=" * 80)
        print("PHASE 4: EMBEDDER MODEL COMPARISON")
        print("=" * 80)

        eval_rows = self.test_pairs
        if self.quick:
            eval_rows = eval_rows[:50]

        train_rows = partition_rows(
            self.pairs, key_fn=lambda r: str(r["query"]), split="train", seed=self.seed
        )
        if self.quick:
            train_rows = train_rows[:100]

        for model_name in POTION_MODELS:
            print(f"\n--- Embedder: {model_name} ---")
            try:
                embedder = Embedder(model_name=model_name)
            except Exception:
                print(f"  SKIPPED: Could not load {model_name}")
                continue

            if embedder.backend_name == "hashed":
                print(f"  SKIPPED: model2vec not available for {model_name}")
                continue

            print(f"  Backend: {embedder.backend_name}")

            for dimension in DIMENSIONS:
                embedder_dim = Embedder(model_name=model_name, dimension=dimension)
                if embedder_dim.backend_name == "hashed":
                    continue

                for strategy_name in ["hybrid", "binary_reranker", "late_interaction"]:
                    print(f"  dim={dimension}, strategy={strategy_name}...", end=" ")

                    reranker = self._build_reranker_for_embedder_test(
                        strategy_name, embedder_dim, train_rows
                    )
                    if reranker is None:
                        print("SKIPPED")
                        continue

                    ndcgs: list[float] = []
                    mrrs: list[float] = []
                    p1s: list[float] = []
                    rerank_latencies: list[float] = []

                    grouped: dict[str, list[dict]] = {}
                    for row in eval_rows:
                        grouped.setdefault(str(row["query"]), []).append(row)
                    query_keys = list(grouped.keys())[: (5 if self.quick else 50)]

                    for q in query_keys:
                        items = grouped[q]
                        docs = [str(item["doc"]) for item in items]
                        label_map = {str(item["doc"]): int(item["score"]) for item in items}

                        tracker = LatencyTracker()
                        with tracker.measure():
                            ranked = reranker.rerank(q, docs)
                        rerank_latencies.append(tracker.summary()["mean"])

                        relevances = [float(label_map.get(result.doc, 0)) for result in ranked]
                        binary_labels = [1 if rel >= 2 else 0 for rel in relevances]

                        ndcgs.append(ndcg_at_k(relevances, 10))
                        mrrs.append(reciprocal_rank(binary_labels))
                        p1s.append(precision_at_k(binary_labels, 1))

                    if not ndcgs:
                        print("SKIPPED")
                        continue

                    ndcg_mean = sum(ndcgs) / len(ndcgs)
                    mrr_mean = sum(mrrs) / len(mrrs)
                    p1_mean = sum(p1s) / len(p1s)
                    lat_mean = np.mean(rerank_latencies)

                    self.results.append(
                        ExperimentResult(
                            experiment_name=f"embedder_{model_name.replace('/', '_')}_dim{dimension}_{strategy_name}",  # noqa: E501
                            strategy=strategy_name,
                            configuration={
                                "model_name": model_name,
                                "dimension": dimension,
                                "backend": embedder_dim.backend_name,
                            },
                            metrics={
                                "ndcg@10": ndcg_mean,
                                "ndcg@10_std": float(np.std(ndcgs)),
                                "mrr": mrr_mean,
                                "mrr_std": float(np.std(mrrs)),
                                "p@1": p1_mean,
                                "p@1_std": float(np.std(p1s)),
                                "latency_mean_ms": lat_mean,
                                "latency_std_ms": float(np.std(rerank_latencies)),
                                "n_queries_evaluated": len(ndcgs),
                            },
                            n_samples=len(eval_rows),
                            embedder_model=model_name,
                        )
                    )
                    print(f"NDCG@10={ndcg_mean:.4f}")

    def _build_reranker_for_embedder_test(
        self,
        strategy: str,
        embedder: Embedder,
        train_rows: list[dict],
    ):
        if not train_rows:
            return None

        if strategy == "hybrid":
            labels = [1 if int(row["score"]) >= 2 else 0 for row in train_rows]
            return HybridFusionReranker(
                adapters=[KeywordMatchAdapter()],
                embedder=embedder,
            ).fit(
                queries=[str(row["query"]) for row in train_rows],
                docs=[str(row["doc"]) for row in train_rows],
                labels=labels,
            )

        if strategy == "binary_reranker":
            labels = [1 if int(row["score"]) >= 2 else 0 for row in train_rows]
            return BinaryQuantizedReranker(embedder=embedder).fit(
                queries=[str(row["query"]) for row in train_rows],
                docs=[str(row["doc"]) for row in train_rows],
                labels=labels,
            )

        if strategy == "late_interaction":
            unique_docs = list({str(row["doc"]) for row in train_rows})
            reranker = StaticColBERTReranker(embedder=embedder)
            reranker.fit(unique_docs)
            return reranker

        return None

    def save_results(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": self.seed,
            "embedder_model": self.embedder_model_name,
            "quick": self.quick,
            "data_counts": {
                "train_pairs": len(self.train_pairs),
                "test_pairs": len(self.test_pairs),
                "train_prefs": len(self.train_prefs),
                "test_prefs": len(self.test_prefs),
                "train_contradictions": len(self.train_contra),
                "test_contradictions": len(self.test_contra),
            },
        }

        results_dict = []
        for r in self.results:
            results_dict.append(
                {
                    "experiment_name": r.experiment_name,
                    "strategy": r.strategy,
                    "configuration": r.configuration,
                    "metrics": r.metrics,
                    "latency_stats": r.latency_stats,
                    "ablation_info": r.ablation_info,
                    "n_samples": r.n_samples,
                    "embedder_model": r.embedder_model,
                }
            )

        output = {"metadata": metadata, "results": results_dict}

        with open(output_dir / "benchmark_results.json", "w") as f:
            json.dump(output, f, indent=2)

        summary = self._generate_summary(metadata)
        with open(output_dir / "benchmark_summary.md", "w") as f:
            f.write(summary)

        print(f"\nResults saved to {output_dir}")
        print("  - benchmark_results.json")
        print("  - benchmark_summary.md")

    def _generate_summary(self, metadata: dict) -> str:
        lines = []
        lines.append("# Unified Benchmark Results")
        lines.append("")
        lines.append(f"Generated: {metadata['timestamp']}")
        lines.append(f"Seed: {metadata['seed']}")
        lines.append(f"Embedder: {metadata['embedder_model']}")
        lines.append(f"Quick mode: {metadata['quick']}")
        dc = metadata["data_counts"]
        lines.append(f"Test pairs: {dc['test_pairs']}")
        lines.append(f"Test preferences: {dc['test_prefs']}")
        lines.append(f"Test contradictions: {dc['test_contradictions']}")
        lines.append("")

        lines.append("## Baseline Results")
        lines.append("")
        lines.append("| Strategy | Experiment | NDCG@10 | MRR | P@1 | Latency (ms) |")
        lines.append("|----------|------------|---------|-----|-----|--------------|")

        for r in self.results:
            if "baseline" in r.experiment_name and r.strategy not in ("consistency", "splade"):
                m = r.metrics
                ndcg = f"{m.get('ndcg@10', 0):.4f} ± {m.get('ndcg@10_std', 0):.4f}"
                mrr = f"{m.get('mrr', 0):.4f} ± {m.get('mrr_std', 0):.4f}"
                p1 = f"{m.get('p@1', 0):.4f} ± {m.get('p@1_std', 0):.4f}"
                lat = f"{m.get('latency_mean_ms', 0):.2f}"
                lines.append(
                    f"| {r.strategy} | {r.experiment_name} | {ndcg} | {mrr} | {p1} | {lat} |"
                )  # noqa: E501

        for r in self.results:
            if r.experiment_name == "distilled_baseline":
                m = r.metrics
                acc = f"{m.get('accuracy', 0):.4f} ± {m.get('accuracy_std', 0):.4f}"
                lat = f"{m.get('latency_mean_ms', 0):.2f}"
                lines.append(f"| distilled | {r.experiment_name} | acc={acc} | - | - | {lat} |")

        for r in self.results:
            if r.experiment_name == "consistency_baseline":
                m = r.metrics
                recall = f"{m.get('recall', 0):.4f}"
                fpr = f"{m.get('false_positive_rate', 0):.4f}"
                lat = f"{m.get('latency_mean_ms', 0):.2f}"
                lines.append(
                    f"| consistency | {r.experiment_name} | recall={recall}, fpr={fpr} | - | - | {lat} |"  # noqa: E501
                )

        splade = [r for r in self.results if r.strategy == "splade"]
        if splade:
            lines.append(
                f"| splade | {splade[0].experiment_name} | SKIPPED (requires sentence-transformers) |"  # noqa: E501
            )

        lines.append("")
        lines.append("## MultiReranker (RRF Fusion)")
        lines.append("")
        lines.append("| Experiment | Rerankers | NDCG@10 | MRR | P@1 | Latency (ms) |")
        lines.append("|------------|-----------|---------|-----|-----|--------------|")

        for r in self.results:
            if r.experiment_name.startswith("multi_"):
                m = r.metrics
                rerankers = r.configuration.get("rerankers", [])
                ndcg = f"{m.get('ndcg@10', 0):.4f} ± {m.get('ndcg@10_std', 0):.4f}"
                mrr = f"{m.get('mrr', 0):.4f} ± {m.get('mrr_std', 0):.4f}"
                p1 = f"{m.get('p@1', 0):.4f} ± {m.get('p@1_std', 0):.4f}"
                lat = f"{m.get('latency_mean_ms', 0):.2f}"
                lines.append(
                    f"| {r.experiment_name} | {', '.join(rerankers)} | {ndcg} | {mrr} | {p1} | {lat} |"  # noqa: E501
                )

        lines.append("")
        lines.append("## Ablation Studies")
        lines.append("")

        for strategy in ["hybrid", "late_interaction", "binary_reranker", "consistency"]:
            strategy_results = [
                r
                for r in self.results
                if r.strategy == strategy and "ablation" in r.experiment_name
            ]  # noqa: E501
            if not strategy_results:
                continue

            lines.append(f"### {strategy}")
            lines.append("")

            baseline = None
            for r in self.results:
                if r.strategy == strategy and "baseline" in r.experiment_name:
                    baseline = r
                    break

            if strategy == "consistency":
                lines.append("| Experiment | Recall | FPR | Δ Recall vs Baseline |")
                lines.append("|------------|--------|-----|---------------------|")
                baseline_recall = baseline.metrics.get("recall", 0) if baseline else 0
                for r in strategy_results:
                    m = r.metrics
                    recall = m.get("recall", 0)
                    fpr = m.get("false_positive_rate", 0)
                    delta = recall - baseline_recall
                    lines.append(
                        f"| {r.experiment_name} | {recall:.4f} | {fpr:.4f} | {delta:+.4f} |"
                    )  # noqa: E501
            else:
                lines.append("| Experiment | NDCG@10 | Δ vs Baseline |")
                lines.append("|------------|---------|---------------|")
                baseline_ndcg = baseline.metrics.get("ndcg@10", 0) if baseline else 0
                for r in strategy_results:
                    ndcg = r.metrics.get("ndcg@10", 0)
                    delta = ndcg - baseline_ndcg
                    lines.append(f"| {r.experiment_name} | {ndcg:.4f} | {delta:+.4f} |")

            lines.append("")

        scaling = [r for r in self.results if r.strategy == "scaling"]
        if scaling:
            lines.append("## Scaling Results (Latency in ms)")
            lines.append("")
            lines.append("| Corpus Size | BM25 (ms) | Hybrid (ms) | ColBERT (ms) | Binary (ms) |")
            lines.append("|-------------|-----------|-------------|--------------|-------------|")
            m = scaling[0].metrics
            for size in [20, 50, 100, 200]:
                bm25 = m.get(f"bm25_{size}ms", 0)
                hybrid = m.get(f"hybrid_{size}ms", 0)
                colbert = m.get(f"colbert_{size}ms", 0)
                binary = m.get(f"binary_{size}ms", 0)
                lines.append(
                    f"| {size} | {bm25:.2f} | {hybrid:.2f} | {colbert:.2f} | {binary:.2f} |"
                )  # noqa: E501
            lines.append("")

        embedder_results = [r for r in self.results if r.experiment_name.startswith("embedder_")]
        if embedder_results:
            lines.append("## Embedder Model Comparison")
            lines.append("")
            lines.append("| Model | Dim | Strategy | NDCG@10 | MRR | P@1 | Latency (ms) |")
            lines.append("|-------|-----|----------|---------|-----|-----|--------------|")
            for r in sorted(embedder_results, key=lambda x: -x.metrics.get("ndcg@10", 0)):
                model = r.configuration.get("model_name", "").split("/")[-1]
                dim = r.configuration.get("dimension", 0)
                strategy = r.strategy
                m = r.metrics
                ndcg = f"{m.get('ndcg@10', 0):.4f}"
                mrr = f"{m.get('mrr', 0):.4f}"
                p1 = f"{m.get('p@1', 0):.4f}"
                lat = f"{m.get('latency_mean_ms', 0):.2f}"
                lines.append(f"| {model} | {dim} | {strategy} | {ndcg} | {mrr} | {p1} | {lat} |")
            lines.append("")

            best = max(embedder_results, key=lambda r: r.metrics.get("ndcg@10", 0))
            lines.append(
                f"**Best NDCG@10**: {best.configuration.get('model_name', '')} "
                f"(dim={best.configuration.get('dimension', 0)}, "
                f"strategy={best.strategy}) = {best.metrics['ndcg@10']:.4f}"
            )
            lines.append("")

        lines.append("## Key Findings")
        lines.append("")

        baseline_results = [
            r for r in self.results if "baseline" in r.experiment_name and "ndcg@10" in r.metrics
        ]
        if baseline_results:
            best = max(baseline_results, key=lambda r: r.metrics.get("ndcg@10", 0))
            fastest = min(
                [r for r in baseline_results if r.metrics.get("latency_mean_ms", 0) > 0],
                key=lambda r: r.metrics.get("latency_mean_ms", float("inf")),
            )
            lines.append(  # noqa: E501
                f"- **Best NDCG@10**: {best.experiment_name} ({best.metrics['ndcg@10']:.4f})"
            )
            lines.append(
                f"- **Fastest**: {fastest.experiment_name} ({fastest.metrics['latency_mean_ms']:.2f}ms)"  # noqa: E501
            )
            lines.append("")

        return "\n".join(lines)
