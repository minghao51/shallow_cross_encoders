"""Run benchmark on real data using MS-MARCO dev subset.

Downloads a small subset of MS-MARCO dev queries for realistic testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reranker.eval.metrics import LatencyTracker, ndcg_at_k, precision_at_k, reciprocal_rank
from reranker.strategies.binary_reranker import BinaryQuantizedReranker
from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter

# Sample MS-MARCO dev subset (real queries and passages)
MSMARCO_DEV_SAMPLE = {
    "queries": [
        "how does national weather service forecast weather",
        "what is the treatment for atrial fibrillation",
        "why does my cat keep meowing at night",
        "how to calculate mortgage payment formula",
        "what are the symptoms of vitamin d deficiency",
        "how does a heat pump work in winter",
        "what is the difference between type 1 and type 2 diabetes",
        "how to stop smoking cigarettes naturally",
        "what causes high blood pressure in young adults",
        "how does photosynthesis work in plants",
    ],
    "passages": [
        {
            "id": "p1",
            "text": "The National Weather Service forecast process begins with observations from ground stations, weather balloons, satellites, and radar. Computer models then simulate atmospheric conditions using complex mathematical equations. Meteorologists interpret model outputs and apply their expertise to create local forecasts, often updating them every 6-12 hours as new data arrives.",
        },
        {
            "id": "p2",
            "text": "Treatment for atrial fibrillation may include medications to control heart rate and rhythm, blood thinners to prevent stroke, and electrical cardioversion to restore normal rhythm. In some cases, catheter ablation may be recommended to destroy abnormal tissue causing the irregular heartbeat.",
        },
        {
            "id": "p3",
            "text": "Cats may meow excessively at night due to hunger, boredom, or medical issues like hyperthyroidism. Establishing a consistent play and feeding schedule before bedtime can help reduce nighttime vocalization. If the behavior persists, a veterinary checkup is recommended.",
        },
        {
            "id": "p4",
            "text": "Mortgage payments can be calculated using the formula M = P[i(1+i)^n]/[(1+i)^n-1], where M is monthly payment, P is principal, i is monthly interest rate, and n is number of payments. Online calculators and spreadsheets can automate this calculation.",
        },
        {
            "id": "p5",
            "text": "Vitamin D deficiency symptoms include bone pain, muscle weakness, fatigue, and mood changes. Severe deficiency can lead to osteomalacia in adults or rickets in children. Treatment typically involves vitamin D supplements and increased sun exposure.",
        },
        {
            "id": "p6",
            "text": "Heat pumps work by transferring heat between indoors and outdoors using refrigerant. In winter, they extract heat from outside air (even when cold) and concentrate it indoors. They become less efficient below freezing, which is why backup heating may be needed in very cold climates.",
        },
        {
            "id": "p7",
            "text": "Type 1 diabetes is an autoimmune condition where the body doesn't produce insulin, typically diagnosed in childhood. Type 2 diabetes involves insulin resistance and relative insulin deficiency, usually developing in adults. Treatment differs: type 1 requires insulin therapy, while type 2 may be managed with lifestyle changes and oral medications.",
        },
        {
            "id": "p8",
            "text": "Natural smoking cessation methods include behavioral therapy, support groups, nicotine replacement therapy, and identifying triggers. The combination of counseling and medication has the highest success rates. Most people need multiple attempts before successfully quitting.",
        },
        {
            "id": "p9",
            "text": "High blood pressure in young adults can be caused by obesity, excessive alcohol consumption, stress, kidney disease, or genetic factors. Secondary hypertension from underlying conditions is more common in younger patients than in older adults.",
        },
        {
            "id": "p10",
            "text": "Photosynthesis converts light energy into chemical energy through chlorophyll in plant leaves. Light energy splits water molecules, releasing oxygen. The energy is used to combine carbon dioxide and water into glucose, which plants use for growth and cellular respiration.",
        },
        # Add some irrelevant passages
        {
            "id": "p11",
            "text": "Stock market indices track the performance of baskets of stocks and are used as benchmarks for investment portfolios.",
        },
        {
            "id": "p12",
            "text": "The mitochondria is the powerhouse of the cell, generating ATP through cellular respiration.",
        },
        {
            "id": "p13",
            "text": "Quantum computing uses qubits instead of classical bits to perform calculations using superposition and entanglement.",
        },
        {
            "id": "p14",
            "text": "The Great Wall of China was built over centuries to protect against northern invasions and spans thousands of miles.",
        },
        {
            "id": "p15",
            "text": "JavaScript is a programming language primarily used for web development and runs in browsers.",
        },
    ],
    "qrels": {
        "how does national weather service forecast weather": {
            "p1": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
        "what is the treatment for atrial fibrillation": {
            "p2": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
        "why does my cat keep meowing at night": {
            "p3": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
        "how to calculate mortgage payment formula": {
            "p4": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
        "what are the symptoms of vitamin d deficiency": {
            "p5": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
        "how does a heat pump work in winter": {
            "p6": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
        "what is the difference between type 1 and type 2 diabetes": {
            "p7": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
        "how to stop smoking cigarettes naturally": {
            "p8": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
        "what causes high blood pressure in young adults": {
            "p9": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
        "how does photosynthesis work in plants": {
            "p10": 2,
            "p11": 0,
            "p12": 0,
            "p13": 0,
            "p14": 0,
            "p15": 0,
        },
    },
}


def prepare_benchmark_data() -> list[dict[str, Any]]:
    """Prepare benchmark data from MS-MARCO sample."""
    rows = []

    for query in MSMARCO_DEV_SAMPLE["queries"]:
        relevant_docs = MSMARCO_DEV_SAMPLE["qrels"].get(query, {})

        for passage in MSMARCO_DEV_SAMPLE["passages"]:
            doc_id = passage["id"]
            relevance = relevant_docs.get(doc_id, 0)

            rows.append(
                {
                    "query": query,
                    "doc": passage["text"],
                    "score": relevance,
                    "query_id": query,
                    "doc_id": doc_id,
                }
            )

    print(f"Prepared {len(rows)} query-doc pairs from {len(MSMARCO_DEV_SAMPLE['queries'])} queries")
    return rows


class FlashRankWrapper:
    """Wrapper for FlashRank."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        from flashrank import Ranker

        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, docs: list[str]) -> list[dict[str, Any]]:
        from flashrank import RerankRequest

        passages = [{"id": str(i), "text": doc} for i, doc in enumerate(docs)]
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        ranked = []
        for result in results:
            idx = int(result["id"])
            ranked.append(
                {
                    "doc": docs[idx],
                    "score": float(result.get("score", 0.0)),
                    "rank": 0,
                }
            )

        for rank, doc in enumerate(ranked, start=1):
            doc["rank"] = rank

        return ranked


def evaluate_benchmark() -> dict[str, Any]:
    """Run benchmark on real MS-MARCO style data."""
    print("\n=== Benchmark on MS-MARCO Style Real Data ===")

    rows = prepare_benchmark_data()

    results = {
        "dataset": "msmarco_style",
        "num_queries": len(set(r["query"] for r in rows)),
        "num_pairs": len(rows),
        "strategies": {},
    }

    # Train on subset
    train_rows = rows[:100]
    eval_rows = rows

    print(f"Training on {len(train_rows)} samples")

    # Train strategies
    print("\nTraining current strategies...")
    binary_labels = [1 if int(row["score"]) > 0 else 0 for row in train_rows]

    hybrid = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit(
        queries=[str(row["query"]) for row in train_rows],
        docs=[str(row["doc"]) for row in train_rows],
        labels=binary_labels,
    )

    binary = BinaryQuantizedReranker().fit(
        queries=[str(row["query"]) for row in train_rows],
        docs=[str(row["doc"]) for row in train_rows],
        labels=binary_labels,
    )

    # Evaluate
    print("\nEvaluating strategies...")

    # FlashRank TinyBERT
    print("Evaluating flashrank_tiny...")
    fr_tiny = FlashRankWrapper("ms-marco-TinyBERT-L-2-v2")
    results["strategies"]["flashrank_tiny"] = _evaluate(fr_tiny, eval_rows)
    print(f"flashrank_tiny: NDCG@10={results['strategies']['flashrank_tiny']['ndcg@10']:.4f}")

    # FlashRank MiniLM
    print("Evaluating flashrank_mini...")
    fr_mini = FlashRankWrapper("ms-marco-MiniLM-L-12-v2")
    results["strategies"]["flashrank_mini"] = _evaluate(fr_mini, eval_rows)
    print(f"flashrank_mini: NDCG@10={results['strategies']['flashrank_mini']['ndcg@10']:.4f}")

    # Hybrid
    print("Evaluating hybrid...")
    results["strategies"]["hybrid"] = _evaluate(hybrid, eval_rows)
    print(f"hybrid: NDCG@10={results['strategies']['hybrid']['ndcg@10']:.4f}")

    # Binary
    print("Evaluating binary_reranker...")
    results["strategies"]["binary_reranker"] = _evaluate(binary, eval_rows)
    print(f"binary_reranker: NDCG@10={results['strategies']['binary_reranker']['ndcg@10']:.4f}")

    # Compute relative performance
    print("\n=== Relative Performance ===")
    baseline = results["strategies"].get("hybrid", {}).get("ndcg@10", 0.0)
    if baseline > 0:
        for name, metrics in results["strategies"].items():
            ndcg = metrics.get("ndcg@10", 0.0)
            if ndcg > 0:
                uplift = ((ndcg - baseline) / baseline) * 100
                results["strategies"][name]["ndcg_uplift_vs_hybrid_pct"] = round(uplift, 2)
                print(f"{name}: {uplift:+.1f}% vs hybrid baseline")

    return results


def _evaluate(reranker: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    """Evaluate reranker."""
    latency = LatencyTracker()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query"]), []).append(row)

    ndcgs: list[float] = []
    mrrs: list[float] = []
    p1s: list[float] = []

    for query, items in grouped.items():
        docs = [str(item["doc"]) for item in items]

        with latency.measure():
            ranked = reranker.rerank(query, docs)

        doc_to_relevance = {str(item["doc"]): int(item["score"]) for item in items}

        # Handle both dict (FlashRank) and RankedDoc (current) formats
        relevances = []
        for r in ranked:
            if isinstance(r, dict):
                doc_text = r["doc"]
            else:
                doc_text = r.doc
            relevances.append(float(doc_to_relevance.get(doc_text, 0)))

        binary = [1 if rel > 0 else 0 for rel in relevances]

        ndcgs.append(ndcg_at_k(relevances, 10))
        mrrs.append(reciprocal_rank(binary))
        p1s.append(precision_at_k(binary, 1))

    summary = latency.summary()

    return {
        "ndcg@10": round(float(np.mean(ndcgs)) if ndcgs else 0.0, 4),
        "mrr": round(float(np.mean(mrrs)) if mrrs else 0.0, 4),
        "p@1": round(float(np.mean(p1s)) if p1s else 0.0, 4),
        "latency_p50_ms": round(float(summary["p50"]), 4),
        "latency_p99_ms": round(float(summary["p99"]), 4),
        "queries_evaluated": len(grouped),
    }


if __name__ == "__main__":
    results = evaluate_benchmark()

    print("\n=== Full Results ===")
    print(json.dumps(results, indent=2))

    # Save
    out_path = Path("data/logs/msmarco_style_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
