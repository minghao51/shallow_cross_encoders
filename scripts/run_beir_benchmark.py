"""Run comprehensive BEIR benchmark with harder negatives.

Loads TREC-COVID or NFCorpus and compares all reranking strategies.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from reranker.eval.metrics import LatencyTracker, ndcg_at_k, precision_at_k, reciprocal_rank
from reranker.strategies.binary_reranker import BinaryQuantizedReranker
from reranker.strategies.hybrid import HybridFusionReranker, KeywordMatchAdapter
from reranker.strategies.late_interaction import StaticColBERTReranker


def load_beir_dataset(dataset_path: Path) -> dict[str, Any]:
    """Load BEIR dataset from path."""
    print(f"Loading dataset from {dataset_path}...")

    corpus = {}
    queries = {}
    qrels = defaultdict(dict)

    # Try to find corpus file
    corpus_file = None
    for possible_name in ["corpus.jsonl", "collection.tsv", "docs.tsv"]:
        test_path = dataset_path / possible_name
        if test_path.exists():
            corpus_file = test_path
            break

    if corpus_file is None:
        # Search subdirectories
        for subdir in dataset_path.rglob("*"):
            if subdir.is_file() and "corpus" in subdir.name.lower():
                corpus_file = subdir
                break

    if corpus_file and corpus_file.suffix == ".jsonl":
        with open(corpus_file) as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("_id", doc.get("doc_id", ""))
                title = doc.get("title", "")
                text = doc.get("text", "")
                corpus[str(doc_id)] = {
                    "_id": str(doc_id),
                    "title": title,
                    "text": f"{title} {text}" if title else text,
                }
        print(f"Loaded {len(corpus)} documents from {corpus_file}")

    elif corpus_file and corpus_file.suffix == ".tsv":
        with open(corpus_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    doc_id = parts[0]
                    text = parts[1] if len(parts) > 1 else ""
                    title = parts[2] if len(parts) > 2 else ""
                    corpus[str(doc_id)] = {
                        "_id": str(doc_id),
                        "title": title,
                        "text": f"{title} {text}" if title else text,
                    }
        print(f"Loaded {len(corpus)} documents from {corpus_file}")

    # Load queries
    queries_file = None
    for possible_name in [
        "queries.jsonl",
        "queries.tsv",
        "test.queries.jsonl",
        "train.queries.jsonl",
    ]:
        test_path = dataset_path / possible_name
        if test_path.exists():
            queries_file = test_path
            break

    if queries_file and queries_file.suffix == ".jsonl":
        with open(queries_file) as f:
            for line in f:
                item = json.loads(line)
                q_id = item.get("_id", item.get("query_id", ""))
                query_text = item.get("text", item.get("query", ""))
                queries[str(q_id)] = query_text
        print(f"Loaded {len(queries)} queries from {queries_file}")

    elif queries_file and queries_file.suffix == ".tsv":
        with open(queries_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    q_id, query_text = parts[0], parts[1]
                    queries[str(q_id)] = query_text
        print(f"Loaded {len(queries)} queries from {queries_file}")

    # Load qrels
    qrels_dir = dataset_path / "qrels"
    if qrels_dir.exists():
        for qrels_file in qrels_dir.glob("*.tsv"):
            with open(qrels_file) as f:
                f.readline()
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        # Handle both formats:
                        # "query-id\tcorpus-id\tscore" and "q_id\t0\tdoc_id\trel"
                        if len(parts) == 3:
                            q_id, doc_id, rel = parts[0], parts[1], int(parts[2])
                        else:
                            q_id, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
                        if rel > 0:
                            qrels[str(q_id)][str(doc_id)] = rel
        print(f"Loaded qrels for {len(qrels)} queries")

    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": dict(qrels),
    }


def prepare_benchmark_data_with_hard_negatives(
    dataset: dict[str, Any],
    num_queries: int = 50,
    docs_per_query: int = 100,
    hard_negative_ratio: float = 0.7,
) -> list[dict[str, Any]]:
    """Prepare benchmark data with BM25 hard negatives.

    Uses BM25 to retrieve top candidates, then adds random hard negatives
    that are similar but not relevant.
    """
    from rank_bm25 import BM25Okapi

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]

    # Limit queries
    query_ids = list(queries.keys())[:num_queries]
    print(f"Using {len(query_ids)} queries for benchmark")

    # Build corpus index
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid]["text"][:500] for cid in corpus_ids]  # Truncate for speed
    tokenized_corpus = [text.lower().split() for text in corpus_texts]

    # Build BM25 index
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    rows = []

    for q_id in query_ids:
        query = queries[q_id]
        relevant_docs = set(qrels.get(q_id, {}).keys())

        # Get BM25 scores
        tokenized_query = query.lower().split()
        doc_scores = bm25.get_scores(tokenized_query)

        # Sort by BM25 score
        sorted_indices = np.argsort(doc_scores)[::-1]

        # Separate relevant and non-relevant
        relevant_in_top = []
        non_relevant_in_top = []

        for idx in sorted_indices[: docs_per_query * 2]:  # Get more candidates
            doc_id = corpus_ids[idx]
            if doc_id in relevant_docs:
                relevant_in_top.append(doc_id)
            else:
                non_relevant_in_top.append(doc_id)

        # Select candidates
        selected_docs = []

        # Add all relevant docs
        selected_docs.extend(relevant_in_top)

        # Add hard negatives (high BM25 score but not relevant)
        num_hard_negatives = int(docs_per_query * hard_negative_ratio)
        hard_negatives = non_relevant_in_top[:num_hard_negatives]
        selected_docs.extend(hard_negatives)

        # Add random docs to fill
        while len(selected_docs) < docs_per_query:
            remaining = [cid for cid in corpus_ids if cid not in selected_docs]
            if not remaining:
                break
            selected_docs.append(np.random.choice(remaining))

        # Create rows
        for doc_id in selected_docs[:docs_per_query]:
            doc = corpus[doc_id]
            relevance = qrels.get(q_id, {}).get(doc_id, 0)

            rows.append(
                {
                    "query": query,
                    "doc": doc["text"][:800],  # Truncate for FlashRank token limit
                    "score": int(relevance),
                    "query_id": q_id,
                    "doc_id": doc_id,
                }
            )

    print(f"Created {len(rows)} query-doc pairs")
    print(f"Relevant pairs: {sum(1 for r in rows if r['score'] > 0)}")
    print(f"Non-relevant pairs: {sum(1 for r in rows if r['score'] == 0)}")

    return rows


class FlashRankWrapper:
    """Wrapper for FlashRank."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        from flashrank import Ranker

        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, docs: list[str]) -> list[dict[str, Any]]:
        from flashrank import RerankRequest

        # Truncate docs for token limit
        truncated_docs = [doc[:500] for doc in docs]
        passages = [{"id": str(i), "text": doc} for i, doc in enumerate(truncated_docs)]
        request = RerankRequest(query=query[:200], passages=passages)  # Truncate query too
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


def evaluate_all_strategies(
    dataset_name: str = "nfcorpus",
    num_queries: int = 50,
    docs_per_query: int = 100,
) -> dict[str, Any]:
    """Run comprehensive benchmark on BEIR dataset."""
    print(f"\n{'=' * 60}")
    print(f"BEIR Benchmark: {dataset_name}")
    print(f"{'=' * 60}")

    # Load dataset
    dataset_path = Path(f"data/beir/{dataset_name}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run download_beir.py first.")

    dataset = load_beir_dataset(dataset_path)

    # Prepare benchmark data
    rows = prepare_benchmark_data_with_hard_negatives(
        dataset,
        num_queries=num_queries,
        docs_per_query=docs_per_query,
    )

    results = {
        "dataset": dataset_name,
        "num_queries": len(set(r["query_id"] for r in rows)),
        "num_pairs": len(rows),
        "relevant_pairs": sum(1 for r in rows if r["score"] > 0),
        "strategies": {},
    }

    # Train on first 70%
    train_size = int(len(rows) * 0.7)
    train_rows = rows[:train_size]
    eval_rows = rows[train_size:]

    print(f"\nTraining on {len(train_rows)} samples, evaluating on {len(eval_rows)}")

    # Train strategies
    print("\n" + "=" * 60)
    print("Training strategies...")
    print("=" * 60)

    binary_labels = [1 if int(row["score"]) > 0 else 0 for row in train_rows]
    negative_count = len(binary_labels) - sum(binary_labels)
    print(f"Training labels: {sum(binary_labels)} positive, {negative_count} negative")

    # Hybrid Fusion
    print("\nTraining HybridFusionReranker...")
    hybrid = HybridFusionReranker(adapters=[KeywordMatchAdapter()]).fit(
        queries=[str(row["query"]) for row in train_rows],
        docs=[str(row["doc"]) for row in train_rows],
        labels=binary_labels,
    )

    # Binary Reranker
    print("Training BinaryQuantizedReranker...")
    binary = BinaryQuantizedReranker().fit(
        queries=[str(row["query"]) for row in train_rows],
        docs=[str(row["doc"]) for row in train_rows],
        labels=binary_labels,
    )

    # Late Interaction
    print("Training StaticColBERTReranker...")
    unique_docs = list(set(str(row["doc"]) for row in train_rows))
    late_interaction = StaticColBERTReranker()
    late_interaction.fit(unique_docs)

    # Distilled (skip for now - needs preferences)
    # distilled = DistilledPairwiseRanker()

    # Evaluate all strategies
    print("\n" + "=" * 60)
    print("Evaluating strategies...")
    print("=" * 60)

    strategies = [
        ("flashrank_tiny", FlashRankWrapper("ms-marco-TinyBERT-L-2-v2")),
        ("flashrank_mini", FlashRankWrapper("ms-marco-MiniLM-L-12-v2")),
        ("hybrid", hybrid),
        ("binary_reranker", binary),
        ("late_interaction", late_interaction),
    ]

    for name, reranker in strategies:
        print(f"\nEvaluating {name}...")
        try:
            metrics = _evaluate_reranker(reranker, eval_rows)
            results["strategies"][name] = metrics
            print(
                f"  NDCG@10={metrics['ndcg@10']:.4f}, P@1={metrics['p@1']:.4f}, "
                f"latency={metrics['latency_p50_ms']:.2f}ms"
            )
        except Exception as e:
            print(f"  Error: {e}")
            results["strategies"][name] = {"error": str(e)}

    # Compute relative performance
    print("\n" + "=" * 60)
    print("Relative Performance (vs Hybrid baseline)")
    print("=" * 60)

    baseline = results["strategies"].get("hybrid", {}).get("ndcg@10", 0.0)
    if baseline > 0:
        for name, metrics in results["strategies"].items():
            ndcg = metrics.get("ndcg@10", 0.0)
            if ndcg > 0:
                uplift = ((ndcg - baseline) / baseline) * 100
                results["strategies"][name]["ndcg_uplift_vs_hybrid_pct"] = round(uplift, 2)
                marker = "✓" if uplift > 5 else ("✗" if uplift < -5 else "~")
                print(f"  {marker} {name}: {uplift:+.1f}% vs hybrid")

    return results


def _evaluate_reranker(reranker: Any, rows: list[dict[str, Any]]) -> dict[str, float]:
    """Evaluate reranker on BEIR rows."""
    latency = LatencyTracker()

    # Group by query
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query_id"]), []).append(row)

    ndcgs: list[float] = []
    mrrs: list[float] = []
    p1s: list[float] = []

    for items in grouped.values():
        query = items[0]["query"]
        docs = [str(item["doc"]) for item in items]

        with latency.measure():
            ranked = reranker.rerank(query, docs)

        # Get relevance scores
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

        # Only compute metrics if we have at least one relevant doc
        if sum(binary) > 0:
            ndcgs.append(ndcg_at_k(relevances, 10))
            mrrs.append(reciprocal_rank(binary))
            p1s.append(precision_at_k(binary, 1))

    if not ndcgs:
        return {
            "ndcg@10": 0.0,
            "mrr": 0.0,
            "p@1": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p99_ms": 0.0,
            "queries_evaluated": 0,
        }

    summary = latency.summary()

    return {
        "ndcg@10": round(float(np.mean(ndcgs)), 4),
        "mrr": round(float(np.mean(mrrs)), 4),
        "p@1": round(float(np.mean(p1s)), 4),
        "latency_p50_ms": round(float(summary["p50"]), 4),
        "latency_p99_ms": round(float(summary["p99"]), 4),
        "queries_evaluated": len(grouped),
    }


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    num_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    results = evaluate_all_strategies(
        dataset_name=dataset,
        num_queries=num_queries,
        docs_per_query=100,
    )

    # Save results
    out_path = Path(f"data/logs/beir_{dataset}_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 60}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Strategy':<20} {'NDCG@10':<10} {'P@1':<10} {'Latency (ms)':<15}")
    print("-" * 60)
    for name, metrics in results["strategies"].items():
        if "error" not in metrics:
            print(
                f"{name:<20} {metrics['ndcg@10']:<10.4f} {metrics['p@1']:<10.4f} "
                f"{metrics['latency_p50_ms']:<15.2f}"
            )
