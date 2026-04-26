"""Hard negative sampling for benchmarking.

This module provides functions to generate hard negatives using BM25,
with caching support to avoid recomputing indexes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


class BM25IndexCache:
    """Cache for BM25 indexes to avoid recomputation.

    This class caches BM25 indexes on disk using content-based hashing
    to avoid rebuilding indexes for the same corpus.

    Example:
        >>> cache = BM25IndexCache(Path("data/cache/bm25"))
        >>> tokenized_corpus = cache.get_or_build(
        ...     corpus_texts,
        ...     build_fn=lambda: [text.lower().split() for text in corpus_texts]
        ... )
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize cache directory.

        Args:
            cache_dir: Directory path for cache storage.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, corpus_texts: list[str]) -> str:
        """Generate cache key from corpus content.

        Args:
            corpus_texts: List of corpus texts.

        Returns:
            16-character hexadecimal cache key.
        """
        # Hash corpus content for stable cache key
        combined = "\n".join(sorted(corpus_texts))
        return sha256(combined.encode()).hexdigest()[:16]

    def get_or_build(
        self, corpus_texts: list[str], build_fn: Callable[[], list[list[str]]]
    ) -> list[list[str]]:
        """Get tokenized corpus from cache or build it.

        Args:
            corpus_texts: Original corpus texts.
            build_fn: Function to build tokenized corpus if not cached.

        Returns:
            Tokenized corpus (list of token lists).
        """
        cache_key = self._get_cache_key(corpus_texts)
        cache_path = self.cache_dir / f"bm25_{cache_key}.json"

        if cache_path.exists():
            print(f"Loading BM25 index from cache: {cache_path}")
            with open(cache_path) as f:
                return json.load(f)

        print("Building BM25 index...")
        tokenized_corpus = build_fn()

        # Save to cache
        with open(cache_path, "w") as f:
            json.dump(tokenized_corpus, f)
        print(f"BM25 index cached to: {cache_path}")

        return tokenized_corpus


def prepare_benchmark_data_with_hard_negatives(
    dataset: dict[str, Any],
    num_queries: int = 50,
    docs_per_query: int = 100,
    hard_negative_ratio: float = 0.7,
    cache_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Prepare benchmark data with BM25 hard negatives.

    This function uses BM25 to retrieve top candidates, then adds hard
    negatives that are similar but not relevant. Includes BM25 index caching.

    Args:
        dataset: BEIR dataset dict with keys 'corpus', 'queries', 'qrels'.
        num_queries: Number of queries to use for benchmark.
        docs_per_query: Number of documents per query.
        hard_negative_ratio: Ratio of hard negatives (0.0-1.0).
        cache_dir: Directory for BM25 index cache. If None, uses default.

    Returns:
        List of benchmark rows with keys:
        - query: Query text
        - doc: Document text
        - score: Relevance score (higher = more relevant)
        - query_id: Query ID
        - doc_id: Document ID

    Raises:
        ValueError: If hard_negative_ratio not in [0, 1].
        ImportError: If rank_bm25 not installed.

    Example:
        >>> data = load_beir_comprehensive(Path("data/beir/nfcorpus"))
        >>> rows = prepare_benchmark_data_with_hard_negatives(data, num_queries=30)
        >>> print(f"Created {len(rows)} query-doc pairs")
    """
    if not 0.0 <= hard_negative_ratio <= 1.0:
        raise ValueError("hard_negative_ratio must be between 0.0 and 1.0")

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise ImportError(
            "rank_bm25 not installed. Install with: uv pip install rank-bm25"
        ) from e

    corpus = dataset["corpus"]
    queries = dataset["queries"]
    qrels = dataset["qrels"]

    # Limit queries
    query_ids = list(queries.keys())[:num_queries]
    print(f"Using {len(query_ids)} queries for benchmark")

    # Setup BM25 cache
    if cache_dir is None:
        cache_dir = Path("data/cache/bm25")

    cache = BM25IndexCache(cache_dir)

    # Build corpus index
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid]["text"][:500] for cid in corpus_ids]

    # Tokenize and cache
    tokenized_corpus = cache.get_or_build(
        corpus_texts,
        build_fn=lambda: [text.lower().split() for text in corpus_texts],
    )

    # Build BM25 index
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

        for idx in sorted_indices[: docs_per_query * 2]:
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
