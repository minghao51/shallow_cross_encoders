"""BEIR dataset loader.

This module provides functions to load BEIR-format datasets for both
simple and comprehensive use cases (e.g., distillation vs benchmarking).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_beir_simple(dataset_name: str = "nfcorpus") -> tuple[dict, dict, dict]:
    """Load BEIR dataset in simple format for distillation.

    This function loads BEIR datasets and returns them in a simplified format
    suitable for ensemble distillation workflows.

    Args:
        dataset_name: Name of BEIR dataset (default: "nfcorpus")

    Returns:
        Tuple of (queries, corpus, qrels) where:
            - queries: Mapping from query_id to query text
            - corpus: Mapping from doc_id to doc dict with _id, title, text
            - qrels: Mapping from query_id to {doc_id: relevance_score}

    Raises:
        FileNotFoundError: If BEIR dataset not found.
        ImportError: If beir package not installed.
        ValueError: If dataset format is invalid.

    Example:
        >>> queries, corpus, qrels = load_beir_simple("nfcorpus")
        >>> print(f"Loaded {len(queries)} queries, {len(corpus)} docs")
    """
    try:
        from beir import util
    except ImportError as e:
        raise ImportError("BEIR not installed. Run: uv pip install beir") from e

    beir_dir = Path("data/beir") / dataset_name

    if not beir_dir.exists():
        url = (
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        )
        print(f"Downloading {dataset_name} from {url}")
        util.download_and_unzip(url, str(beir_dir.parent))

    # Parse corpus.jsonl
    corpus_file = beir_dir / "corpus.jsonl"
    if not corpus_file.exists():
        raise FileNotFoundError(f"corpus.jsonl not found in {beir_dir}")

    corpus = {}
    with open(corpus_file) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "_id": doc["_id"],
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }

    # Parse queries.jsonl
    queries_file = beir_dir / "queries.jsonl"
    if not queries_file.exists():
        raise FileNotFoundError(f"queries.jsonl not found in {beir_dir}")

    queries = {}
    with open(queries_file) as f:
        for line in f:
            item = json.loads(line)
            queries[item["_id"]] = item["text"]

    # Parse qrels/test.tsv
    qrels_file = beir_dir / "qrels" / "test.tsv"
    if not qrels_file.exists():
        raise FileNotFoundError(f"qrels/test.tsv not found in {beir_dir}")

    qrels_dict: defaultdict[str, dict[str, int]] = defaultdict(dict)
    with open(qrels_file) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                qrels_dict[query_id][doc_id] = score

    qrels = dict(qrels_dict)
    return queries, corpus, qrels


def load_beir_comprehensive(dataset_path: Path) -> dict[str, Any]:
    """Load BEIR dataset in comprehensive format for benchmarking.

    This function loads BEIR datasets with support for both JSONL and TSV
    formats, providing a more comprehensive representation for advanced benchmarking.

    Args:
        dataset_path: Path to BEIR dataset directory.

    Returns:
        Dictionary with keys:
        - corpus: Mapping from doc_id to doc dict with _id, title, text
        - queries: Mapping from query_id to query text
        - qrels: Mapping from query_id to {doc_id: relevance_score}

    Raises:
        FileNotFoundError: If dataset or required files not found.
        ValueError: If dataset format is invalid.

    Example:
        >>> data = load_beir_comprehensive(Path("data/beir/nfcorpus"))
        >>> print(f"Loaded {len(data['queries'])} queries")
    """
    print(f"Loading dataset from {dataset_path}...")

    corpus: dict[str, Any] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)

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

    if corpus_file is None:
        raise FileNotFoundError(f"corpus file not found in {dataset_path}")

    # Load corpus
    if corpus_file.suffix == ".jsonl":
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

    elif corpus_file.suffix == ".tsv":
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

    if queries_file is None:
        raise FileNotFoundError(f"queries file not found in {dataset_path}")

    if queries_file.suffix == ".jsonl":
        with open(queries_file) as f:
            for line in f:
                item = json.loads(line)
                q_id = item.get("_id", item.get("query_id", ""))
                query_text = item.get("text", item.get("query", ""))
                queries[str(q_id)] = query_text
        print(f"Loaded {len(queries)} queries from {queries_file}")

    elif queries_file.suffix == ".tsv":
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
                            q_id, _, doc_id, rel = (
                                parts[0],
                                parts[1],
                                parts[2],
                                int(parts[3]),
                            )
                        if rel > 0:
                            qrels[str(q_id)][str(doc_id)] = rel
        print(f"Loaded qrels for {len(qrels)} queries")

    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": dict(qrels),
    }
