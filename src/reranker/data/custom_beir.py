"""Custom BEIR dataset loader for domain-specific datasets.

This module provides functionality to load custom BEIR-format JSON files
for ensemble distillation. Users can provide their own datasets in BEIR format.
"""

from __future__ import annotations

from pathlib import Path

from reranker.utils import read_json


def load_custom_beir(path: Path | str) -> dict:
    """Load a custom BEIR-format dataset from a JSON file.

    Expected format:
    {
        "queries": {
            "q1": "query text 1",
            "q2": "query text 2"
        },
        "corpus": {
            "doc1": {"text": "document text 1"},
            "doc2": {"text": "document text 2"}
        },
        "qrels": {
            "q1": {"doc1": 2, "doc2": 1},
            "q2": {"doc2": 2}
        }
    }

    Args:
        path: Path to the JSON file containing BEIR-format data.

    Returns:
        Dictionary with keys:
        - "queries": Dict mapping query IDs to query text
        - "corpus": Dict mapping doc IDs to extracted document text
        - "qrels": Dict mapping query IDs to dicts of doc IDs to relevance scores

    Raises:
        ValueError: If the JSON is invalid, missing required keys, or contains empty data.
    """
    path = Path(path)

    # Load and validate JSON
    try:
        data = read_json(path)
    except Exception as e:
        raise ValueError(f"Invalid JSON file: {e}") from e

    # Validate required top-level keys
    required_keys = {"queries", "corpus", "qrels"}
    missing_keys = required_keys - data.keys()
    if missing_keys:
        raise ValueError(f"Missing required keys: {', '.join(sorted(missing_keys))}")

    # Extract and validate queries
    queries = data["queries"]
    if not queries:
        raise ValueError("'queries' dictionary is empty")

    # Extract and validate corpus
    corpus_raw = data["corpus"]
    if not corpus_raw:
        raise ValueError("'corpus' dictionary is empty")

    # Normalize corpus to standard BEIR format: {"_id": ..., "title": ..., "text": ...}
    corpus = {}
    for doc_id, doc_data in corpus_raw.items():
        # Handle both string format and dict format
        if isinstance(doc_data, str):
            # Simple string format - treat as text
            corpus[doc_id] = {
                "_id": doc_id,
                "title": "",
                "text": doc_data
            }
        elif isinstance(doc_data, dict):
            # Dict format - extract fields with defaults
            corpus[doc_id] = {
                "_id": doc_id,
                "title": doc_data.get("title", ""),
                "text": doc_data.get("text", "")
            }
        else:
            raise ValueError(f"Invalid corpus data format for doc {doc_id}: {type(doc_data)}")

    # Extract qrels
    qrels = data["qrels"]

    return {
        "queries": queries,
        "corpus": corpus,
        "qrels": qrels,
    }
