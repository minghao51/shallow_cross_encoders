"""Tests for custom BEIR dataset loader."""

from pathlib import Path

import pytest

from reranker.data.custom_beir import load_custom_beir


def test_load_custom_beir_valid(tmp_path: Path) -> None:
    """Valid BEIR JSON loads correctly and extracts corpus texts."""
    # Create a valid BEIR-format JSON file
    data = {
        "queries": {
            "q1": "What is machine learning?",
            "q2": "How does deep learning work?",
        },
        "corpus": {
            "doc1": {"text": "Machine learning is a subset of AI."},
            "doc2": {"text": "Deep learning uses neural networks."},
            "doc3": {
                "text": "Neural networks have multiple layers.",
                "title": "Neural Networks",
            },
        },
        "qrels": {
            "q1": {"doc1": 2, "doc3": 1},
            "q2": {"doc2": 2, "doc3": 1},
        },
    }

    beir_file = tmp_path / "custom_beir.json"
    import json

    beir_file.write_text(json.dumps(data))

    # Load the data
    result = load_custom_beir(beir_file)

    # Verify structure
    assert "queries" in result
    assert "corpus" in result
    assert "qrels" in result

    # Verify queries
    assert result["queries"]["q1"] == "What is machine learning?"
    assert result["queries"]["q2"] == "How does deep learning work?"

    # Verify corpus texts are extracted (normalized to dict format)
    assert result["corpus"]["doc1"] == {
        "_id": "doc1",
        "title": "",
        "text": "Machine learning is a subset of AI.",
    }
    assert result["corpus"]["doc2"] == {
        "_id": "doc2",
        "title": "",
        "text": "Deep learning uses neural networks.",
    }
    assert result["corpus"]["doc3"] == {
        "_id": "doc3",
        "title": "Neural Networks",
        "text": "Neural networks have multiple layers.",
    }

    # Verify qrels
    assert result["qrels"]["q1"]["doc1"] == 2
    assert result["qrels"]["q1"]["doc3"] == 1
    assert result["qrels"]["q2"]["doc2"] == 2


def test_load_custom_beir_missing_keys(tmp_path: Path) -> None:
    """Missing required keys raises ValueError."""
    import json

    # Test missing 'queries'
    data = {
        "corpus": {"doc1": {"text": "Some text"}},
        "qrels": {"q1": {"doc1": 1}},
    }
    beir_file = tmp_path / "missing_queries.json"
    beir_file.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Missing required keys: queries"):
        load_custom_beir(beir_file)

    # Test missing 'corpus'
    data = {
        "queries": {"q1": "query"},
        "qrels": {"q1": {"doc1": 1}},
    }
    beir_file = tmp_path / "missing_corpus.json"
    beir_file.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Missing required keys: corpus"):
        load_custom_beir(beir_file)

    # Test missing 'qrels'
    data = {
        "queries": {"q1": "query"},
        "corpus": {"doc1": {"text": "text"}},
    }
    beir_file = tmp_path / "missing_qrels.json"
    beir_file.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="Missing required keys: qrels"):
        load_custom_beir(beir_file)


def test_load_custom_beir_empty(tmp_path: Path) -> None:
    """Empty queries or corpus raises ValueError."""
    import json

    # Test empty queries
    data = {
        "queries": {},
        "corpus": {"doc1": {"text": "text"}},
        "qrels": {"q1": {"doc1": 1}},
    }
    beir_file = tmp_path / "empty_queries.json"
    beir_file.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="'queries' dictionary is empty"):
        load_custom_beir(beir_file)

    # Test empty corpus
    data = {
        "queries": {"q1": "query"},
        "corpus": {},
        "qrels": {"q1": {"doc1": 1}},
    }
    beir_file = tmp_path / "empty_corpus.json"
    beir_file.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="'corpus' dictionary is empty"):
        load_custom_beir(beir_file)


def test_load_custom_beir_invalid_json(tmp_path: Path) -> None:
    """Invalid JSON raises ValueError."""
    beir_file = tmp_path / "invalid.json"
    beir_file.write_text("{invalid json content")

    with pytest.raises(ValueError, match="Invalid JSON file"):
        load_custom_beir(beir_file)
