"""Tests for BEIR loader."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


def test_load_beir_comprehensive_file_not_found():
    """Test load_beir_comprehensive raises FileNotFoundError for missing dataset."""
    from reranker.data.beir_loader import load_beir_comprehensive

    with pytest.raises(FileNotFoundError, match="corpus file not found"):
        load_beir_comprehensive(Path("/nonexistent/path"))


def test_load_beir_simple_downloads_and_parses_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test load_beir_simple triggers download and parses BEIR files."""
    from reranker.data.beir_loader import load_beir_simple

    called: dict[str, str] = {}

    def fake_download_and_unzip(url: str, save_path: str) -> None:
        called["url"] = url
        called["save_path"] = save_path
        dataset_dir = Path(save_path) / "nfcorpus"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "corpus.jsonl").write_text(
            json.dumps({"_id": "d1", "title": "Doc", "text": "alpha beta"}) + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "queries.jsonl").write_text(
            json.dumps({"_id": "q1", "text": "alpha"}) + "\n",
            encoding="utf-8",
        )
        qrels_dir = dataset_dir / "qrels"
        qrels_dir.mkdir(parents=True, exist_ok=True)
        (qrels_dir / "test.tsv").write_text(
            "query-id\tcorpus-id\tscore\nq1\td1\t2\n",
            encoding="utf-8",
        )

    fake_beir = types.ModuleType("beir")
    fake_beir.util = types.SimpleNamespace(download_and_unzip=fake_download_and_unzip)
    monkeypatch.setitem(sys.modules, "beir", fake_beir)
    monkeypatch.chdir(tmp_path)

    queries, corpus, qrels = load_beir_simple("nfcorpus")

    assert "nfcorpus.zip" in called["url"]
    assert called["save_path"].endswith("data/beir")
    assert queries == {"q1": "alpha"}
    assert corpus["d1"]["text"] == "alpha beta"
    assert qrels == {"q1": {"d1": 2}}
