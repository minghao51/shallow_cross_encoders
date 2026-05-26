"""Focused unit tests for cli.train helper functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from reranker.cli import train
from reranker.config import clear_settings_override, get_settings


def test_auto_label_meta_router_categories() -> None:
    rows = [
        {"query": "hi"},
        {"query": "this is medium"},
        {"query": "this query has many tokens and should be long"},
    ]
    assert train._auto_label_meta_router_categories(rows) == [0, 1, 2]


def test_partition_train_rows_falls_back_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    class Eval:
        train_ratio = 0.7
        validation_ratio = 0.15
        test_ratio = 0.15

    class Settings:
        eval = Eval()

    rows = [{"query": "q1", "score": 1}, {"query": "q2", "score": 2}]

    monkeypatch.setattr(
        "reranker.data.splits.partition_rows",
        lambda _rows, key_fn, split, ratios: [rows[0]],
    )
    selected = train._partition_train_rows(
        Settings(),
        rows,
        key_fn=lambda row: str(row["query"]),
        fallback_needed=lambda items: len(items) < 2,
    )
    assert selected == rows


def test_partition_train_rows_uses_partition_when_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    class Eval:
        train_ratio = 0.7
        validation_ratio = 0.15
        test_ratio = 0.15

    class Settings:
        eval = Eval()

    rows = [{"query": "q1", "score": 1}, {"query": "q2", "score": 2}]
    monkeypatch.setattr(
        "reranker.data.splits.partition_rows",
        lambda _rows, key_fn, split, ratios: list(rows),
    )
    selected = train._partition_train_rows(
        Settings(),
        rows,
        key_fn=lambda row: str(row["query"]),
        fallback_needed=lambda items: len(items) < 2,
    )
    assert selected == rows


def test_apply_config_sets_override(tmp_path: Path) -> None:
    cfg = tmp_path / "override.yaml"
    cfg.write_text("embedder:\n  model_name: unit/test-model\n", encoding="utf-8")
    try:
        train._apply_config(cfg)
        assert get_settings().embedder.model_name == "unit/test-model"
    finally:
        clear_settings_override()


def test_print_train_summary_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        "reranker.eval.runner.evaluate_strategy",
        lambda strategy, split, data_root, model_root: {"ndcg@10": 0.42},
    )
    train._print_train_summary("hybrid", Path("/tmp/model.pkl"), 10, Path("/tmp"), Path("/tmp"))
    captured = capsys.readouterr()
    assert "saved_model=/tmp/model.pkl" in captured.out
    assert "test_ndcg@10=0.4200" in captured.out


def test_print_train_summary_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("reranker.eval.runner.evaluate_strategy", _raise)
    train._print_train_summary("hybrid", Path("/tmp/model.pkl"), 10, Path("/tmp"), Path("/tmp"))
    captured = capsys.readouterr()
    assert "saved_model=/tmp/model.pkl" in captured.out
    assert "evaluation_skipped: boom" in captured.err


def _settings_with_model_dir(model_dir: Path):
    class Eval:
        train_ratio = 0.7
        validation_ratio = 0.15
        test_ratio = 0.15

    class Paths:
        raw_data_dir = ""
        model_dir = Path(".")

    Paths.raw_data_dir = str(model_dir / "raw")
    Paths.model_dir = model_dir

    class Settings:
        eval = Eval()
        paths = Paths()

    return Settings()


def test_train_hybrid_runs_with_stubbed_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from reranker.strategies import hybrid as hybrid_module

    class StubReranker:
        model_backend = "sklearn"

        def fit_pointwise(self, queries, docs, scores):
            assert queries and docs and scores

        def save(self, path):
            Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(hybrid_module, "HybridFusionReranker", lambda adapters=None: StubReranker())
    monkeypatch.setattr(train, "_print_train_summary", lambda *args, **kwargs: None)
    rows = [{"query": "q", "doc": "d", "score": 3}]
    settings = _settings_with_model_dir(tmp_path / "models")
    monkeypatch.setattr(train, "_load_rows", lambda *args, **kwargs: (settings, tmp_path, rows))
    monkeypatch.setattr(train, "_partition_train_rows", lambda *args, **kwargs: rows)
    train.train_hybrid(output=tmp_path / "hybrid.pkl")
    assert (tmp_path / "hybrid.pkl").exists()


def test_train_distilled_and_binary_run_with_stubs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from reranker.strategies import binary_reranker as binary_module
    from reranker.strategies import distilled as distilled_module

    class StubDistilled:
        def fit(self, queries, doc_as, doc_bs, labels):
            assert queries and doc_as and doc_bs and labels
            return self

        def save(self, path):
            Path(path).write_text("ok", encoding="utf-8")

    class StubBinary:
        def fit(self, queries, docs, labels):
            assert queries and docs and labels
            return self

        def save(self, path):
            Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr(distilled_module, "DistilledPairwiseRanker", StubDistilled)
    monkeypatch.setattr(binary_module, "BinaryQuantizedReranker", StubBinary)
    monkeypatch.setattr(train, "_print_train_summary", lambda *args, **kwargs: None)

    pref_rows = [{"query": "q", "doc_a": "a", "doc_b": "b", "preferred": "A"}]
    pair_rows = [{"query": "q", "doc": "d", "score": 2}]
    settings = _settings_with_model_dir(tmp_path / "models")

    monkeypatch.setattr(
        train, "_load_rows", lambda *args, **kwargs: (settings, tmp_path, pref_rows)
    )
    monkeypatch.setattr(train, "_partition_train_rows", lambda *args, **kwargs: pref_rows)
    train.train_distilled(output=tmp_path / "distilled.pkl")
    assert (tmp_path / "distilled.pkl").exists()

    monkeypatch.setattr(
        train, "_load_rows", lambda *args, **kwargs: (settings, tmp_path, pair_rows)
    )
    monkeypatch.setattr(train, "_partition_train_rows", lambda *args, **kwargs: pair_rows)
    train.train_binary(output=tmp_path / "binary.pkl")
    assert (tmp_path / "binary.pkl").exists()


def test_train_late_interaction_splade_cascade_meta_router(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from reranker.strategies import cascade as cascade_module
    from reranker.strategies import hybrid as hybrid_module
    from reranker.strategies import late_interaction as li_module
    from reranker.strategies import meta_router as mr_module
    from reranker.strategies import splade as splade_module

    class StubLI:
        def fit(self, docs):
            assert docs

        def save(self, path):
            Path(path).write_text("ok", encoding="utf-8")

    class StubSplade:
        def __init__(self, top_k_terms=128):
            self.top_k_terms = top_k_terms

        def fit(self, docs):
            assert docs and self.top_k_terms == 64

        def save(self, path):
            Path(path).write_text("ok", encoding="utf-8")

    class StubHybrid:
        def __init__(self, adapters=None):
            self.adapters = adapters

        def fit_pointwise(self, queries, docs, scores):
            assert queries and docs and scores

        def save(self, path):
            Path(path).write_text("ok", encoding="utf-8")

    class StubRouter:
        def __init__(self):
            self.embedder = type("E", (), {"model_name": "stub"})()
            self.n_categories = 3
            self.min_samples_leaf = 2
            self.model = {"ok": True}
            self.is_fitted = False

        def fit(self, queries, categories):
            assert queries and categories
            self.is_fitted = True

    monkeypatch.setattr(li_module, "StaticColBERTReranker", StubLI)
    monkeypatch.setattr(splade_module, "SPLADEReranker", StubSplade)
    monkeypatch.setattr(hybrid_module, "HybridFusionReranker", StubHybrid)
    monkeypatch.setattr(cascade_module, "CascadeReranker", lambda **kwargs: object())
    monkeypatch.setattr(mr_module, "MetaRouter", StubRouter)

    saved_paths: list[Path] = []

    def _save_safe(path, artifact_type, metadata, weights):
        del artifact_type, metadata, weights
        saved_paths.append(Path(path))
        Path(path).write_text("ok", encoding="utf-8")

    monkeypatch.setattr("reranker.persistence.save_safe", _save_safe)
    monkeypatch.setattr(train, "_print_train_summary", lambda *args, **kwargs: None)

    rows = [
        {"query": "query text", "doc": "doc one", "score": 2},
        {"query": "query text", "doc": "doc two", "score": 1},
    ]
    settings = _settings_with_model_dir(tmp_path / "models")
    monkeypatch.setattr(train, "_load_rows", lambda *args, **kwargs: (settings, tmp_path, rows))
    monkeypatch.setattr(train, "_partition_train_rows", lambda *args, **kwargs: rows)

    train.train_late_interaction(output=tmp_path / "li.pkl")
    train.train_splade(output=tmp_path / "splade.pkl", top_k_terms=64)
    train.train_cascade(output=tmp_path / "cascade.pkl", confidence_threshold=0.7)
    train.train_meta_router(output=tmp_path / "router.pkl")

    assert (tmp_path / "li.pkl").exists()
    assert (tmp_path / "splade.pkl").exists()
    assert (tmp_path / "cascade.pkl").exists()
    assert (tmp_path / "router.pkl").exists()
    assert saved_paths

    output_text = capsys.readouterr().out
    assert "saved_model=" in output_text
