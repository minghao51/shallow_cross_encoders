import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from reranker.config import (
    clear_settings_override,
    get_settings,
    reset_settings_cache,
    settings_from_dict,
)


def _load_benchmark_sweep_module():
    module_path = Path(__file__).resolve().parents[2] / "benchmarks" / "run_sweep.py"
    spec = importlib.util.spec_from_file_location("benchmark_sweep_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    benchmark_sweep = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark_sweep
    spec.loader.exec_module(benchmark_sweep)
    return benchmark_sweep


def test_settings_from_dict_applies_variant_weighting_mode(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pairs.jsonl").write_text(
        json.dumps({"query": "q", "doc": "d", "score": 1}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_sweep",
                "variants:",
                "  learned_variant:",
                "    hybrid:",
                '      weighting_mode: "learned"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RERANKER_PATHS__RAW_DATA_DIR", str(raw_dir))
    reset_settings_cache()

    try:
        from reranker.config import load_yaml_config

        yaml_data = load_yaml_config(config_path)
        variant_config = yaml_data["variants"]["learned_variant"]
        override = settings_from_dict(variant_config)

        assert override.hybrid.weighting_mode == "learned"
    finally:
        clear_settings_override()
        reset_settings_cache()


def test_run_sweep_cli_executable(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pairs.jsonl").write_text(
        json.dumps({"query": "test query", "doc": "test doc", "score": 1}) + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_sweep",
                "variants:",
                "  baseline_variant:",
                "    hybrid:",
                '      weighting_mode: "static"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RERANKER_PATHS__RAW_DATA_DIR", str(raw_dir))

    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "benchmarks/run_sweep.py", "--config", str(config_path)],
        capture_output=True,
        text=True,
        cwd=str(project_root),
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "test_sweep" in combined
    assert "baseline_variant" in combined


def test_run_sweep_reports_variant_specific_metrics(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pairs.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "doc": "doc one", "score": 3}),
                json.dumps({"query": "q1", "doc": "doc two", "score": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: metric_sweep",
                "variants:",
                "  static_variant:",
                "    hybrid:",
                '      weighting_mode: "static"',
                "  learned_variant:",
                "    hybrid:",
                '      weighting_mode: "learned"',
                "    late_interaction:",
                '      quantization: "4bit"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RERANKER_PATHS__RAW_DATA_DIR", str(raw_dir))
    reset_settings_cache()
    benchmark_sweep = _load_benchmark_sweep_module()

    def fake_build_hybrid(config_override, embedder):
        del config_override, embedder
        return object()

    def fake_evaluate_hybrid(reranker, pairs):
        del reranker, pairs
        mode = get_settings().hybrid.weighting_mode
        return {"ndcg@10": 0.9 if mode == "learned" else 0.4, "n_queries": 1.0}

    def fake_measure_latency(reranker, query, docs, n_runs=5):
        del reranker, query, docs, n_runs
        return 1.5 if get_settings().hybrid.weighting_mode == "learned" else 3.0

    def fake_evaluate_colbert(config_override, pairs, embedder):
        del config_override, pairs, embedder
        return {"ndcg@10": 0.7, "n_queries": 1.0}

    monkeypatch.setattr(benchmark_sweep, "_build_hybrid_for_variant", fake_build_hybrid)
    monkeypatch.setattr(benchmark_sweep, "_evaluate_hybrid", fake_evaluate_hybrid)
    monkeypatch.setattr(benchmark_sweep, "_measure_latency_generic", fake_measure_latency)
    monkeypatch.setattr(benchmark_sweep, "_evaluate_colbert", fake_evaluate_colbert)

    try:
        results = benchmark_sweep.run_sweep(config_path)
    finally:
        clear_settings_override()
        reset_settings_cache()

    assert [result.metrics["ndcg@10"] for result in results] == [0.4, 0.9]
    assert results[1].metrics["colbert_ndcg@10"] == 0.7
    assert results[1].metrics["colbert_n_queries"] == 1.0


def test_run_sweep_rebuilds_embedder_after_each_variant_override(
    monkeypatch, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pairs.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "doc": "doc one", "score": 3}),
                json.dumps({"query": "q1", "doc": "doc two", "score": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: embedder_sweep",
                "variants:",
                "  first_variant:",
                "    embedder:",
                '      model_name: "model/one"',
                "  second_variant:",
                "    embedder:",
                '      model_name: "model/two"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RERANKER_PATHS__RAW_DATA_DIR", str(raw_dir))
    reset_settings_cache()
    benchmark_sweep = _load_benchmark_sweep_module()
    embedder_models: list[str] = []

    class StubEmbedder:
        def __init__(self) -> None:
            embedder_models.append(get_settings().embedder.model_name)

    monkeypatch.setattr(benchmark_sweep, "Embedder", StubEmbedder)
    monkeypatch.setattr(
        benchmark_sweep,
        "_build_hybrid_for_variant",
        lambda config_override, embedder: object(),
    )
    monkeypatch.setattr(
        benchmark_sweep,
        "_evaluate_hybrid",
        lambda reranker, pairs: {"ndcg@10": 0.5, "n_queries": 1.0},
    )
    monkeypatch.setattr(
        benchmark_sweep,
        "_measure_latency_generic",
        lambda reranker, query, docs, n_runs=5: 1.0,
    )

    try:
        benchmark_sweep.run_sweep(config_path)
    finally:
        clear_settings_override()
        reset_settings_cache()

    assert embedder_models == ["model/one", "model/two"]


def test_run_sweep_validates_unknown_variant_key(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pairs.jsonl").write_text(
        json.dumps({"query": "q1", "doc": "d1", "score": 1}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: invalid_key_sweep",
                "variants:",
                "  bad_variant:",
                "    hybrid:",
                '      weighting_mode: "static"',
                "    unknown_block:",
                "      foo: bar",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RERANKER_PATHS__RAW_DATA_DIR", str(raw_dir))
    benchmark_sweep = _load_benchmark_sweep_module()

    try:
        try:
            benchmark_sweep.run_sweep(config_path)
            raise AssertionError("Expected ValueError for unknown variant key")
        except ValueError as exc:
            assert "Unsupported keys in variant" in str(exc)
    finally:
        clear_settings_override()
        reset_settings_cache()


def test_run_sweep_smoke_binary_cascade_distilled_pipeline(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pairs.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "doc": "d1", "score": 3}),
                json.dumps({"query": "q1", "doc": "d2", "score": 1}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "preferences.jsonl").write_text(
        json.dumps({"query": "q1", "doc_a": "d1", "doc_b": "d2", "preferred": "A"}) + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: smoke_all_sweeps",
                "variants:",
                "  binary_variant:",
                "    binary:",
                '      quantization: "int8"',
                "      hamming_top_k: 10",
                "      bilinear_top_k: 2",
                "  cascade_variant:",
                "    cascade:",
                '      confidence_metric: "MAX_PROB"',
                "      confidence_threshold: 0.6",
                "  distilled_variant:",
                "    distilled:",
                '      loss_type: "logistic"',
                "      C: 1.0",
                "      max_iter: 100",
                "  pipeline_variant:",
                "    pipeline:",
                '      stages: ["bm25"]',
                "      top_ks: [10]",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("RERANKER_PATHS__RAW_DATA_DIR", str(raw_dir))
    benchmark_sweep = _load_benchmark_sweep_module()

    class StubEmbedder:
        pass

    monkeypatch.setattr(benchmark_sweep, "Embedder", StubEmbedder)
    monkeypatch.setattr(benchmark_sweep, "_build_binary_for_variant", lambda *_args: object())
    monkeypatch.setattr(benchmark_sweep, "_build_cascade_for_variant", lambda *_args: object())
    monkeypatch.setattr(benchmark_sweep, "_build_distilled_for_variant", lambda *_args: object())
    monkeypatch.setattr(benchmark_sweep, "_build_pipeline_for_variant", lambda *_args: object())
    monkeypatch.setattr(
        benchmark_sweep,
        "_evaluate_binary",
        lambda _reranker, _pairs: {"ndcg@10": 0.5, "n_queries": 1.0},
    )
    monkeypatch.setattr(
        benchmark_sweep,
        "_evaluate_cascade",
        lambda _reranker, _pairs: {"ndcg@10": 0.6, "n_queries": 1.0},
    )
    monkeypatch.setattr(
        benchmark_sweep,
        "_evaluate_distilled",
        lambda _reranker, _prefs: {"accuracy": 1.0, "n_comparisons": 1.0},
    )
    monkeypatch.setattr(
        benchmark_sweep,
        "_evaluate_pipeline",
        lambda _reranker, _pairs: {"ndcg@10": 0.4, "n_queries": 1.0},
    )
    monkeypatch.setattr(benchmark_sweep, "_measure_latency_generic", lambda *_args, **_kwargs: 1.0)

    try:
        results = benchmark_sweep.run_sweep(config_path)
    finally:
        clear_settings_override()
        reset_settings_cache()

    assert len(results) == 4
    by_name = {r.variant_name: r for r in results}
    assert by_name["binary_variant"].metrics["ndcg@10"] == 0.5
    assert by_name["cascade_variant"].metrics["ndcg@10"] == 0.6
    assert by_name["distilled_variant"].metrics["accuracy"] == 1.0
    assert by_name["pipeline_variant"].metrics["ndcg@10"] == 0.4
