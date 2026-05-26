"""Integration tests for reranker CLI commands — exercise real command logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reranker.cli import app

runner = CliRunner()


class TestHelpCommands:
    def test_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output
        assert "eval" in result.output
        assert "generate" in result.output
        assert "serve" in result.output
        assert "benchmark" in result.output
        assert "doctor" in result.output

    def test_train_help(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        for name in [
            "hybrid",
            "distilled",
            "binary",
            "late_interaction",
            "cascade",
            "meta_router",
            "splade",
        ]:
            assert name in result.output

    def test_eval_help(self) -> None:
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_generate_help(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "pairs" in result.output
        assert "preferences" in result.output
        assert "contradictions" in result.output

    def test_serve_help(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "start" in result.output

    def test_benchmark_help(self) -> None:
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "sweep" in result.output
        assert "full" in result.output
        assert "compare" in result.output

    def test_doctor_help(self) -> None:
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "check" in result.output


class TestServeCommand:
    def test_serve_start_exits_with_phase11_message(self) -> None:
        result = runner.invoke(app, ["serve", "start"])
        assert result.exit_code == 1
        assert "Phase 11" in result.output

    def test_serve_start_default_host_is_0000(self) -> None:
        result = runner.invoke(app, ["serve", "start"])
        assert result.exit_code == 1
        assert "0.0.0.0:8000" in result.output

    def test_serve_start_custom_host_port(self) -> None:
        result = runner.invoke(app, ["serve", "start", "--host", "127.0.0.1", "--port", "9000"])
        assert result.exit_code == 1
        assert "127.0.0.1:9000" in result.output

    def test_serve_start_help_options(self) -> None:
        result = runner.invoke(app, ["serve", "start", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output


class TestDoctorCommand:
    def test_doctor_check_runs(self) -> None:
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 0
        assert "numpy" in result.output
        assert "pydantic" in result.output
        assert "typer" in result.output

    def test_doctor_check_shows_directories(self) -> None:
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 0
        assert "Data dir:" in result.output
        assert "Model dir:" in result.output

    def test_doctor_check_shows_backends(self) -> None:
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 0
        assert "Embedder backend:" in result.output
        assert "GBDT backend:" in result.output


class TestBenchmarkCompareCommand:
    def test_compare_missing_results_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.json"
        result = runner.invoke(
            app,
            ["benchmark", "compare", "hybrid", "distilled", "--results", str(missing)],
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_compare_mismatched_per_query_lengths(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.json"
        payload = {
            "results": [
                {
                    "strategy": "hybrid",
                    "metrics": {"ndcg@10": 0.7},
                    "per_query_metrics": {"per_query_ndcg@10": [0.9, 0.5, 0.7]},
                },
                {
                    "strategy": "distilled",
                    "metrics": {"ndcg@10": 0.6},
                    "per_query_metrics": {"per_query_ndcg@10": [0.8, 0.4]},
                },
            ]
        }
        results_file.write_text(json.dumps(payload))
        result = runner.invoke(
            app,
            ["benchmark", "compare", "hybrid", "distilled", "--results", str(results_file)],
        )
        assert result.exit_code == 1
        assert "different lengths" in result.output

    def test_compare_unknown_strategy(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.json"
        payload = {
            "results": [
                {
                    "strategy": "hybrid",
                    "metrics": {"ndcg@10": 0.7},
                    "per_query_metrics": {"per_query_ndcg@10": [0.9, 0.5, 0.7]},
                },
            ]
        }
        results_file.write_text(json.dumps(payload))
        result = runner.invoke(
            app,
            ["benchmark", "compare", "hybrid", "nonexistent", "--results", str(results_file)],
        )
        assert result.exit_code == 1
        assert "nonexistent" in result.output

    def test_compare_successful(self, tmp_path: Path) -> None:
        results_file = tmp_path / "results.json"
        payload = {
            "results": [
                {
                    "strategy": "hybrid",
                    "metrics": {"ndcg@10": 0.7},
                    "per_query_metrics": {"per_query_ndcg@10": [0.9, 0.5, 0.7, 0.6, 0.8]},
                },
                {
                    "strategy": "binary_reranker",
                    "metrics": {"ndcg@10": 0.6},
                    "per_query_metrics": {"per_query_ndcg@10": [0.8, 0.4, 0.6, 0.5, 0.7]},
                },
            ]
        }
        results_file.write_text(json.dumps(payload))
        result = runner.invoke(
            app,
            ["benchmark", "compare", "hybrid", "binary_reranker", "--results", str(results_file)],
        )
        assert result.exit_code == 0
        assert "Comparison: hybrid vs binary_reranker" in result.output
        assert "Wilcoxon p:" in result.output
        assert "Significant:" in result.output


class TestBenchmarkExecutionCommands:
    def test_benchmark_run_quick_executes_baselines_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchmarks import runner as runner_module
        from reranker import config as config_module

        calls: list[str] = []

        class Paths:
            raw_data_dir = "/tmp/raw"
            model_dir = "/tmp/models"

        class Settings:
            paths = Paths()

        class StubRunner:
            def __init__(self, **kwargs):
                calls.append(f"init:{kwargs['quick']}:{kwargs['profiling_enabled']}")

            def run_baselines(self):
                calls.append("baselines")

            def run_ablations(self):
                calls.append("ablations")

            def run_scaling(self):
                calls.append("scaling")

            def run_embedder_comparison(self):
                calls.append("embedder")

            def save_results(self, output_dir):
                calls.append(f"save:{output_dir}")

        monkeypatch.setattr(config_module, "get_settings", lambda: Settings())
        monkeypatch.setattr(runner_module, "BenchmarkRunner", StubRunner)
        result = runner.invoke(app, ["benchmark", "run", "--quick"])
        assert result.exit_code == 0
        assert calls == [
            "init:True:False",
            "baselines",
            "save:benchmarks/results",
        ]

    def test_benchmark_run_config_path_executes_sweep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from benchmarks import run_sweep as run_sweep_module

        called: dict[str, object] = {"config": None, "printed": False}

        def _run_sweep(config: str):
            called["config"] = config
            return []

        monkeypatch.setattr(run_sweep_module, "run_sweep", _run_sweep)
        monkeypatch.setattr(
            run_sweep_module,
            "print_comparison_table",
            lambda results: called.__setitem__("printed", True),
        )
        result = runner.invoke(
            app,
            ["benchmark", "run", "--config", "benchmarks/configs/sweep_hybrid.yaml"],
        )
        assert result.exit_code == 0
        assert called["config"] == "benchmarks/configs/sweep_hybrid.yaml"
        assert called["printed"] is True

    def test_benchmark_sweep_writes_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from benchmarks import run_sweep as run_sweep_module

        class ResultObj:
            def __init__(self):
                self.variant_name = "v1"
                self.metrics = {"ndcg@10": 0.6}
                self.latency_ms = 1.5

        monkeypatch.setattr(run_sweep_module, "run_sweep", lambda config: [ResultObj()])
        monkeypatch.setattr(run_sweep_module, "print_comparison_table", lambda results: None)
        output = tmp_path / "results" / "sweep.json"
        result = runner.invoke(
            app,
            [
                "benchmark",
                "sweep",
                "--config",
                "benchmarks/configs/sweep_hybrid.yaml",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0
        assert output.exists()
        assert "Results saved to" in result.output

    def test_benchmark_full_passes_quick_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from benchmarks import run as run_module

        captured: dict[str, object] = {}

        def _main():
            import sys

            captured["argv"] = list(sys.argv)

        monkeypatch.setattr(run_module, "main", _main)
        result = runner.invoke(app, ["benchmark", "full", "--quick"])
        assert result.exit_code == 0
        assert captured["argv"] == ["benchmarks/run.py", "full", "--quick"]


class TestGeneratePairsCommand:
    def test_generate_pairs_basic(self, tmp_path: Path) -> None:
        out = tmp_path / "pairs.jsonl"
        result = runner.invoke(
            app,
            ["generate", "pairs", "--count", "5", "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            row = json.loads(line)
            assert "query" in row
            assert "doc" in row

    def test_generate_pairs_with_seed(self, tmp_path: Path) -> None:
        out_a = tmp_path / "pairs_a.jsonl"
        out_b = tmp_path / "pairs_b.jsonl"
        runner.invoke(
            app, ["generate", "pairs", "--count", "3", "--seed", "99", "--output", str(out_a)]
        )
        runner.invoke(
            app, ["generate", "pairs", "--count", "3", "--seed", "99", "--output", str(out_b)]
        )
        assert out_a.read_text() == out_b.read_text()

    def test_generate_pairs_default_count(self, tmp_path: Path) -> None:
        out = tmp_path / "pairs.jsonl"
        result = runner.invoke(
            app,
            ["generate", "pairs", "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 100


class TestGeneratePreferencesCommand:
    def test_generate_preferences_basic(self, tmp_path: Path) -> None:
        out = tmp_path / "prefs.jsonl"
        result = runner.invoke(
            app,
            ["generate", "preferences", "--count", "50", "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) >= 1
        for line in lines:
            row = json.loads(line)
            assert "query" in row
            assert "doc_a" in row
            assert "doc_b" in row
            assert "preferred" in row


class TestGenerateContradictionsCommand:
    def test_generate_contradictions_basic(self, tmp_path: Path) -> None:
        out = tmp_path / "contradictions.jsonl"
        result = runner.invoke(
            app,
            ["generate", "contradictions", "--count", "3", "--output", str(out)],
        )
        assert result.exit_code == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) >= 3


class TestEvalRunCommand:
    def test_eval_run_help_options(self) -> None:
        result = runner.invoke(app, ["eval", "run", "--help"])
        assert result.exit_code == 0
        assert "--split" in result.output
        assert "--metrics" in result.output

    def test_eval_run_uses_requested_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from reranker import config as config_module
        from reranker.eval import runner as runner_module

        class Paths:
            raw_data_dir = "/tmp/raw"
            model_dir = "/tmp/models"

        class Settings:
            paths = Paths()

        monkeypatch.setattr(config_module, "get_settings", lambda: Settings())
        monkeypatch.setattr(
            runner_module,
            "evaluate_strategy",
            lambda **kwargs: {
                "strategy": kwargs["strategy"],
                "ndcg@10": 0.5,
                "map": 0.4,
                "latency_p50_ms": 1.2,
            },
        )
        result = runner.invoke(
            app,
            ["eval", "run", "hybrid", "--metrics", "ndcg,map"],
        )
        assert result.exit_code == 0
        assert "ndcg@10: 0.5" in result.output
        assert "map: 0.4" in result.output
        assert "latency_p50_ms" not in result.output

    def test_eval_run_accepts_explicit_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from reranker.eval import runner as runner_module

        captured: dict[str, str] = {}

        def _evaluate_strategy(strategy: str, split: str, data_root: Path, model_root: Path):
            captured["strategy"] = strategy
            captured["split"] = split
            captured["data_root"] = str(data_root)
            captured["model_root"] = str(model_root)
            return {"ndcg@10": 0.9}

        monkeypatch.setattr(runner_module, "evaluate_strategy", _evaluate_strategy)
        data_root = tmp_path / "raw"
        model_root = tmp_path / "models"
        result = runner.invoke(
            app,
            [
                "eval",
                "run",
                "distilled",
                "--split",
                "validation",
                "--dataset",
                str(data_root),
                "--model-root",
                str(model_root),
            ],
        )
        assert result.exit_code == 0
        assert "ndcg@10: 0.9" in result.output
        assert captured == {
            "strategy": "distilled",
            "split": "validation",
            "data_root": str(data_root),
            "model_root": str(model_root),
        }


class TestTrainSubcommandHelp:
    def test_train_hybrid_help_options(self) -> None:
        result = runner.invoke(app, ["train", "hybrid", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output
        assert "--output" in result.output
        assert "--config" in result.output

    def test_train_distilled_help_options(self) -> None:
        result = runner.invoke(app, ["train", "distilled", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output

    def test_train_binary_help_options(self) -> None:
        result = runner.invoke(app, ["train", "binary", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output

    def test_train_cascade_help_has_threshold(self) -> None:
        result = runner.invoke(app, ["train", "cascade", "--help"])
        assert result.exit_code == 0
        assert "--threshold" in result.output

    def test_train_splade_help_has_top_k(self) -> None:
        result = runner.invoke(app, ["train", "splade", "--help"])
        assert result.exit_code == 0
        assert "--top-k" in result.output

    def test_train_meta_router_help_options(self) -> None:
        result = runner.invoke(app, ["train", "meta_router", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output
