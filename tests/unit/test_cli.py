"""Tests for the typer-based CLI."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from reranker.cli import app

runner = CliRunner()


class TestRoot:
    def test_no_args_shows_help(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output
        assert "eval" in result.output
        assert "benchmark" in result.output
        assert "generate" in result.output
        assert "doctor" in result.output


class TestTrain:
    def test_train_help(self) -> None:
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "hybrid" in result.output
        assert "distilled" in result.output
        assert "binary" in result.output
        assert "late_interaction" in result.output
        assert "cascade" in result.output
        assert "meta_router" in result.output
        assert "splade" in result.output

    def test_train_hybrid_help(self) -> None:
        result = runner.invoke(app, ["train", "hybrid", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output
        assert "--output" in result.output
        assert "--config" in result.output

    def test_train_cascade_help(self) -> None:
        result = runner.invoke(app, ["train", "cascade", "--help"])
        assert result.exit_code == 0
        assert "--threshold" in result.output

    def test_train_splade_help(self) -> None:
        result = runner.invoke(app, ["train", "splade", "--help"])
        assert result.exit_code == 0
        assert "--top-k" in result.output


class TestEval:
    def test_eval_help(self) -> None:
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_eval_run_help(self) -> None:
        result = runner.invoke(app, ["eval", "run", "--help"])
        assert result.exit_code == 0
        assert "--split" in result.output
        assert "--metrics" in result.output
        assert "--dataset" in result.output


class TestBenchmark:
    def test_benchmark_help(self) -> None:
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "sweep" in result.output
        assert "full" in result.output

    def test_benchmark_run_help(self) -> None:
        result = runner.invoke(app, ["benchmark", "run", "--help"])
        assert result.exit_code == 0
        assert "--quick" in result.output

    def test_benchmark_compare_with_paired_per_query_metrics(self, tmp_path) -> None:
        results_file = tmp_path / "benchmark_results.json"
        payload = {
            "results": [
                {
                    "strategy": "hybrid",
                    "metrics": {"ndcg@10": 0.7},
                    "per_query_metrics": {"per_query_ndcg@10": [0.9, 0.5, 0.7]},
                },
                {
                    "strategy": "binary_reranker",
                    "metrics": {"ndcg@10": 0.6},
                    "per_query_metrics": {"per_query_ndcg@10": [0.8, 0.4, 0.6]},
                },
            ]
        }
        results_file.write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "benchmark",
                "compare",
                "hybrid",
                "binary_reranker",
                "--results",
                str(results_file),
                "--metric",
                "ndcg@10",
            ],
        )
        assert result.exit_code == 0
        assert "Comparison: hybrid vs binary_reranker" in result.output

    def test_benchmark_compare_rejects_aggregate_only_results(self, tmp_path) -> None:
        results_file = tmp_path / "benchmark_results.json"
        payload = {
            "results": [
                {"strategy": "hybrid", "metrics": {"ndcg@10": 0.7}},
                {"strategy": "binary_reranker", "metrics": {"ndcg@10": 0.6}},
            ]
        }
        results_file.write_text(json.dumps(payload), encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "benchmark",
                "compare",
                "hybrid",
                "binary_reranker",
                "--results",
                str(results_file),
                "--metric",
                "ndcg@10",
            ],
        )
        assert result.exit_code == 1
        assert "no paired per-query metric" in result.output


class TestGenerate:
    def test_generate_help(self) -> None:
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "pairs" in result.output
        assert "preferences" in result.output
        assert "contradictions" in result.output

    def test_generate_pairs_help(self) -> None:
        result = runner.invoke(app, ["generate", "pairs", "--help"])
        assert result.exit_code == 0
        assert "--count" in result.output
        assert "--seed" in result.output

    def test_generate_preferences_help(self) -> None:
        result = runner.invoke(app, ["generate", "preferences", "--help"])
        assert result.exit_code == 0
        assert "--count" in result.output

    def test_generate_contradictions_help(self) -> None:
        result = runner.invoke(app, ["generate", "contradictions", "--help"])
        assert result.exit_code == 0
        assert "--count" in result.output


class TestDoctor:
    def test_doctor_help(self) -> None:
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "check" in result.output

    def test_doctor_check(self) -> None:
        result = runner.invoke(app, ["doctor", "check"])
        assert result.exit_code == 0
        assert "numpy" in result.output
        assert "pydantic" in result.output


class TestServe:
    def test_serve_help(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "start" in result.output

    def test_serve_start_is_phase11_preview(self) -> None:
        result = runner.invoke(app, ["serve", "start"])
        assert result.exit_code == 1
        assert "Phase 11" in result.output
