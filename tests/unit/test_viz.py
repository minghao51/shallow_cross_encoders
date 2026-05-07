"""Tests for visualization utilities.

These tests validate that viz functions don't crash and return
expected output types. Visual correctness requires manual inspection.
"""

from __future__ import annotations

from pathlib import Path

from reranker.eval.viz import generate_comparison_table, plot_pareto_frontier, plot_radar


def _sample_results() -> list[dict]:
    return [
        {
            "experiment_name": "hybrid_baseline",
            "strategy": "hybrid",
            "metrics": {
                "ndcg@10": 0.85,
                "ndcg@10_std": 0.05,
                "map@10": 0.82,
                "mrr": 0.88,
                "p@1": 0.75,
                "latency_mean_ms": 12.5,
                "throughput_qps": 80.0,
            },
        },
        {
            "experiment_name": "colbert_baseline",
            "strategy": "late_interaction",
            "metrics": {
                "ndcg@10": 0.78,
                "ndcg@10_std": 0.06,
                "map@10": 0.74,
                "mrr": 0.80,
                "p@1": 0.70,
                "latency_mean_ms": 25.3,
                "throughput_qps": 40.0,
            },
        },
        {
            "experiment_name": "binary_baseline",
            "strategy": "binary_reranker",
            "metrics": {
                "ndcg@10": 0.72,
                "ndcg@10_std": 0.07,
                "map@10": 0.68,
                "mrr": 0.75,
                "p@1": 0.65,
                "latency_mean_ms": 8.1,
                "throughput_qps": 123.0,
            },
        },
    ]


class TestPlotParetoFrontier:
    def test_returns_none_when_no_matplotlib(self, monkeypatch):
        monkeypatch.setattr("reranker.eval.viz._check_matplotlib", lambda: False)
        result = plot_pareto_frontier(_sample_results(), "/tmp/_test_pareto.png")
        assert result is None

    def test_returns_path_when_matplotlib(self, tmp_path: Path):
        output = tmp_path / "pareto.png"
        result = plot_pareto_frontier(_sample_results(), str(output))
        if result is not None:
            assert Path(result).exists()

    def test_empty_results_returns_none(self, tmp_path: Path):
        output = tmp_path / "empty.png"
        result = plot_pareto_frontier([], str(output))
        assert result is None

    def test_no_latency_results(self, tmp_path: Path):
        results = [
            {
                "experiment_name": "test",
                "strategy": "test",
                "metrics": {"ndcg@10": 0.5, "latency_mean_ms": 0.0},
            }
        ]
        output = tmp_path / "no_lat.png"
        result = plot_pareto_frontier(results, str(output))
        assert result is None


class TestPlotRadar:
    def test_returns_none_when_no_matplotlib(self, monkeypatch):
        monkeypatch.setattr("reranker.eval.viz._check_matplotlib", lambda: False)
        result = plot_radar(_sample_results(), "/tmp/_test_radar.png")
        assert result is None

    def test_returns_path_when_matplotlib(self, tmp_path: Path):
        output = tmp_path / "radar.png"
        result = plot_radar(_sample_results(), str(output))
        if result is not None:
            assert Path(result).exists()

    def test_less_than_two_strategies_returns_none(self, tmp_path: Path):
        results = [
            {
                "experiment_name": "only_baseline",
                "strategy": "hybrid",
                "metrics": {
                    "ndcg@10": 0.85,
                    "map@10": 0.82,
                    "mrr": 0.88,
                    "p@1": 0.75,
                    "latency_mean_ms": 12.5,
                },
            }
        ]
        output = tmp_path / "single.png"
        result = plot_radar(results, str(output))
        assert result is None

    def test_empty_results_returns_none(self, tmp_path: Path):
        output = tmp_path / "empty.png"
        result = plot_radar([], str(output))
        assert result is None


class TestGenerateComparisonTable:
    def test_returns_markdown_table(self):
        results = _sample_results()
        table = generate_comparison_table(results)
        assert "## Strategy Comparison" in table
        assert "| Strategy |" in table
        assert "| hybrid |" in table
        assert "| late_interaction |" in table
        assert "| binary_reranker |" in table

    def test_empty_results_returns_empty(self):
        table = generate_comparison_table([])
        assert table == ""

    def test_no_baseline_results_returns_empty(self):
        results = [
            {
                "experiment_name": "some_other",
                "strategy": "hybrid",
                "metrics": {"ndcg@10": 0.85},
            }
        ]
        table = generate_comparison_table(results)
        assert table == ""

    def test_writes_to_file_when_path_provided(self, tmp_path: Path):
        output = tmp_path / "table.md"
        generate_comparison_table(_sample_results(), str(output))
        assert Path(output).exists()
        content = output.read_text(encoding="utf-8")
        assert "| hybrid |" in content

    def test_includes_ci_column(self):
        results = _sample_results()
        table = generate_comparison_table(results)
        assert "CI 95%" in table
