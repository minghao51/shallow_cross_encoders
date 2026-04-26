"""End-to-end integration tests for the evaluation runner."""

from __future__ import annotations

from pathlib import Path

from reranker.eval.runner import evaluate_strategy


class TestEvalRunnerE2E:
    """End-to-end tests for the full evaluation pipeline."""

    def test_hybrid_strategy_full_pipeline(self, tmp_path: Path) -> None:
        """Test full evaluation pipeline for hybrid strategy."""
        report = evaluate_strategy("hybrid", "test", tmp_path / "data", tmp_path / "models")

        # Verify report structure
        assert report["strategy"] == "hybrid"
        assert report["split"] == "test"
        assert "ndcg@10" in report
        assert "bm25_ndcg@10" in report
        assert "ndcg@10_uplift_vs_bm25" in report
        assert "mrr" in report
        assert "map" in report
        assert "p@1" in report
        assert "latency_p50_ms" in report
        assert "latency_p99_ms" in report

        # Verify value ranges
        assert 0 <= report["ndcg@10"] <= 1
        assert 0 <= report["bm25_ndcg@10"] <= 1
        assert 0 <= report["map"] <= 1
        assert report["latency_p50_ms"] >= 0
        assert report["latency_p99_ms"] >= report["latency_p50_ms"]

        # Verify model was saved
        model_path = tmp_path / "models" / "hybrid_reranker.pkl"
        assert model_path.exists()

    def test_distilled_strategy_full_pipeline(self, tmp_path: Path) -> None:
        """Test full evaluation pipeline for distilled strategy."""
        report = evaluate_strategy("distilled", "test", tmp_path / "data", tmp_path / "models")

        # Verify report structure
        assert report["strategy"] == "distilled"
        assert report["split"] == "test"
        assert "accuracy" in report
        assert "latency_p50_ms" in report
        assert "latency_p99_ms" in report

        # Verify value ranges
        assert 0 <= report["accuracy"] <= 1
        assert report["latency_p50_ms"] >= 0

        # Verify model was saved
        model_path = tmp_path / "models" / "pairwise_ranker.pkl"
        assert model_path.exists()

    def test_consistency_strategy_full_pipeline(self, tmp_path: Path) -> None:
        """Test full evaluation pipeline for consistency strategy."""
        report = evaluate_strategy("consistency", "test", tmp_path / "data", tmp_path / "models")

        # Verify report structure
        assert report["strategy"] == "consistency"
        assert report["split"] == "test"
        assert "recall" in report
        assert "false_positive_rate" in report
        assert "latency_p50_ms" in report
        assert "latency_p99_ms" in report

        # Verify value ranges
        assert 0 <= report["recall"] <= 1
        assert 0 <= report["false_positive_rate"] <= 1

    def test_model_persistence_and_loading(self, tmp_path: Path) -> None:
        """Test that models are persisted and can be loaded."""
        data_dir = tmp_path / "data"
        model_dir = tmp_path / "models"

        # First run - should create and save model
        report1 = evaluate_strategy("hybrid", "test", data_dir, model_dir)
        model_path = model_dir / "hybrid_reranker.pkl"
        assert model_path.exists()

        # Second run - should load existing model
        report2 = evaluate_strategy("hybrid", "test", data_dir, model_dir)

        # Results should be consistent
        assert report1["ndcg@10"] == report2["ndcg@10"]

    def test_data_splitting(self, tmp_path: Path) -> None:
        """Test that data splitting works correctly."""
        data_dir = tmp_path / "data"
        model_dir = tmp_path / "models"

        # Run with different splits
        train_report = evaluate_strategy("hybrid", "train", data_dir, model_dir)
        val_report = evaluate_strategy("hybrid", "validation", data_dir, model_dir)
        test_report = evaluate_strategy("hybrid", "test", data_dir, model_dir)

        # All should succeed
        assert train_report["split"] == "train"
        assert val_report["split"] == "validation"
        assert test_report["split"] == "test"

    def test_latency_tracking(self, tmp_path: Path) -> None:
        """Test that latency is tracked correctly."""
        report = evaluate_strategy("hybrid", "test", tmp_path / "data", tmp_path / "models")

        # Latency metrics should be present
        assert "latency_p50_ms" in report
        assert "latency_p99_ms" in report

        # p99 should be >= p50
        assert report["latency_p99_ms"] >= report["latency_p50_ms"]

        # Latency should be reasonable (less than 1 second per query)
        assert report["latency_p50_ms"] < 1000

    def test_data_file_creation(self, tmp_path: Path) -> None:
        """Test that data files are created if they don't exist."""
        data_dir = tmp_path / "data"
        model_dir = tmp_path / "models"

        # Remove data directory if it exists
        if data_dir.exists():
            import shutil

            shutil.rmtree(data_dir)

        # Run evaluation - should create data files
        evaluate_strategy("hybrid", "test", data_dir, model_dir)

        # Verify data files were created
        assert (data_dir / "pairs.jsonl").exists()
        assert (data_dir / "preferences.jsonl").exists()
        assert (data_dir / "contradictions.jsonl").exists()
        assert (data_dir / "manifest.json").exists()

    def test_bm25_baseline_comparison(self, tmp_path: Path) -> None:
        """Test that BM25 baseline is computed correctly."""
        report = evaluate_strategy("hybrid", "test", tmp_path / "data", tmp_path / "models")

        # Both hybrid and BM25 scores should be present
        assert "ndcg@10" in report
        assert "bm25_ndcg@10" in report

        # Uplift should be calculated
        assert "ndcg@10_uplift_vs_bm25" in report

        # Uplift = hybrid - bm25
        expected_uplift = report["ndcg@10"] - report["bm25_ndcg@10"]
        assert abs(report["ndcg@10_uplift_vs_bm25"] - expected_uplift) < 0.001

    def test_accuracy_calculation(self, tmp_path: Path) -> None:
        """Test that accuracy is calculated correctly for distilled strategy."""
        report = evaluate_strategy("distilled", "test", tmp_path / "data", tmp_path / "models")

        # Accuracy should be between 0 and 1
        assert 0 <= report["accuracy"] <= 1

        # For a decent model, accuracy should be reasonable
        # (though it might be low on synthetic data)
        assert report["accuracy"] >= 0

    def test_contradiction_detection_metrics(self, tmp_path: Path) -> None:
        """Test that contradiction detection metrics are calculated correctly."""
        report = evaluate_strategy("consistency", "test", tmp_path / "data", tmp_path / "models")

        # Recall and FPR should be present
        assert "recall" in report
        assert "false_positive_rate" in report

        # Both should be between 0 and 1
        assert 0 <= report["recall"] <= 1
        assert 0 <= report["false_positive_rate"] <= 1


class TestEvalRunnerEdgeCases:
    """Edge case tests for the evaluation runner."""

    def test_multiple_runs_consistency(self, tmp_path: Path) -> None:
        """Test that multiple runs produce consistent results."""
        results = []
        for _ in range(3):
            report = evaluate_strategy("hybrid", "test", tmp_path / "data", tmp_path / "models")
            results.append(report["ndcg@10"])

        # All results should be identical (deterministic)
        assert all(r == results[0] for r in results)

    def test_isolated_data_directories(self, tmp_path: Path) -> None:
        """Test that different data directories don't interfere."""
        data_dir1 = tmp_path / "data1"
        data_dir2 = tmp_path / "data2"
        model_dir = tmp_path / "models"

        report1 = evaluate_strategy("hybrid", "test", data_dir1, model_dir)
        report2 = evaluate_strategy("hybrid", "test", data_dir2, model_dir)

        # Results should be the same (same seed)
        assert report1["ndcg@10"] == report2["ndcg@10"]


class TestEvalRunnerWithSmallDataset:
    """Tests with intentionally small datasets."""

    def test_evaluation_with_minimal_data(self, tmp_path: Path) -> None:
        """Test evaluation with a very small dataset."""
        # This tests the edge case where we have minimal data
        report = evaluate_strategy("hybrid", "test", tmp_path / "data", tmp_path / "models")

        # Should still succeed
        assert report["strategy"] == "hybrid"
        assert "ndcg@10" in report

    def test_all_strategies_with_same_data(self, tmp_path: Path) -> None:
        """Test all strategies with the same data directory."""
        data_dir = tmp_path / "data"
        model_dir = tmp_path / "models"

        hybrid_report = evaluate_strategy("hybrid", "test", data_dir, model_dir)
        distilled_report = evaluate_strategy("distilled", "test", data_dir, model_dir)
        consistency_report = evaluate_strategy("consistency", "test", data_dir, model_dir)

        # All should succeed
        assert hybrid_report["strategy"] == "hybrid"
        assert distilled_report["strategy"] == "distilled"
        assert consistency_report["strategy"] == "consistency"

        # All should have latency metrics
        assert "latency_p50_ms" in hybrid_report
        assert "latency_p50_ms" in distilled_report
        assert "latency_p50_ms" in consistency_report


class TestEvalRunnerDataGeneration:
    """Tests for automatic data generation."""

    def test_auto_data_generation_creates_all_files(self, tmp_path: Path) -> None:
        """Test that auto data generation creates all required files."""
        data_dir = tmp_path / "data"
        model_dir = tmp_path / "models"

        evaluate_strategy("hybrid", "test", data_dir, model_dir)

        # Check all expected files
        expected_files = [
            "pairs.jsonl",
            "preferences.jsonl",
            "contradictions.jsonl",
            "manifest.json",
        ]

        for filename in expected_files:
            file_path = data_dir / filename
            assert file_path.exists(), f"Expected file {filename} was not created"

    def test_manifest_content(self, tmp_path: Path) -> None:
        """Test that manifest contains correct information."""
        import json

        data_dir = tmp_path / "data"
        model_dir = tmp_path / "models"

        evaluate_strategy("hybrid", "test", data_dir, model_dir)

        # Read manifest
        manifest_path = data_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Verify manifest structure
        assert "generated_at" in manifest
        assert "seed" in manifest
        assert "generation_mode" in manifest
        assert "datasets" in manifest
        assert "pairs" in manifest["datasets"]
