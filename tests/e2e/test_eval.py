from pathlib import Path

from reranker.eval.runner import evaluate_strategy


def test_eval_runner_hybrid(tmp_path: Path) -> None:
    report = evaluate_strategy("hybrid", "test", tmp_path / "data", tmp_path / "models")
    assert report["strategy"] == "hybrid"
    assert "ndcg@10" in report
    assert "bm25_ndcg@10" in report
    assert "ndcg@10_uplift_vs_bm25" in report


def test_eval_runner_distilled(tmp_path: Path) -> None:
    report = evaluate_strategy("distilled", "test", tmp_path / "data", tmp_path / "models")
    assert report["strategy"] == "distilled"
    assert "accuracy" in report
    assert report["split"] == "test"


def test_eval_runner_consistency(tmp_path: Path) -> None:
    report = evaluate_strategy("consistency", "test", tmp_path / "data", tmp_path / "models")
    assert report["strategy"] == "consistency"
    assert "recall" in report
    assert report["split"] == "test"


def test_eval_runner_model_caching(tmp_path: Path) -> None:
    """Test that models are cached and reused on subsequent runs."""
    model_dir = tmp_path / "models"

    # First run - creates model
    report1 = evaluate_strategy("hybrid", "test", tmp_path / "data", model_dir)
    model_path = model_dir / "hybrid_reranker.pkl"
    assert model_path.exists()

    # Second run - uses cached model
    report2 = evaluate_strategy("hybrid", "test", tmp_path / "data", model_dir)

    # Results should be identical
    assert report1["ndcg@10"] == report2["ndcg@10"]


def test_eval_runner_missing_data_file_handling(tmp_path: Path) -> None:
    """Test that missing data files are handled gracefully."""
    # Remove data directory if it exists
    import shutil

    data_dir = tmp_path / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    # Should create data files automatically
    report = evaluate_strategy("hybrid", "test", data_dir, tmp_path / "models")

    # Verify data was created
    assert (data_dir / "pairs.jsonl").exists()
    assert report["strategy"] == "hybrid"
