import pytest

from reranker.lexical import BM25Engine
from reranker.strategies.binary_reranker import BinaryQuantizedReranker
from reranker.strategies.late_interaction import StaticColBERTReranker
from reranker.strategies.pipeline import PipelineReranker, PipelineStage


class TestPipelineReranker:
    def test_empty_pipeline(self):
        pipeline = PipelineReranker()
        result = pipeline.rerank("python dataclass", ["doc1", "doc2"])
        assert len(result) == 2
        assert result[0].metadata["strategy"] == "passthrough"

    def test_empty_docs(self):
        pipeline = PipelineReranker()
        assert pipeline.rerank("query", []) == []

    def test_single_stage(self):
        docs = [
            "python dataclass field default factory",
            "javascript array spread operator",
        ]
        stage = PipelineStage(
            name="colbert",
            reranker=StaticColBERTReranker(),
            top_k=2,
        )
        pipeline = PipelineReranker(stages=[stage])
        result = pipeline.rerank("python dataclass", docs)
        assert len(result) == 2
        assert result[0].metadata["strategy"] == "pipeline"

    def test_multi_stage_cascade(self):
        docs = [
            "python dataclass field default factory mutable list",
            "javascript array spread operator clone object",
            "python typing annotations type hints",
            "rust ownership borrow checker memory safety",
            "go concurrency goroutines channels",
        ]
        bm25 = BM25Engine()
        bm25.fit(docs)

        stages = [
            PipelineStage(name="bm25", reranker=bm25, top_k=3),
            PipelineStage(name="colbert", reranker=StaticColBERTReranker(), top_k=2),
        ]
        pipeline = PipelineReranker(stages=stages)
        result = pipeline.run_pipeline("python dataclass", docs)
        assert len(result.final_ranking) == 2
        assert len(result.stage_results) == 2
        assert result.stage_results[0]["input_count"] == 5
        assert result.stage_results[0]["output_count"] == 3
        assert result.stage_results[1]["input_count"] == 3
        assert result.stage_results[1]["output_count"] == 2
        assert result.total_latency_ms > 0

    def test_add_stage(self):
        pipeline = PipelineReranker()
        pipeline.add_stage("stage1", StaticColBERTReranker(), top_k=10)
        pipeline.add_stage("stage2", BinaryQuantizedReranker(), top_k=5)
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0].top_k == 10
        assert pipeline.stages[1].top_k == 5

    def test_stage_latency_tracking(self):
        docs = ["python dataclass field", "javascript array"]
        stage = PipelineStage(
            name="colbert",
            reranker=StaticColBERTReranker(),
            top_k=2,
        )
        pipeline = PipelineReranker(stages=[stage])
        result = pipeline.run_pipeline("python", docs)
        assert len(result.stage_results) == 1
        assert "latency_ms" in result.stage_results[0]
        assert result.stage_results[0]["latency_ms"] >= 0

    def test_early_termination(self):
        docs = ["doc1", "doc2", "doc3"]

        class EmptyReranker:
            def rerank(self, query, docs):
                return []

        stages = [
            PipelineStage(name="empty", reranker=EmptyReranker(), top_k=2),
            PipelineStage(name="colbert", reranker=StaticColBERTReranker(), top_k=1),
        ]
        pipeline = PipelineReranker(stages=stages)
        result = pipeline.run_pipeline("query", docs)
        assert len(result.final_ranking) == 0
        assert len(result.stage_results) == 1

    def test_save_load_structure(self, tmp_path):
        docs = ["python dataclass field", "javascript array"]
        colbert = StaticColBERTReranker()
        colbert.fit(docs)
        pipeline = PipelineReranker()
        pipeline.add_stage("colbert", colbert, top_k=2)
        path = tmp_path / "pipeline.pkl"
        pipeline.save(path)
        with pytest.raises(ValueError, match="not provided"):
            PipelineReranker.load(path)
        loaded = PipelineReranker.load(path, stage_rerankers={"colbert": colbert})
        assert len(loaded.stages) == 1
        assert loaded.stages[0].name == "colbert"

    def test_default_top_k(self):
        pipeline = PipelineReranker(default_top_k=50)
        pipeline.add_stage("colbert", StaticColBERTReranker())
        assert pipeline.stages[0].top_k == 50
