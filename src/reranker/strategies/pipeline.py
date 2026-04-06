from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reranker.config import get_settings
from reranker.protocols import RankedDoc


@dataclass
class PipelineStage:
    """A single stage in a reranking pipeline."""

    name: str
    reranker: Any
    top_k: int

    def run(self, query: str, docs: list[str]) -> tuple[list[RankedDoc], float]:
        start = time.perf_counter()
        ranked = self.reranker.rerank(query, docs)
        elapsed = (time.perf_counter() - start) * 1000
        return ranked, elapsed


@dataclass
class PipelineResult:
    """Full pipeline execution result with per-stage metadata."""

    final_ranking: list[RankedDoc]
    stage_results: list[dict[str, Any]]
    total_latency_ms: float


class PipelineReranker:
    """Multi-stage reranking pipeline that cascades candidates through stages.

    Each stage filters and re-ranks the candidate set, passing only top-k
    documents to the next stage. This enables combining fast lexical/binary
    filters with expensive semantic rerankers efficiently.

    Example pipeline:
        Stage 1: BM25 → Top-500 (fast lexical filter)
        Stage 2: BinaryQuantizedReranker → Top-200 (fast binary semantic)
        Stage 3: HybridFusionReranker → Top-50 (GBDT reranker)
        Stage 4: StaticColBERTReranker → Top-20 (late interaction)
        Stage 5: DistilledPairwiseRanker → Final (pairwise tournament)
    """

    def __init__(
        self,
        stages: list[PipelineStage] | None = None,
        default_top_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self.stages = stages or []
        self.default_top_k = (
            default_top_k if default_top_k is not None else settings.pipeline.default_stage_top_k
        )

    def add_stage(
        self,
        name: str,
        reranker: Any,
        top_k: int | None = None,
    ) -> PipelineReranker:
        self.stages.append(
            PipelineStage(
                name=name,
                reranker=reranker,
                top_k=top_k or self.default_top_k,
            )
        )
        return self

    def rerank(self, query: str, docs: list[str]) -> list[RankedDoc]:
        if not docs:
            return []
        if not self.stages:
            return [
                RankedDoc(doc=doc, score=0.0, rank=idx, metadata={"strategy": "passthrough"})
                for idx, doc in enumerate(docs, start=1)
            ]

        result = self.run_pipeline(query, docs)
        return result.final_ranking

    def run_pipeline(self, query: str, docs: list[str]) -> PipelineResult:
        if not docs:
            return PipelineResult(
                final_ranking=[],
                stage_results=[],
                total_latency_ms=0.0,
            )

        current_docs = list(docs)
        stage_results: list[dict[str, Any]] = []
        total_start = time.perf_counter()

        for stage in self.stages:
            ranked, latency_ms = stage.run(query, current_docs)
            top_k = min(stage.top_k, len(ranked))
            passed = ranked[:top_k]

            stage_results.append(
                {
                    "stage_name": stage.name,
                    "input_count": len(current_docs),
                    "output_count": len(passed),
                    "latency_ms": round(latency_ms, 2),
                    "top_score": float(passed[0].score) if passed else 0.0,
                }
            )

            current_docs = [r.doc for r in passed]
            if not current_docs:
                break

        total_latency = (time.perf_counter() - total_start) * 1000

        final_ranking = [
            RankedDoc(
                doc=doc,
                score=ranked_doc.score if any(r.doc == doc for r in passed) else 0.0,
                rank=idx,
                metadata={
                    "strategy": "pipeline",
                    "stages": stage_results,
                },
            )
            for idx, (doc, ranked_doc) in enumerate(
                zip(current_docs, passed, strict=False), start=1
            )
        ]

        return PipelineResult(
            final_ranking=final_ranking,
            stage_results=stage_results,
            total_latency_ms=round(total_latency, 2),
        )

    def save(self, path: str | Path) -> None:
        from reranker.utils import build_artifact_metadata, dump_pickle

        stage_data = []
        for stage in self.stages:
            stage_path = Path(path).parent / f"{stage.name}.pkl"
            if hasattr(stage.reranker, "save"):
                stage.reranker.save(stage_path)
            stage_data.append(
                {
                    "name": stage.name,
                    "top_k": stage.top_k,
                    "model_path": str(stage_path),
                }
            )

        dump_pickle(
            path,
            build_artifact_metadata(
                "pipeline_reranker",
                format_name="pickle",
                embedder_model_name="multiple",
                extra={
                    "stages": stage_data,
                    "default_top_k": self.default_top_k,
                },
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        stage_rerankers: dict[str, Any] | None = None,
    ) -> PipelineReranker:
        from reranker.utils import load_pickle, validate_artifact_metadata

        payload = load_pickle(path)
        validate_artifact_metadata(
            payload,
            expected_type="pipeline_reranker",
            expected_formats={"pickle"},
        )
        instance = cls(
            default_top_k=payload.get("default_top_k"),
        )

        stage_rerankers = stage_rerankers or {}
        for stage_info in payload.get("stages", []):
            name = stage_info["name"]
            top_k = stage_info["top_k"]
            reranker = stage_rerankers.get(name)
            if reranker is None:
                raise ValueError(
                    f"Reranker for stage '{name}' not provided. "
                    f"Pass stage_rerankers={{'{name}': <reranker_instance>}}"
                )
            instance.add_stage(name=name, reranker=reranker, top_k=top_k)

        return instance
