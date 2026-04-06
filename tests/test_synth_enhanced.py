import pytest

from reranker.data.synth import (
    HardNegativeRecord,
    ListwisePreferenceRecord,
    QueryExpansionRecord,
    SyntheticDataGenerator,
)


@pytest.mark.unit
class TestEnhancedSyntheticData:
    def test_generate_hard_negatives_offline(self):
        pairs = [
            {"query": "python dataclass", "doc": "Use field(default_factory=list)", "score": 3},
            {"query": "bm25 retrieval", "doc": "BM25 emphasizes exact term overlap", "score": 2},
        ]
        generator = SyntheticDataGenerator()
        records = generator.generate_hard_negatives(pairs, target_count=4, use_teacher=False)
        assert len(records) == 4
        for record in records:
            assert "query" in record
            assert "positive" in record
            assert "hard_negative" in record
            assert "easy_negative" in record
            assert record["generation_mode"] == "offline"

    def test_generate_listwise_preferences_offline(self):
        pairs = [
            {"query": "python dataclass", "doc": "Use field(default_factory=list)", "score": 3},
            {"query": "python dataclass", "doc": "Dataclasses reduce boilerplate", "score": 2},
            {"query": "python dataclass", "doc": "Python has many features", "score": 1},
            {"query": "python dataclass", "doc": "JavaScript is different", "score": 0},
        ]
        generator = SyntheticDataGenerator()
        records = generator.generate_listwise_preferences(pairs, target_count=2, use_teacher=False)
        assert len(records) >= 1
        for record in records:
            assert "query" in record
            assert "docs" in record
            assert "scores" in record
            assert len(record["docs"]) == len(record["scores"])
            assert record["generation_mode"] == "offline"

    def test_generate_query_expansions_offline(self):
        pairs = [
            {"query": "python dataclass", "doc": "Use field(default_factory=list)", "score": 3},
            {"query": "bm25 retrieval", "doc": "BM25 emphasizes exact term overlap", "score": 2},
        ]
        generator = SyntheticDataGenerator()
        records = generator.generate_query_expansions(pairs, target_count=2, use_teacher=False)
        assert len(records) == 2
        for record in records:
            assert "original_query" in record
            assert "expanded_queries" in record
            assert len(record["expanded_queries"]) == 3
            assert record["generation_mode"] == "offline"

    def test_hard_negative_record_validation(self):
        valid_record = {
            "query": "test query",
            "positive": "positive doc",
            "hard_negative": "hard negative doc",
            "easy_negative": "easy negative doc",
            "generation_seed": 42,
            "generation_mode": "offline",
            "teacher_model": None,
        }
        record = HardNegativeRecord(**valid_record)
        assert record.query == "test query"
        assert record.generation_mode == "offline"

    def test_listwise_preference_record_validation(self):
        valid_record = {
            "query": "test query",
            "docs": ["doc1", "doc2", "doc3"],
            "scores": [0.5, 0.3, 0.2],
            "generation_seed": 42,
            "generation_mode": "offline",
            "teacher_model": None,
        }
        record = ListwisePreferenceRecord(**valid_record)
        assert len(record.docs) == 3
        assert len(record.scores) == 3

    def test_query_expansion_record_validation(self):
        valid_record = {
            "original_query": "test query",
            "expanded_queries": ["alt1", "alt2", "alt3"],
            "generation_seed": 42,
            "generation_mode": "offline",
            "teacher_model": None,
        }
        record = QueryExpansionRecord(**valid_record)
        assert len(record.expanded_queries) == 3

    def test_hard_negative_diversity(self):
        pairs = [
            {"query": "python dataclass", "doc": "Use field(default_factory=list)", "score": 3},
        ]
        generator = SyntheticDataGenerator()
        records = generator.generate_hard_negatives(pairs, target_count=6, use_teacher=False)
        hard_negatives = [r["hard_negative"] for r in records]
        easy_negatives = [r["easy_negative"] for r in records]
        unique_hard = set(hard_negatives)
        unique_easy = set(easy_negatives)
        assert len(unique_hard) > 1
        assert len(unique_easy) > 1

    def test_listwise_scores_descending(self):
        pairs = [
            {"query": "python dataclass", "doc": "Use field(default_factory=list)", "score": 3},
            {"query": "python dataclass", "doc": "Dataclasses reduce boilerplate", "score": 2},
            {"query": "python dataclass", "doc": "Python has many features", "score": 1},
            {"query": "python dataclass", "doc": "JavaScript is different", "score": 0},
        ]
        generator = SyntheticDataGenerator()
        records = generator.generate_listwise_preferences(pairs, target_count=1, use_teacher=False)
        if records:
            scores = records[0]["scores"]
            assert scores == sorted(scores, reverse=True)
