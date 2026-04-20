
from reranker.strategies.consistency import ConsistencyEngine


def test_consistency_engine_detects_conflict() -> None:
    engine = ConsistencyEngine()
    docs = [
        "Project Atlas reports release_year as 2025. The rest of the setup is unchanged.",
        "Project Atlas reports release_year as 2026. The rest of the setup is unchanged.",
    ]
    contradictions = engine.check(engine.extract_claims(docs))
    assert contradictions


def test_consistency_engine_reranks_lower_penalty_first() -> None:
    engine = ConsistencyEngine()
    docs = [
        "Project Atlas reports release_year as 2025. The rest of the setup is unchanged.",
        "Project Atlas reports release_year as 2026. The rest of the setup is unchanged.",
        "Northwind Clinic reports screening_status as approved.",
    ]
    ranked = engine.rerank("", docs)
    assert "Northwind Clinic" in ranked[0].doc


def test_consistency_engine_extracts_structured_claim_variants() -> None:
    engine = ConsistencyEngine()
    docs = [
        "Project Atlas's release year is 2025. Northwind Clinic: screening_status = approved.",
        "For Project Atlas, release-year remains 2026; latency_ms for Model2Vec Potion-8M is 7.",
    ]

    claim_sets = engine.extract_claims(docs, ["atlas", "mixed"])

    first_claims = {
        (claim.entity, claim.attribute, str(claim.value)) for claim in claim_sets[0].claims
    }
    second_claims = {
        (claim.entity, claim.attribute, str(claim.value)) for claim in claim_sets[1].claims
    }

    assert ("Project Atlas", "release_year", "2025") in first_claims
    assert ("Northwind Clinic", "screening_status", "approved") in first_claims
    assert ("Project Atlas", "release_year", "2026") in second_claims
    assert ("Model2Vec Potion-8M", "latency", "7") in second_claims


def test_consistency_engine_detects_conflict_from_structured_heuristics() -> None:
    engine = ConsistencyEngine()
    docs = [
        "Project Atlas's release year is 2025.",
        "For Project Atlas, release_year remains 2026.",
    ]

    contradictions = engine.check(engine.extract_claims(docs))

    assert contradictions


def test_consistency_engine_normalizes_teacher_latency_variants() -> None:
    engine = ConsistencyEngine()
    docs = [
        "Testing of Model2Vec Potion-8M reveals a latency of only 2 milliseconds.",
        "The latest tests indicate that Model2Vec Potion-8M has a latency of 7 milliseconds.",
    ]

    contradictions = engine.check(engine.extract_claims(docs))

    assert contradictions


def test_consistency_engine_extracts_release_and_metric_variants() -> None:
    engine = ConsistencyEngine()
    docs = [
        "Project Atlas is set to release in 2025.",
        "The best metric for the HybridFusionReranker is NDCG@10.",
    ]

    claim_sets = engine.extract_claims(docs, ["release", "metric"])

    release_claims = {
        (claim.entity, claim.attribute, str(claim.value)) for claim in claim_sets[0].claims
    }
    metric_claims = {
        (claim.entity, claim.attribute, str(claim.value)) for claim in claim_sets[1].claims
    }

    assert ("Project Atlas", "release_year", "2025") in release_claims
    assert ("HybridFusionReranker", "best_metric", "NDCG@10") in metric_claims


def test_consistency_engine_multiple_claims_detection() -> None:
    """Test detection of contradictions across multiple claims."""
    engine = ConsistencyEngine()
    docs = [
        "Project Atlas reports release_year as 2025 and latency_ms as 5.",
        "Project Atlas reports release_year as 2026 and latency_ms as 5.",
        "Project Atlas reports release_year as 2025 and latency_ms as 10.",
    ]

    contradictions = engine.check(engine.extract_claims(docs))

    # Should detect contradictions
    assert len(contradictions) > 0


def test_consistency_engine_value_conflict_detection() -> None:
    """Test value conflict detection logic."""
    engine = ConsistencyEngine()

    # Test numeric value conflict
    assert engine._values_conflict(2025, 2026) is True
    assert engine._values_conflict(2025, 2025) is False

    # Test string value conflict
    assert engine._values_conflict("approved", "pending") is True
    assert engine._values_conflict("approved", "approved") is False

    # Test type conversion
    assert engine._values_conflict("2025", 2026) is True


def test_consistency_engine_semantic_similarity_threshold() -> None:
    """Test semantic similarity threshold for contradiction detection."""
    engine = ConsistencyEngine(sim_threshold=0.95)

    docs = [
        "Project Atlas reports release_year as 2025.",
        "Project Atlas reports release_year as 2026.",
    ]

    contradictions = engine.check(engine.extract_claims(docs))

    # Should detect contradiction with high similarity
    assert len(contradictions) > 0


def test_consistency_engine_no_contradiction_consistent_values() -> None:
    """Test that no contradictions are detected when values are consistent."""
    engine = ConsistencyEngine()
    docs = [
        "Project Atlas reports release_year as 2025.",
        "Project Atlas reports release_year as 2025.",
    ]

    contradictions = engine.check(engine.extract_claims(docs))

    # Should not detect contradictions
    assert len(contradictions) == 0
