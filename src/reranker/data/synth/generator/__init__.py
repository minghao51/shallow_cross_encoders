"""Internal helpers backing the SyntheticDataGenerator facade."""

from reranker.data.synth.generator.artifacts import (
    distribution_report,
    materialize_all,
    refresh_metadata,
    write_distribution_summary,
    write_manifest,
)
from reranker.data.synth.generator.contradictions import iter_contradictions
from reranker.data.synth.generator.core import (
    chunk_specs,
    log_cost,
    parallel_teacher_batches,
    require_teacher,
    should_use_teacher,
    stabilize_contradiction_record,
    validate_record,
)
from reranker.data.synth.generator.enhanced import (
    iter_hard_negatives,
    iter_listwise_preferences,
    iter_query_expansions,
)
from reranker.data.synth.generator.pairs import get_expanded_seeds, iter_pairs
from reranker.data.synth.generator.preferences import apply_preference_swap, iter_preferences

__all__ = [
    "apply_preference_swap",
    "chunk_specs",
    "distribution_report",
    "get_expanded_seeds",
    "iter_contradictions",
    "iter_hard_negatives",
    "iter_listwise_preferences",
    "iter_pairs",
    "iter_preferences",
    "iter_query_expansions",
    "log_cost",
    "materialize_all",
    "parallel_teacher_batches",
    "refresh_metadata",
    "require_teacher",
    "should_use_teacher",
    "stabilize_contradiction_record",
    "validate_record",
    "write_distribution_summary",
    "write_manifest",
]
