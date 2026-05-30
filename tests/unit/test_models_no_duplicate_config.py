"""Regression test for C-6: no duplicate model_config in _models.py."""

import pytest
from pydantic_core._pydantic_core import ValidationError

from reranker.data.synth._models import DatasetManifest


def test_dataset_manifest_valid_construction():
    manifest = DatasetManifest(
        generated_at="2026-01-01",
        root="data",
        seed=42,
        generation_mode="offline",
        datasets={},
    )
    assert manifest.generated_at == "2026-01-01"
    assert manifest.seed == 42
    assert manifest.datasets == {}


def test_dataset_manifest_rejects_extra_fields():
    with pytest.raises(ValidationError):
        DatasetManifest(
            generated_at="2026-01-01",
            root="data",
            seed=42,
            generation_mode="offline",
            datasets={},
            unexpected_field="should_fail",
        )
