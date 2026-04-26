"""Safe model persistence layer.

Replaces ad-hoc pickle dumps with structured joblib + JSON metadata.
For backward compatibility, loading still supports legacy pickle files
but emits a security warning.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib

from reranker.utils import build_artifact_metadata, ensure_parent, validate_artifact_metadata

SAFE_FORMAT_VERSION = 2


def _meta_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".meta.json")


def _weights_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".weights.joblib")


def save_safe(
    path: str | Path,
    artifact_type: str,
    metadata: dict[str, Any],
    weights: dict[str, Any],
) -> None:
    """Save model weights as joblib and metadata as JSON."""
    target = Path(path)
    ensure_parent(target)

    meta = build_artifact_metadata(
        artifact_type,
        format_name="safe-joblib",
        extra={**metadata, "safe_format_version": SAFE_FORMAT_VERSION},
    )
    meta_path = _meta_path(target)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    weights_path = _weights_path(target)
    joblib.dump(weights, weights_path)

    # Create an empty marker file with the original extension so callers
    # that check path.exists() continue to work.
    target.touch(exist_ok=True)


def load_safe(
    path: str | Path,
    *,
    expected_type: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load metadata and weights from safe format."""
    target = Path(path)
    meta_path = _meta_path(target)
    weights_path = _weights_path(target)

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_artifact_metadata(meta, expected_type=expected_type, expected_formats={"safe-joblib"})

    weights: dict[str, Any] = joblib.load(weights_path)
    return meta, weights


def try_load_safe_or_warn(
    path: str | Path,
    *,
    expected_type: str,
    legacy_loader: Any,
) -> Any:
    """Attempt safe load; fall back to legacy pickle with a security warning."""
    target = Path(path)
    meta_path = _meta_path(target)
    weights_path = _weights_path(target)

    if meta_path.exists() and weights_path.exists():
        meta, weights = load_safe(target, expected_type=expected_type)
        payload = dict(meta)
        payload.update(weights)
        return payload

    warnings.warn(
        f"Loading legacy pickle artifact from {target}. "
        "Pickle files can execute arbitrary code on deserialization. "
        "Re-save the model using the current format to remove this warning.",
        UserWarning,
        stacklevel=3,
    )
    return legacy_loader(target)
