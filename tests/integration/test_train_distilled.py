from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_train_distilled_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "train_distilled.py"
    spec = importlib.util.spec_from_file_location("train_distilled_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_should_generate_synthetic_data_accepts_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_distilled = _load_train_distilled_module()
    monkeypatch.setenv("RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA", "yes")

    assert train_distilled._should_generate_synthetic_data("Proceed?") is True


def test_should_generate_synthetic_data_defaults_to_no_without_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_distilled = _load_train_distilled_module()
    monkeypatch.delenv("RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA", raising=False)
    monkeypatch.setattr(train_distilled.sys.stdin, "isatty", lambda: False)
    prompt_calls: list[str] = []
    monkeypatch.setattr(train_distilled, "_prompt_user", lambda prompt: prompt_calls.append(prompt))

    assert train_distilled._should_generate_synthetic_data("Proceed?") is False
    assert prompt_calls == []


def test_should_generate_synthetic_data_rejects_invalid_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_distilled = _load_train_distilled_module()
    monkeypatch.setenv("RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA", "maybe")

    with pytest.raises(ValueError, match="RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA"):
        train_distilled._should_generate_synthetic_data("Proceed?")
