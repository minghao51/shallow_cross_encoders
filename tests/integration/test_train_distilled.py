from __future__ import annotations

import pytest

from reranker.cli import train as train_cli


def test_should_generate_synthetic_data_accepts_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA", "yes")
    assert train_cli._should_generate_synthetic_data("Proceed?") is True


def test_should_generate_synthetic_data_defaults_to_no_without_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA", raising=False)
    monkeypatch.setattr(train_cli.sys.stdin, "isatty", lambda: False)
    prompt_calls: list[str] = []
    monkeypatch.setattr(train_cli, "_prompt_user", lambda prompt: prompt_calls.append(prompt))

    assert train_cli._should_generate_synthetic_data("Proceed?") is False
    assert prompt_calls == []


def test_should_generate_synthetic_data_rejects_invalid_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA", "maybe")
    with pytest.raises(ValueError, match="RERANKER_AUTO_CONFIRM_SYNTHETIC_DATA"):
        train_cli._should_generate_synthetic_data("Proceed?")
